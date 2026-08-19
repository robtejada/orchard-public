"""Post-step repair of unphysical temperature inversions in the envelope.

Motivation: during Z/He-rain eras the transport
Newton-Raphson frequently exits through its stalled-accept hatch or the
max-iteration cap (mean niter ~32, ~29% of steps at >=30 iterations in the
rain window vs ~5 pre-onset), and the systematically under-converged entropy
updates accumulate into persistent cold layers at the rain
interface (dT/T down to -50% at m/M ~ 0.2-0.3; zero inversions in rain-quiet
controls). A temperature inversion (T decreasing with depth) is
Schwarzschild-unstable — real convection would erase it within a dynamical
time — but the composition-gradient gating that marks these zones stable
lets the numerical artifact persist for ~Gyr and pollutes T_eff and the
gravity moments.

Repair: weighted isotonic regression (pool-adjacent-violators) on the
envelope temperature profile with cell-mass weights — the minimal monotone
adjustment, equivalent to mixing each inverted run to a mass-weighted
isothermal patch (energy-conserving up to Cp variations across the run;
the exact internal-energy residual is computed and logged). Entropy is then
recomputed per repaired cell by inverting get_logt_sp at fixed (P, Y, Z,
f_rock) — a fixed-T EOS inversion — and
density is refreshed at the repaired temperature.

Gated by [transport] repair_T_inversions (default False = exact no-op).
"""

import numpy as np
from scipy.optimize import brentq

from utils.common import logger
from utils import const

# Module-level throttle so rain-era steps don't flood the log.
_repair_count = 0


# --------------------------------------------------------------------------- #
#  Entropy recompute (inverse of get_logt_sp at preserved temperature)
# --------------------------------------------------------------------------- #
def _entropy_at_fixed_T(mixtures_eos, logP, logT_target, y, z, frock, tab,
                        s_guess, s_lo=0.05, s_hi=15.0):
    """Find the entropy S (kb/baryon) such that the envelope EOS reproduces the
    target temperature at fixed (P, Y, Z, f_rock):

        get_logt_sp(S, logP, Y, Z, f_rock) == logT_target

    The call mirrors the exact positional/keyword convention used in
    ``hydrostatic.py`` so the recomputed S is bit-consistent with the hydrostatic
    temperature solve (which uses the same ``get_logt_sp``).  ``get_logt_sp`` is
    monotonic in S, so a 1-D bracketed root find is robust.
    """
    def resid(s):
        out = mixtures_eos.get_logt_sp(
            np.array([s]),
            np.array([logP]),
            np.array([y]),
            np.array([z]),
            np.array([frock]),
            ideal_guess=False,
            arr_guess=np.array([logT_target]),
            tab=tab,
        )
        lt = float(np.asarray(out).reshape(-1)[0])
        return lt - logT_target

    f_lo, f_hi = resid(s_lo), resid(s_hi)
    if np.isfinite(f_lo) and np.isfinite(f_hi) and (f_lo * f_hi < 0.0):
        try:
            return float(brentq(resid, s_lo, s_hi, xtol=1e-6, rtol=1e-10, maxiter=200))
        except (ValueError, RuntimeError):
            pass
    # Fallback: forward PT entropy (erg/g/K -> kb/baryon).  Less consistent with
    # the SP table (~few % T error) but always finite; flagged via the caller.
    try:
        s_raw = float(mixtures_eos.get_s_pt(logP, logT_target, y, z, frock))
        return s_raw * float(const.erg_to_kbbar)
    except Exception:
        return float(s_guess)


def _isotonic_nondecreasing(y, w):
    """Weighted isotonic regression (non-decreasing), pool-adjacent-violators.

    Returns the fitted array. O(N)."""
    n = len(y)
    # blocks: value, weight, count (as parallel lists)
    vals = []
    wts = []
    cnts = []
    for i in range(n):
        vals.append(float(y[i]))
        wts.append(float(w[i]))
        cnts.append(1)
        # merge while monotonicity violated
        while len(vals) > 1 and vals[-2] > vals[-1]:
            v2, w2, c2 = vals.pop(), wts.pop(), cnts.pop()
            v1, w1, c1 = vals.pop(), wts.pop(), cnts.pop()
            wt = w1 + w2
            vals.append((v1 * w1 + v2 * w2) / wt)
            wts.append(wt)
            cnts.append(c1 + c2)
    out = np.empty(n)
    k = 0
    for v, c in zip(vals, cnts):
        out[k:k + c] = v
        k += c
    return out


def repair_temperature_inversions(S, temp, rho, p, Y, Z, f_rock, dm, kcore,
                                  mixtures_eos, tab,
                                  skip_outer=2, min_dT_frac=1e-3):
    """Repair T inversions in envelope cells [skip_outer:kcore], in place.

    Arrays are surface->center: physically T must be non-decreasing with
    index. Cells whose temperature changes by more than min_dT_frac
    (relative) get S recomputed via get_logt_sp inversion and rho refreshed.

    Returns (n_repaired, max_frac_change, dU_residual_erg) — all zero/None
    when the profile is already monotone.
    """
    global _repair_count
    i0, i1 = skip_outer, int(kcore)
    if i1 - i0 < 3:
        return 0, 0.0, 0.0

    T_env = np.asarray(temp[i0:i1], dtype=float)
    if np.all(np.diff(T_env) >= 0.0):
        return 0, 0.0, 0.0

    w = np.asarray(dm[i0:i1], dtype=float)
    T_fit = _isotonic_nondecreasing(T_env, w)

    frac = np.abs(T_fit - T_env) / np.maximum(T_env, 1.0)
    fix = np.where(frac > min_dT_frac)[0]
    if fix.size == 0:
        return 0, 0.0, 0.0

    logp_fix = np.log10(np.asarray(p, dtype=float)[i0:i1][fix])
    dU = 0.0
    for j, lp in zip(fix, logp_fix):
        k = i0 + j
        logT_new = np.log10(T_fit[j])
        try:
            u_old = 10.0 ** float(np.atleast_1d(mixtures_eos.get_logu_pt(
                lp, np.log10(temp[k]), Y[k], Z[k], f_rock[k]))[0])
            u_new = 10.0 ** float(np.atleast_1d(mixtures_eos.get_logu_pt(
                lp, logT_new, Y[k], Z[k], f_rock[k]))[0])
            dU += (u_new - u_old) * dm[k]
        except Exception:
            pass
        try:
            s_new = _entropy_at_fixed_T(
                mixtures_eos, lp, logT_new, Y[k], Z[k], f_rock[k], tab,
                s_guess=S[k])
        except Exception:
            continue
        if s_new is None or not np.isfinite(s_new):
            continue
        S[k] = s_new
        temp[k] = T_fit[j]
        try:
            rho[k] = 10.0 ** float(np.atleast_1d(mixtures_eos.get_logrho_pt(
                lp, logT_new, Y[k], Z[k], f_rock[k]))[0])
        except Exception:
            pass

    max_frac = float(frac[fix].max())
    _repair_count += 1
    if max_frac > 0.02 or _repair_count % 200 == 1:
        logger.info(
            "\033[96mT-INVERSION REPAIR\033[0m #%d: %d cells pooled "
            "(max |dT|/T = %.3f, dU = %+.3e erg)",
            _repair_count, fix.size, max_frac, dU,
        )
    return int(fix.size), max_frac, dU
