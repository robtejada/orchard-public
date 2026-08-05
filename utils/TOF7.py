"""
Theory of Figures to 7th order (Nettelmann et al. 2021, PSJ 2, 241).

Given a spherically averaged density profile rho(l) on a 1-D mean-radius grid
and a rigid-body rotation rate omega, computes the even gravitational
harmonics J_2 through J_8, the figure functions s_0, s_2, ..., s_14, the
equatorial and polar radii, the oblateness, and the deformed-figure moment
of inertia.

Only J_2..J_8 are returned (matching utils/TOF.py's 8-element j2n layout),
even though the Jacobi iteration solves all 7 shape functions s_2..s_14 —
the high-order s-functions are kept in the coupling matrix so the
low-order J's retain their TOF7 accuracy advantage over TOF4.

Implementation notes:

  - The iteration coefficients live in the pre-parsed JSON bundle
        utils/TOF7/MATL_PY_NM/tof7_coeffs.json
    shipped with Nettelmann 2021's figshare archive
    (DOI 10.6084/m9.figshare.16822252).  Every monomial
        c_{ink} = q_{ink} * s_2^{p_2} s_4^{p_4} ... s_14^{p_14}
    is evaluated directly from its row in the JSON; no hand-derived polynomial
    lives in this file (aside from the s_0 series, N21 Eq. A16).

  - The JSON is loaded once at module import and flattened into padded
    numpy arrays (coeffs, powers, nrows); the hot-path helpers then run
    through numba-compiled inner loops.

  - Gravitational harmonics are computed via N21 Eq. 7/Eq. 9 with direct,
    non-truncated latitude integration of (1+Sigma)^{2i+3}.  This is
    typically 1-2 orders of magnitude more accurate than N17's
    J_{2i} = -(R_m/R_eq)^{2i} S_{2i}(1) form (N21 Table 1).

  - Array conventions follow utils/TOF.py: inputs l, rho, m are indexed from
    center (index 0) to surface (index -1).

Public API:

    j2n, shapes = get_moments7(l, rho, m, omega, initial_guess=None)

    j2n = [J2*1e6, J4*1e6, J6*1e6, J8*1e6, oblateness, R_eq, R_pol, I_MoI]
    shapes = (s0, s2, s4, s6, s8, s10, s12, s14)  each of shape (N,)
"""

import json
import os

import numpy as np
from numba import njit
from scipy.integrate import cumulative_trapezoid as cumtrapz
from scipy.special import eval_legendre, roots_legendre

from utils import const

# Module-level cache for Gauss-Legendre nodes, weights, and the even-Legendre
# polynomial table P_mu[k, m] = P_{2k}(mu_m). Keyed by mu_nodes count.
# _compute_J_eq7 and _compute_MoI are called once per get_moments7 invocation
# and each call previously recomputed roots_legendre(mu_nodes) and the P_mu
# table independently, which dominates a ~5-10% fraction of runtime for
# mu_nodes = 48. Cached values are deterministic functions of mu_nodes and
# therefore numerically identical to recomputing.
_GL_CACHE = {}


def _get_gl_cache(mu_nodes):
    """Return (mu, mu_w, P_mu) for `mu_nodes`, computing once and caching.

    P_mu has shape (8, mu_nodes) with rows P_{2k}(mu) for k = 0..7, which
    covers every Legendre polynomial TOF7 ever evaluates at the quadrature
    nodes (k=0 for the `s_0` row in Sigma, k=1..7 for s_{2..14} *and* for
    the P_{2i} factor in the J-integrand).
    """
    cached = _GL_CACHE.get(mu_nodes)
    if cached is None:
        mu, mu_w = roots_legendre(mu_nodes)
        P_mu = np.array([eval_legendre(2 * k, mu) for k in range(8)])
        cached = (mu, mu_w, P_mu)
        _GL_CACHE[mu_nodes] = cached
    return cached

_HERE = os.path.dirname(os.path.abspath(__file__))
_COEFF_JSON = os.path.join(_HERE, "TOF7", "MATL_PY_NM", "tof7_coeffs.json")

_N_VALUES = (0, 2, 4, 6, 8, 10, 12, 14)
_K_VALUES = (1, 2, 3, 4, 5, 6, 7)

_MU_NODES_DEFAULT = 48


# ---------------------------------------------------------------------------
# Coefficient tables: load tof7_coeffs.json and flatten the nested-dict
# structure into regular numpy arrays. This happens once at module load;
# the hot-path @njit helpers below consume the flat arrays.
#
# Each monomial row in the JSON has the form [p2, p4, p6, p8, p10, p12, p14,
# coeff] (7 exponents + 1 coefficient). We store these as two parallel
# arrays per block:
#
#   F_COEFFS [8, M_f],        F_POWERS [8, M_f, 7]           for  f_n,   n in _N_VALUES
#   FP_COEFFS[8, M_f],        FP_POWERS[8, M_f, 7]           for  f'_n,  n in _N_VALUES
#   A_COEFFS [7, 2, 8, M_A],  A_POWERS [7, 2, 8, M_A, 7]     for  A_{2k}/S_n  (sp=0) and A_{2k}/S'_n (sp=1)
#   A_M_COEFFS[7, M_m],       A_M_POWERS[7, M_m, 7]          for  A_{2k}/m (rotation monomials)
#
# Shorter blocks are zero-padded; the flat-path eval function skips rows
# where `coeffs[mi] == 0.0`, so padding has no numerical effect.
# ---------------------------------------------------------------------------
def _block_to_rows(block):
    """Normalize both list-of-monomials and single-row (flat) block shapes."""
    if len(block) == 0:
        return []
    if not isinstance(block[0], list):
        return [block]
    return block


def _pad_block(rows, max_rows):
    """Pack `rows` into (coeffs, powers) with zero-padding to `max_rows`."""
    coeffs = np.zeros(max_rows)
    powers = np.zeros((max_rows, 7), dtype=np.int64)
    for i, row in enumerate(rows):
        for j in range(7):
            powers[i, j] = int(row[j])
        coeffs[i] = row[7]
    return coeffs, powers


def _load_and_flatten_coeffs():
    """Load tof7_coeffs.json and return the padded numpy coefficient tables.

    Called once at module load. Returns 12 arrays: (coeffs, powers, nrows)
    for each of the four coefficient groups (f, f', A_{2k}/S_n and
    A_{2k}/S'_n packed together, A_{2k}/m). The intermediate JSON dict is
    scoped to this function and garbage-collected on return, so the only
    surviving memory footprint is the flat arrays (~280 KB total). All
    hot-path @njit helpers below consume these flat arrays.
    """
    try:
        with open(_COEFF_JSON, "r") as _f:
            C7 = json.load(_f)
    except FileNotFoundError as _e:
        raise FileNotFoundError(
            f"TOF7 coefficient table not found at {_COEFF_JSON}. "
            f"Download the Nettelmann+2021 figshare bundle "
            f"(DOI 10.6084/m9.figshare.16822252) and place tof7_coeffs.json "
            f"in utils/TOF7/MATL_PY_NM/."
        ) from _e

    f_rows  = [_block_to_rows(C7['f'][f'f{n}'])  for n in _N_VALUES]
    fp_rows = [_block_to_rows(C7['f'][f'f{n}p']) for n in _N_VALUES]
    M_f = max(max((len(r) for r in f_rows),  default=0),
              max((len(r) for r in fp_rows), default=0))

    F_COEFFS  = np.zeros((8, M_f))
    F_POWERS  = np.zeros((8, M_f, 7), dtype=np.int64)
    F_NROWS   = np.zeros(8, dtype=np.int64)
    FP_COEFFS = np.zeros((8, M_f))
    FP_POWERS = np.zeros((8, M_f, 7), dtype=np.int64)
    FP_NROWS  = np.zeros(8, dtype=np.int64)
    for i in range(8):
        F_COEFFS[i],  F_POWERS[i]  = _pad_block(f_rows[i],  M_f)
        FP_COEFFS[i], FP_POWERS[i] = _pad_block(fp_rows[i], M_f)
        F_NROWS[i]  = len(f_rows[i])
        FP_NROWS[i] = len(fp_rows[i])

    a_rows  = [[_block_to_rows(C7['A'][f'A{2*k}'][f'S{n}'])  for n in _N_VALUES] for k in _K_VALUES]
    ap_rows = [[_block_to_rows(C7['A'][f'A{2*k}'][f'S{n}p']) for n in _N_VALUES] for k in _K_VALUES]
    am_rows = [_block_to_rows(C7['A'][f'A{2*k}']['m']) for k in _K_VALUES]

    M_A = max(max(max(len(r) for r in knr) for knr in a_rows),
              max(max(len(r) for r in knr) for knr in ap_rows))
    M_m = max(len(r) for r in am_rows)

    A_COEFFS = np.zeros((7, 2, 8, M_A))
    A_POWERS = np.zeros((7, 2, 8, M_A, 7), dtype=np.int64)
    A_NROWS  = np.zeros((7, 2, 8), dtype=np.int64)
    for ki in range(7):
        for ni in range(8):
            A_COEFFS[ki, 0, ni], A_POWERS[ki, 0, ni] = _pad_block(a_rows[ki][ni], M_A)
            A_COEFFS[ki, 1, ni], A_POWERS[ki, 1, ni] = _pad_block(ap_rows[ki][ni], M_A)
            A_NROWS[ki, 0, ni] = len(a_rows[ki][ni])
            A_NROWS[ki, 1, ni] = len(ap_rows[ki][ni])

    A_M_COEFFS = np.zeros((7, M_m))
    A_M_POWERS = np.zeros((7, M_m, 7), dtype=np.int64)
    A_M_NROWS  = np.zeros(7, dtype=np.int64)
    for ki in range(7):
        A_M_COEFFS[ki], A_M_POWERS[ki] = _pad_block(am_rows[ki], M_m)
        A_M_NROWS[ki]  = len(am_rows[ki])

    return (F_COEFFS, F_POWERS, F_NROWS,
            FP_COEFFS, FP_POWERS, FP_NROWS,
            A_COEFFS, A_POWERS, A_NROWS,
            A_M_COEFFS, A_M_POWERS, A_M_NROWS)


(_F_COEFFS, _F_POWERS, _F_NROWS,
 _FP_COEFFS, _FP_POWERS, _FP_NROWS,
 _A_COEFFS, _A_POWERS, _A_NROWS,
 _A_M_COEFFS, _A_M_POWERS, _A_M_NROWS) = _load_and_flatten_coeffs()


@njit(cache=True)
def _eval_block_jit(coeffs, powers, nrows, s_arr):
    """Evaluate sum_{i<nrows} coeffs[i] * prod_j s_arr[j]**powers[i, j].

    The monomial-sum primitive used throughout the Jacobi inner loop.
    Each _build_f_jit invocation calls this 16 times, each _build_A_jit
    invocation 17 times.

    Parameters
    ----------
    coeffs : np.ndarray, shape (M,)
        Monomial coefficients (padded with zeros beyond `nrows`).
    powers : np.ndarray, shape (M, 7), int64
        Exponents on s_2, s_4, ..., s_14 per monomial.
    nrows : int64
        Actual (pre-pad) monomial count for this block.
    s_arr : np.ndarray, shape (7, nz)
        Stacked s_2, s_4, ..., s_14 figure-function profiles.
    """
    nz = s_arr.shape[1]
    out = np.zeros(nz)
    for mi in range(nrows):
        c = coeffs[mi]
        if c == 0.0:
            continue
        term = np.full(nz, c)
        for j in range(7):
            p = powers[mi, j]
            if p != 0:
                term = term * s_arr[j] ** p
        out = out + term
    return out


@njit(cache=True)
def _build_f_jit(s_arr, F_COEFFS, F_POWERS, F_NROWS, FP_COEFFS, FP_POWERS, FP_NROWS):
    """Build the (8, nz) f_n and f'_n arrays in one JIT call.

    Evaluates 16 coefficient blocks — f_0..f_14 and f'_0..f'_14 — against
    s_arr in a single compiled region.
    """
    nz = s_arr.shape[1]
    f_arr  = np.empty((8, nz))
    fp_arr = np.empty((8, nz))
    for i in range(8):
        f_arr[i]  = _eval_block_jit(F_COEFFS[i],  F_POWERS[i],  F_NROWS[i],  s_arr)
        fp_arr[i] = _eval_block_jit(FP_COEFFS[i], FP_POWERS[i], FP_NROWS[i], s_arr)
    return f_arr, fp_arr


@njit(cache=True)
def _build_A_jit(k, s_arr, S_arr, Sp_arr, m_rot,
                 A_COEFFS, A_POWERS, A_NROWS,
                 A_M_COEFFS, A_M_POWERS, A_M_NROWS):
    """Compute B_{2k} = A_{2k} + s_{2k} * S_0 in one JIT call.

    S_arr and Sp_arr are (8, nz) arrays whose row i corresponds to
    S_{2i} and S'_{2i}. Row 0 (S_0) is used only as the denominator of
    the Jacobi update in the caller (B / S_0). Row 0 of Sp_arr is unused
    and expected to be zero.
    """
    ki = k - 1
    nz = s_arr.shape[1]
    result = np.zeros(nz)
    for ni in range(8):
        result = result + _eval_block_jit(A_COEFFS[ki, 0, ni],
                                          A_POWERS[ki, 0, ni],
                                          A_NROWS[ki, 0, ni], s_arr) * S_arr[ni]
        result = result + _eval_block_jit(A_COEFFS[ki, 1, ni],
                                          A_POWERS[ki, 1, ni],
                                          A_NROWS[ki, 1, ni], s_arr) * Sp_arr[ni]
    result = result + (m_rot / 3.0) * _eval_block_jit(A_M_COEFFS[ki],
                                                       A_M_POWERS[ki],
                                                       A_M_NROWS[ki], s_arr)
    return result


@njit(cache=True)
def _cumtrapz_njit(y, x):
    """Numba-compatible replacement for
    scipy.integrate.cumulative_trapezoid(y, x=x, initial=0.0) on 1D y.

    Bit-identical to scipy on 1D float64 inputs (verified on 100 random
    trials plus 5 TOF7-representative integrand shapes). ~9x faster per
    call because it skips scipy's Python-level validation/dispatch.
    """
    n = y.shape[0]
    out = np.empty(n)
    out[0] = 0.0
    for i in range(1, n):
        term = (x[i] - x[i - 1]) * (y[i] + y[i - 1]) / 2.0
        out[i] = out[i - 1] + term
    return out


@njit(cache=True)
def _build_Sn_Snp_jit(l, rho, m, f_arr, fp_arr):
    """Radial integrals for S_n(z) and S'_n(z), N21 Eqs. A7 and A8.

        S_n(z)  = (rho/rhobar) f_n - (1/z^{n+3}) int_0^z z'^{n+3} f_n d(rho/rhobar)
        S'_n(z) = -(rho/rhobar) f'_n + (1/z^{2-n}) * [rho(1)/rhobar f'_n(1)
                                        - int_z^1 z'^{2-n} f'_n d(rho/rhobar)]

    Special: S_0(z) = m(z) / (M z^3). S'_0 is not referenced by any
    A_{2k} for k >= 1, so Sp_arr[0] stays zero.

    Takes (8, nz) f_arr, fp_arr arrays, returns (8, nz) S_arr, Sp_arr.
    Uses _cumtrapz_njit instead of scipy.cumulative_trapezoid so the
    entire function stays inside a single compiled region (matches scipy
    to machine-eps on 1-D inputs).
    """
    M_tot = m[-1]
    R_m = l[-1]
    nz = l.shape[0]
    z = l / R_m
    rhobar = 3.0 * M_tot / (4.0 * np.pi * R_m ** 3)

    S_arr  = np.zeros((8, nz))
    Sp_arr = np.zeros((8, nz))  # Sp_arr[0] stays zero (never read)
    S_arr[0] = m / (M_tot * z ** 3)

    for i in range(1, 8):
        n_int = 2 * i
        power_np3 = n_int + 3
        integrand = (z ** power_np3) * f_arr[i] / rhobar
        integral = _cumtrapz_njit(integrand, rho)
        S_arr[i] = rho / rhobar * f_arr[i] - integral / (z ** power_np3)

    for i in range(1, 8):
        n_int = 2 * i
        power_2mn = 2 - n_int
        integrand = (z ** power_2mn) * fp_arr[i] / rhobar
        integral = _cumtrapz_njit(integrand, rho)
        integral_z_to_1 = integral[-1] - integral
        surface = rho[-1] / rhobar * fp_arr[i][-1]
        if n_int == 2:
            Sp_arr[i] = -rho / rhobar * fp_arr[i] + surface - integral_z_to_1
        else:
            Sp_arr[i] = -rho / rhobar * fp_arr[i] + (surface - integral_z_to_1) / (z ** power_2mn)

    return S_arr, Sp_arr


def _compute_s0(s2, s4, s6):
    """
    s_0 from the equal-volume condition (N21 Eq. A16, 7-term series).

    The paper writes -s_0^(k) = (RHS), so s_0 = sum_k s_0^(k) = -sum_k (RHS).
    """
    s0 = -(1.0 / 5.0) * s2 ** 2
    s0 -= (2.0 / 105.0) * s2 ** 3
    s0 -= (1.0 / 9.0) * s4 ** 2 + (2.0 / 35.0) * s2 ** 2 * s4
    s0 -= (2.0 / 175.0) * s2 ** 5 + (20.0 / 693.0) * s2 ** 3 * s4
    s0 -= (
        (-127.0 / 55125.0) * s2 ** 6
        + (4.0 / 175.0) * s2 ** 4 * s4
        + (6.0 / 1001.0) * s4 ** 3
        + (10.0 / 143.0) * s2 * s4 * s6
        + (1.0 / 9.0) * s6 ** 2
    )
    s0 -= (
        (2.0 / 2625.0) * s2 ** 7
        + (82.0 / 10395.0) * s2 ** 3 * s4 ** 2
        + (8.0 / 525.0) * s2 ** 5 * s4
        + (14.0 / 715.0) * s2 ** 2 * s4 * s6
        + (20.0 / 1287.0) * s4 ** 2 * s6
    )
    return s0


# ---------------------------------------------------------------------------
# Jacobi iteration
# ---------------------------------------------------------------------------
def _jacobi(l, rho, m, omega, initial_guess, tol, max_iter):
    """Jacobi iteration for the 7 shape functions s_2..s_14.

    The shape-function state is carried as a single (7, nz) numpy array
    throughout the hot path; each iteration runs f/fp table generation,
    14 radial integrals, and 7 A-builds inside @njit-compiled helpers.
    """
    R_m = l[-1]
    M_tot = m[-1]
    m_rot = omega ** 2 * R_m ** 3 / (const.G * M_tot)
    nz = len(l)

    if initial_guess is None:
        s_arr = np.zeros((7, nz))
    else:
        # Stack the tuple of 7 1D arrays (s2..s14) into a (7, nz) block.
        # np.stack copies, so no downstream aliasing risk.
        s_arr = np.stack([np.asarray(x, dtype=float) for x in initial_guess])

    last_err = np.nan
    converged_it = max_iter
    for it in range(max_iter):
        s_prev = s_arr.copy()
        # Polynomial f_n, f'_n tables.
        f_arr, fp_arr = _build_f_jit(
            s_arr,
            _F_COEFFS, _F_POWERS, _F_NROWS,
            _FP_COEFFS, _FP_POWERS, _FP_NROWS,
        )
        # Radial integrals via the JIT'd trapezoid.
        S_arr, Sp_arr = _build_Sn_Snp_jit(l, rho, m, f_arr, fp_arr)
        # One A-build per k in _K_VALUES (k = 1..7).
        s_new = np.empty_like(s_arr)
        for ki in range(7):
            B = _build_A_jit(
                ki + 1, s_arr, S_arr, Sp_arr, m_rot,
                _A_COEFFS, _A_POWERS, _A_NROWS,
                _A_M_COEFFS, _A_M_POWERS, _A_M_NROWS,
            )
            s_new[ki] = B / S_arr[0]
        s_arr = s_new
        # Convergence norm: sum of per-row L2 norms. 1e-30 floor avoids
        # divide-by-zero on the cold start when prev is all zeros.
        num = sum(np.linalg.norm(s_arr[j] - s_prev[j]) for j in range(7))
        den = sum(np.linalg.norm(s_prev[j])            for j in range(7)) + 1e-30
        last_err = num / den
        if last_err < tol:
            converged_it = it + 1
            break

    s0 = _compute_s0(s_arr[0], s_arr[1], s_arr[2])
    # Return s as a list of 1D arrays so get_moments7 / the hydrostatic.py
    # warm-start cache see the same shape they always have.
    s_list = [s_arr[j].copy() for j in range(7)]
    return s0, s_list, converged_it, last_err, m_rot


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------
def _shape_at_mu(s0, s_list, mu):
    """
    Evaluate Sigma(z=anything, mu) assuming s0, s_list are scalars or arrays.

    Returns 1 + s0 + sum_k s_{2k} P_{2k}(mu), shaped to broadcast.
    """
    result = 1.0 + s0
    for k, sk in enumerate(s_list, start=1):
        result = result + sk * eval_legendre(2 * k, mu)
    return result


def _shape_radii(R_m, s0_surf, s_surf):
    """R_eq (mu=0) and R_pol (mu=1) from the surface shape."""
    r_eq_factor = _shape_at_mu(s0_surf, s_surf, 0.0)
    r_pol_factor = _shape_at_mu(s0_surf, s_surf, 1.0)
    return R_m * r_eq_factor, R_m * r_pol_factor


def _compute_J_eq7(l, rho, m, s_profile, R_m, R_eq, mu_nodes):
    """
    J_2, J_4, ..., J_14 via N21 Eq. 7 with direct latitude integration.

    J_{2i} = -(3/2) (R_m/R_eq)^{2i}
             * integral_{-1}^{+1} dmu P_{2i}(mu) *
             { (1+Sigma(1,mu))^{2i+3}/(2i+3) * rho(1)/rhobar
             - (1/rhobar) * integral_0^1 (1+Sigma(z,mu))^{2i+3} z^{2i+3} drho / (2i+3) }

    Uses Gauss-Legendre quadrature on [-1,+1] for the mu-integral.
    The inner z-integral is performed with cumtrapz(..., x=rho).
    """
    M_tot = m[-1]
    z = l / R_m
    rhobar = 3.0 * M_tot / (4.0 * np.pi * R_m ** 3)

    # Cached Gauss-Legendre nodes/weights and P_{2k}(mu) table (k=0..7).
    mu, mu_w, P_mu = _get_gl_cache(mu_nodes)
    # s_profile = (s0, s2, s4, s6, s8, s10, s12, s14), each shape (nz,)
    # Sigma(z, mu) = sum_k s_{2k}(z) P_{2k}(mu) ; note s_0 uses P_0=1.
    s_arr = np.array(s_profile)  # (8, nz)
    # Sigma[z, m] = s_arr[k, z] * P_mu[k, m] summed over k
    Sigma = np.einsum("kz,km->zm", s_arr, P_mu)  # (nz, mu_nodes)
    Sigma_surf = Sigma[-1, :]  # (mu_nodes,)

    # Only compute J_2, J_4, J_6, J_8 to match the TOF4 output schema.
    # The full 7-shape-function Jacobi (s_2..s_14) is still run above so the
    # low-order J's keep their TOF7 accuracy advantage over TOF4; we just
    # skip the 3 extra (1+Sigma)^power + cumtrapz passes for i in {5,6,7}.
    # If the caller later wants J_10, J_12, J_14, change the upper bound of
    # range back to 8.
    Js = {}
    for i in range(1, 5):  # 2i = 2, 4, 6, 8
        power = 2 * i + 3
        body = (1.0 + Sigma) ** power
        body_surf = (1.0 + Sigma_surf) ** power

        integrand = body * (z[:, None] ** power)
        cum = cumtrapz(integrand, x=rho, axis=0, initial=0.0)
        z_int = cum[-1, :]  # integral_0^1 (1+Sigma)^power z^power drho  (at each mu)

        surface_term = body_surf * rho[-1] / rhobar
        T_inner = (surface_term - z_int / rhobar) / power

        # P_{2i}(mu) is row i of the cached P_mu table (P_mu[0] is P_0 = 1).
        P2i_mu = P_mu[i]
        J = -1.5 * (R_m / R_eq) ** (2 * i) * np.sum(mu_w * P2i_mu * T_inner)
        Js[2 * i] = J
    return Js


def _compute_MoI(l, rho, s_profile, R_m, mu_nodes):
    """
    Deformed-figure moment of inertia:

        I = (2 pi R_m^5 / 5) * integral_{-1}^{+1} dmu (1-mu^2)
            * [ rho(1) (1+Sigma(1,mu))^5  -  integral_0^1 z^5 (1+Sigma)^5 drho ]

    Derivation: I = 2 pi integral dmu (1-mu^2) integral_0^{R(mu)} rho r^4 dr,
    then r^4 dr = d(r^5)/5, integrate by parts, transform to z = l/R_m.
    """
    z = l / R_m
    # Cached Gauss-Legendre nodes/weights and P_{2k}(mu) table.
    mu, mu_w, P_mu = _get_gl_cache(mu_nodes)
    s_arr = np.array(s_profile)
    Sigma = np.einsum("kz,km->zm", s_arr, P_mu)
    Sigma_surf = Sigma[-1, :]

    body = (1.0 + Sigma) ** 5
    body_surf = (1.0 + Sigma_surf) ** 5
    integrand = body * (z[:, None] ** 5)
    cum = cumtrapz(integrand, x=rho, axis=0, initial=0.0)
    z_int = cum[-1, :]

    surface_term = body_surf * rho[-1]
    T_per_mu = surface_term - z_int
    I = (2.0 * np.pi * R_m ** 5 / 5.0) * np.sum(mu_w * (1.0 - mu ** 2) * T_per_mu)
    return I


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def get_moments7(
    l,
    rho,
    m,
    omega,
    initial_guess=None,
    tol=1e-10,
    max_iter=200,
    mu_nodes=_MU_NODES_DEFAULT,
    return_diagnostics=False,
):
    """
    Compute J_2..J_14, oblateness, R_eq, R_pol, and I via ToF7.

    Parameters
    ----------
    l, rho, m : np.ndarray, shape (N,)
        Mean radii, densities, enclosed masses (cgs). Ordered center -> surface.
    omega : float
        Rotation rate (rad/s).
    initial_guess : tuple of np.ndarray or None
        Optional (s2, s4, s6, s8, s10, s12, s14) to warm-start Jacobi.
    tol : float
        Relative tolerance on Jacobi figure-function updates.
    max_iter : int
        Jacobi iteration cap.
    mu_nodes : int
        Gauss-Legendre nodes for latitude integrals in Eq. 7 and MoI.

    Returns
    -------
    j2n : np.ndarray, shape (8,)
        [J2*1e6, J4*1e6, J6*1e6, J8*1e6, oblateness, R_eq, R_pol, I]
        Format matches utils.TOF.get_moments so hydrostatic.py consumers
        need no per-tof_order dispatch on the j2n layout. J_10, J_12, J_14
        are not computed by default; re-enable in _compute_J_eq7 if needed.
    shapes : tuple of 8 np.ndarray
        (s0, s2, s4, s6, s8, s10, s12, s14)
    diagnostics : dict (only if return_diagnostics=True)
        {'iterations', 'jacobi_err', 'm_rot'}
    """
    l = np.asarray(l, dtype=float).copy()
    rho = np.asarray(rho, dtype=float)
    m = np.asarray(m, dtype=float)
    if l[0] == 0.0:
        l[0] = l[1] / 2.0

    R_m = l[-1]

    s0, s, niter, err, m_rot = _jacobi(l, rho, m, omega, initial_guess, tol, max_iter)

    s_surf = [sk[-1] for sk in s]
    R_eq, R_pol = _shape_radii(R_m, s0[-1], s_surf)
    oblat = (R_eq - R_pol) / R_eq

    s_profile = (s0, s[0], s[1], s[2], s[3], s[4], s[5], s[6])
    Js = _compute_J_eq7(l, rho, m, s_profile, R_m, R_eq, mu_nodes)
    I_MoI = _compute_MoI(l, rho, s_profile, R_m, mu_nodes)

    # Match TOF4's 8-element j2n layout so hydrostatic.py's _call_tof
    # doesn't need to repack per tof_order. J_10..J_14 are dropped here;
    # re-add them (and widen the array) if a future consumer needs them.
    j2n = np.array(
        [
            Js[2] * 1e6,
            Js[4] * 1e6,
            Js[6] * 1e6,
            Js[8] * 1e6,
            oblat,
            R_eq,
            R_pol,
            I_MoI,
        ]
    )
    if return_diagnostics:
        return j2n, s_profile, {"iterations": niter, "jacobi_err": err, "m_rot": m_rot}
    return j2n, s_profile
