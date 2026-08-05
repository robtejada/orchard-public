from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional

import numpy as np
from scipy.interpolate import interp1d


# Unit conversions
GPA_TO_DYN_CM2 = 1.0e10
W_MK_TO_CGS = 1.0e5  # W/(m K) -> erg/(s cm K)

# Pressure-domain blending: keep French (2012) exactly up to the top of its table,
# then transition smoothly to the Becker (2018) composite over this factor in P.
FRENCH_TO_BECKER_BLEND_FACTOR = 3.0

MM2S_TO_CM2S = 1.0e-2  # mm^2/s -> cm^2/s

# French et al. (2012) J11-8a envelope helium mass fractions.
Y_FRENCH_OUTER = 0.238   # P < 300 GPa (Galileo probe)
Y_FRENCH_INNER = 0.278   # P >= 300 GPa (protosolar constraint)
P_FRENCH_Y_BOUNDARY_GPA = 300.0


DATA_DIR = Path(__file__).resolve().parent / "data"


@dataclass(frozen=True)
class PressureAdiabat:
    name: str
    p_gpa: np.ndarray
    lambda_wmk: np.ndarray
    _loglog_interp_obj: interp1d = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        object.__setattr__(
            self,
            "_loglog_interp_obj",
            interp1d(
                np.log10(self.p_gpa),
                np.log10(self.lambda_wmk),
                kind="linear",
                fill_value="extrapolate",
                assume_sorted=True,
            ),
        )

    @property
    def pmin_gpa(self) -> float:
        return float(np.min(self.p_gpa))

    @property
    def pmax_gpa(self) -> float:
        return float(np.max(self.p_gpa))

    @property
    def pchar_gpa(self) -> float:
        # Characteristic pressure for pressure-only interpolation between adiabats.
        return float(np.sqrt(self.pmin_gpa * self.pmax_gpa))

    def eval_wmk(self, p_gpa: np.ndarray | float, *, clip: bool = False) -> np.ndarray | float:
        p = np.asarray(p_gpa, dtype=float)
        p_eval = np.clip(p, self.pmin_gpa, self.pmax_gpa) if clip else p
        log10p = np.log10(np.clip(p_eval, np.finfo(float).tiny, None))
        lam = 10.0 ** np.asarray(self._loglog_interp_obj(log10p), dtype=float)
        return _match_shape(lam, p_gpa)

    def in_range_mask(self, p_gpa: np.ndarray | float) -> np.ndarray:
        p = np.asarray(p_gpa, dtype=float)
        return (p >= self.pmin_gpa) & (p <= self.pmax_gpa)


@dataclass(frozen=True)
class FrenchPressureConductivity:
    p_gpa_sorted: np.ndarray
    lambda_wmk_sorted: np.ndarray
    _loglog_interp_obj: interp1d = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        object.__setattr__(
            self,
            "_loglog_interp_obj",
            interp1d(
                np.log10(self.p_gpa_sorted),
                np.log10(self.lambda_wmk_sorted),
                kind="linear",
                fill_value="extrapolate",
                assume_sorted=True,
            ),
        )

    @property
    def pmin_gpa(self) -> float:
        return float(self.p_gpa_sorted[0])

    @property
    def pmax_gpa(self) -> float:
        return float(self.p_gpa_sorted[-1])

    def eval_wmk(self, p_gpa: np.ndarray | float) -> np.ndarray | float:
        p = np.asarray(p_gpa, dtype=float)
        log10p = np.log10(np.clip(p, np.finfo(float).tiny, None))
        lam = 10.0 ** np.asarray(self._loglog_interp_obj(log10p), dtype=float)
        return _match_shape(lam, p_gpa)


@dataclass(frozen=True)
class PreisingAdiabat:
    """Preising et al. (2023) MF20 Saturn model: λ(P), Y(P), D_H(P), D_He(P)."""

    lam_adiabat: PressureAdiabat
    y_of_p: interp1d       # Y(logP) interpolator
    dh_of_p: interp1d      # D_H(logP) interpolator [mm^2/s]
    dhe_of_p: interp1d     # D_He(logP) interpolator [mm^2/s]
    pmin_gpa: float
    pmax_gpa: float

    def eval_y(self, p_gpa: np.ndarray | float) -> np.ndarray | float:
        p = np.asarray(p_gpa, dtype=float)
        log10p = np.log10(np.clip(p, np.finfo(float).tiny, None))
        return _match_shape(np.asarray(self.y_of_p(log10p), dtype=float), p_gpa)

    def eval_dh_mm2s(self, p_gpa: np.ndarray | float) -> np.ndarray | float:
        p = np.asarray(p_gpa, dtype=float)
        log10p = np.log10(np.clip(p, np.finfo(float).tiny, None))
        return _match_shape(10.0 ** np.asarray(self.dh_of_p(log10p), dtype=float), p_gpa)

    def eval_dhe_mm2s(self, p_gpa: np.ndarray | float) -> np.ndarray | float:
        p = np.asarray(p_gpa, dtype=float)
        log10p = np.log10(np.clip(p, np.finfo(float).tiny, None))
        return _match_shape(10.0 ** np.asarray(self.dhe_of_p(log10p), dtype=float), p_gpa)


def _match_shape(arr: np.ndarray, template: np.ndarray | float) -> np.ndarray | float:
    t = np.asarray(template)
    return arr.item() if t.ndim == 0 else arr


def _smoothstep01(x: np.ndarray) -> np.ndarray:
    xx = np.clip(x, 0.0, 1.0)
    return xx * xx * (3.0 - 2.0 * xx)


def _blend_logspace(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> np.ndarray:
    tiny = np.finfo(float).tiny
    la = np.log(np.clip(a, tiny, None))
    lb = np.log(np.clip(b, tiny, None))
    return np.exp((1.0 - w) * la + w * lb)


def _numeric_table(path: Path) -> np.ndarray:
    return np.genfromtxt(path, comments="#")


def _structured_table(path: Path):
    return np.genfromtxt(path, dtype=None, encoding=None, comments="#")


def _load_french_curves() -> FrenchPressureConductivity:
    # French et al. (2012), Jupiter model J11-8a.
    # Table 1 provides P(R); Table 2 provides lambda(R). We map table-2 radii onto pressure.
    french1 = _numeric_table(DATA_DIR / "French_table1.txt")
    french2 = _numeric_table(DATA_DIR / "French_table2.txt")

    r_f1 = french1[:, 0]
    p_f1_gpa = french1[:, 3]
    r_f2 = french2[:, 0]
    lambda_f2_wmk = french2[:, 3]

    p_of_r = interp1d(r_f1, p_f1_gpa, kind="cubic", fill_value="extrapolate", assume_sorted=True)
    p_f2_gpa = np.asarray(p_of_r(r_f2), dtype=float)

    # Sort ascending in pressure for interpolation.
    idx = np.argsort(p_f2_gpa)
    p_sorted = p_f2_gpa[idx]
    lam_sorted = lambda_f2_wmk[idx]
    return FrenchPressureConductivity(p_sorted, lam_sorted)


def _split_becker_adiabats() -> Dict[str, PressureAdiabat]:
    # Becker et al. (2018): table1 gives P,T,... per object; table2 gives lambda,... per object.
    t1 = _structured_table(DATA_DIR / "becker_table1.txt")
    t2 = _structured_table(DATA_DIR / "becker_table2.txt")

    names1 = np.asarray(t1["f0"])
    names2 = np.asarray(t2["f0"])
    objects = sorted(set(names1.tolist()))

    adiabats: Dict[str, PressureAdiabat] = {}
    for obj in objects:
        m1 = names1 == obj
        m2 = names2 == obj
        if np.count_nonzero(m1) != np.count_nonzero(m2):
            raise ValueError(
                f"Becker table row-count mismatch for {obj}: table1={np.count_nonzero(m1)}, "
                f"table2={np.count_nonzero(m2)}"
            )

        # Column mapping from screenshot / machine-readable rows:
        # table1: Object, r, m, P(GPa), T(K), rho, ...
        # table2: Object, r, m, T(K), sigma, beta, lambda(W/m/K), ...
        p_gpa = np.asarray(t1["f3"][m1], dtype=float)
        lam_wmk = np.asarray(t2["f6"][m2], dtype=float)

        # Sanity-check row alignment by (r, m, T) triplets.
        r1 = np.asarray(t1["f1"][m1], dtype=float)
        mobj1 = np.asarray(t1["f2"][m1], dtype=float)
        t1k = np.asarray(t1["f4"][m1], dtype=float)
        r2 = np.asarray(t2["f1"][m2], dtype=float)
        mobj2 = np.asarray(t2["f2"][m2], dtype=float)
        t2k = np.asarray(t2["f3"][m2], dtype=float)
        if not (np.allclose(r1, r2, rtol=0, atol=5e-4) and np.allclose(mobj1, mobj2, rtol=0, atol=5e-6) and np.allclose(t1k, t2k, rtol=0, atol=1.0)):
            raise ValueError(f"Becker table1/table2 rows do not align for {obj}.")

        idx = np.argsort(p_gpa)
        adiabats[obj] = PressureAdiabat(name=obj, p_gpa=p_gpa[idx], lambda_wmk=lam_wmk[idx])

    return adiabats


def _load_preising_adiabat() -> PreisingAdiabat:
    """Load Preising et al. (2023) MF20 Saturn model data."""
    data = _numeric_table(DATA_DIR / "preising_saturn_mf20.txt")
    # Columns: R_RS, P_GPa, lambda, Y_He, D_H, D_He
    p_gpa = data[:, 1]
    lam_wmk = data[:, 2]
    y_he = data[:, 3]
    dh_mm2s = data[:, 4]
    dhe_mm2s = data[:, 5]

    idx = np.argsort(p_gpa)
    p_sorted = p_gpa[idx]
    lam_sorted = lam_wmk[idx]
    y_sorted = y_he[idx]
    dh_sorted = dh_mm2s[idx]
    dhe_sorted = dhe_mm2s[idx]

    log10p = np.log10(p_sorted)
    lam_adi = PressureAdiabat(name="Preising-Saturn-MF20", p_gpa=p_sorted, lambda_wmk=lam_sorted)
    y_interp = interp1d(log10p, y_sorted, kind="linear", fill_value=(y_sorted[0], y_sorted[-1]), bounds_error=False)
    dh_interp = interp1d(log10p, np.log10(dh_sorted), kind="linear", fill_value="extrapolate")
    dhe_interp = interp1d(log10p, np.log10(dhe_sorted), kind="linear", fill_value="extrapolate")

    return PreisingAdiabat(
        lam_adiabat=lam_adi,
        y_of_p=y_interp,
        dh_of_p=dh_interp,
        dhe_of_p=dhe_interp,
        pmin_gpa=float(p_sorted[0]),
        pmax_gpa=float(p_sorted[-1]),
    )


@lru_cache(maxsize=1)
def _get_french_curve() -> FrenchPressureConductivity:
    return _load_french_curves()


@lru_cache(maxsize=1)
def _get_becker_adiabats() -> Dict[str, PressureAdiabat]:
    return _split_becker_adiabats()


@lru_cache(maxsize=1)
def _get_ordered_becker_adiabats() -> List[PressureAdiabat]:
    # Pressure-only ordering by characteristic (geometric-mean) pressure.
    adis = list(_get_becker_adiabats().values())
    adis.sort(key=lambda a: a.pchar_gpa)
    return adis


@lru_cache(maxsize=1)
def _get_preising_adiabat() -> PreisingAdiabat:
    return _load_preising_adiabat()


@lru_cache(maxsize=1)
def _get_french_diffusion() -> tuple[interp1d, interp1d, np.ndarray]:
    """Return French D_H(logP) and D_He(logP) interpolators (log-log) and pressure array.

    D values are in mm^2/s (log-space), pressure in GPa.
    """
    french1 = _numeric_table(DATA_DIR / "French_table1.txt")
    french2 = _numeric_table(DATA_DIR / "French_table2.txt")

    r_f1 = french1[:, 0]
    p_f1_gpa = french1[:, 3]
    r_f2 = french2[:, 0]
    # Columns 7 and 8 in French table2 are D_H and D_He (mm^2/s).
    dh_f2 = french2[:, 7]
    dhe_f2 = french2[:, 8]

    p_of_r = interp1d(r_f1, p_f1_gpa, kind="cubic", fill_value="extrapolate", assume_sorted=True)
    p_f2_gpa = np.asarray(p_of_r(r_f2), dtype=float)

    idx = np.argsort(p_f2_gpa)
    p_sorted = p_f2_gpa[idx]
    log10p = np.log10(p_sorted)

    dh_interp = interp1d(log10p, np.log10(dh_f2[idx]), kind="linear", fill_value="extrapolate")
    dhe_interp = interp1d(log10p, np.log10(dhe_f2[idx]), kind="linear", fill_value="extrapolate")
    return dh_interp, dhe_interp, p_sorted


def get_becker_adiabat_lambdas(peval_dyn_cm2: Iterable[float] | np.ndarray, *, clip: bool = True) -> Dict[str, np.ndarray]:
    """
    Return per-adiabat Becker conductivity curves (W/m/K) evaluated at pressure.
    Pressure input is in dyn/cm^2, matching orchard conventions.
    """
    p_dyn = np.asarray(peval_dyn_cm2, dtype=float)
    p_gpa = p_dyn / GPA_TO_DYN_CM2
    out: Dict[str, np.ndarray] = {}
    for name, adiabat in _get_becker_adiabats().items():
        out[name] = np.asarray(adiabat.eval_wmk(p_gpa, clip=clip), dtype=float)
    return out


def _becker_composite_lambda_wmk(p_gpa: np.ndarray | float) -> np.ndarray | float:
    """
    Smooth pressure-only composite of Becker (2018) adiabats.

    We separate the KOI-889b / Corot-3b / Gliese-229b adiabats, build a log-log
    lambda(P) interpolation for each, and then blend between them in log-pressure
    using the characteristic pressure of each adiabat.
    """
    p = np.asarray(p_gpa, dtype=float)
    adis = _get_ordered_becker_adiabats()
    if len(adis) < 2:
        lam = np.asarray(adis[0].eval_wmk(p, clip=True), dtype=float)
        return _match_shape(lam, p_gpa)

    chars = np.array([a.pchar_gpa for a in adis], dtype=float)
    lams = [np.asarray(a.eval_wmk(p, clip=True), dtype=float) for a in adis]

    # Piecewise smooth blending between neighboring adiabats across their
    # characteristic-pressure intervals in log P.
    if len(adis) == 3:
        l0, l1, l2 = lams
        c0, c1, c2 = chars
        out = np.array(l1, copy=True)

        mask0 = p <= c0
        mask2 = p >= c2
        out[mask0] = l0[mask0]
        out[mask2] = l2[mask2]

        m01 = (~mask0) & (p < c1)
        if np.any(m01):
            u = (np.log(p[m01]) - np.log(c0)) / (np.log(c1) - np.log(c0))
            w = _smoothstep01(u)
            out[m01] = _blend_logspace(l0[m01], l1[m01], w)

        m12 = (p >= c1) & (~mask2)
        if np.any(m12):
            u = (np.log(p[m12]) - np.log(c1)) / (np.log(c2) - np.log(c1))
            w = _smoothstep01(u)
            out[m12] = _blend_logspace(l1[m12], l2[m12], w)

        return _match_shape(out, p_gpa)

    # Generic fallback for any other adiabat count: nearest characteristic pressure.
    out = np.zeros_like(p, dtype=float)
    logp = np.log(np.clip(p, np.finfo(float).tiny, None))
    logc = np.log(chars)
    idx_near = np.argmin(np.abs(logp[:, None] - logc[None, :]), axis=1) if p.ndim else int(np.argmin(np.abs(logp - logc)))
    if p.ndim == 0:
        out = lams[idx_near].item()
    else:
        for i, lam in enumerate(lams):
            out[idx_near == i] = lam[idx_near == i]
    return _match_shape(np.asarray(out, dtype=float), p_gpa)


def _french_y_of_p(p_gpa: np.ndarray) -> np.ndarray:
    """French J11-8a reference helium mass fraction Y(P)."""
    return np.where(p_gpa < P_FRENCH_Y_BOUNDARY_GPA, Y_FRENCH_OUTER, Y_FRENCH_INNER)


def _y_blend_weight(y_local: np.ndarray, y_french: np.ndarray, y_preising: np.ndarray) -> np.ndarray:
    """Compute blend weight w in [0, 1] for Y-dependent interpolation.

    w = 0 → French, w = 1 → Preising.
    """
    dy = y_preising - y_french
    # Avoid division by zero where both references are equal.
    safe_dy = np.where(np.abs(dy) > 1e-6, dy, np.sign(dy) * 1e-6)
    w = (y_local - y_french) / safe_dy
    return np.clip(w, 0.0, 1.0)


def get_Lambda_cond(
    peval_dyn_cm2: Iterable[float] | np.ndarray,
    *,
    Y: Optional[np.ndarray | float] = None,
) -> np.ndarray:
    """
    Pressure-based conductivity (CGS) used by orchard for the non-water component.

    When *Y* is ``None`` (default), behavior is identical to the original
    French (2012) + Becker (2018) pressure-only composite — fully backward
    compatible.

    When *Y* (helium mass fraction array, same shape as *peval_dyn_cm2*) is
    provided, the function blends between French (Y~0.24) and Preising (2023)
    Saturn MF20 (Y~0.07--0.91) conductivities in log-lambda space, weighted by
    the local helium fraction.  This captures:

    - He-depleted envelopes (Y << 0.24): Saturn-like, lower lambda.
    - He-rich layers (Y >> 0.24): dramatic conductivity suppression.

    Parameters
    ----------
    peval_dyn_cm2 : array-like
        Pressure in dyn/cm^2 (orchard standard units).
    Y : array-like or None
        Helium mass fraction.  If provided, enables Y-dependent blending.

    Returns
    -------
    np.ndarray
        Thermal conductivity in erg/(s * cm * K).
    """
    p_dyn = np.asarray(peval_dyn_cm2, dtype=float)
    p_gpa = p_dyn / GPA_TO_DYN_CM2

    french = _get_french_curve()
    lam_f = np.asarray(french.eval_wmk(p_gpa), dtype=float)

    # ------ French + Becker composite (pressure-only baseline) ------
    p_french_max = french.pmax_gpa
    lam_wmk = np.array(lam_f, copy=True)

    high = p_gpa > p_french_max
    if np.any(high):
        lam_b = np.asarray(_becker_composite_lambda_wmk(p_gpa), dtype=float)

        # Blend from French -> Becker above the French maximum pressure.
        p_blend_end = min(FRENCH_TO_BECKER_BLEND_FACTOR * p_french_max, max(a.pmax_gpa for a in _get_ordered_becker_adiabats()))
        if p_blend_end <= p_french_max:
            lam_wmk[high] = lam_b[high]
        else:
            mid = high & (p_gpa < p_blend_end)
            top = p_gpa >= p_blend_end
            if np.any(mid):
                u = (np.log(p_gpa[mid]) - np.log(p_french_max)) / (np.log(p_blend_end) - np.log(p_french_max))
                w = _smoothstep01(u)
                lam_wmk[mid] = _blend_logspace(lam_f[mid], lam_b[mid], w)
            if np.any(top):
                lam_wmk[top] = lam_b[top]

    # ------ Y-dependent Preising blending (if Y provided) ------
    if Y is not None:
        y_local = np.asarray(Y, dtype=float)
        p_arr = np.atleast_1d(p_gpa)
        lam_arr = np.atleast_1d(lam_wmk)

        preising = _get_preising_adiabat()
        # Only blend where Preising data covers the pressure range.
        in_range = (p_arr >= preising.pmin_gpa) & (p_arr <= preising.pmax_gpa)
        if np.any(in_range):
            p_ir = p_arr[in_range]
            y_loc = np.atleast_1d(y_local)[in_range] if y_local.ndim > 0 else y_local

            lam_p = np.asarray(preising.lam_adiabat.eval_wmk(p_ir), dtype=float)
            y_f = _french_y_of_p(p_ir)
            y_p = np.asarray(preising.eval_y(p_ir), dtype=float)

            w = _y_blend_weight(y_loc, y_f, y_p)
            lam_arr[in_range] = _blend_logspace(lam_arr[in_range], lam_p, w)

        lam_wmk = lam_arr if p_gpa.ndim > 0 else lam_arr.item()

    return np.asarray(lam_wmk * W_MK_TO_CGS, dtype=float)


def _get_diffusion_coeff(
    peval_dyn_cm2: Iterable[float] | np.ndarray,
    *,
    Y: Optional[np.ndarray | float] = None,
    species: str = "He",
) -> np.ndarray:
    """Pressure-dependent self-diffusion coefficient in cm^2/s.

    Parameters
    ----------
    peval_dyn_cm2 : array-like
        Pressure in dyn/cm^2.
    Y : array-like or None
        Helium mass fraction for Y-dependent blending.
    species : {"H", "He"}
        Which diffusion coefficient to return.

    Returns
    -------
    np.ndarray
        Self-diffusion coefficient in cm^2/s.
    """
    p_dyn = np.asarray(peval_dyn_cm2, dtype=float)
    p_gpa = p_dyn / GPA_TO_DYN_CM2

    # French baseline (log10-log10 interpolation in pressure).
    dh_french_interp, dhe_french_interp, p_french_sorted = _get_french_diffusion()
    log10p = np.log10(np.clip(p_gpa, np.finfo(float).tiny, None))
    if species == "H":
        d_french_mm2s = 10.0 ** np.asarray(dh_french_interp(log10p), dtype=float)
    else:
        d_french_mm2s = 10.0 ** np.asarray(dhe_french_interp(log10p), dtype=float)

    d_mm2s = np.array(d_french_mm2s, copy=True)

    # Y-dependent Preising blending.
    if Y is not None:
        y_local = np.asarray(Y, dtype=float)
        p_arr = np.atleast_1d(p_gpa)
        d_arr = np.atleast_1d(d_mm2s)

        preising = _get_preising_adiabat()
        in_range = (p_arr >= preising.pmin_gpa) & (p_arr <= preising.pmax_gpa)
        if np.any(in_range):
            p_ir = p_arr[in_range]
            y_loc = np.atleast_1d(y_local)[in_range] if y_local.ndim > 0 else y_local

            if species == "H":
                d_p = np.asarray(preising.eval_dh_mm2s(p_ir), dtype=float)
            else:
                d_p = np.asarray(preising.eval_dhe_mm2s(p_ir), dtype=float)

            y_f = _french_y_of_p(p_ir)
            y_p = np.asarray(preising.eval_y(p_ir), dtype=float)
            w = _y_blend_weight(y_loc, y_f, y_p)
            d_arr[in_range] = _blend_logspace(d_arr[in_range], d_p, w)

        d_mm2s = d_arr if p_gpa.ndim > 0 else d_arr.item()

    return np.asarray(d_mm2s * MM2S_TO_CM2S, dtype=float)


def get_D_H(
    peval_dyn_cm2: Iterable[float] | np.ndarray,
    *,
    Y: Optional[np.ndarray | float] = None,
) -> np.ndarray:
    """Hydrogen self-diffusion coefficient D_H(P, Y) in cm^2/s.

    Based on French et al. (2012) and Preising et al. (2023) ab initio data.
    """
    return _get_diffusion_coeff(peval_dyn_cm2, Y=Y, species="H")


def get_D_He(
    peval_dyn_cm2: Iterable[float] | np.ndarray,
    *,
    Y: Optional[np.ndarray | float] = None,
) -> np.ndarray:
    """Helium self-diffusion coefficient D_He(P, Y) in cm^2/s.

    Based on French et al. (2012) and Preising et al. (2023) ab initio data.
    """
    return _get_diffusion_coeff(peval_dyn_cm2, Y=Y, species="He")


def preising_pressure_range_gpa() -> tuple[float, float]:
    p = _get_preising_adiabat()
    return p.pmin_gpa, p.pmax_gpa


def get_Z_ref_for_adiabats(
    peval_dyn_cm2: Iterable[float] | np.ndarray,
    *,
    Y: Optional[np.ndarray | float] = None,
) -> np.ndarray | float:
    """
    Reference heavy-element mass fraction Z_ref(P) for the French/Becker pressure-only
    conductivity baseline used by `get_Lambda_cond`.

    `Z` here is the heavy-element *mass fraction* (not Fe/H and not a number fraction).
    The values are taken from:
    - French et al. (2012), Jupiter model J11-8a envelope heavy-element mass fractions:
      Z1=0.038 (outer), Z2=0.128 (inner), with envelope boundary at 8 Mbar (=800 GPa)
    - Becker et al. (2018) Table 1 X/Y/Z triplets (mass fractions):
      KOI-889b (Z=0.04), Corot-3b (Z=0.02), Gliese-229b (Z=0.02)
    - Preising et al. (2023) Saturn MF20 model: Z_oxygen ~ 0.07

    When *Y* is provided, Z_ref is blended between the French value and
    the Preising value (0.07) in the same Y-dependent manner as
    ``get_Lambda_cond``.

    Parameters
    ----------
    peval_dyn_cm2 : array-like
        Pressure in dyn/cm^2.
    Y : array-like or None
        Helium mass fraction for Y-dependent blending.

    Returns
    -------
    np.ndarray or float
        Reference heavy-element mass fraction Z_ref(P).
    """
    p_dyn = np.asarray(peval_dyn_cm2, dtype=float)
    p_gpa = p_dyn / GPA_TO_DYN_CM2
    scalar = p_gpa.ndim == 0
    p = np.atleast_1d(np.asarray(p_gpa, dtype=float))

    # French et al. (2012) J11-8a envelope heavy-element mass fractions (mass, not number).
    Z_FRENCH_OUTER = 0.038
    Z_FRENCH_INNER = 0.128
    P_FRENCH_ENV_BOUNDARY_GPA = 800.0  # 8 Mbar
    z_french = np.where(p < P_FRENCH_ENV_BOUNDARY_GPA, Z_FRENCH_OUTER, Z_FRENCH_INNER)

    # Becker et al. (2018) Table 1 X/Y/Z triplets (mass fractions).
    z_becker_lookup = {
        "KOI-889b": 0.04,
        "Corot-3b": 0.02,
        "Gliese-229b": 0.02,
    }
    becker_ranges = dict(becker_adiabat_ranges_gpa())

    # Match the pressure-only Becker composite weighting in this module using
    # geometric-mean characteristic pressures.
    ordered = []
    for name, zref in z_becker_lookup.items():
        if name not in becker_ranges:
            continue
        pmin, pmax = becker_ranges[name]
        ordered.append((np.sqrt(pmin * pmax), zref))
    ordered.sort(key=lambda t: t[0])

    if len(ordered) == 0:
        z_becker = np.full_like(p, Z_FRENCH_INNER, dtype=float)
    elif len(ordered) == 1:
        z_becker = np.full_like(p, ordered[0][1], dtype=float)
    elif len(ordered) == 3:
        c0, z0 = ordered[0]
        c1, z1 = ordered[1]
        c2, z2 = ordered[2]
        z_becker = np.full_like(p, z1, dtype=float)

        m0 = p <= c0
        m2 = p >= c2
        z_becker[m0] = z0
        z_becker[m2] = z2

        m01 = (~m0) & (p < c1)
        if np.any(m01):
            u = (np.log(p[m01]) - np.log(c0)) / (np.log(c1) - np.log(c0))
            w = _smoothstep01(u)
            z_becker[m01] = (1.0 - w) * z0 + w * z1

        m12 = (p >= c1) & (~m2)
        if np.any(m12):
            u = (np.log(p[m12]) - np.log(c1)) / (np.log(c2) - np.log(c1))
            w = _smoothstep01(u)
            z_becker[m12] = (1.0 - w) * z1 + w * z2
    else:
        # Fallback: nearest characteristic-pressure adiabat.
        chars = np.array([row[0] for row in ordered], dtype=float)
        zvals = np.array([row[1] for row in ordered], dtype=float)
        psafe = np.clip(p, np.finfo(float).tiny, None)
        idx = np.argmin(np.abs(np.log(psafe)[:, None] - np.log(chars)[None, :]), axis=1)
        z_becker = zvals[idx]

    # Mirror the French->Becker pressure-domain blend used in get_Lambda_cond.
    _, p_french_max = french_pressure_range_gpa()
    p_becker_max = max(rng[1] for rng in becker_ranges.values()) if becker_ranges else p_french_max
    p_blend_end = min(FRENCH_TO_BECKER_BLEND_FACTOR * p_french_max, p_becker_max)

    z_ref = np.array(z_french, copy=True)
    high = p > p_french_max
    if p_blend_end <= p_french_max:
        z_ref[high] = z_becker[high]
    else:
        mid = high & (p < p_blend_end)
        top = p >= p_blend_end
        if np.any(mid):
            u = (np.log(p[mid]) - np.log(p_french_max)) / (np.log(p_blend_end) - np.log(p_french_max))
            w = _smoothstep01(u)
            z_ref[mid] = (1.0 - w) * z_french[mid] + w * z_becker[mid]
        if np.any(top):
            z_ref[top] = z_becker[top]

    # Y-dependent Preising blending (mirrors get_Lambda_cond).
    if Y is not None:
        Z_PREISING = 0.07  # oxygen mass fraction in MF20 Saturn model
        y_local = np.asarray(Y, dtype=float)
        y_arr = np.atleast_1d(y_local)

        preising = _get_preising_adiabat()
        in_range = (p >= preising.pmin_gpa) & (p <= preising.pmax_gpa)
        if np.any(in_range):
            p_ir = p[in_range]
            y_loc = y_arr[in_range] if y_arr.ndim > 0 else y_arr
            y_f = _french_y_of_p(p_ir)
            y_p = np.asarray(preising.eval_y(p_ir), dtype=float)
            w = _y_blend_weight(y_loc, y_f, y_p)
            z_ref[in_range] = (1.0 - w) * z_ref[in_range] + w * Z_PREISING

    return _match_shape(np.asarray(z_ref, dtype=float), peval_dyn_cm2)


def french_pressure_range_gpa() -> tuple[float, float]:
    f = _get_french_curve()
    return f.pmin_gpa, f.pmax_gpa


def becker_adiabat_ranges_gpa() -> Mapping[str, tuple[float, float]]:
    return {name: (adi.pmin_gpa, adi.pmax_gpa) for name, adi in _get_becker_adiabats().items()}
