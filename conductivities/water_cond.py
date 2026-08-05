import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u

# French (2019) thermal-conductivity fit validity bounds (upper clipping only).
# The published dataset spans roughly rho <= 6 g/cc and T <= 50,000 K.
FRENCH19_RHO_MAX_GCC = 6.0
FRENCH19_T_MAX_K = 5.0e4

# Out-of-range handling modes for the French (2019) fits:
#   0 = legacy/raw evaluation (no clipping; polynomial extrapolation)
#   1 = clip rho and T at the published upper bounds
#   2 = extrapolate linearly in ln(lambda) from the clipped boundary
FRENCH19_EXTRAP_MODE_LEGACY = 0
FRENCH19_EXTRAP_MODE_CLIP = 1
FRENCH19_EXTRAP_MODE_LOG = 2

# Safe default: clip.
_FRENCH19_EXTRAP_MODE = FRENCH19_EXTRAP_MODE_CLIP


def set_french19_extrapolation_mode(mode):
    """Set the module-default out-of-range handling mode for French (2019) fits."""
    global _FRENCH19_EXTRAP_MODE
    mode_i = int(mode)
    if mode_i not in (0, 1, 2):
        raise ValueError(f"Invalid French19 extrapolation mode {mode_i}. Expected 0, 1, or 2.")
    _FRENCH19_EXTRAP_MODE = mode_i


def get_french19_extrapolation_mode():
    return int(_FRENCH19_EXTRAP_MODE)


def _resolve_french19_mode(mode):
    return get_french19_extrapolation_mode() if mode is None else int(mode)


def _clip_french19_domain(rho, T):
    """Clip French (2019) fit inputs to the published upper validity range."""
    rho_arr, T_arr = np.broadcast_arrays(np.asarray(rho, dtype=float), np.asarray(T, dtype=float))
    return np.minimum(rho_arr, FRENCH19_RHO_MAX_GCC), np.minimum(T_arr, FRENCH19_T_MAX_K)

def _as_output_shape(x):
    arr = np.asarray(x)
    return arr.item() if arr.ndim == 0 else arr


def _french19_log_slopes(eval_raw, rho_c, T_c, h=1.0e-3):
    """Numerical slopes d ln(lambda)/d ln(rho) and d ln(lambda)/d ln(T) at clipped points."""
    tiny = np.finfo(float).tiny
    rho_c = np.asarray(rho_c, dtype=float)
    T_c = np.asarray(T_c, dtype=float)

    rho_lo = np.maximum(rho_c * np.exp(-h), tiny)
    rho_hi = np.minimum(rho_c * np.exp(+h), FRENCH19_RHO_MAX_GCC)
    T_lo = np.maximum(T_c * np.exp(-h), tiny)
    T_hi = np.minimum(T_c * np.exp(+h), FRENCH19_T_MAX_K)

    f_rho_lo = np.asarray(eval_raw(rho_lo, T_c), dtype=float)
    f_rho_hi = np.asarray(eval_raw(rho_hi, T_c), dtype=float)
    f_T_lo = np.asarray(eval_raw(rho_c, T_lo), dtype=float)
    f_T_hi = np.asarray(eval_raw(rho_c, T_hi), dtype=float)

    ln_rho_lo = np.log(np.maximum(f_rho_lo, tiny))
    ln_rho_hi = np.log(np.maximum(f_rho_hi, tiny))
    ln_T_lo = np.log(np.maximum(f_T_lo, tiny))
    ln_T_hi = np.log(np.maximum(f_T_hi, tiny))

    dlnrho = np.log(np.maximum(rho_hi, tiny)) - np.log(np.maximum(rho_lo, tiny))
    dlnT = np.log(np.maximum(T_hi, tiny)) - np.log(np.maximum(T_lo, tiny))

    slope_rho = np.zeros_like(rho_c, dtype=float)
    slope_T = np.zeros_like(T_c, dtype=float)
    mask_rho = dlnrho > 0.0
    mask_T = dlnT > 0.0
    slope_rho[mask_rho] = (ln_rho_hi[mask_rho] - ln_rho_lo[mask_rho]) / dlnrho[mask_rho]
    slope_T[mask_T] = (ln_T_hi[mask_T] - ln_T_lo[mask_T]) / dlnT[mask_T]
    return slope_rho, slope_T


def _apply_french19_mode(eval_raw, rho, T, mode=None):
    mode_i = _resolve_french19_mode(mode)
    if mode_i not in (0, 1, 2):
        raise ValueError(f"Invalid French19 extrapolation mode {mode_i}. Expected 0, 1, or 2.")

    rho_arr, T_arr = np.broadcast_arrays(np.asarray(rho, dtype=float), np.asarray(T, dtype=float))

    if mode_i == FRENCH19_EXTRAP_MODE_LEGACY:
        return _as_output_shape(eval_raw(rho_arr, T_arr))

    rho_c, T_c = _clip_french19_domain(rho_arr, T_arr)
    base = np.asarray(eval_raw(rho_c, T_c), dtype=float)

    if mode_i == FRENCH19_EXTRAP_MODE_CLIP:
        return _as_output_shape(base)

    # Log-space extension from the clipped boundary:
    # ln lambda ~ ln lambda(rho_c, T_c) + s_rho * ln(rho/rho_c) + s_T * ln(T/T_c)
    out = np.array(base, copy=True, dtype=float)
    over = (rho_arr > FRENCH19_RHO_MAX_GCC) | (T_arr > FRENCH19_T_MAX_K)
    active = over & (base > 0.0)
    if np.any(active):
        tiny = np.finfo(float).tiny
        slope_rho, slope_T = _french19_log_slopes(eval_raw, rho_c, T_c)

        dlnrho = np.zeros_like(rho_arr, dtype=float)
        dlnT = np.zeros_like(T_arr, dtype=float)
        rho_over = rho_arr > FRENCH19_RHO_MAX_GCC
        T_over = T_arr > FRENCH19_T_MAX_K
        dlnrho[rho_over] = np.log(np.maximum(rho_arr[rho_over], tiny) / FRENCH19_RHO_MAX_GCC)
        dlnT[T_over] = np.log(np.maximum(T_arr[T_over], tiny) / FRENCH19_T_MAX_K)

        ln_base = np.log(np.maximum(base, tiny))
        ln_ext = ln_base + slope_rho * dlnrho + slope_T * dlnT
        out[active] = np.exp(ln_ext[active])
    return _as_output_shape(out)


def _Lambda_e_raw(rho, T):
    """
    Compute the electronic contribution to Lambda, returning zero for T < 4000 K.
    """
    a0 = 45.22
    a1 = 1.624
    a2 = 0.8743
    a3 = -0.1757
    a4 = -0.07007
    b0_1 = -871.1
    b1_1 = 449.4
    b00 = 82.65
    b10 = -122.4
    b01 = 1.414
    b11 = 7.086

    # Take logs of rho and T; be sure T > 0
    x = np.log(rho)
    y = np.log(T)

    A = a0 * np.arctan(a1 + a2*x + a3*y + a4*x*y)
    B = b0_1 / y + b1_1*x / y + b00 + b10*x + b01*y + b11*x*y

    # Compute the nominal Lambda_e
    lam_e = np.exp(A + B)

    # For T < 4000, force lam_e to 0. Use np.where so scalar inputs are safe too.
    return np.where(T < 4000, 0.0, lam_e)

### Proton contribution to thermal conductivity from French (2019) ###

def _Lambda_p_raw(rho, T):
    a0 = 1.05431
    a1 = -0.457948
    a2 = -7708.55
    a3 = 1692.09
    c00 = 338.278
    c01 = -108.114
    c02 = 11.3921
    c03 = -0.393551
    c10 = -45.8201
    c11 = 16.7430
    c12 = -1.92584
    c13 = 0.0712254
    c20 = -0.944639
    c21 = 0.161103
    c22 = -0.00678351
    c23 = 0.0

    s0 = 5.636
    y0 = 8.5
    rho0 = 1.117

    y = np.log(T)
    A = c00 + c01*y + c02*y**2 + c03*y**3 + \
        c10*rho + c11*rho*y + c12*rho*y**2 + c13*rho*y**3 + \
        c20*rho**2 + c21*rho**2 * y + c22*rho**2 * y**2
    B = a0*np.exp(-s0*(y - y0)**2 - rho/rho0)
    C = a1*np.log(rho + 1) + a2/T + a3*np.log(rho + 1)/T

    return np.exp(A + B + C)


def Lambda_e(rho, T, mode=None):
    """Electronic thermal conductivity with selectable out-of-range handling (0/1/2)."""
    return _apply_french19_mode(_Lambda_e_raw, rho, T, mode=mode)


def Lambda_p(rho, T, mode=None):
    """Nuclear thermal conductivity (French 2019 fit) with selectable handling (0/1/2)."""
    return _apply_french19_mode(_Lambda_p_raw, rho, T, mode=mode)

# Viscosity and heat capacity from French & Nettelmann (2019)

def get_eta(rho, T):

    C00 = -12.77
    C01 = -0.1032
    C02 = 0.06499
    C10 = 3.38
    C11 = -0.5129
    C12 = 0.01799
    C2_1 = 51.17
    C20 = -8.923
    C21 = 0.3831
    C3_3 = -58.49
    C30 = 0.05828

    y = np.log(T)

    log_eta = C00 + C01 * y + C02 * y ** 2 + \
              C10 * rho + C11 * rho * y + C12 * rho * y ** 2 + \
              C2_1 * rho ** 2 / y + C20 * rho ** 2 + C21 * rho ** 2 * y + \
              C3_3 * rho ** 3 / (y ** 3) + C30 * rho ** 3

    return np.exp(log_eta)

def get_c_p(rho, T):
    C00 = 41.9
    C01 = -10.3
    C02 = 0.6813
    C10 = 8.715
    C11 = -1.393
    C122 = 0.0324

    y0 = 8.9
    a0 = 3.036
    s0 = 1.335

    y = np.log(T)
    x = np.log(rho)

    cp_sum = C00 + C01 * y + C02 * y ** 2 + \
             C10 * rho + C11 * rho * y + C122 * rho ** 1.2 * y ** 2

    return cp_sum + a0/(x ** 0.25) * np.exp(-s0 * (y - y0) ** 2)
