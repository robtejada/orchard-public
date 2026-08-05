import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import newton, brenth
from scipy.interpolate import RegularGridInterpolator as RGI

def _transition_weight(logP, logPs, width_dex, kind="tanh"):
    """
    Return weights w in [0, 1] with w≈1 for logP << logPs and w≈0 for logP >> logPs.
    width_dex controls how fast the transition is.
    """
    d = (logP - logPs) / width_dex  # dimensionless distance in log10 P

    if kind == "tanh":
        # what you already have
        return 0.5 * (1.0 - np.tanh(d))

    elif kind == "logistic":
        # classic sigmoid, very similar to tanh but a bit “sharper”
        # w = 1 / (1 + exp(d))
        return 1.0 / (1.0 + np.exp(d))

    elif kind == "cosine":
        # compact support: exactly 1 below logPs - width, exactly 0 above logPs + width
        z = np.clip(0.5 * d + 0.5, 0.0, 1.0)   # maps d=-1→0, d=+1→1
        return 0.5 * (1.0 + np.cos(np.pi * z))  # smooth C^1 “bump” from 1→0

    elif kind == "smoothstep":
        # polynomial “smoothstep” with compact support, C^1 continuous
        z = np.clip(0.5 * d + 0.5, 0.0, 1.0)
        s = z * z * (3.0 - 2.0 * z)           # 3z^2 - 2z^3, goes 0→1
        return 1.0 - s                        # flip to go 1→0

    elif kind == "smootherstep":
        # C^2 version: 6z^5 - 15z^4 + 10z^3
        z = np.clip(0.5 * d + 0.5, 0.0, 1.0)
        s = z**3 * (z * (6.0 * z - 15.0) + 10.0)
        return 1.0 - s

    else:
        raise ValueError(f"Unknown weight kind: {kind}")


def _modify_Tc_with_melt(P, Tc_raw, *,
                         Tc_mode="original",
                         Tc_P_switch=12.0,
                         Tc_blend_width_dex=0.25,
                         Tc_avg_weight=0.5,
                         interp_curve='logistic'):
    """
    Modify the crest temperature Tc(P) using the MgSiO3 melting curve.

    Parameters
    ----------
    P : array
        Pressure [GPa].
    Tc_raw : array
        Original Gilmore+ crest temperature E*(1 + P/D) [K].
    Tc_mode : {"original", "follow_melt", "avg_melt"}
        - "original": use Tc_raw unchanged.
        - "follow_melt": Tc(P) follows the larger of Tc_raw and Tmelt(P),
          or smoothly transitions to Tmelt if Tc_P_switch is given.
        - "avg_melt": above Tc_P_switch, Tc tends toward an average of
          Tc_raw and Tmelt.
    Tc_P_switch : float, optional
        Characteristic transition pressure [GPa] in the *miscibility*
        curve where you want to start modifying Tc. If None:
          - for "follow_melt": use a simple max(Tc_raw, Tmelt).
          - for "avg_melt": default to 10 GPa (you can override with a
            Y,Z-specific crossing found earlier).
    Tc_blend_width_dex : float, optional
        Width of the transition in log10(P) for smooth blending.
    Tc_avg_weight : float in [0, 1], optional
        For "avg_melt": at high P, Tc_high = w*Tc_raw + (1-w)*Tmelt
        with w = Tc_avg_weight (w=0.5 → simple average).

    Returns
    -------
    Tc_mod : array
        Modified crest temperature [K], same shape as Tc_raw.
    """
    if Tc_mode in (None, "original"):
        return Tc_raw

    P = np.asarray(P, float)
    Tc_raw = np.asarray(Tc_raw, float)
    Tmelt = get_Tmelt_mgsio3(P)

    # --------------------------
    # Mode 1: follow the melt
    # --------------------------
    if Tc_mode == "follow_melt":
        if Tc_P_switch is None:
            # Easiest: crest never below melt; at high P Tc ≈ Tmelt.
            return np.maximum(Tc_raw, Tmelt)

        # Smooth transition from Tc_raw (low P) to Tmelt (high P)
        logP = np.log10(P)
        logPs = np.log10(Tc_P_switch)
        # w≈1 below P_switch, w≈0 above.
        w = _transition_weight(logP, logPs, Tc_blend_width_dex, kind=interp_curve)
        Tc_mod = w * Tc_raw + (1.0 - w) * Tmelt
        Tc_mod = np.maximum(Tc_mod, Tmelt)
       # Tc_mod[P > Tc_P_switch] = Tmelt[P > Tc_P_switch]
        return Tc_mod

    # --------------------------
    # Mode 2: averaged crest
    # --------------------------
    if Tc_mode == "avg_melt":
        if Tc_P_switch is None:
            # You can replace 10.0 with the Y,Z-specific crossing
            # from your earlier find_P_cross_mgsio3().
            Tc_P_switch = 10.0

        logP = np.log10(P)
        logPs = np.log10(Tc_P_switch)
        w = 0.5 * (1.0 - np.tanh((logP - logPs) / Tc_blend_width_dex))

        # Target high-P crest: between original and melt.
        Tc_high = Tc_avg_weight * Tc_raw + (1.0 - Tc_avg_weight) * Tmelt
        Tc_mod = w * Tc_raw + (1.0 - w) * Tc_high
        return Tc_mod

    raise ValueError(f"Unknown Tc_mode = {Tc_mode!r}")

def get_tmisc(P_GPa, Y, Z, *,
              eps=1e-12,
              M_H2=2.01588,
              M_He=4.002602,
              M_rock=100.387,
              # NEW: options for modifying Tc with the melt curve
              Tc_mode="original",
              Tc_P_switch=12.0,
              Tc_blend_width_dex=0.25,
              Tc_avg_weight=0.5,
              interp_curve='tanh',):
    """
    Binodal (H2–MgSiO3) miscibility temperature from Gilmore et al. 2025, App. A,
    generalized to an H–He–rock (MgSiO3) ternary mixture.

    Inputs:
      P_GPa : float or array
          Pressure [GPa].
      Y     : float or array
          Helium mass fraction.
      Z     : float or array
          Rock (MgSiO3) mass fraction.

    Assumes:
      X + Y + Z = 1, where X is the hydrogen mass fraction.
      Helium is treated as an inert spectator for the miscibility curve:
      the fit is applied using the H2–rock mole fraction only.

    Options:
      return_log10 : if True, return log10(T_b). Else return T_b [K].
      M_H2, M_He, M_rock : molar masses [g/mol].

    Returns:
      T_b [K] or log10(T_b), same shape as broadcast(P_GPa, Y, Z).
    """
    # Broadcast inputs
    P = np.asarray(P_GPa, float)
    Y = np.asarray(Y, float)
    Z = np.asarray(Z, float)
    P, Y, Z = np.broadcast_arrays(P, Y, Z)

    # Infer hydrogen mass fraction
    X = 1.0 - Y - Z

    # Basic sanity check on composition
    if np.any(X < -1e-6) or np.any(X > 1.0 + 1e-6):
        raise ValueError("Invalid composition: X = 1 - Y - Z must be in [0, 1].")

    # Clamp to avoid zero / one limits
    X = np.clip(X, eps, 1.0 - eps)
    Z = np.clip(Z, eps, 1.0 - eps)

    # mass -> mole fraction conversion for the *binary* H2–rock subsystem
    nH2 = X / M_H2
    nR  = Z / M_rock
    x   = nH2 / (nH2 + nR)  # hydrogen mole fraction in H2+rock subsystem

    # ================== Gilmore+ 2025 Appendix A coefficients ==================
    xc = 0.73913
    D = -35.0        # GPa
    E = 4223.0       # K
    a2, a3, a4, a5 = -4.515523, 0.075651, -0.933822, -2.206251
    b2, b3, b4, b5 =  0.544371, 30.217687,  2.504075, -1.712032
    y0 = -5.0

    # Crest temperature (A6) – ORIGINAL fit:
    Tc_raw = E * (1.0 + P / D)

    Tc = _modify_Tc_with_melt(
        P, Tc_raw,
        Tc_mode=Tc_mode,
        Tc_P_switch=Tc_P_switch,
        Tc_blend_width_dex=Tc_blend_width_dex,
        Tc_avg_weight=Tc_avg_weight,
        interp_curve=interp_curve,
    )
    logTc = np.log10(Tc)

    # a1, b1 chosen so f(0)=g1(0)=log10(Tc) (A5, A9)
    a1 = logTc - a2 * (1.0 + a3 * np.exp(a4 * a5)) ** (-1.0 / a3)
    b1 = logTc - b2 * (1.0 + b3 * np.exp(b4 * b5)) ** (-1.0 / b3)

    logTb = np.empty_like(P)

    # Branch x <= xc : f(x̃) (A1–A4)
    m1 = x <= xc
    if np.any(m1):
        xt = np.log10(np.clip(x[m1] / xc, eps, None))  # (A2)
        logTb[m1] = a1[m1] + a2 * (1.0 + a3 * np.exp(-a4 * (xt - a5))) ** (-1.0 / a3)

    # Branch x > xc : g(ỹ) (A1, A7–A11)
    m2 = ~m1
    if np.any(m2):
        yt = np.log10(np.clip((1.0 - x[m2]) / (1.0 - xc), eps, None))  # (A3)
        g1 = b1[m2] + b2 * (1.0 + b3 * np.exp(-b4 * (yt - b5))) ** (-1.0 / b3)  # (A8)
        g1_y0 = b1[m2] + b2 * (1.0 + b3 * np.exp(-b4 * (y0 - b5))) ** (-1.0 / b3)
        beta0 = g1_y0 - 4.0 * y0  # (A11)
        logTb[m2] = np.where(yt >= y0, g1, 4.0 * yt + beta0)  # (A7, A10)

    return np.power(10.0, logTb), Tc


def get_Tmelt_mgsio3(P_GPa):
    Tm = 6295 * (P_GPa / 140) ** 0.317 # Fei et al. (2021)
    return Tm


# ---------- Helpers for inversion ----------

def x_from_YZ(Y, Z, *, M_H2=2.01588, M_rock=100.387):
    """
    Hydrogen mole fraction in the H2+rock subsystem, for fixed Y, Z.
    """
    X = 1.0 - Y - Z
    nH2 = X / M_H2
    nR  = Z / M_rock
    return nH2 / (nH2 + nR)


def find_Zcrit_for_Y(Y, *, xc=0.73913, M_H2=2.01588, M_rock=100.387,
                     eps=1e-12):
    """
    For a given helium mass fraction Y, find the rock mass fraction Z_crit
    such that x_H2(Y, Z_crit) = x_crit.

    Because x(Y, Z) is monotonic in Z (more rock => lower x), there is a
    unique Z_crit in (0, 1 - Y).
    """
    Zmin = eps
    Zmax = 1.0 - Y - eps
    if Zmax <= Zmin:
        raise ValueError("No valid Z range for given Y when finding Z_crit.")

    def fZ(Z):
        return x_from_YZ(Y, Z, M_H2=M_H2, M_rock=M_rock) - xc

    # x(Zmin) ~ 1, x(Zmax) ~ 0, so fZ changes sign
    return brenth(fZ, Zmin, Zmax, maxiter=100)


def invert_tmisc_hhe_mgsio3(P_GPa, Y, T_target, *,
                             xc=0.73913,
                             M_H2=2.01588, M_He=4.002602, M_rock=100.387,
                             Z_eps=1e-10, tol=1e-6,
                             max_bracket_subdiv=200,
                             Tc_mode="original",
                             Tc_P_switch=8,
                             Tc_blend_width_dex=0.25,
                             Tc_avg_weight=0.7,
                             interp_curve='tanh'
                             ):
    """
    Invert the miscibility curve for fixed P and Y:

        get_tmisc(P, Y, Z_i) = T_target

    and return the two rock mass fractions Z1 (low-Z branch, H-rich) and
    Z2 (high-Z branch, rock-rich), which lie on opposite sides of the
    critical composition x = x_crit.

    Uses Newton's method as a first attempt on each branch, then falls back
    to a brenth root finder with adaptive bracketing.

    Parameters
    ----------
    P_GPa : float
        Pressure [GPa].
    Y : float
        Helium mass fraction.
    T_target : float
        Temperature [K] for which to invert the miscibility curve.

    Returns
    -------
    Z1, Z2 : floats
        Rock mass fractions on the low-Z and high-Z branches, respectively.

    Raises
    ------
    RuntimeError
        If no root is found on a branch (e.g., (P, T_target) outside miscibility dome).
    """

    P = float(P_GPa)
    Y = float(Y)
    T_target = float(T_target)

    # Valid Z range from X = 1 - Y - Z in (0, 1)
    Zmin = Z_eps
    Zmax = 1.0 - Y - Z_eps
    if Zmax <= Zmin:
        raise ValueError("No valid Z range for given Y in inversion.")

    # Find Z where x_H2 = x_crit: this splits the two branches
    Zcrit = find_Zcrit_for_Y(Y, xc=xc, M_H2=M_H2, M_rock=M_rock, eps=Z_eps)

    # Residual function
    def f(Z):
        return get_tmisc(P, Y, Z,
                                    M_H2=M_H2, M_He=M_He, M_rock=M_rock,
                                    Tc_mode=Tc_mode,
                                    Tc_P_switch=Tc_P_switch,
                                    Tc_blend_width_dex=Tc_blend_width_dex,
                                    Tc_avg_weight=Tc_avg_weight,
                                    interp_curve=interp_curve)[0] - T_target

    def find_branch_root(Z_lo, Z_hi, initial_bias=0.5):
        """
        Try Newton first, then fall back to brenth with an adaptive bracket.
        """
        # First guess somewhere inside the interval
        Z0 = Z_lo + initial_bias * (Z_hi - Z_lo)

        # Newton attempt
        try:
            z_newton = newton(f, Z0, maxiter=50)
            if Z_lo <= z_newton <= Z_hi:
                if abs(f(z_newton)) < tol * max(1.0, abs(T_target)):
                    return z_newton
        except Exception:
            pass

        # Bracketing fallback: subdivide [Z_lo, Z_hi] and look for sign changes
        N = max_bracket_subdiv
        zs = np.linspace(Z_lo, Z_hi, N + 1)
        fzs = np.array([f(z) for z in zs])

        for i in range(N):
            fi, fj = fzs[i], fzs[i + 1]
            if not np.isfinite(fi) or not np.isfinite(fj):
                continue
            if fi == 0.0:
                return zs[i]
            if fi * fj < 0.0:
                try:
                    root = brenth(f, zs[i], zs[i + 1], maxiter=100)
                    if Z_lo <= root <= Z_hi:
                        return root
                except Exception:
                    continue

        raise RuntimeError("No root found in branch; (P, T) may be outside miscibility dome.")

    # Low-Z (H-rich) branch: Z in [Zmin, Zcrit]
    Z1 = find_branch_root(Zmin, Zcrit, initial_bias=0.3)

    # High-Z (rock-rich) branch: Z in [Zcrit, Zmax]
    Z2 = find_branch_root(Zcrit, Zmax, initial_bias=0.7)

    return Z1, Z2

def get_zmisc_inv(P_GPa, T_target, Y, *,
                                 xc=0.73913,
                                 M_H2=2.01588, M_He=4.002602, M_rock=100.387,
                                 Z_eps=1e-10, tol=1e-6,
                                 max_bracket_subdiv=200,
                                 error_on_fail=False,
                                 fill_value=np.nan,
                                Tc_mode='original',
                                Tc_P_switch=8,
                                Tc_blend_width_dex=0.25,
                                Tc_avg_weight=0.7,
                                interp_curve='tanh'):
    """
    Vectorized inversion of the miscibility curve for arrays of P and T.

    For each element of (P, Y, T_target), solves
        get_tmisc(P, Y, Z_i) = T_target
    and returns two rock mass fractions Z1 (low-Z branch) and Z2 (high-Z branch).

    Parameters
    ----------
    P_GPa : float or array
        Pressure [GPa].
    Y : float or array
        Helium mass fraction. Can be scalar or broadcastable with P_GPa and T_target.
    T_target : float or array
        Temperature [K] for which to invert the miscibility curve.

    Other parameters are passed through to the scalar inverter.

    Returns
    -------
    Z1, Z2 : arrays
        Rock mass fractions on the low-Z and high-Z branches. Shape is the
        broadcast shape of (P_GPa, Y, T_target).

    Notes
    -----
    - By default, if a given (P, Y, T) is outside the miscibility dome or the
      solver fails, that element is set to np.nan. Set error_on_fail=True to
      raise instead.
    """

    # Broadcast P, Y, T to a common shape
    P = np.asarray(P_GPa, float)
    Y = np.asarray(Y, float)
    T = np.asarray(T_target, float)
    P_b, Y_b, T_b = np.broadcast_arrays(P, Y, T)

    Z1 = np.empty_like(P_b, dtype=float)
    Z2 = np.empty_like(P_b, dtype=float)

    # Flattened iteration with multi_index back to original shape
    it = np.nditer(P_b, flags=['multi_index'])
    for _ in it:
        idx = it.multi_index
        p = float(P_b[idx])
        y = float(Y_b[idx])
        t = float(T_b[idx])

        try:
            z1, z2 = invert_tmisc_hhe_mgsio3(
                p, y, t,
                xc=xc,
                M_H2=M_H2, M_He=M_He, M_rock=M_rock,
                Z_eps=Z_eps, tol=tol,
                max_bracket_subdiv=max_bracket_subdiv,
                Tc_mode=Tc_mode,
                Tc_P_switch=Tc_P_switch,
                Tc_blend_width_dex=Tc_blend_width_dex,
                Tc_avg_weight=Tc_avg_weight,
                interp_curve=interp_curve
            )
        except Exception as e:
            if error_on_fail:
                raise
            z1, z2 = fill_value, fill_value  # out-of-bounds indicator

        Z1[idx] = z1
        Z2[idx] = z2

    return Z1, Z2

#zmisc_table = np.load('misc/ztab_gilmore_misc_follow_melt.npz')
zmisc_table = np.load('misc/ztab_gilmore_misc.npz')

tgrid = zmisc_table['tgrid']
pgrid = zmisc_table['pgrid']
y_grid = zmisc_table['ygrid']
Z1_tab = zmisc_table['Z1']
Z2_tab = zmisc_table['Z2']

Z1_interp = RGI((y_grid, pgrid, tgrid), Z1_tab, method='linear',
                bounds_error=False, fill_value=None)
Z2_interp = RGI((y_grid, pgrid, tgrid), Z2_tab, method='linear',
                bounds_error=False, fill_value=None)

def get_zmisc_tab(p_GPa, t_K, y):
    args = (y, p_GPa, t_K)
    v_args = [np.atleast_1d(arg) for arg in args]
    pts = np.column_stack(v_args)
    result1 = Z1_interp(pts)
    result2 = Z2_interp(pts)
    if all(np.isscalar(arg) for arg in args):
        return result1.item(), result2.item()
    else:
        return result1, result2

def get_zmisc(P_GPa, T, y_array, fill_value=1.0, tab=True):
    """
    Get the miscibility parameter Z for given P, T, and y.
    Returns two Z values (Z1, Z2) for the miscibility gap.
    If no valid Z is found, returns fill_value for both.
    """
    # Ensure inputs are numpy arrays
    P_GPa = np.atleast_1d(P_GPa)
    T = np.atleast_1d(T)
    y_array = np.atleast_1d(y_array)

    # Get the shape of the broadcasted inputs
    out_shape = np.broadcast_shapes(P_GPa.shape, T.shape, y_array.shape)

    # Get the miscibility parameters from the table
    if tab:
        # Use the precomputed table for efficiency
        Z1, Z2 = get_zmisc_tab(P_GPa, T, y_array)
    else:
        Z1, Z2 = get_zmisc_inv(P_GPa, T, y_array, fill_value=fill_value)

    # If any value is NaN or out of bounds, replace with fill_value
    Z1 = np.where(np.isfinite(Z1), Z1, fill_value)
    Z2 = np.where(np.isfinite(Z2), Z2, fill_value)

    return Z1.reshape(out_shape), Z2.reshape(out_shape)

def get_dzdt_misc(P_GPa, T, y_array, dT=0.1, fill_value=1.0, tab=True):
    # T0 = T
    T_high = T*(1 + dT)
    T_low = T

    # T1 = T + dT
    # T2 = T - dT

    Z1_high, Z2_high = get_zmisc(P_GPa, T_high, y_array, fill_value=fill_value, tab=tab)
    Z1_low, Z2_low = get_zmisc(P_GPa, T_low, y_array, fill_value=fill_value, tab=tab)

    return (Z1_high - Z1_low)/(T_high - T_low), (Z2_high - Z2_low)/(T_high - T_low)

def get_dzdp_misc(P_GPa, T, y_array, dP=0.1, fill_value=1.0, tab=True):
    P_low = P_GPa
    P_high = P_GPa*(1 + dP)

    Z1_high, Z2_high = get_zmisc(P_high, T, y_array, fill_value=fill_value, tab=tab)
    Z1_low, Z2_low = get_zmisc(P_low, T, y_array, fill_value=fill_value, tab=tab)
    return (Z1_high - Z1_low)/(P_high - P_low), (Z2_high - Z2_low)/(P_high - P_low)