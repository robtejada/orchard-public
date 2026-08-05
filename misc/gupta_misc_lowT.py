import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar, newton
from scipy.interpolate import RegularGridInterpolator as RGI
from math import isfinite

# --- Setup and Constants ---

mh = 2.016 # Mass of molecular hydrogen in g/mol
mhe = 4.0026 # Mass of atomic helium in g/mol
mz = 18.01528 # Mass of water in g/mol

# Load the CSV file containing the interpolated data.
PTX_CC_intp = pd.read_csv('misc/Interpolated_Hydrogen_water_CC_est.csv')
# Create a copy for further modifications (if needed).
data_df = PTX_CC_intp.copy()

# --- Define Constants ---

# Constants related to the thermodynamic properties of the system.
# W_H and W_S appear to be energy-related constants (multiplied by 2 for appropriate scaling).
W_H = -299.54 * 2
W_S = -8.04 * 2
R_gas = 8.314  # Universal gas constant in J/(mol-K)

# --- Compute Derived Quantities ---

# Compute W_V for each row. This variable is derived using the critical temperature (T_c) and pressure (P_c)
# from the data. Note that pandas operations are applied element-wise.
data_df['W_V'] = (2 * R_gas * data_df['T_c'] - (W_H - W_S * data_df['T_c'])) / data_df['P_c']
W_V_interp = interp1d(data_df['T_c'], data_df['W_V'] , kind='linear', fill_value='extrapolate')

def get_pmisc(T, X):
    """
    T: Temperature in Kelvin
    X: is the number fraction of hydrogen...
    Returns: Pressure of demixing in GPa
    """
    temp1 = W_V_interp(T)
    temp3 = W_H - T * W_S
    temp2 = (R_gas * T * np.log(X / (1 - X))) / (2 * X - 1)
    return (temp2 - temp3) / temp1

def x_H(
    Y: float,
    Z: float,
    water_molar_mass: float = mz,
    hydrogen_molar_mass: float = mh,
    helium_molar_mass: float = mhe
    ) -> float:
    """
    Calculate the number fraction of water molecules in a hydrogen-helium-water mixture.

    In this model, the total mass is normalized to 1. The mass fractions are defined as follows:
      - Z: mass fraction of water (H2O).
      - Y: mass fraction of helium.
      - The remaining mass is due to hydrogen:
          hydrogen_mass_frac = (1 - Z) - Y * (1 - Z)
                              = (1 - Z) * (1 - Y).

    The number of moles for each component is computed by dividing the mass fraction by its
    respective molar mass. The water number fraction is then the ratio of the water moles to
    the total moles of all components.

    Args:
        Y (float): Mass fraction of helium (0 ≤ Y ≤ 1).
        Z (float): Mass fraction of water (0 ≤ Z ≤ 1).
        water_molar_mass (float): Molar mass of water (e.g., ~18.01528 g/mol).
        hydrogen_molar_mass (float, optional): Molar mass of hydrogen (default ~1.00794 g/mol).
        helium_molar_mass (float, optional): Molar mass of helium (default ~4.002602 g/mol).

    Returns:
        float: The number fraction (mole fraction) of water in the mixture.

    Raises:
        ValueError: If the computed total number of moles is zero (indicative of an input error).
    """
    # Compute effective mass fractions:
    # Non-water mass is divided between hydrogen and helium.
    # Here, (1 - Z) is the total mass fraction of hydrogen + helium.
    hydrogen_mass_frac = (1 - Z) * (1 - Y)
    helium_effective_mass_frac = (1 - Z) * Y

    # Calculate number of moles for each component.
    moles_hydrogen = hydrogen_mass_frac / hydrogen_molar_mass
    moles_helium   = helium_effective_mass_frac / helium_molar_mass
    moles_water    = Z / water_molar_mass

    # Total number of moles in the mixture.
    total_moles = moles_hydrogen + moles_helium + moles_water

    if np.any(total_moles) == 0:
        raise ValueError("The computed total number of moles is zero. "
                         "Check that mass fraction inputs are valid (non-negative and ≤ 1).")

    # Return the number fraction of water molecules.
    return moles_hydrogen / total_moles

def x_Z(
    Y: float,
    Z: float,
    water_molar_mass: float = mz,
    hydrogen_molar_mass: float = mh,
    helium_molar_mass: float = mhe
    ) -> float:
    """
    Calculate the number fraction of water molecules in a hydrogen-helium-water mixture.

    In this model, the total mass is normalized to 1. The mass fractions are defined as follows:
      - Z: mass fraction of water (H2O).
      - Y: mass fraction of helium.
      - The remaining mass is due to hydrogen:
          hydrogen_mass_frac = (1 - Z) - Y * (1 - Z)
                              = (1 - Z) * (1 - Y).

    The number of moles for each component is computed by dividing the mass fraction by its
    respective molar mass. The water number fraction is then the ratio of the water moles to
    the total moles of all components.

    Args:
        Y (float): Mass fraction of helium (0 ≤ Y ≤ 1).
        Z (float): Mass fraction of water (0 ≤ Z ≤ 1).
        water_molar_mass (float): Molar mass of water (e.g., ~18.01528 g/mol).
        hydrogen_molar_mass (float, optional): Molar mass of hydrogen (default ~1.00794 g/mol).
        helium_molar_mass (float, optional): Molar mass of helium (default ~4.002602 g/mol).

    Returns:
        float: The number fraction (mole fraction) of water in the mixture.

    Raises:
        ValueError: If the computed total number of moles is zero (indicative of an input error).
    """
    # Compute effective mass fractions:
    # Non-water mass is divided between hydrogen and helium.
    # Here, (1 - Z) is the total mass fraction of hydrogen + helium.
    hydrogen_mass_frac = (1 - Z) * (1 - Y)
    helium_effective_mass_frac = (1 - Z) * Y

    # Calculate number of moles for each component.
    moles_hydrogen = hydrogen_mass_frac / hydrogen_molar_mass
    moles_helium   = helium_effective_mass_frac / helium_molar_mass
    moles_water    = Z / water_molar_mass

    # Total number of moles in the mixture.
    total_moles = moles_hydrogen + moles_helium + moles_water

    if np.any(total_moles) == 0:
        raise ValueError("The computed total number of moles is zero. "
                         "Check that mass fraction inputs are valid (non-negative and ≤ 1).")

    # Return the number fraction of water molecules.
    return moles_water / total_moles

def x_to_Z(
    xZ: float,
    xY: float,
    water_molar_mass: float = mz,
    hydrogen_molar_mass: float = mh,
    helium_molar_mass: float = mz
) -> float:
    """
    Calculate the mass fraction of water in a hydrogen-helium-water mixture from the given number fractions.

    This function computes the water mass fraction (Z) by first calculating the mass contributions
    of water, helium, and hydrogen. The hydrogen number fraction is determined from the fact that the
    total number fraction must sum to 1:
        hydrogen_number_fraction = 1 - xZ - xY

    The mass fraction of water is then obtained by:

        Z = (xZ * water_molar_mass) /
            (hydrogen_number_fraction * hydrogen_molar_mass +
             xY * helium_molar_mass +
             xZ * water_molar_mass)

    Args:
        xZ (float): Mole fraction of water in the mixture (0 ≤ xZ ≤ 1).
        xY (float): Mole fraction of helium in the mixture (0 ≤ xY ≤ 1).
        water_molar_mass (float, optional): Molar mass of water. Default is 18.01528 g/mol.
        hydrogen_molar_mass (float, optional): Molar mass of hydrogen. Default is 1.00794 g/mol.
        helium_molar_mass (float, optional): Molar mass of helium. Default is 4.002602 g/mol.

    Returns:
        float: Mass fraction of water in the mixture.

    Raises:
        ValueError: If the sum of water and helium number fractions exceeds 1.
    """
    # Determine the number fraction of hydrogen
    hydrogen_number_fraction = 1 - xZ - xY

    # Ensure that the mixture fractions are physically sensible.
    if np.any(hydrogen_number_fraction) < 0:
        raise ValueError("The sum of water and helium number fractions exceeds 1, "
                         "resulting in a negative hydrogen number fraction.")

    # Calculate the total mass of the mixture using the molar masses:
    total_mass = (hydrogen_number_fraction * hydrogen_molar_mass +
                  xY * helium_molar_mass +
                  xZ * water_molar_mass)

    # Compute and return the water mass fraction:
    Z = (xZ * water_molar_mass) / total_mass
    return Z


def get_pmisc(T: float, Y: float, Z: float) -> float:

    """
    Calculate the pressure of demixing (in GPa) for a hydrogen-water mixture at a given temperature
    and hydrogen number (mole) fraction.

    Parameters:
        T: temperature (float): Temperature in Kelvin.
        X: hydrogen_mole_fraction (float): Number (mole) fraction of hydrogen in the mixture.
            (Should be between 0 and 1. Note that a value of 0.5 must be avoided as it causes a singularity.)

    Returns:
        float: Demixing pressure in GPa.

    Raises:
        ValueError: If the hydrogen mole fraction is near 0.5, where the calculation would result in division by zero.

    Calculation Details:
      - It uses an interpolated value of W_V from the critical data using the function W_V_interp.
      - The term (W_H - temperature * W_S) represents a temperature-dependent enthalpic contribution.
      - A logarithmic term based on the hydrogen mole fraction quantifies the chemical potential difference.
      - Finally, the demixing pressure is computed as:

            P_demixing = ( [R_gas * temperature * ln(hydrogen_mole_fraction/(1-hydrogen_mole_fraction))] / (2*hydrogen_mole_fraction - 1)
                           - (W_H - temperature*W_S) ) / W_V_interp(temperature)
    """

    # Interpolated value of W_V at the provided temperature.

    X = x_H(Y, Z)

    interpolated_W_V = W_V_interp(T)

    # Compute the enthalpic term.
    enthalpy_term = W_H - T * W_S

    # Compute the logarithmic term involving the hydrogen mole fraction.
    log_term = (R_gas * T * np.log(X / (1 - X))) / (2 * (X+1e-6) - 1)

    # Calculate and return the demixing pressure in GPa.
    P = (log_term - enthalpy_term) / interpolated_W_V
    valid_range = (T < 6000) & (T > 100)
    t_res = T[valid_range]
    p_res = P[valid_range]
    y_res = Y[valid_range]
    z_res = Z[valid_range]

    return p_res, t_res, y_res, z_res, valid_range

def get_pgap(P_GPa, T, Y, Z):
    """ This function returns the P-T miscibility gap points.
    This should return two points, P1 and P2, and the region
    between P1 and P2 is the H-He immiscbility region."""

    pmisc, tmisc, ymisc, zmisc, valid_idx_t = get_pmisc(T, Y, Z) # in GPa

    idx_critical = np.argwhere(np.diff(np.sign(P_GPa[valid_idx_t] - pmisc))).flatten()

    if len(idx_critical) == 0:
        return None, None

    return pmisc[idx_critical], tmisc[idx_critical]

#################### INVERSION TO GET ZMISC(P, T, Y) ####################

EPS_X          = 1e-10       # for clipping X
SING_EPS       = 1e-6        # threshold for |2X-1| to use analytic limit
ADAPT_MAX_DEPTH= 12
ADAPT_MIN_SIZE = 1e-4        # minimum bracket width in z
MERGE_TOL      = 5e-5
BOUNDARY_PAD   = 5e-4        # discard roots too close to z_min/z_max unless only one found

def safe_log_fraction_term(X, T):
    # X assumed in (0,1)
    denom = 2.0*X - 1.0
    if abs(denom) < SING_EPS:
        return 2.0 * R_gas * T   # analytic limit
    return R_gas * T * np.log(X/(1.0 - X)) / denom

def pmisc_scalar(T: float, Y: float, Z: float) -> float:
    # hydrogen mole fraction
    X = x_H(Y, Z)
    # enforce domain
    if not np.isfinite(X) or X <= 0.0 or X >= 1.0:
        return np.nan

    X = min(1.0 - EPS_X, max(EPS_X, X))
    log_term = safe_log_fraction_term(X, T)
    enthalpy_term = W_H - T * W_S
    W_V = W_V_interp(T)
    return (log_term - enthalpy_term) / W_V

# def pmisc_scalar(T: float, Y: float, Z: float) -> float:

#     # Interpolated value of W_V at the provided temperature.

#     X = x_H(Y, Z)

#     interpolated_W_V = W_V_interp(T)

#     # Compute the enthalpic term.
#     enthalpy_term = W_H - T * W_S

#     # Compute the logarithmic term involving the hydrogen mole fraction.
#     log_term = (R_gas * T * np.log(X / (1 - X))) / (2 * (X+1e-6) - 1)

#     # Calculate and return the demixing pressure in GPa.
#     P = (log_term - enthalpy_term) / interpolated_W_V
#     return P

def adaptive_brackets(f, a, b, f_a=None, f_b=None, depth=0, max_depth=ADAPT_MAX_DEPTH, out=None):
    """
    Recursively find sign-change brackets for f on [a,b].
    """
    if out is None:
        out = []
    if f_a is None:
        f_a = f(a)
    if f_b is None:
        f_b = f(b)
    if not (np.isfinite(f_a) and np.isfinite(f_b)):
        return out
    # No sign change and function large => stop
    if f_a * f_b > 0:
        # Optionally subdivide if interval still large and magnitude small (to catch missed zero)
        if depth < max_depth and (b - a) > ADAPT_MIN_SIZE and (abs(f_a) < 5 or abs(f_b) < 5):
            m = 0.5*(a+b)
            f_m = f(m)
            adaptive_brackets(f, a, m, f_a, f_m, depth+1, max_depth, out)
            adaptive_brackets(f, m, b, f_m, f_b, depth+1, max_depth, out)
        return out
    # Sign change bracket
    out.append((a,b))
    return out

def filter_roots(z_roots, z_min, z_max):
    z_roots = np.array(sorted(z_roots))
    # merge
    filtered = []
    for z in z_roots:
        if not filtered or abs(z - filtered[-1]) > MERGE_TOL:
            filtered.append(z)
    # boundary filter
    tmp = []
    for z in filtered:
        if (z - z_min) < BOUNDARY_PAD or (z_max - z) < BOUNDARY_PAD:
            # keep only if we have no other good interior roots
            continue
        tmp.append(z)
    # fallback if all removed
    if not tmp and filtered:
        tmp = filtered
    # enforce at most two: extremes
    if len(tmp) > 2:
        tmp = [tmp[0], tmp[-1]]
    return tmp

def get_zmisc_inv(p_array, t_array, y_array,
              z_min=0.0001, z_max=0.9999,
              method='brentq',
              coarse_points=32,
              ):
    """
    Return array shape (2, ...) of (Z1, Z2) roots to pmisc_scalar(T,Y,Z)=p.
    If <2 roots, fill remaining slot(s) with 1.0.
    """
    p_array   = np.atleast_1d(p_array)
    t_array   = np.atleast_1d(t_array)
    y_array   = np.atleast_1d(y_array)
    p_array, t_array, y_array = np.broadcast_arrays(p_array, t_array, y_array)
    out_shape = p_array.shape

    p_flat = p_array.ravel()
    t_flat = t_array.ravel()
    y_flat = y_array.ravel()

    n = p_flat.size
    out = np.ones((n, 2), dtype=float)

    # Precompute a coarse z grid (used to seed adaptivity per point)
    coarse_z = np.linspace(z_min, z_max, coarse_points)

    for i in range(n):
        p_val = p_flat[i]; T = t_flat[i]; Y = y_flat[i]

        def f(z):
            Pm = pmisc_scalar(T, Y, z)
            if not np.isfinite(Pm):
                return np.nan
            return Pm - p_val

        # Quick sanity: if function invalid everywhere, skip
        f_vals = [f(zc) for zc in coarse_z]
        if all((not np.isfinite(v)) for v in f_vals):
            continue

        # Build initial candidate brackets adaptively over whole interval
        brackets = adaptive_brackets(f, z_min, z_max)

        # For low-noise, also *seed* with sign changes found in coarse grid
        for k in range(len(coarse_z)-1):
            zL, zR = coarse_z[k], coarse_z[k+1]
            fL, fR = f_vals[k], f_vals[k+1]
            if np.isfinite(fL) and np.isfinite(fR) and fL * fR < 0:
                brackets.append((zL, zR))

        # Deduplicate brackets
        uniq = []
        for a,b in brackets:
            if a > b: a,b = b,a
            keep = True
            for (u,v) in uniq:
                if abs(a-u) < 1e-6 and abs(b-v) < 1e-6:
                    keep = False
                    break
            if keep:
                uniq.append((a,b))
        brackets = uniq

        roots = []
        for (a,b) in brackets:
            try:
                fa = f(a); fb = f(b)
                if not (np.isfinite(fa) and np.isfinite(fb)):
                    continue
                if fa * fb > 0:
                    continue
                sol = root_scalar(f, bracket=[a,b], method=method, xtol=1e-10, rtol=1e-8, maxiter=100)
                if sol.converged and z_min <= sol.root <= z_max and isfinite(f(sol.root)):
                    roots.append(sol.root)
            except Exception:
                pass

        roots = filter_roots(roots, z_min, z_max)

        if len(roots) == 1:
            out[i,0] = roots[0]
        elif len(roots) >= 2:
            out[i,0], out[i,1] = roots[0], roots[1]

    # Reshape to (2, ...)
    return out.T.reshape((2,) + out_shape)



zmisc_table = np.load('misc/ztab_gupta_misc_lowT.npz')

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
        Z1, Z2 = get_zmisc_inv(P_GPa, T, y_array)

    # If any value is NaN or out of bounds, replace with fill_value
    Z1 = np.where(np.isfinite(Z1), Z1, fill_value)
    Z2 = np.where(np.isfinite(Z2), Z2, fill_value)

    return Z1.reshape(out_shape), Z2.reshape(out_shape)

# def find_sign_change_intervals(func, a, b, num_points=20):
#     """
#     Sample `func(z)` at `num_points` between a and b.
#     Return list of (z_left, z_right) brackets where func changes sign.
#     """
#     z_vals = np.linspace(a, b, num_points)
#     f_vals = [func(z) for z in z_vals]

#     intervals = []
#     for i in range(len(z_vals) - 1):
#         if f_vals[i] * f_vals[i+1] < 0:
#             intervals.append((z_vals[i], z_vals[i+1]))
#     return intervals


# def get_zmisc(p_array, t_array, y_array,
#               z_min=0.001, z_max=0.999,
#               method='brentq',
#               num_brackets=250,
#               # Default guesses for Newton’s method:
#               guess1=0.3,
#               guess2=0.8,
#               fill_value=1.0,
#               tab=False):
#     """
#     Solve p_misc(T, y, z) = p for z, allowing up to TWO solutions per point (p, T, y).

#     Tries Newton’s method first with two initial guesses; if that fails (or yields
#     out-of-bounds or duplicate roots), falls back to the brentq bracket approach.

#     Returns an array of shape (2, ...) with up to two roots for each point.
#       - The first row is the first root,
#       - The second row is the second root,
#     in the same broadcast shape as p_array, t_array, y_array.

#     If no root is found, store [1.0, 1.0].
#     If exactly one root is found, store [z_root, 1.0].
#     If two or more are found, store the first two.
#     """
#     # 1) Convert to np arrays, broadcast to common shape
#     p_array   = np.atleast_1d(p_array)
#     t_array   = np.atleast_1d(t_array)
#     y_array   = np.atleast_1d(y_array)

#     p_array, t_array, y_array = np.broadcast_arrays(p_array, t_array, y_array)
#     out_shape = p_array.shape  # shape of the broadcast

#     # 2) Flatten so we can loop once
#     p_flat = p_array.ravel()
#     t_flat = t_array.ravel()
#     y_flat = y_array.ravel()

#     # We'll store two solutions per point in a 2D array: shape (Npoints, 2)
#     n_points = p_flat.size
#     z_solutions_flat = np.ones((n_points, 2), dtype=float)  # default fill: 1.0

#     # 3) Loop over each point
#     for i in range(n_points):
#         p_val = p_flat[i]
#         t_val = t_flat[i]
#         y_val = y_flat[i]

#         # We'll define the root function f(z) = pmisc_scalar(T,y,z) - p
#         def f(z):
#             return pmisc_scalar(t_val, y_val, z) - p_val

#         # -- Step A: Attempt two-root solution via Newton’s method --
#         use_brentq = False
#         newton_roots = []

#         # Helper for checking validity of a root from Newton
#         def is_valid_root(zr):
#             return (z_min <= zr <= z_max) and np.isfinite(zr)

#         try:
#             # 1) Root from guess1
#             root1 = newton(f, guess1, tol=1e-8, maxiter=200)
#             if not is_valid_root(root1):
#                 raise ValueError("Newton root1 out-of-bounds")

#             # 2) Root from guess2
#             root2 = newton(f, guess2, tol=1e-8, maxiter=200)
#             if not is_valid_root(root2):
#                 raise ValueError("Newton root2 out-of-bounds")

#             # Check if the two solutions are distinct enough
#             # if abs(root1 - root2) < 1e-10:
#             #     # If they collapse to same solution, treat as single solution or fail
#             #     raise ValueError("Newton found only a single unique solution")

#             # Both valid solutions found
#             newton_roots = sorted([root1, root2])  # sort if you like
#         except Exception:
#             # If any newton call fails, we’ll fallback to brentq
#             use_brentq = True

#         # -- Step B: If Newton’s method fails or gives <2 solutions, fallback to brentq --
#         if use_brentq or len(newton_roots) < 2:
#             # 4) Find intervals in [z_min, z_max] where f changes sign
#             intervals = find_sign_change_intervals(f, z_min, z_max, num_points=num_brackets)

#             z_roots = []
#             for (z_a, z_b) in intervals:
#                 sol = root_scalar(f, bracket=[z_a, z_b], method=method)
#                 if sol.converged:
#                     z_roots.append(sol.root)
#             if len(z_roots) == 0:
#                 # No solutions => store [1.0, 1.0]
#                 z_solutions_flat[i, :] = [fill_value, fill_value]
#             elif len(z_roots) == 1:
#                 # One solution => [z_root, 1.0]
#                 z_solutions_flat[i, :] = [z_roots[0], fill_value]
#             else:
#                 # 2 or more solutions => take the first two
#                 z_solutions_flat[i, 0] = z_roots[0]
#                 z_solutions_flat[i, 1] = z_roots[1]
#         else:
#             # We got two valid, distinct Newton solutions
#             z_solutions_flat[i, 0] = newton_roots[0]
#             z_solutions_flat[i, 1] = newton_roots[1]

#     # 6) Reshape to produce final array of shape (2, out_shape...)
#     z_solutions_out = z_solutions_flat.T  # shape = (2, n_points)
#     z_solutions_out = z_solutions_out.reshape((2,) + out_shape)

#     return z_solutions_out

def get_dzdt_misc(P_GPa, T, y_array, dT=0.1, tab=True):
    # T0 = T
    T_high = T*(1 + dT)
    T_low = T

    # T1 = T + dT
    # T2 = T - dT

    Z1_high, Z2_high = get_zmisc(P_GPa, T_high, y_array, tab=tab)
    Z1_low, Z2_low = get_zmisc(P_GPa, T_low, y_array, tab=tab)

    return (Z1_high - Z1_low)/(T_high - T_low), (Z2_high - Z2_low)/(T_high - T_low)

def get_dzdp_misc(P_GPa, T, y_array, dP=0.1, tab=True):
    P_low = P_GPa
    P_high = P_GPa*(1 + dP)

    Z1_high, Z2_high = get_zmisc(P_high, T, y_array, tab=tab)
    Z1_low, Z2_low = get_zmisc(P_low, T, y_array, tab=tab)
    return (Z1_high - Z1_low)/(P_high - P_low), (Z2_high - Z2_low)/(P_high - P_low)