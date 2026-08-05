import numpy as np
from scipy.interpolate import interp1d
from scipy.interpolate import RegularGridInterpolator as RGI
from scipy.optimize import root_scalar, newton

#The following are the parameters estimated based on DFT calculations
W_V = -13.06
W_U = -299.54
W_S = -8.04
W_V2 = 490.89
lambda_X1 = 2.62
lambda_X2 = -0.68

T_c_mock_arr = np.linspace(750,6000,1000)
P_c_mock_arr = np.zeros(len(T_c_mock_arr))
X_c_mock_arr = np.zeros(len(T_c_mock_arr))

R_const = 8.314


mh = 2.016
mhe = 4.0026
mz = 18.01528

def Y_to_x(_z):
    ''' Change between mass and number fraction OF WATER'''
    return ((_z/mz)/(((1 - _z)/mh) + (_z/mz)))

def x_H(_y, _z, mz):
    ''' Obtains number fraction of hydrogen in a XYZ mixture'''
    Ntot = (1-_y)*(1-_z)/mh + (_y*(1-_z)/mhe) + _z/mz
    return (1-_y)*(1-_z)/mh/Ntot

def x_Z(_y, _z, mz):
    Ntot = (1-_y)*(1-_z)/mh + (_y*(1-_z)/mhe) + _z/mz
    return (_z/mz)/Ntot

################################################################################################
# This function is basically f(X_H2,T,P) = 0; X_H2O = 1 - X_H2
################################################################################################
def est_W_params_w_Y(T_d, Y_d, W_V, W_U, W_S, W_V2=0, W_V3=0, W_V4=0, flag_return_value=0):

    temp1 = (W_V + (((T_d/1000)**(-2))*W_V2))
    temp2 = (R_const*T_d*(np.log( Y_d/(1-Y_d) )) )/(2.*(2*Y_d - 1))
    temp3 = W_U - (T_d)*W_S
    P_est = (temp2 - temp3)/temp1

    if flag_return_value == 1:
        return temp1 # W_V_eff
    if flag_return_value == 2:
        return temp3 # W_0 or W_U - T*W_S
    else:
        return P_est
################################################################################################
################################################################################################
def est_lambda_X_eff(T_d, lambda_X1, lambda_X2, flag_return_value=0):

    lambda_X_eff = lambda_X1 + (lambda_X2/(T_d/1000))

    return lambda_X_eff
################################################################################################
################################################################################################

def get_pmisc(T, y, z):
    """
    If `T` is an array, returns (p_range, t_range) arrays.
    If `T` is a float, returns a single float p_value.
    """
    # Safely convert `T` to an array but remember if it was scalar:
    t_arr = np.atleast_1d(T)

    n_x = x_H(y, z, mz)
    n_z = x_Z(y, z, mz)
    lambda_X_eff = est_lambda_X_eff(t_arr, lambda_X1, lambda_X2)

    Y = n_x/(n_x + lambda_X_eff*n_z)  # This might need broadcasting if t_arr is bigger

    pmisc_array = est_W_params_w_Y(t_arr, Y, W_V, W_U, W_S, W_V2=W_V2)

    # If we got a single T, return pmisc_array as a float
    if t_arr.size == 1:
        return pmisc_array.item()  # Convert array of shape (1,) to scalar

    # Otherwise, do the existing logic that filters over T, etc.
    p_res = pmisc_array
    # e.g. p_res = np.array([np.min(pmisc_array[i]) for i in range(len(t_arr))])
    # or if pmisc_array is already 1D, just use p_res = pmisc_array

    valid_idx_t = (T >= 750) & (T <= 6000)

    p_range = p_res[valid_idx_t]

    valid_idx_p = (p_range >= 0.2) & (p_range <= 2000)
    t_range = T[valid_idx_t]

    return p_range[valid_idx_p], t_range[valid_idx_p], y[valid_idx_t][valid_idx_p], z[valid_idx_t][valid_idx_p], valid_idx_t # returns in GPa

def get_pgap(P_GPa, T, y, z):
    """ This function returns the P-T miscibility gap points.
    This should return two points, P1 and P2, and the region
    between P1 and P2 is the H-He immiscbility region."""

    pmisc, tmisc, ymisc, zmisc, valid_idx_t, valid_idx_p = get_pmisc(T, y, z) # in GPa
    idx_critical = np.argwhere(np.diff(np.sign(P_GPa[valid_idx_t][valid_idx_p] - pmisc))).flatten()

    if len(idx_critical) == 0:
        return None, None

    return pmisc[idx_critical], tmisc[idx_critical]

def pmisc_scalar(T, y, z):
    """
    Returns the immiscibility pressure (in GPa) at *one* temperature T,
    for given mass fractions y (He) and z (H2O).
    """
    # 1) Compute number fractions
    n_x = x_H(y, z, mz)
    n_z = x_Z(y, z, mz)

    # 2) Effective lambda
    lambda_eff = est_lambda_X_eff(T, lambda_X1, lambda_X2)

    # 3) The Y needed in W-params
    Y_val = n_x / (n_x + lambda_eff * n_z)

    # 4) Call the W-params routine to get one pressure
    p_val = est_W_params_w_Y(T, Y_val, W_V, W_U, W_S, W_V2=W_V2)

    return p_val # in GPa

def get_tmisc(p_array, y_array, z_array,
              T_min=750, T_max=6000,
              method='brentq'):
    """
    Solve p_misc(T, y, z) = p for T (P in GPa),
    for each element of p_array, y_array, z_array (broadcasted).
    Returns an array of same shape.
    """
    # Make sure inputs are np arrays
    p_array = np.atleast_1d(p_array)
    y_array = np.atleast_1d(y_array)
    z_array = np.atleast_1d(z_array)

    # Broadcast them to a common shape
    p_array, y_array, z_array = np.broadcast_arrays(p_array, y_array, z_array)
    out_shape = p_array.shape

    # Prepare a 1D list to fill with solutions
    T_solutions_flat = np.empty(p_array.size, dtype=float)

    # Flatten so we can loop once and then reshape back
    p_flat = p_array.ravel()
    y_flat = y_array.ravel()
    z_flat = z_array.ravel()

    # Loop over each point in flattened arrays
    for i in range(p_flat.size):
        pval = p_flat[i]
        yval = y_flat[i]
        zval = z_flat[i]

        # The function we want to be zero
        def f_root(T):
            return pmisc_scalar(T, yval, zval) - pval

        # Solve on [T_min, T_max]
        sol = root_scalar(f_root, bracket=[T_min, T_max], method=method)

        if not sol.converged:
            raise RuntimeError(
                f"Could not converge for p={pval}, y={yval}, z={zval} "
                f"within T=[{T_min}, {T_max}]"
            )

        T_solutions_flat[i] = sol.root

    # Reshape to original broadcast shape
    return T_solutions_flat.reshape(out_shape)

def find_sign_change_intervals(func, a, b, num_points=20):
    """
    Sample `func(z)` at `num_points` between a and b.
    Return list of (z_left, z_right) brackets where func changes sign.
    """
    z_vals = np.linspace(a, b, num_points)
    f_vals = [func(z) for z in z_vals]

    intervals = []
    for i in range(len(z_vals) - 1):
        if f_vals[i] * f_vals[i+1] < 0:
            intervals.append((z_vals[i], z_vals[i+1]))
    return intervals


def get_zmisc_inv(p_array, t_array, y_array,
              z_min=0.001, z_max=0.999,
              method='brentq',
              num_brackets=250,
              # Default guesses for Newton’s method:
              guess1=0.3,
              guess2=0.8,
              fill_value=1.0):
    """
    Solve p_misc(T, y, z) = p for z, allowing up to TWO solutions per point (p, T, y).

    Tries Newton’s method first with two initial guesses; if that fails (or yields
    out-of-bounds or duplicate roots), falls back to the brentq bracket approach.

    Returns an array of shape (2, ...) with up to two roots for each point.
      - The first row is the first root,
      - The second row is the second root,
    in the same broadcast shape as p_array, t_array, y_array.

    If no root is found, store [1.0, 1.0].
    If exactly one root is found, store [z_root, 1.0].
    If two or more are found, store the first two.
    """
    # 1) Convert to np arrays, broadcast to common shape
    p_array   = np.atleast_1d(p_array)
    t_array   = np.atleast_1d(t_array)
    y_array   = np.atleast_1d(y_array)

    p_array, t_array, y_array = np.broadcast_arrays(p_array, t_array, y_array)
    out_shape = p_array.shape  # shape of the broadcast

    # 2) Flatten so we can loop once
    p_flat = p_array.ravel()
    t_flat = t_array.ravel()
    y_flat = y_array.ravel()

    # We'll store two solutions per point in a 2D array: shape (Npoints, 2)
    n_points = p_flat.size
    z_solutions_flat = np.ones((n_points, 2), dtype=float)  # default fill: 1.0

    # 3) Loop over each point
    for i in range(n_points):
        p_val = p_flat[i]
        t_val = t_flat[i]
        y_val = y_flat[i]

        # We'll define the root function f(z) = pmisc_scalar(T,y,z) - p
        def f(z):
            return pmisc_scalar(t_val, y_val, z) - p_val

        # -- Step A: Attempt two-root solution via Newton’s method --
        use_brentq = False
        newton_roots = []

        # Helper for checking validity of a root from Newton
        def is_valid_root(zr):
            return (z_min <= zr <= z_max) and np.isfinite(zr)

        try:
            # 1) Root from guess1
            root1 = newton(f, guess1, tol=1e-8, maxiter=200)
            if not is_valid_root(root1):
                raise ValueError("Newton root1 out-of-bounds")

            # 2) Root from guess2
            root2 = newton(f, guess2, tol=1e-8, maxiter=200)
            if not is_valid_root(root2):
                raise ValueError("Newton root2 out-of-bounds")

            # Check if the two solutions are distinct enough
            # if abs(root1 - root2) < 1e-10:
            #     # If they collapse to same solution, treat as single solution or fail
            #     raise ValueError("Newton found only a single unique solution")

            # Both valid solutions found
            newton_roots = sorted([root1, root2])  # sort if you like
        except Exception:
            # If any newton call fails, we’ll fallback to brentq
            use_brentq = True

        # -- Step B: If Newton’s method fails or gives <2 solutions, fallback to brentq --
        if use_brentq or len(newton_roots) < 2:
            # 4) Find intervals in [z_min, z_max] where f changes sign
            intervals = find_sign_change_intervals(f, z_min, z_max, num_points=num_brackets)

            z_roots = []
            for (z_a, z_b) in intervals:
                sol = root_scalar(f, bracket=[z_a, z_b], method=method)
                if sol.converged:
                    z_roots.append(sol.root)
            if len(z_roots) == 0:
                # No solutions => store [1.0, 1.0]
                z_solutions_flat[i, :] = [fill_value, fill_value]
            elif len(z_roots) == 1:
                # One solution => [z_root, 1.0]
                z_solutions_flat[i, :] = [z_roots[0], fill_value]
            else:
                # 2 or more solutions => take the first two
                z_solutions_flat[i, 0] = z_roots[0]
                z_solutions_flat[i, 1] = z_roots[1]
        else:
            # We got two valid, distinct Newton solutions
            z_solutions_flat[i, 0] = newton_roots[0]
            z_solutions_flat[i, 1] = newton_roots[1]

    # 6) Reshape to produce final array of shape (2, out_shape...)
    z_solutions_out = z_solutions_flat.T  # shape = (2, n_points)
    z_solutions_out = z_solutions_out.reshape((2,) + out_shape)

    return z_solutions_out

zmisc_table = np.load('misc/ztab_gupta_misc.npz')

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