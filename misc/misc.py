from misc import misc_sch, misc_schmod, misc_lor
import numpy as np
from scipy.interpolate import interp1d
from scipy.interpolate import RegularGridInterpolator as RGI
from scipy.optimize import root, root_scalar

""" This file calls the individual miscibility curves and 
    contains wrappers to get the miscibility curves, the
    immiscibility regions, and the immiscible helium mass
    fractions. """

mh = 1
mhe = 4.0026

def x_to_Y(x):
    # x is the helium number fraction
    # converts number fraction to mass fraction
    return (mhe*x/(mh*(1-x) + mhe*x))

def Y_to_x(Y):
    return (Y/mhe/(Y/mhe + (1-Y)/mh))

def get_misc_p(logp, Y, misc, delta_T, bry_sigma=0, interp='linear', pad=False):
    """ This function returns the P-T miscibility 
        curve. Arrays only"""
    deltaT_mod = delta_T*1e-3
    if misc=='s':
        pmisc, tmisc = misc_sch.get_misc_curve(logp, Y, misc_interp=interp, pad=pad)
        if pmisc is None:
            return None, None
        return pmisc, tmisc+deltaT_mod
    # elif misc=='smod':
    #     pmisc, tmisc = misc_schmod.get_misc_curve(logp, Y, misc_interp=interp, pad=pad)
    #     if pmisc is None:
    #         return None, None
    elif misc=='l':
        pmisc, tmisc = misc_lor.get_misc_curve(logp, Y, misc_interp=interp, pad=pad)
        if pmisc is None:
            return None, None
        return pmisc, tmisc+deltaT_mod
    else:
        print('misc. curve must be s, or l')

def get_pgap(logp, logt, y, misc = 's', delta_T = 0, sigma=0, interp='linear', pad=False):
    """ This function returns the P-T miscibility gap points. 
    This should return two points, P1 and P2, and the region
    between P1 and P2 is the H-He immiscbility region."""
    # delta_T in 10^3 K
    interp_t = interp1d(10**(logp-12), 10**(logt-3), kind='linear')
    pmisc, tmisc = get_misc_p(logp, y, misc, delta_T, bry_sigma=sigma, interp=interp, pad=pad)
    if pmisc is None:
        return None
    tnew = interp_t(pmisc[pmisc < max(10**(logp-12))])
    tmisc = tmisc[pmisc < max(10**(logp-12))]
    #pnew = pmisc[pmisc < max(10**(logp-12))]

    idx = np.argwhere(np.diff(np.sign(tnew - tmisc))).flatten()
    
    if len(idx) > 2:
        return pmisc[idx[-2:]]*1e12
    elif len(idx) == 2:
        return pmisc[idx]*1e12
    elif len(idx) == 1:
        return np.array([1e12, pmisc[int(idx)]*1e12])
    elif len(idx) == 0:
        return None

def get_t_py(logp, y, deltaT, misc_curve, interp='linear'):
    # delta T needs to be in K now
    xval = Y_to_x(y)

    if misc_curve == 's':
        misc_mod = misc_sch
    elif misc_curve == 'l':
        misc_mod = misc_lor

    elif misc_curve == 'smod':
        #return misc_schmod.get_t_misc(logp, y, misc_interp=interp)*1000 + deltaT
        misc_mod = misc_schmod

    return misc_mod.get_t_misc(logp, y, misc_interp=interp)*1000 + deltaT

def x_err(yval, logpval, logtval, deltaT, misc_curve, interp):

    tres = get_t_py(logpval, yval, deltaT, misc_curve=misc_curve, interp=interp)

    return (tres/10**logtval) - 1

def get_y_misc(logp, logt, deltaT, misc_curve, interp='linear'):
    """ This function inverts T(P, Y) and returns Y(P, T) for
    the miscibility curves."""
    #p, t = 10**(logp-12), 10**(logt-3)
    if np.isscalar(logp):
        logp, logt, deltaT = np.array([logp]), np.array([logt]), np.array([deltaT])
        sol = root(x_err, [0.2], tol=1e-10, method='hybr', args=(logp, logt, deltaT, misc_curve, interp))
        return float(sol.x)

    else:
        sol = root(x_err, np.zeros(len(logp))+0.2, tol=1e-10, method='hybr', args=(logp, logt, deltaT, misc_curve, interp))
        Ym_calc = sol.x

        return Ym_calc

logpgrid_l = np.linspace(12, np.log10(24*1e12), 100) # from 1 to 24 Mbar
logtgrid_l = np.linspace(np.log10(3e3), np.log10(15e3), 150) # from 3kK to 15kK (tempporary)
deltaT_grid_l = np.arange(-2000, 5000, 50) # in Kelvin

y_misc_res_linear_l = np.load('misc/lorenzen_ymisc_pt_tables_deltaT_updated.npy')

get_y_misc_rgi_linear_l = RGI((logpgrid_l, deltaT_grid_l, logtgrid_l), y_misc_res_linear_l, method='linear', \
            bounds_error=False, fill_value=None)

logpgrid_s = np.linspace(12, np.log10(24e12), 100)
logtgrid_s = np.linspace(np.log10(2e3), np.log10(15e3), 150)

deltaT_grid_s = np.arange(-1000, 7000, 50)

y_misc_res_linear_s = np.load('misc/sr18_ymisc_pt_tables_deltaT_updated.npy')

get_y_misc_rgi_linear_s = RGI((logpgrid_s, deltaT_grid_s, logtgrid_s), y_misc_res_linear_s, method='linear', \
            bounds_error=False, fill_value=None)

logtgrid_smod = np.linspace(np.log10(2e3), np.log10(15e3), 150)

y_misc_res_linear_smod = np.load('misc/sr18mod_ymisc_pt_tables_deltaT_updated.npy')

get_y_misc_rgi_linear_smod = RGI((logpgrid_s, deltaT_grid_s, logtgrid_smod), y_misc_res_linear_smod, method='linear', \
            bounds_error=False, fill_value=None)

def _shift_logp(logp, deltaP):
    """Shift the miscibility curve to higher pressure by deltaP (dyn/cm^2).

    The curve is evaluated at P_eff = P - deltaP, so its features appear
    deltaP deeper in the planet. Where P_eff falls at or below zero the
    effective log-pressure is set to 0, which the logp_eff < 12 clamps in
    get_y_misc_pt then mark as fully miscible (Ym = 1).
    """
    if deltaP == 0.0:
        return logp
    p_eff = 10.0 ** np.asarray(logp, dtype=float) - deltaP
    logp_eff = np.where(p_eff > 1.0, np.log10(np.maximum(p_eff, 1.0)), 0.0)
    return float(logp_eff) if np.isscalar(logp) else logp_eff

# Maximum deltaT covered by the precomputed tables (see deltaT_grid_* above).
_DELTAT_TAB_MAX = {'l': 4950.0, 's': 6950.0, 'smod': 6950.0}

def get_y_misc_pt(logp, logt, deltaT, misc_curve='l', tab=True, deltaP=0.0):
    """
    logp in log10 of dyn/cm^2 and logt in log 10 of K.
    deltaP (dyn/cm^2, default 0) shifts the curve to higher pressure: the
    lookup happens at P - deltaP. The high-temperature validity ceiling
    shifts with deltaT, consistent with the curve being displaced by
    deltaT in temperature.

    deltaT beyond the precomputed table grid is applied exactly via the
    identity Y_misc(P, T; deltaT) = Y_misc(P, T - dT_extra; deltaT_tab):
    the in-grid part uses the table's deltaT dimension and the remainder
    shifts the temperature argument. Within the grid, behavior is
    unchanged.
    """
    logp = _shift_logp(logp, deltaP)
    dt_cap = _DELTAT_TAB_MAX.get(misc_curve, 0.0)
    dT_extra = max(deltaT - dt_cap, 0.0)
    if dT_extra > 0.0:
        deltaT = dt_cap
        t_eff = 10.0 ** np.asarray(logt, dtype=float) - dT_extra
        logt_shifted = np.where(t_eff > 1.0, np.log10(np.maximum(t_eff, 1.0)), 0.0)
        # logt_shifted = 0 (T_eff <= 1 K) falls below the cold-end clamps,
        # which mark the region fully miscible (Ym = 1).
        logt = float(logt_shifted) if np.isscalar(logt) else logt_shifted
    if not tab:
        deltaT_arr = np.full_like(logp, deltaT)
        #if misc_curve=='l'
        return get_y_misc(logp, logt, deltaT_arr, misc_curve=misc_curve, interp='linear')
    else:
        #if interp=='linear':
        if np.isscalar(logp):
            deltaT_arr = np.full_like(logp, deltaT)
            if misc_curve == 'l':
                Ym_tab_res = float(get_y_misc_rgi_linear_l(np.array([logp, deltaT_arr, logt]).T))

            elif misc_curve == 's':
                Ym_tab_res = float(get_y_misc_rgi_linear_s(np.array([logp, deltaT_arr, logt]).T))

            elif misc_curve == 'smod':
                Ym_tab_res = float(get_y_misc_rgi_linear_smod(np.array([logp, deltaT_arr, logt]).T))
            else:
                raise Exception('Only l and s allowed here!')

            return Ym_tab_res
        else:
            deltaT_arr = np.full_like(logp, deltaT)
            if misc_curve == 'l':
                Ym_tab_res = get_y_misc_rgi_linear_l(np.array([logp, deltaT_arr, logt]).T)

                Ym_tab_res[logp < 12] = 1.0
                Ym_tab_res[logt > np.log10(14e3 + deltaT)] = 1.0
                Ym_tab_res[logt < np.log10(4e3)] = 1.0

                return Ym_tab_res
            elif misc_curve == 's':
                Ym_tab_res = get_y_misc_rgi_linear_s(np.array([logp, deltaT_arr, logt]).T)

                Ym_tab_res[logp < 12] = 1.0
                Ym_tab_res[logt > np.log10(12e3 + deltaT)] = 1.0
                Ym_tab_res[logt < np.log10(2e3)] = 1.0

                return Ym_tab_res
            elif misc_curve == 'smod':
                Ym_tab_res = get_y_misc_rgi_linear_smod(np.array([logp, deltaT_arr, logt]).T)

                Ym_tab_res[logp < 12] = 1.0
                Ym_tab_res[logt > np.log10(12e3 + deltaT)] = 1.0
                Ym_tab_res[logt < np.log10(2e3)] = 1.0

                return Ym_tab_res
            else:
                raise Exception('Only l and s allowed here!')


### composition derivatives for implicit scheme ###
def get_dydt_misc(logp, logt, deltaT, misc_curve, dt=0.1, tab=False, deltaP=0.0):
    T0 = 10**logt
    T1 = T0*(1+dt)
    #if interp == 'linear':
    Y0 = get_y_misc_pt(logp, logt, deltaT, misc_curve, tab, deltaP=deltaP)
    Y1 = get_y_misc_pt(logp, np.log10(T1), deltaT, misc_curve, tab, deltaP=deltaP)

    deriv = (Y1 - Y0)/(T1 - T0)

    # zero below the (shifted) 1 Mbar floor of the curve
    logp_eff = _shift_logp(logp, deltaP)
    deriv[logp_eff < 12] = np.zeros_like(logp[logp_eff < 12])

    return deriv

def get_dydp_misc(logp, logt, deltaT, misc_curve, dp=0.1, tab=False, deltaP=0.0):
    P0 = 10**logp
    P1 = P0*(1+dp)
    #if interp == 'linear':
    Y0 = get_y_misc_pt(logp, logt, deltaT, misc_curve, tab, deltaP=deltaP)
    Y1 = get_y_misc_pt(np.log10(P1), logt, deltaT, misc_curve, tab, deltaP=deltaP)

    deriv = (Y1 - Y0)/(P1 - P0)

    # zero below the (shifted) 1 Mbar floor of the curve
    logp_eff = _shift_logp(logp, deltaP)
    deriv[logp_eff < 12] = np.zeros_like(logp[logp_eff < 12])

    return deriv

def get_dyds_misc(logp, logt, misc_curve, dt=0.1):
    T0 = 10**logt
    T1 = T0*(1+dt)
    Y0 = get_y_misc(logp, logt, misc_curve)
    Y1 = get_y_misc(logp, np.log10(T1), misc_curve)

    S0 = cms_eos.get_s_pt(logp, logt, Y0)
    S1 = cms_eos.get_s_pt(logp, logt, Y1)

    return (Y1 - Y0)/(S1 - S0)