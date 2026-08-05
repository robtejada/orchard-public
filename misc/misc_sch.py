import numpy as np
from scipy.interpolate import interp1d
from astropy import units as u
from astropy.constants import u as amu
import pandas as pd
from astropy.constants import k_B, m_e, m_p, m_n
from scipy.interpolate import RegularGridInterpolator as RGI
import pdb

"""This module is for the Schottler & Redmer (2018) miscibility data"""

# Reading data

s05 = pd.read_csv('misc/schoettler_nonideal_data/sr18_0_5.csv', names = ['x', 'T'], dtype=float).sort_values('x')
s1 = pd.read_csv('misc/schoettler_nonideal_data/sr18_1.csv', names = ['x', 'T'], dtype=float).sort_values('x')
s1_2 = pd.read_csv('misc/schoettler_nonideal_data/sr18_1_2.csv', names = ['x', 'T'], dtype=float).sort_values('x')
s1_5 = pd.read_csv('misc/schoettler_nonideal_data/sr18_1_5_pad.csv', names = ['x', 'T'], dtype=float).sort_values('x')
s2 = pd.read_csv('misc/schoettler_nonideal_data/sr18_2.csv', names = ['x', 'T'], dtype=float).sort_values('x')
s4 = pd.read_csv('misc/schoettler_nonideal_data/sr18_4.csv', names = ['x', 'T'], dtype=float).sort_values('x')
s10 = pd.read_csv('misc/schoettler_nonideal_data/sr18_10.csv', names = ['x', 'T'], dtype=float).sort_values('x')
s24 = pd.read_csv('misc/schoettler_nonideal_data/sr18_24.csv', names = ['x', 'T'], dtype=float).sort_values('x')

# list of isobars and corresponding pressures
isobars = [s05, s1, s1_2, s1_5, s2, s4, s10, s24]
pressures = np.array([0.5, 1, 1.2, 1.5, 2, 4, 10, 24])

mh = 1
mhe = 4.0026
def x_to_Y(x):
    # x is the helium number fraction
    # converts number fraction to mass fraction
    #return (mhe*x/(mh*(1-x) + mhe*x)).value
    return (mhe*x/(mh*(1-x) + mhe*x))

def Y_to_x(Y):
    #return (mh*Y/(mhe*(1-Y) + mh*Y))
    return (Y/mhe/(Y/mhe + (1-Y)/mh))

interp_list = []
for table in isobars:
    interp = interp1d(table['x'], table['T'], kind='linear', fill_value='extrapolate')
    interp_list.append(interp)
    
xgrid = np.arange(np.min(s05['x']), np.max(s05['x']), 0.001)

T_res = []
for i, p in enumerate(pressures):

    t = interp_list[i](xgrid)
    T_res.append(t)

get_t_x_rgi_cubic = RGI((np.log10(pressures)+12, xgrid), T_res, method='cubic', bounds_error=False, fill_value=None)
get_t_x_rgi_linear = RGI((np.log10(pressures)+12, xgrid), T_res, method='linear', bounds_error=False, fill_value=None)


def get_t_misc(logp, Y, misc_interp='linear'):
    x = Y_to_x(Y)
    if misc_interp == 'linear':
        return get_t_x_rgi_linear(np.array([logp, x]).T)

    else:
        return get_t_x_rgi_cubic(np.array([logp, x]).T)

def get_misc_curve(logp, Y, misc_interp='linear', pad=False):
    x = Y_to_x(Y)
    p_prof = 10**(logp-12)

    pinit_cut = 0.5
    p0_cut = 1.0
    p1_cut = 4.0 # works with 4 but has trouble at 5 Gyr, trying it out with 10
    p2_cut = 24.0

    zone1 = (p_prof > 0.5) & (p_prof <= 1.0)
    zone2 = (p_prof > 1.0) & (p_prof <= 4.0)
    zone3 = (p_prof > 4.0) & (p_prof <= 24.0)

    t_low = get_t_misc(np.log10(p_prof[zone1])+12, Y[zone1])
    t_med = get_t_misc(np.log10(p_prof[zone2])+12, Y[zone2])
    t_high = get_t_misc(np.log10(p_prof[zone3])+12, Y[zone3])

    t_new = np.concatenate([t_low, t_med, t_high])
    pmisc = p_prof[(p_prof > 0.5) & (p_prof <= 24.0)]

    if len(pmisc) == 0: return None, None

    if not pad:

        t_interp_smooth = interp1d(pmisc, t_new, kind='linear')

        return pmisc, t_interp_smooth(pmisc)

    elif pad:
        t_interp_smooth = interp1d(pmisc, t_new, kind='linear')

        pres = pmisc[(pmisc > 1.01) & (pmisc < 24.0)]
        tres = t_interp_smooth(pmisc[(pmisc > 1.01) &  (pmisc < 24.0)])

        pres = np.insert(pres, 0, 1.0)
        tres = np.insert(tres, 0, 0)

        return pres, tres