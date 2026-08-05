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

sr18_mod = np.load('misc/schoettler_nonideal_data/sr18mod.npy')


mh = 1
mhe = 4.0026
def x_to_Y(x):
    # x is the helium number fraction
    # converts number fraction to mass fraction
    return (mhe*x/(mh*(1-x) + mhe*x))

def Y_to_x(Y):
    #return (mh*Y/(mhe*(1-Y) + mh*Y))
    return (Y/mhe/(Y/mhe + (1-Y)/mh))

pgrid1 = np.arange(0.5, 1.0, 0.01)
pgrid2 = np.arange(1.01, 1.5, 0.01)
pgrid3 = np.arange(1.51, 2.0, 0.01)
pgrid4 = np.arange(2.01, 4.0, 0.01)
pgrid5 = np.arange(4.01, 24.0, 1.0)

pgrid = np.concatenate([pgrid1, pgrid2, pgrid3, pgrid4])

ymisc_mock= np.linspace(0, 1, 500)

T_res = np.load('misc/schoettler_nonideal_data/sr18mod_tempgrid.npy')


get_t_x_rgi_cubic = RGI((np.log10(pgrid)+12, ymisc_mock), T_res, method='cubic', bounds_error=False, fill_value=None)
get_t_x_rgi_linear = RGI((np.log10(pgrid)+12, ymisc_mock), T_res, method='linear', bounds_error=False, fill_value=None)


def get_t_misc(logp, Y, misc_interp='linear'):
    #x = Y_to_x(Y)
    # logp = np.log10(press_Mbar+12)
    if misc_interp == 'linear':
        return get_t_x_rgi_linear(np.array([logp, Y]).T)

    else:
        return get_t_x_rgi_cubic(np.array([logp, Y]).T)

def get_misc_curve(logp, Y, misc_interp='linear', pad=False):
    #x = Y_to_x(Y)
    tmisc = get_t_misc(logp, Y, misc_interp=misc_interp)
    p_prof = 10**(logp-12)

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