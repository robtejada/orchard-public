import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import pandas as pd
from scipy.interpolate import interp1d
from scipy.interpolate import RectBivariateSpline as RBS
from scipy.interpolate import RegularGridInterpolator as RGI

from astropy import units as u
from astropy.constants import k_B, m_e, m_p, m_n
from astropy.constants import G, M_jup, M_earth, R_jup, R_earth
from astropy.constants import u as amu
import pdb
from tqdm import tqdm

erg_to_kbbar = (u.erg/u.Kelvin/u.gram).to(k_B/amu)
MJ_to_kbbar = (u.MJ/u.Kelvin/u.kg).to(k_B/amu)
dyn_to_bar = (u.dyne/(u.cm)**2).to('bar')
erg_to_MJ = (u.erg/u.Kelvin/u.gram).to(u.MJ/u.Kelvin/u.kg)

mp = amu.to('g') # grams
kb = k_B.to('erg/K') # ergs/KG=G.to('cm^3/g s^2').value # Newton's constant
R_jup = R_jup.to('cm').value
M_jup = M_jup.to('g').value

l1 = pd.read_csv('misc/lorenzen_misc_data/lor_1Mbar_top.csv', names=['x', 'T'], dtype=float)
l1['T'] /= 1000
l2 = pd.read_csv('misc/lorenzen_misc_data/lor_2Mbar_top.csv', names=['x', 'T'], dtype=float)
l2['T'] /= 1000
l4 = pd.read_csv('misc/lorenzen_misc_data/lorenzen_4mbar.csv', dtype=float).sort_values('x')
l4['T'] /= 1000
l10 = pd.read_csv('misc/lorenzen_misc_data/lorenzen_10mbar.csv', dtype=float).sort_values('x')
l10['T'] /= 1000
l24 = pd.read_csv('misc/lorenzen_misc_data/lorenzen_24mbar.csv', dtype=float).sort_values('x')
l24['T'] /= 1000

mh = 1
mhe = 4.0026

he_fracs = [l1, l2, l4, l10, l24]
pressures = np.array([1, 2, 4, 10, 24])

def x_to_Y(x):
    # x is the helium number fraction
    return (mhe*x/(mh*(1-x) + mhe*x))

def Y_to_x(Y):
    return (Y/mhe/(Y/mhe + (1-Y)/mh))

interp_1Mbar = interp1d(l1['x'], l1['T'], kind='linear', fill_value='extrapolate')
interp_2Mbar = interp1d(l2['x'], l2['T'], kind='linear', fill_value='extrapolate')
interp_4Mbar = interp1d(l4['x'], l4['T'], kind='linear', fill_value='extrapolate')
interp_10Mbar = interp1d(l10['x'], l10['T'], kind='linear', fill_value='extrapolate')
interp_24Mbar = interp1d(l24['x'], l24['T'], kind='linear', fill_value='extrapolate')

xgrid = np.arange(np.min(l1['x']), np.max(l1['x']), 0.001) 

interp_list = [interp_1Mbar, interp_2Mbar, interp_4Mbar, interp_10Mbar, interp_24Mbar]

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

    zone1 = (p_prof > 0.5) & (p_prof <= 1.0)
    zone2 = (p_prof > 1.0) & (p_prof <= 4.0)
    zone3 = (p_prof > 4.0) & (p_prof <= 24.0)

    t_low = get_t_misc(np.log10(p_prof[zone1])+12, Y[zone1])
    t_med = get_t_misc(np.log10(p_prof[zone2])+12, Y[zone2])
    t_high = get_t_misc(np.log10(p_prof[zone3])+12, Y[zone3])

    t_new = np.concatenate([t_low, t_med, t_high])
    pmisc = p_prof[(p_prof > 0.5) & (p_prof <= 24.0)]

    if len(pmisc) <= 1: return None, None

    if not pad:

        t_interp_smooth = interp1d(pmisc, t_new, kind='linear')

        return pmisc, t_interp_smooth(pmisc)

    elif pad:
        t_interp_smooth = interp1d(pmisc, t_new, kind='linear')

        pres = pmisc[(pmisc > 1.001) & (pmisc < 24.0)]
        tres = t_interp_smooth(pmisc[(pmisc > 1.001) &  (pmisc < 24.0)])

        pres = np.insert(pres, 0, 1.0)
        tres = np.insert(tres, 0, 0)

        return pres, tres

    