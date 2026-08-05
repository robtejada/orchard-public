import numpy as np
import astropy.units as u
from scipy.interpolate import interp1d
from scipy.interpolate import RegularGridInterpolator as RGI
from astropy.constants import k_B
from astropy.constants import u as amu
from scipy.optimize import brentq, brenth
from scipy.integrate import quad
import pdb
import os
from pathlib import Path

J_K_kg_to_erg_K_g = (u.J / (u.kg * u.K)).to('erg/(K*g)') # specific entropy conversion
J_kg_to_erg_g = (u.J / u.kg).to('erg/g') # specific energy conversion
dyn_to_Pa = (u.dyn/u.cm**2).to('Pa') # dyn/cm² to Pa conversion
kb = k_B.to('erg/K') # ergs/K
erg_to_kbbar = (u.erg/u.Kelvin/u.gram).to(k_B/amu)

CURR_DIR = os.path.dirname(os.path.realpath(__file__))
MANTLE_PATH = str((Path(CURR_DIR) / '..' / '..' / 'CMAPPER_rock' / 'EoS' / 'mantle').resolve()) + '/'

p_solidus_liquidus = np.loadtxt(MANTLE_PATH + 'solid_P.txt')

p_grid = p_solidus_liquidus[:,0][:1500].copy()
T_liq = np.loadtxt(MANTLE_PATH + 'T_liq_Py_1500GPa.txt')
rho_liq = np.loadtxt(MANTLE_PATH + 'rho_liq_Py_1500GPa.txt')
CP_liq = np.loadtxt(MANTLE_PATH + 'CP_liq_Py_1500GPa.txt') * J_K_kg_to_erg_K_g
alpha_liq = np.loadtxt(MANTLE_PATH + 'alpha_liq_Py_1500GPa.txt')
y_grid = np.loadtxt(MANTLE_PATH + 'y.txt')

# Create interpolator objects
rgi_kwargs = {
    'method': 'linear',
    'bounds_error': False,
    'fill_value': None
}
get_T_Py_liq_rgi = RGI((p_grid, y_grid), T_liq, **rgi_kwargs)
get_rho_Py_liq_rgi = RGI((p_grid, y_grid), rho_liq, **rgi_kwargs)

#### ENSTATITE, PEROVSKITE, AND POST-PEROVSKITE GRIDS AND INTERPOLATORS ####

# Temperature, density, heat capacity, and thermal expansivity mixture and solid tables
# for enstatite (en), perovskite (pv), and post-perovskite (ppv) phases of liquid-solid mixtures

# all pressures in Pa
P_grid_en = np.loadtxt(MANTLE_PATH + 'P_en.txt')
P_grid_pv = np.loadtxt(MANTLE_PATH + 'P_pv.txt')
P_grid_ppv = np.loadtxt(MANTLE_PATH + 'P_ppv.txt')

# all temperatures in K
T_mix_en = np.loadtxt(MANTLE_PATH + 'T_mix_en_Py.txt')
T_sol_en = np.loadtxt(MANTLE_PATH + 'T_sol_en_Py.txt')

T_mix_pv = np.loadtxt(MANTLE_PATH + 'T_mix_pv_Py.txt')
T_sol_pv = np.loadtxt(MANTLE_PATH + 'T_sol_pv_Py.txt')

T_mix_ppv = np.loadtxt(MANTLE_PATH + 'T_mix_ppv_Py.txt')
T_sol_ppv = np.loadtxt(MANTLE_PATH + 'T_sol_ppv_Py.txt')

# all densities in kg/m³
rho_mix_en = np.loadtxt(MANTLE_PATH + 'rho_mix_en_Py.txt')
rho_sol_en = np.loadtxt(MANTLE_PATH + 'rho_sol_en_Py.txt')

rho_mix_pv = np.loadtxt(MANTLE_PATH + 'rho_mix_pv_Py.txt')
rho_sol_pv = np.loadtxt(MANTLE_PATH + 'rho_sol_pv_Py.txt')

rho_mix_ppv = np.loadtxt(MANTLE_PATH + 'rho_mix_ppv_Py.txt')
rho_sol_ppv = np.loadtxt(MANTLE_PATH + 'rho_sol_ppv_Py.txt')

# all specific heats converted to erg/(g·K) from J/(kg·K)
CP_mix_en = np.loadtxt(MANTLE_PATH + 'CP_mix_en_Py.txt') * J_K_kg_to_erg_K_g
CP_mix_pv = np.loadtxt(MANTLE_PATH + 'CP_mix_pv_Py.txt') * J_K_kg_to_erg_K_g
CP_mix_ppv = np.loadtxt(MANTLE_PATH + 'CP_mix_ppv_Py.txt') * J_K_kg_to_erg_K_g

alpha_mix_en = np.loadtxt(MANTLE_PATH + 'alpha_mix_en_Py.txt')
alpha_sol_en = np.loadtxt(MANTLE_PATH + 'alpha_sol_en_Py.txt')

alpha_mix_pv = np.loadtxt(MANTLE_PATH + 'alpha_mix_pv_Py.txt')
alpha_sol_pv = np.loadtxt(MANTLE_PATH + 'alpha_sol_pv_Py.txt')

alpha_mix_ppv = np.loadtxt(MANTLE_PATH + 'alpha_mix_ppv_Py.txt')
alpha_sol_ppv = np.loadtxt(MANTLE_PATH + 'alpha_sol_ppv_Py.txt')

########### CREATE REGULAR GRID INTERPOLATORS FOR EACH PHASE ###########

get_T_Py_mix_en = RGI((P_grid_en, y_grid), T_mix_en, **rgi_kwargs)
get_T_Py_sol_en = RGI((P_grid_en, y_grid), T_sol_en, **rgi_kwargs)

get_T_Py_mix_pv = RGI((P_grid_pv, y_grid), T_mix_pv, **rgi_kwargs)
get_T_Py_sol_pv = RGI((P_grid_pv, y_grid), T_sol_pv, **rgi_kwargs)

get_T_Py_mix_ppv = RGI((P_grid_ppv, y_grid), T_mix_ppv, **rgi_kwargs)
get_T_Py_sol_ppv = RGI((P_grid_ppv, y_grid), T_sol_ppv, **rgi_kwargs)

get_rho_Py_mix_en = RGI((P_grid_en, y_grid), rho_mix_en, **rgi_kwargs)
get_rho_Py_sol_en = RGI((P_grid_en, y_grid), rho_sol_en, **rgi_kwargs)

get_rho_Py_mix_pv = RGI((P_grid_pv, y_grid), rho_mix_pv, **rgi_kwargs)
get_rho_Py_sol_pv = RGI((P_grid_pv, y_grid), rho_sol_pv, **rgi_kwargs)

get_rho_Py_mix_ppv = RGI((P_grid_ppv, y_grid), rho_mix_ppv, **rgi_kwargs)
get_rho_Py_sol_ppv = RGI((P_grid_ppv, y_grid), rho_sol_ppv, **rgi_kwargs)

get_CP_Py_mix_en = RGI((P_grid_en, y_grid), CP_mix_en, **rgi_kwargs)
get_CP_Py_mix_pv = RGI((P_grid_pv, y_grid), CP_mix_pv, **rgi_kwargs)
get_CP_Py_mix_ppv = RGI((P_grid_ppv, y_grid), CP_mix_ppv, **rgi_kwargs)
get_CP_Py_liq = RGI((p_grid, y_grid), CP_liq, **rgi_kwargs)

get_alpha_Py_mix_en = RGI((P_grid_en, y_grid), alpha_mix_en, **rgi_kwargs)
get_alpha_Py_sol_en = RGI((P_grid_en, y_grid), alpha_sol_en, **rgi_kwargs)

get_alpha_Py_mix_pv = RGI((P_grid_pv, y_grid), alpha_mix_pv, **rgi_kwargs)
get_alpha_Py_sol_pv = RGI((P_grid_pv, y_grid), alpha_sol_pv, **rgi_kwargs)

get_alpha_Py_mix_ppv = RGI((P_grid_ppv, y_grid), alpha_mix_ppv, **rgi_kwargs)
get_alpha_Py_sol_ppv = RGI((P_grid_ppv, y_grid), alpha_sol_ppv, **rgi_kwargs)

get_alpha_Py_liq = RGI((p_grid, y_grid), alpha_liq, **rgi_kwargs)


# --- Phase-specific user-defined functions for the rocky mantle ---

def get_Tmix_en_Py(P, y):
    """
    Return mixture temperature T (K) at pressure P (Pa) and composition y for enstatite phase.
    """
    scalar = np.isscalar(P) and np.isscalar(y)
    P_arr = np.array(P, ndmin=1)
    y_arr = np.array(y, ndmin=1)
    if P_arr.shape != y_arr.shape:
        P_arr, y_arr = np.broadcast_arrays(P_arr, y_arr)
    pts = np.stack((P_arr.ravel(), y_arr.ravel()), axis=-1)
    vals = get_T_Py_mix_en(pts).reshape(P_arr.shape)
    return float(vals) if scalar else vals


def get_rhomix_en_Py(P, y):
    """
    Return mixture density rho (kg/m^3) at pressure P (Pa) and composition y for enstatite phase.
    """
    scalar = np.isscalar(P) and np.isscalar(y)
    P_arr = np.array(P, ndmin=1)
    y_arr = np.array(y, ndmin=1)
    if P_arr.shape != y_arr.shape:
        P_arr, y_arr = np.broadcast_arrays(P_arr, y_arr)
    pts = np.stack((P_arr.ravel(), y_arr.ravel()), axis=-1)
    vals = get_rho_Py_mix_en(pts).reshape(P_arr.shape)
    return float(vals) if scalar else vals


def get_CPmix_en_Py(P, y):
    """
    Return mixture specific heat CP (J·kg⁻¹·K⁻¹) at pressure P (Pa) and composition y for enstatite phase.
    """
    scalar = np.isscalar(P) and np.isscalar(y)
    P_arr = np.array(P, ndmin=1)
    y_arr = np.array(y, ndmin=1)
    if P_arr.shape != y_arr.shape:
        P_arr, y_arr = np.broadcast_arrays(P_arr, y_arr)
    pts = np.stack((P_arr.ravel(), y_arr.ravel()), axis=-1)
    vals = get_CP_Py_mix_en(pts).reshape(P_arr.shape)
    return float(vals) if scalar else vals


def get_alphamix_en_Py(P, y):
    """
    Return mixture thermal expansivity alpha (1/K) at pressure P (Pa) and composition y for enstatite phase.
    """
    scalar = np.isscalar(P) and np.isscalar(y)
    P_arr = np.array(P, ndmin=1)
    y_arr = np.array(y, ndmin=1)
    if P_arr.shape != y_arr.shape:
        P_arr, y_arr = np.broadcast_arrays(P_arr, y_arr)
    pts = np.stack((P_arr.ravel(), y_arr.ravel()), axis=-1)
    vals = get_alpha_Py_mix_en(pts).reshape(P_arr.shape)
    return float(vals) if scalar else vals


def get_Tmix_pv_Py(P, y):
    # perovskite temperature
    scalar = np.isscalar(P) and np.isscalar(y)
    P_arr = np.array(P, ndmin=1)
    y_arr = np.array(y, ndmin=1)
    if P_arr.shape != y_arr.shape:
        P_arr, y_arr = np.broadcast_arrays(P_arr, y_arr)
    pts = np.stack((P_arr.ravel(), y_arr.ravel()), axis=-1)
    vals = get_T_Py_mix_pv(pts).reshape(P_arr.shape)
    return float(vals) if scalar else vals


def get_rhomix_pv_Py(P, y):
    # perovskite density
    scalar = np.isscalar(P) and np.isscalar(y)
    P_arr = np.array(P, ndmin=1)
    y_arr = np.array(y, ndmin=1)
    if P_arr.shape != y_arr.shape:
        P_arr, y_arr = np.broadcast_arrays(P_arr, y_arr)
    pts = np.stack((P_arr.ravel(), y_arr.ravel()), axis=-1)
    vals = get_rho_Py_mix_pv(pts).reshape(P_arr.shape)
    return float(vals) if scalar else vals


def get_CPmix_pv_Py(P, y):
    # perovskite specific heat
    scalar = np.isscalar(P) and np.isscalar(y)
    P_arr = np.array(P, ndmin=1)
    y_arr = np.array(y, ndmin=1)
    if P_arr.shape != y_arr.shape:
        P_arr, y_arr = np.broadcast_arrays(P_arr, y_arr)
    pts = np.stack((P_arr.ravel(), y_arr.ravel()), axis=-1)
    vals = get_CP_Py_mix_pv(pts).reshape(P_arr.shape)
    return float(vals) if scalar else vals


def get_alphamix_pv_Py(P, y):
    # perovskite thermal expansivity
    scalar = np.isscalar(P) and np.isscalar(y)
    P_arr = np.array(P, ndmin=1)
    y_arr = np.array(y, ndmin=1)
    if P_arr.shape != y_arr.shape:
        P_arr, y_arr = np.broadcast_arrays(P_arr, y_arr)
    pts = np.stack((P_arr.ravel(), y_arr.ravel()), axis=-1)
    vals = get_alpha_Py_mix_pv(pts).reshape(P_arr.shape)
    return float(vals) if scalar else vals


def get_Tmix_ppv_Py(P, y):
    # post-perovskite temperature
    scalar = np.isscalar(P) and np.isscalar(y)
    P_arr = np.array(P, ndmin=1)
    y_arr = np.array(y, ndmin=1)
    if P_arr.shape != y_arr.shape:
        P_arr, y_arr = np.broadcast_arrays(P_arr, y_arr)
    pts = np.stack((P_arr.ravel(), y_arr.ravel()), axis=-1)
    vals = get_T_Py_mix_ppv(pts).reshape(P_arr.shape)
    return float(vals) if scalar else vals


def get_rhomix_ppv_Py(P, y):
    # post-perovskite density
    scalar = np.isscalar(P) and np.isscalar(y)
    P_arr = np.array(P, ndmin=1)
    y_arr = np.array(y, ndmin=1)
    if P_arr.shape != y_arr.shape:
        P_arr, y_arr = np.broadcast_arrays(P_arr, y_arr)
    pts = np.stack((P_arr.ravel(), y_arr.ravel()), axis=-1)
    vals = get_rho_Py_mix_ppv(pts).reshape(P_arr.shape)
    return float(vals) if scalar else vals


def get_CPmix_ppv_Py(P, y):
    # post-perovskite specific heat
    scalar = np.isscalar(P) and np.isscalar(y)
    P_arr = np.array(P, ndmin=1)
    y_arr = np.array(y, ndmin=1)
    if P_arr.shape != y_arr.shape:
        P_arr, y_arr = np.broadcast_arrays(P_arr, y_arr)
    pts = np.stack((P_arr.ravel(), y_arr.ravel()), axis=-1)
    vals = get_CP_Py_mix_ppv(pts).reshape(P_arr.shape)
    return float(vals) if scalar else vals


def get_alphamix_ppv_Py(P, y):
    # post-perovskite thermal expansivity
    scalar = np.isscalar(P) and np.isscalar(y)
    P_arr = np.array(P, ndmin=1)
    y_arr = np.array(y, ndmin=1)
    if P_arr.shape != y_arr.shape:
        P_arr, y_arr = np.broadcast_arrays(P_arr, y_arr)
    pts = np.stack((P_arr.ravel(), y_arr.ravel()), axis=-1)
    vals = get_alpha_Py_mix_ppv(pts).reshape(P_arr.shape)
    return float(vals) if scalar else vals

# --- General wrappers selecting phase by pressure ---

# Phase boundaries (in Pa)
_EN_BOUND = 23e9
_PV_BOUND = 125e9  # 125 GPa


def get_Tmix_Py(P, y):
    """
    General mixture temperature: selects en, pv, or ppv interpolator based on P.
    """
    scalar = np.isscalar(P) and np.isscalar(y)
    P_arr = np.array(P, ndmin=1)
    y_arr = np.array(y, ndmin=1)
    if P_arr.shape != y_arr.shape:
        P_arr, y_arr = np.broadcast_arrays(P_arr, y_arr)

    T_out = np.empty_like(P_arr, dtype=float)
    mask_en = P_arr < _EN_BOUND
    mask_pv = (P_arr >= _EN_BOUND) & (P_arr < _PV_BOUND)
    mask_ppv = P_arr >= _PV_BOUND

    if np.any(mask_en):
        T_out[mask_en] = get_Tmix_en_Py(P_arr[mask_en], y_arr[mask_en])
    if np.any(mask_pv):
        T_out[mask_pv] = get_Tmix_pv_Py(P_arr[mask_pv], y_arr[mask_pv])
    if np.any(mask_ppv):
        T_out[mask_ppv] = get_Tmix_ppv_Py(P_arr[mask_ppv], y_arr[mask_ppv])

    return float(T_out) if scalar else T_out


def get_rhomix_Py(P, y):
    """
    General mixture density rho: selects en, pv, or ppv interpolator based on P.
    """
    scalar = np.isscalar(P) and np.isscalar(y)
    P_arr = np.array(P, ndmin=1)
    y_arr = np.array(y, ndmin=1)
    if P_arr.shape != y_arr.shape:
        P_arr, y_arr = np.broadcast_arrays(P_arr, y_arr)

    rho_out = np.empty_like(P_arr, dtype=float)
    mask_en = P_arr < _EN_BOUND
    mask_pv = (P_arr >= _EN_BOUND) & (P_arr < _PV_BOUND)
    mask_ppv = P_arr >= _PV_BOUND

    if np.any(mask_en):
        rho_out[mask_en] = get_rhomix_en_Py(P_arr[mask_en], y_arr[mask_en])
    if np.any(mask_pv):
        rho_out[mask_pv] = get_rhomix_pv_Py(P_arr[mask_pv], y_arr[mask_pv])
    if np.any(mask_ppv):
        rho_out[mask_ppv] = get_rhomix_ppv_Py(P_arr[mask_ppv], y_arr[mask_ppv])

    return float(rho_out) if scalar else rho_out


def get_CPmix_Py(P, y):
    """
    General mixture specific heat CP: selects en, pv, or ppv interpolator based on P.
    """
    scalar = np.isscalar(P) and np.isscalar(y)
    P_arr = np.array(P, ndmin=1)
    y_arr = np.array(y, ndmin=1)
    if P_arr.shape != y_arr.shape:
        P_arr, y_arr = np.broadcast_arrays(P_arr, y_arr)

    CP_out = np.empty_like(P_arr, dtype=float)
    mask_en = P_arr < _EN_BOUND
    mask_pv = (P_arr >= _EN_BOUND) & (P_arr < _PV_BOUND)
    mask_ppv = P_arr >= _PV_BOUND

    if np.any(mask_en):
        CP_out[mask_en] = get_CPmix_en_Py(P_arr[mask_en], y_arr[mask_en])
    if np.any(mask_pv):
        CP_out[mask_pv] = get_CPmix_pv_Py(P_arr[mask_pv], y_arr[mask_pv])
    if np.any(mask_ppv):
        CP_out[mask_ppv] = get_CPmix_ppv_Py(P_arr[mask_ppv], y_arr[mask_ppv])

    return float(CP_out) if scalar else CP_out


def get_alphamix_Py(P, y):
    """
    General mixture thermal expansivity alpha: selects en, pv, or ppv interpolator based on P.
    """
    scalar = np.isscalar(P) and np.isscalar(y)
    P_arr = np.array(P, ndmin=1)
    y_arr = np.array(y, ndmin=1)
    if P_arr.shape != y_arr.shape:
        P_arr, y_arr = np.broadcast_arrays(P_arr, y_arr)

    alpha_out = np.empty_like(P_arr, dtype=float)
    mask_en = P_arr < _EN_BOUND
    mask_pv = (P_arr >= _EN_BOUND) & (P_arr < _PV_BOUND)
    mask_ppv = P_arr >= _PV_BOUND

    if np.any(mask_en):
        alpha_out[mask_en] = get_alphamix_en_Py(P_arr[mask_en], y_arr[mask_en])
    if np.any(mask_pv):
        alpha_out[mask_pv] = get_alphamix_pv_Py(P_arr[mask_pv], y_arr[mask_pv])
    if np.any(mask_ppv):
        alpha_out[mask_ppv] = get_alphamix_ppv_Py(P_arr[mask_ppv], y_arr[mask_ppv])

    return float(alpha_out) if scalar else alpha_out

# smoothing widths: 10% of each boundary
_frac     = 0.25
_delta_en = _frac * _EN_BOUND
_delta_pv = _frac * _PV_BOUND

# EoS parameters for γ
_OL_PARAMS  = {
    'rho0':     3213.7,   # kg/m³
    'gamma0':   1.31,
    'lambda':   3.2
}
_PV_PARAMS  = {
    'rho0':      4105.9,  # kg/m³
    'gamma0':    1.506,
    'gamma_inf': 1.148,
    'lambda':    7.025
}
_PPV_PARAMS = {
    'rho0':      1374.7831663193047,  # kg/m³
    'gamma0':    1.495,
    'gamma_inf': 0.818,
    'lambda':    2.627
}

def get_gamma_Py(P, y):
    """
    Smooth, vectorized Grüneisen parameter γ(P,y):
       • Olivine   for P < 23 GPa
       • Perovskite for 23 GPa ≤ P < 125 GPa
       • Post‑pv   for P ≥ 125 GPa

    Smooth blending (tanh) over ±5% of each boundary.
    """
    # 1) broadcast inputs
    scalar = np.isscalar(P) and np.isscalar(y)
    P_arr = np.array(P, ndmin=1)
    y_arr = np.array(y, ndmin=1)
    if P_arr.shape != y_arr.shape:
        P_arr, y_arr = np.broadcast_arrays(P_arr, y_arr)

    # 2) get density with your existing routine
    rho = get_rhomix_Py(P_arr, y_arr)

    # 3) compute “pure” γ for each phase
    #  olivine (eq.19)
    p = _OL_PARAMS
    x_ol    = rho / p['rho0']
    gamma_ol = p['gamma0'] * x_ol**(-p['lambda'])

    #  perovskite (eq.20)
    p = _PV_PARAMS
    x_pv    = rho / p['rho0']
    gamma_pv = p['gamma_inf'] + (p['gamma0'] - p['gamma_inf']) * x_pv**(-p['lambda'])

    #  post‑perovskite (modified exponent to use λ directly, so it decays)
    p = _PPV_PARAMS
    x_pp    = rho / p['rho0']
    gamma_pp = p['gamma_inf'] + (p['gamma0'] - p['gamma_inf']) * x_pp**(-p['lambda'])
    # → since λ<0, x**λ → decays at high ρ, so γ→γ∞

    # 4) build smooth blend weights
    w_op = 0.5 * (1 + np.tanh((P_arr - _EN_BOUND) / _delta_en))
    w_pp = 0.5 * (1 + np.tanh((P_arr - _PV_BOUND) / _delta_pv))

    # 5) first blend olivine↔pv, then blend that↔ppv
    gamma_op = (1 - w_op)*gamma_ol + w_op*gamma_pv
    gamma    = (1 - w_pp)*gamma_op + w_pp*gamma_pp

    return float(gamma) if scalar else gamma

def get_CVmix_Py(P, y):
    """
    General mixture CV: selects en, pv, or ppv interpolator based on P.
    """
    scalar = np.isscalar(P) and np.isscalar(y)
    P_arr = np.array(P, ndmin=1)
    y_arr = np.array(y, ndmin=1)
    if P_arr.shape != y_arr.shape:
        P_arr, y_arr = np.broadcast_arrays(P_arr, y_arr)

    CV_out = np.empty_like(P_arr, dtype=float)

    CP_mix = get_CPmix_Py(P_arr, y_arr)
    alpha_mix = get_alphamix_Py(P_arr, y_arr)
    T_mix = get_Tmix_Py(P_arr, y_arr)
    gamma = get_gamma_Py(P_arr, y_arr)
    CV_out = CP_mix / (1 + gamma * alpha_mix * T_mix)

    return float(CV_out) if scalar else CV_out

s_solidus = p_solidus_liquidus[:,1][:1500].copy() * J_K_kg_to_erg_K_g
s_liquidus = p_solidus_liquidus[:,2][:1500].copy() * J_K_kg_to_erg_K_g

s_melt_solidus = interp1d(p_grid, s_solidus, bounds_error=False, fill_value='extrapolate')
s_melt_liquidus = interp1d(p_grid, s_liquidus, bounds_error=False, fill_value='extrapolate')

# to update temperature and density from pressure and y of the liquid phase
def get_Tliq_Py(P, y):
    """
    Return liquidus temperature T at pressure P and entropy y.
    Can handle floats or numpy arrays (broadcasting shapes).
    """
    scalar = np.isscalar(P) and np.isscalar(y)
    P_arr = np.array(P, ndmin=1)
    y_arr = np.array(y, ndmin=1)
    if P_arr.shape != y_arr.shape:
        P_arr, y_arr = np.broadcast_arrays(P_arr, y_arr)
    pts = np.stack((P_arr.ravel(), y_arr.ravel()), axis=-1)
    vals = get_T_Py_liq_rgi(pts).reshape(P_arr.shape)
    return float(vals) if scalar else vals

# to update density from pressure and entropy measure
def get_rholiq_Py(P, y):
    """
    Return liquid density rho at pressure P and entropy y.
    Can handle floats or numpy arrays (broadcasting shapes).
    """
    scalar = np.isscalar(P) and np.isscalar(y)
    P_arr = np.array(P, ndmin=1)
    y_arr = np.array(y, ndmin=1)
    if P_arr.shape != y_arr.shape:
        P_arr, y_arr = np.broadcast_arrays(P_arr, y_arr)
    pts = np.stack((P_arr.ravel(), y_arr.ravel()), axis=-1)
    vals = get_rho_Py_liq_rgi(pts).reshape(P_arr.shape)
    return float(vals) if scalar else vals

def get_CPliq_Py(P, y):
    """
    Return liquid specific heat CP at pressure P and entropy y.
    Can handle floats or numpy arrays (broadcasting shapes).
    """
    scalar = np.isscalar(P) and np.isscalar(y)
    P_arr = np.array(P, ndmin=1)
    y_arr = np.array(y, ndmin=1)
    if P_arr.shape != y_arr.shape:
        P_arr, y_arr = np.broadcast_arrays(P_arr, y_arr)
    pts = np.stack((P_arr.ravel(), y_arr.ravel()), axis=-1)
    vals = get_CP_Py_liq(pts).reshape(P_arr.shape)
    return float(vals) if scalar else vals

def get_alphaliq_Py(P, y):
    """
    Return liquid thermal expansivity alpha at pressure P and entropy y.
    Can handle floats or numpy arrays (broadcasting shapes).
    """
    scalar = np.isscalar(P) and np.isscalar(y)
    P_arr = np.array(P, ndmin=1)
    y_arr = np.array(y, ndmin=1)
    if P_arr.shape != y_arr.shape:
        P_arr, y_arr = np.broadcast_arrays(P_arr, y_arr)
    pts = np.stack((P_arr.ravel(), y_arr.ravel()), axis=-1)
    vals = get_alpha_Py_liq(pts).reshape(P_arr.shape)
    return float(vals) if scalar else vals

def get_CVliq_Py(P, y):
    """
    General mixture CV: selects en, pv, or ppv interpolator based on P.
    """
    scalar = np.isscalar(P) and np.isscalar(y)
    P_arr = np.array(P, ndmin=1)
    y_arr = np.array(y, ndmin=1)
    if P_arr.shape != y_arr.shape:
        P_arr, y_arr = np.broadcast_arrays(P_arr, y_arr)

    CV_out = np.empty_like(P_arr, dtype=float)
    # mask_en = P_arr < _EN_BOUND
    # mask_pv = (P_arr >= _EN_BOUND) & (P_arr < _PV_BOUND)
    # mask_ppv = P_arr >= _PV_BOUND

    CP_liq = get_CPliq_Py(P_arr, y_arr)
    alpha_liq = get_alphaliq_Py(P_arr, y_arr)
    T_liq = get_Tliq_Py(P_arr, y_arr)
    gamma = get_gamma_Py(P_arr, y_arr)
    CV_out = CP_liq / (1 + gamma  * alpha_liq * T_liq)

    return float(CV_out) if scalar else CV_out

def get_KT_liq(P, y):
    """
    Calculate the isothermal bulk modulus K_T from pressure P and specific volume y.

    Parameters:
    P : array-like
        Pressure values in Pa.

    Equation 10 from Zhang et al. 2022: alpha = gamma * rho * Cv / K_T
    Rearranged to solve for K_T:
        K_T = gamma * rho * Cv / alpha
    Returns:
    K_T : array-like
        Isothermal bulk modulus values in GPa.
    """
    scalar = np.isscalar(P) and np.isscalar(y)
    P_arr = np.array(P, ndmin=1)
    y_arr = np.array(y, ndmin=1)
    if P_arr.shape != y_arr.shape:
        P_arr, y_arr = np.broadcast_arrays(P_arr, y_arr)

    CV_out = np.empty_like(P_arr, dtype=float)
    # mask_en = P_arr < _EN_BOUND
    # mask_pv = (P_arr >= _EN_BOUND) & (P_arr < _PV_BOUND)
    # mask_ppv = P_arr >= _PV_BOUND

    CP_liq = get_CPliq_Py(P_arr, y_arr) / J_K_kg_to_erg_K_g # SI units
    alpha_liq = get_alphaliq_Py(P_arr, y_arr)
    T_liq = get_Tliq_Py(P_arr, y_arr)
    gamma = get_gamma_Py(P_arr, y_arr)
    CV_out = CP_liq / (1 + gamma  * alpha_liq * T_liq)

    rho_liq = get_rholiq_Py(P_arr, y_arr) #* 1e-3 # in g/cm³
    K_T = gamma * rho_liq * CV_out / alpha_liq

    return float(K_T) if scalar else K_T # output in Pa

def get_KT_sol(P, y):
    """
    Calculate the isothermal bulk modulus K_T from pressure P and specific volume y.

    Parameters:
    P : array-like
        Pressure values in Pa.

    Equation 10 from Zhang et al. 2022: alpha = gamma * rho * Cv / K_T
    Rearranged to solve for K_T:
        K_T = gamma * rho * Cv / alpha
    Returns:
    K_T : array-like
        Isothermal bulk modulus values in GPa.
    """
    scalar = np.isscalar(P) and np.isscalar(y)
    P_arr = np.array(P, ndmin=1)
    y_arr = np.array(y, ndmin=1)
    if P_arr.shape != y_arr.shape:
        P_arr, y_arr = np.broadcast_arrays(P_arr, y_arr)

    CV_out = np.empty_like(P_arr, dtype=float)
    # mask_en = P_arr < _EN_BOUND
    # mask_pv = (P_arr >= _EN_BOUND) & (P_arr < _PV_BOUND)
    # mask_ppv = P_arr >= _PV_BOUND

    CP_sol = get_CPsol_Py(P_arr, y_arr) / J_K_kg_to_erg_K_g # SI units
    alpha_sol = get_alphasol_Py(P_arr, y_arr)
    T_sol = get_Tsol_Py(P_arr, y_arr)
    gamma = get_gamma_Py(P_arr, y_arr)
    CV_out = CP_sol / (1 + gamma  * alpha_sol * T_sol)
    rho_sol = get_rhosol_Py(P_arr, y_arr) #* 1e-3 # in g/cm³
    K_T = gamma * rho_sol * CV_out / alpha_sol

    return float(K_T) if scalar else K_T # output in Pa

# internal energy integrals

def get_u_liq_from_Py(P, y, P_ref=p_grid[0], y_ref = y_grid[0], u_ref=0.0, n1=200, n2=200, return_path_mismatch=False):
    """
    Compute specific internal energy u(P, y) for liquid MgSiO3 relative to a reference (P_ref, y_ref).

    Thermodynamic differential used:
        du = c_v dT + [P - T * alpha * K_T] / rho^2 * d(rho)

    Path: (P_ref, y_ref) -> (P_ref, y) -> (P, y)

    Parameters
    ----------
    P, y : float or np.ndarray
        Target pressure and 'y' (your EOS entropy-like coordinate) in the same shapes.
    P_ref, y_ref : float
        Reference state in the same units as P, y. u(P_ref, y_ref) is set to u_ref.
    u_ref : float, optional
        Internal energy at the reference state. Default 0.0 (J/kg, or your working unit).
    n1, n2 : int, optional
        Number of integration segments for leg-1 (vary y at fixed P_ref) and leg-2 (vary P at fixed y).
    return_path_mismatch : bool, optional
        If True, also compute the reversed-leg path and return their difference (diagnostic).

    Returns
    -------
    u : float or np.ndarray
        Specific internal energy at (P, y), same shape as inputs P and y.
    (optional) mismatch : float or np.ndarray
        Difference between two-leg orders (should be ~0 within quadrature error).

    Notes
    -----
    - Units must be consistent:
        rho : kg/m^3
        P, K_T : Pa
        T : K
        alpha : 1/K
        c_v : J/(kg K)
      so that u comes out in J/kg.
    - This function relies on:
        get_Tliq_Py(P, y), get_rholiq_Py(P, y),
        get_CVliq_Py(P, y), get_alphaliq_Py(P, y), get_KT_liq(P, y).
    - For the solid, make an identical wrapper swapping in the solid getters.
    """

    # Broadcast targets
    P_arr = np.array(P, ndmin=1)
    y_arr = np.array(y, ndmin=1)
    if P_arr.shape != y_arr.shape:
        P_arr, y_arr = np.broadcast_arrays(P_arr, y_arr)

    def _integrate_along_polyline(P_path, y_path):
        """Trapezoidal line integral of du along a polyline in (P, y)."""
        # Evaluate state along the path
        T_path   = get_Tliq_Py(P_path, y_path)
        rho_path = get_rholiq_Py(P_path, y_path)
        cv_path  = get_CVliq_Py(P_path, y_path) / J_K_kg_to_erg_K_g # SI units
        a_path   = get_alphaliq_Py(P_path, y_path)
        KT_path  = get_KT_liq(P_path, y_path)

        # Midpoint values for the integrand
        T_mid   = 0.5 * (T_path[:-1]   + T_path[1:])
        rho_mid = 0.5 * (rho_path[:-1] + rho_path[1:])
        cv_mid  = 0.5 * (cv_path[:-1]  + cv_path[1:])
        a_mid   = 0.5 * (a_path[:-1]   + a_path[1:])
        KT_mid  = 0.5 * (KT_path[:-1]  + KT_path[1:])
        P_mid   = 0.5 * (P_path[:-1]   + P_path[1:])

        # Increments along the path
        dT   = np.diff(T_path)
        drho = np.diff(rho_path)

        # Differential work: du = cv dT + [P - T alpha K_T] / rho^2 * d(rho)
        term1 = cv_mid * dT
        term2 = (P_mid - T_mid * a_mid * KT_mid) / (rho_mid**2) * drho

        return np.sum(term1 + term2)

    def _two_leg(Pt, yt):
        # Leg 1: (P_ref, y_ref) -> (P_ref, yt)
        if n1 < 2:
            du1 = 0.0
        else:
            y_leg1 = np.linspace(y_ref, yt, n1+1)
            P_leg1 = np.full_like(y_leg1, P_ref, dtype=float)
            du1 = _integrate_along_polyline(P_leg1, y_leg1)

        # Leg 2: (P_ref, yt) -> (Pt, yt)
        if n2 < 2:
            du2 = 0.0
        else:
            P_leg2 = np.linspace(P_ref, Pt, n2+1)
            y_leg2 = np.full_like(P_leg2, yt, dtype=float)
            du2 = _integrate_along_polyline(P_leg2, y_leg2)

        return du1 + du2

    # Vectorize over elements
    u_out = np.empty_like(P_arr, dtype=float)
    mismatch = None

    for idx in np.ndindex(P_arr.shape):
        Pt, yt = P_arr[idx], y_arr[idx]
        du_forward = _two_leg(Pt, yt)
        u_out[idx] = u_ref + du_forward

        if return_path_mismatch:
            # Reverse legs order: (P_ref,y_ref)->(Pt,y_ref)->(Pt,yt)
            def _two_leg_rev(Pt_, yt_):
                if n2 < 2:
                    duA = 0.0
                else:
                    P_legA = np.linspace(P_ref, Pt_, n2+1)
                    y_legA = np.full_like(P_legA, y_ref, dtype=float)
                    duA = _integrate_along_polyline(P_legA, y_legA)
                if n1 < 2:
                    duB = 0.0
                else:
                    y_legB = np.linspace(y_ref, yt_, n1+1)
                    P_legB = np.full_like(y_legB, Pt_, dtype=float)
                    duB = _integrate_along_polyline(P_legB, y_legB)
                return duA + duB

            du_rev = _two_leg_rev(Pt, yt)
            if mismatch is None:
                mismatch = np.empty_like(P_arr, dtype=float)
            mismatch[idx] = (u_ref + du_forward) - (u_ref + du_rev)

    if np.isscalar(P) and np.isscalar(y):
        return (float(u_out), float(mismatch) if return_path_mismatch else None) if return_path_mismatch else float(u_out)
    return (u_out, mismatch) if return_path_mismatch else u_out

def get_u_sol_from_Py(P, y, P_ref=p_grid[0], y_ref = y_grid[0], u_ref=0.0, n1=200, n2=200, return_path_mismatch=False):
    """
    Compute specific internal energy u(P, y) for liquid MgSiO3 relative to a reference (P_ref, y_ref).

    Thermodynamic differential used:
        du = c_v dT + [P - T * alpha * K_T] / rho^2 * d(rho)

    Path: (P_ref, y_ref) -> (P_ref, y) -> (P, y)

    Parameters
    ----------
    P, y : float or np.ndarray
        Target pressure and 'y' (your EOS entropy-like coordinate) in the same shapes.
    P_ref, y_ref : float
        Reference state in the same units as P, y. u(P_ref, y_ref) is set to u_ref.
    u_ref : float, optional
        Internal energy at the reference state. Default 0.0 (J/kg, or your working unit).
    n1, n2 : int, optional
        Number of integration segments for leg-1 (vary y at fixed P_ref) and leg-2 (vary P at fixed y).
    return_path_mismatch : bool, optional
        If True, also compute the reversed-leg path and return their difference (diagnostic).

    Returns
    -------
    u : float or np.ndarray
        Specific internal energy at (P, y), same shape as inputs P and y.
    (optional) mismatch : float or np.ndarray
        Difference between two-leg orders (should be ~0 within quadrature error).

    Notes
    -----
    - Units must be consistent:
        rho : kg/m^3
        P, K_T : Pa
        T : K
        alpha : 1/K
        c_v : J/(kg K)
      so that u comes out in J/kg.
    - This function relies on:
        get_Tliq_Py(P, y), get_rholiq_Py(P, y),
        get_CVliq_Py(P, y), get_alphaliq_Py(P, y), get_KT_liq(P, y).
    - For the solid, make an identical wrapper swapping in the solid getters.
    """

    # Broadcast targets
    P_arr = np.array(P, ndmin=1)
    y_arr = np.array(y, ndmin=1)
    if P_arr.shape != y_arr.shape:
        P_arr, y_arr = np.broadcast_arrays(P_arr, y_arr)

    def _integrate_along_polyline(P_path, y_path):
        """Trapezoidal line integral of du along a polyline in (P, y)."""
        # Evaluate state along the path
        T_path   = get_Tsol_Py(P_path, y_path)
        rho_path = get_rhosol_Py(P_path, y_path)
        cv_path  = get_CVsol_Py(P_path, y_path) / J_K_kg_to_erg_K_g # SI units
        a_path   = get_alphasol_Py(P_path, y_path)
        KT_path  = get_KT_sol(P_path, y_path)

        # Midpoint values for the integrand
        T_mid   = 0.5 * (T_path[:-1]   + T_path[1:])
        rho_mid = 0.5 * (rho_path[:-1] + rho_path[1:])
        cv_mid  = 0.5 * (cv_path[:-1]  + cv_path[1:])
        a_mid   = 0.5 * (a_path[:-1]   + a_path[1:])
        KT_mid  = 0.5 * (KT_path[:-1]  + KT_path[1:])
        P_mid   = 0.5 * (P_path[:-1]   + P_path[1:])

        # Increments along the path
        dT   = np.diff(T_path)
        drho = np.diff(rho_path)

        # Differential work: du = cv dT + [P - T alpha K_T] / rho^2 * d(rho)
        term1 = cv_mid * dT
        term2 = (P_mid - T_mid * a_mid * KT_mid) / (rho_mid**2) * drho

        return np.sum(term1 + term2)

    def _two_leg(Pt, yt):
        # Leg 1: (P_ref, y_ref) -> (P_ref, yt)
        if n1 < 2:
            du1 = 0.0
        else:
            y_leg1 = np.linspace(y_ref, yt, n1+1)
            P_leg1 = np.full_like(y_leg1, P_ref, dtype=float)
            du1 = _integrate_along_polyline(P_leg1, y_leg1)

        # Leg 2: (P_ref, yt) -> (Pt, yt)
        if n2 < 2:
            du2 = 0.0
        else:
            P_leg2 = np.linspace(P_ref, Pt, n2+1)
            y_leg2 = np.full_like(P_leg2, yt, dtype=float)
            du2 = _integrate_along_polyline(P_leg2, y_leg2)

        return du1 + du2

    # Vectorize over elements
    u_out = np.empty_like(P_arr, dtype=float)
    mismatch = None

    for idx in np.ndindex(P_arr.shape):
        Pt, yt = P_arr[idx], y_arr[idx]
        du_forward = _two_leg(Pt, yt)
        u_out[idx] = u_ref + du_forward

        if return_path_mismatch:
            # Reverse legs order: (P_ref,y_ref)->(Pt,y_ref)->(Pt,yt)
            def _two_leg_rev(Pt_, yt_):
                if n2 < 2:
                    duA = 0.0
                else:
                    P_legA = np.linspace(P_ref, Pt_, n2+1)
                    y_legA = np.full_like(P_legA, y_ref, dtype=float)
                    duA = _integrate_along_polyline(P_legA, y_legA)
                if n1 < 2:
                    duB = 0.0
                else:
                    y_legB = np.linspace(y_ref, yt_, n1+1)
                    P_legB = np.full_like(y_legB, Pt_, dtype=float)
                    duB = _integrate_along_polyline(P_legB, y_legB)
                return duA + duB

            du_rev = _two_leg_rev(Pt, yt)
            if mismatch is None:
                mismatch = np.empty_like(P_arr, dtype=float)
            mismatch[idx] = (u_ref + du_forward) - (u_ref + du_rev)

    if np.isscalar(P) and np.isscalar(y):
        return (float(u_out), float(mismatch) if return_path_mismatch else None) if return_path_mismatch else float(u_out)
    return (u_out, mismatch) if return_path_mismatch else u_out

def get_s_liq_from_Py(P, y, P_ref=p_grid[0], y_ref = y_grid[0], s_ref=0.0, n1=200, n2=200, return_path_mismatch=False,
                  use_cp_form_check=False):
    """
    Compute specific entropy s(P, y) for liquid MgSiO3 relative to a reference (P_ref, y_ref).

    Exact differential used:
        ds = (c_v/T) dT  -  (alpha * K_T / rho^2) d(rho)

    Path: (P_ref, y_ref) -> (P_ref, y) -> (P, y)

    Parameters
    ----------
    P, y : float or np.ndarray
        Target pressure and EOS 'y' coordinate (your adiabat/entropy label), broadcastable shapes.
    P_ref, y_ref : float
        Reference state where s(P_ref, y_ref) is anchored to s_ref.
    s_ref : float, optional
        Specific entropy at the reference state (J/kg/K). Default 0.0.
    n1, n2 : int, optional
        Number of segments for leg-1 (vary y at fixed P_ref) and leg-2 (vary P at fixed y).
    return_path_mismatch : bool, optional
        If True, also compute the reversed-leg order and return the difference (diagnostic).
    use_cp_form_check : bool, optional
        If True, also evaluate the integral using the (c_p/T, alpha/rho) form on the SAME polyline
        and return (s_cp - s_cv) as an additional diagnostic.

    Returns
    -------
    s : float or np.ndarray
        Specific entropy at (P, y), same shape as inputs.
    (optional) mismatch : float or np.ndarray
        Path-order mismatch (should be ~0 within quadrature error).
    (optional) form_diff : float or np.ndarray
        Difference between cp-form and cv-form integrals on the same polyline (should be ~0).

    Notes
    -----
    Units must be consistent:
        T [K], rho [kg/m^3], alpha [1/K], K_T [Pa], c_v & c_p [J/(kg K)].
    Required getters (liquid variants):
        get_Tliq_Py, get_rholiq_Py, get_CVliq_Py, get_CPliq_Py, get_alphaliq_Py, get_KT_liq.
    For solids, clone this and swap in your solid getters.
    """

    # Broadcast targets
    P_arr = np.array(P, ndmin=1)
    y_arr = np.array(y, ndmin=1)
    if P_arr.shape != y_arr.shape:
        P_arr, y_arr = np.broadcast_arrays(P_arr, y_arr)

    def _integrate_polyline_cv(P_path, y_path):
        """Trapezoidal line integral of ds using the (cv/T, alpha*KT/rho^2) form."""
        T_path   = get_Tliq_Py(P_path, y_path)
        rho_path = get_rholiq_Py(P_path, y_path)
        cv_path  = get_CVliq_Py(P_path, y_path) / J_K_kg_to_erg_K_g # SI units
        a_path   = get_alphaliq_Py(P_path, y_path)
        KT_path  = get_KT_liq(P_path, y_path)

        # Midpoints
        Tm   = 0.5 * (T_path[:-1]   + T_path[1:])
        rhom = 0.5 * (rho_path[:-1] + rho_path[1:])
        cvm  = 0.5 * (cv_path[:-1]  + cv_path[1:])
        am   = 0.5 * (a_path[:-1]   + a_path[1:])
        KTm  = 0.5 * (KT_path[:-1]  + KT_path[1:])

        dT   = np.diff(T_path)
        drho = np.diff(rho_path)

        term_T   = (cvm / Tm) * dT
        term_rho = - (am * KTm) / (rhom**2) * drho
        return np.sum(term_T + term_rho)

    def _integrate_polyline_cp(P_path, y_path):
        """Same polyline, but use the (cp/T, alpha/rho) form: ds = (cp/T)dT - (alpha/rho)dP."""
        T_path   = get_Tliq_Py(P_path, y_path)
        rho_path = get_rholiq_Py(P_path, y_path)
        cp_path  = get_CPliq_Py(P_path, y_path) / J_K_kg_to_erg_K_g # SI units
        a_path   = get_alphaliq_Py(P_path, y_path)

        Tm   = 0.5 * (T_path[:-1]   + T_path[1:])
        rhom = 0.5 * (rho_path[:-1] + rho_path[1:])
        cpm  = 0.5 * (cp_path[:-1]  + cp_path[1:])
        am   = 0.5 * (a_path[:-1]   + a_path[1:])
        dT   = np.diff(T_path)
        dP   = np.diff(P_path)

        term_T = (cpm / Tm) * dT
        term_P = - (am / rhom) * dP
        return np.sum(term_T + term_P)

    def _two_leg(Pt, yt, use_cp=False):
        # Leg 1: (P_ref, y_ref) -> (P_ref, yt)
        if n1 >= 2:
            y_leg1 = np.linspace(y_ref, yt, n1 + 1)
            P_leg1 = np.full_like(y_leg1, P_ref, dtype=float)
            ds1 = _integrate_polyline_cp(P_leg1, y_leg1) if use_cp else _integrate_polyline_cv(P_leg1, y_leg1)
        else:
            ds1 = 0.0

        # Leg 2: (P_ref, yt) -> (Pt, yt)
        if n2 >= 2:
            P_leg2 = np.linspace(P_ref, Pt, n2 + 1)
            y_leg2 = np.full_like(P_leg2, yt, dtype=float)
            ds2 = _integrate_polyline_cp(P_leg2, y_leg2) if use_cp else _integrate_polyline_cv(P_leg2, y_leg2)
        else:
            ds2 = 0.0

        return ds1 + ds2

    s_out = np.empty_like(P_arr, dtype=float)
    mismatch = None
    form_diff = None

    for idx in np.ndindex(P_arr.shape):
        Pt, yt = P_arr[idx], y_arr[idx]

        ds_cv = _two_leg(Pt, yt, use_cp=False)
        s_out[idx] = s_ref + ds_cv

        if return_path_mismatch:
            # Reverse leg order: (P_ref,y_ref)->(Pt,y_ref)->(Pt,yt)
            if n2 >= 2:
                P_legA = np.linspace(P_ref, Pt, n2 + 1)
                y_legA = np.full_like(P_legA, y_ref, dtype=float)
                dsA = _integrate_polyline_cv(P_legA, y_legA)
            else:
                dsA = 0.0
            if n1 >= 2:
                y_legB = np.linspace(y_ref, yt, n1 + 1)
                P_legB = np.full_like(y_legB, Pt, dtype=float)
                dsB = _integrate_polyline_cv(P_legB, y_legB)
            else:
                dsB = 0.0

            if mismatch is None:
                mismatch = np.empty_like(P_arr, dtype=float)
            mismatch[idx] = (s_ref + ds_cv) - (s_ref + dsA + dsB)

        if use_cp_form_check:
            ds_cp = _two_leg(Pt, yt, use_cp=True)
            if form_diff is None:
                form_diff = np.empty_like(P_arr, dtype=float)
            form_diff[idx] = (s_ref + ds_cp) - (s_ref + ds_cv)

    if np.isscalar(P) and np.isscalar(y):
        if return_path_mismatch and use_cp_form_check:
            return float(s_out), float(mismatch), float(form_diff)
        if return_path_mismatch:
            return float(s_out), float(mismatch)
        if use_cp_form_check:
            return float(s_out), float(form_diff)
        return float(s_out)

    if return_path_mismatch and use_cp_form_check:
        return s_out, mismatch, form_diff
    if return_path_mismatch:
        return s_out, mismatch
    if use_cp_form_check:
        return s_out, form_diff
    return s_out

def get_s_sol_from_Py(P, y, P_ref=p_grid[0], y_ref = y_grid[0], s_ref=0.0, n1=200, n2=200, return_path_mismatch=False,
                  use_cp_form_check=False):
    """
    Compute specific entropy s(P, y) for liquid MgSiO3 relative to a reference (P_ref, y_ref).

    Exact differential used:
        ds = (c_v/T) dT  -  (alpha * K_T / rho^2) d(rho)

    Path: (P_ref, y_ref) -> (P_ref, y) -> (P, y)

    Parameters
    ----------
    P, y : float or np.ndarray
        Target pressure and EOS 'y' coordinate (your adiabat/entropy label), broadcastable shapes.
    P_ref, y_ref : float
        Reference state where s(P_ref, y_ref) is anchored to s_ref.
    s_ref : float, optional
        Specific entropy at the reference state (J/kg/K). Default 0.0.
    n1, n2 : int, optional
        Number of segments for leg-1 (vary y at fixed P_ref) and leg-2 (vary P at fixed y).
    return_path_mismatch : bool, optional
        If True, also compute the reversed-leg order and return the difference (diagnostic).
    use_cp_form_check : bool, optional
        If True, also evaluate the integral using the (c_p/T, alpha/rho) form on the SAME polyline
        and return (s_cp - s_cv) as an additional diagnostic.

    Returns
    -------
    s : float or np.ndarray
        Specific entropy at (P, y), same shape as inputs.
    (optional) mismatch : float or np.ndarray
        Path-order mismatch (should be ~0 within quadrature error).
    (optional) form_diff : float or np.ndarray
        Difference between cp-form and cv-form integrals on the same polyline (should be ~0).

    Notes
    -----
    Units must be consistent:
        T [K], rho [kg/m^3], alpha [1/K], K_T [Pa], c_v & c_p [J/(kg K)].
    Required getters (liquid variants):
        get_Tliq_Py, get_rholiq_Py, get_CVliq_Py, get_CPliq_Py, get_alphaliq_Py, get_KT_liq.
    For solids, clone this and swap in your solid getters.
    """

    # Broadcast targets
    P_arr = np.array(P, ndmin=1)
    y_arr = np.array(y, ndmin=1)
    if P_arr.shape != y_arr.shape:
        P_arr, y_arr = np.broadcast_arrays(P_arr, y_arr)

    def _integrate_polyline_cv(P_path, y_path):
        """Trapezoidal line integral of ds using the (cv/T, alpha*KT/rho^2) form."""
        T_path   = get_Tsol_Py(P_path, y_path)
        rho_path = get_rhosol_Py(P_path, y_path)
        cv_path  = get_CVsol_Py(P_path, y_path) / J_K_kg_to_erg_K_g # SI units
        a_path   = get_alphasol_Py(P_path, y_path)
        KT_path  = get_KT_sol(P_path, y_path)
        # Midpoints
        Tm   = 0.5 * (T_path[:-1]   + T_path[1:])
        rhom = 0.5 * (rho_path[:-1] + rho_path[1:])
        cvm  = 0.5 * (cv_path[:-1]  + cv_path[1:])
        am   = 0.5 * (a_path[:-1]   + a_path[1:])
        KTm  = 0.5 * (KT_path[:-1]  + KT_path[1:])

        dT   = np.diff(T_path)
        drho = np.diff(rho_path)

        term_T   = (cvm / Tm) * dT
        term_rho = - (am * KTm) / (rhom**2) * drho
        return np.sum(term_T + term_rho)

    def _integrate_polyline_cp(P_path, y_path):
        """Same polyline, but use the (cp/T, alpha/rho) form: ds = (cp/T)dT - (alpha/rho)dP."""
        T_path   = get_Tsol_Py(P_path, y_path)
        rho_path = get_rhosol_Py(P_path, y_path)
        cp_path  = get_CPsol_Py(P_path, y_path) / J_K_kg_to_erg_K_g # SI units
        a_path   = get_alphasol_Py(P_path, y_path)

        Tm   = 0.5 * (T_path[:-1]   + T_path[1:])
        rhom = 0.5 * (rho_path[:-1] + rho_path[1:])
        cpm  = 0.5 * (cp_path[:-1]  + cp_path[1:])
        am   = 0.5 * (a_path[:-1]   + a_path[1:])
        dT   = np.diff(T_path)
        dP   = np.diff(P_path)

        term_T = (cpm / Tm) * dT
        term_P = - (am / rhom) * dP
        return np.sum(term_T + term_P)

    def _two_leg(Pt, yt, use_cp=False):
        # Leg 1: (P_ref, y_ref) -> (P_ref, yt)
        if n1 >= 2:
            y_leg1 = np.linspace(y_ref, yt, n1 + 1)
            P_leg1 = np.full_like(y_leg1, P_ref, dtype=float)
            ds1 = _integrate_polyline_cp(P_leg1, y_leg1) if use_cp else _integrate_polyline_cv(P_leg1, y_leg1)
        else:
            ds1 = 0.0

        # Leg 2: (P_ref, yt) -> (Pt, yt)
        if n2 >= 2:
            P_leg2 = np.linspace(P_ref, Pt, n2 + 1)
            y_leg2 = np.full_like(P_leg2, yt, dtype=float)
            ds2 = _integrate_polyline_cp(P_leg2, y_leg2) if use_cp else _integrate_polyline_cv(P_leg2, y_leg2)
        else:
            ds2 = 0.0

        return ds1 + ds2

    s_out = np.empty_like(P_arr, dtype=float)
    mismatch = None
    form_diff = None

    for idx in np.ndindex(P_arr.shape):
        Pt, yt = P_arr[idx], y_arr[idx]

        ds_cv = _two_leg(Pt, yt, use_cp=False)
        s_out[idx] = s_ref + ds_cv

        if return_path_mismatch:
            # Reverse leg order: (P_ref,y_ref)->(Pt,y_ref)->(Pt,yt)
            if n2 >= 2:
                P_legA = np.linspace(P_ref, Pt, n2 + 1)
                y_legA = np.full_like(P_legA, y_ref, dtype=float)
                dsA = _integrate_polyline_cv(P_legA, y_legA)
            else:
                dsA = 0.0
            if n1 >= 2:
                y_legB = np.linspace(y_ref, yt, n1 + 1)
                P_legB = np.full_like(y_legB, Pt, dtype=float)
                dsB = _integrate_polyline_cv(P_legB, y_legB)
            else:
                dsB = 0.0

            if mismatch is None:
                mismatch = np.empty_like(P_arr, dtype=float)
            mismatch[idx] = (s_ref + ds_cv) - (s_ref + dsA + dsB)

        if use_cp_form_check:
            ds_cp = _two_leg(Pt, yt, use_cp=True)
            if form_diff is None:
                form_diff = np.empty_like(P_arr, dtype=float)
            form_diff[idx] = (s_ref + ds_cp) - (s_ref + ds_cv)

    if np.isscalar(P) and np.isscalar(y):
        if return_path_mismatch and use_cp_form_check:
            return float(s_out), float(mismatch), float(form_diff)
        if return_path_mismatch:
            return float(s_out), float(mismatch)
        if use_cp_form_check:
            return float(s_out), float(form_diff)
        return float(s_out)

    if return_path_mismatch and use_cp_form_check:
        return s_out, mismatch, form_diff
    if return_path_mismatch:
        return s_out, mismatch
    if use_cp_form_check:
        return s_out, form_diff
    return s_out

# Thermodynamic consistency check:

def eos_consistency_report_liq(
    P, y,
    P_ref=p_grid[0], y_ref=y_grid[0],
    u_ref=0.0, s_ref=0.0,
    n1=200, n2=200,
    eps_rel_P=1e-4, eps_abs_P=1e3,
    eps_rel_y=1e-5, eps_abs_y=1e-6,
    tol_rel_cp_cv=1e-3,
    tol_rel_gamma=1e-3,
    tol_rel_maxwell=1e-3,
    tol_abs_path=1e-6,
):
    """
    Thermodynamic consistency tests (liquid MgSiO3) on points (P, y).

    Tests performed at each point (all SI units):
      1) Heat-capacity identity:   cp - cv ?= T * alpha^2 * K_T / rho
      2) Grüneisen identity:       gamma ?= alpha * K_T / (rho * cv)
      3) Maxwell (Helmholtz):      (∂P/∂T)_rho ?= alpha * K_T
         (via chain rule using T(P,y), rho(P,y) partials)
      4) Path independence (u):    two-leg path mismatch ~ 0
      5) Path independence (s):    two-leg path mismatch ~ 0
      6) Entropy form check:       ∮ ds_cp - ds_cv ≈ 0 on same polyline
      7) Basic positivity:         rho>0, T>0, K_T>0, cv>0, cp>cv

    Returns
    -------
    summary : dict
        'max_rel' and 'rms_rel' for (cp_cv, gamma, maxwell);
        'max_abs' for (u_path, s_path, s_form);
        'positivity_fail_frac' for each positivity check; and
        per-point arrays under 'per_point'.
    """

    # Broadcast targets
    P_arr = np.array(P, ndmin=1, dtype=float)
    y_arr = np.array(y, ndmin=1, dtype=float)
    if P_arr.shape != y_arr.shape:
        P_arr, y_arr = np.broadcast_arrays(P_arr, y_arr)

    shp = P_arr.shape

    # Storage
    rel_cp_cv = np.empty(shp)
    rel_gamma = np.empty(shp)
    rel_maxwell = np.empty(shp)
    abs_u_path = np.empty(shp)
    abs_s_path = np.empty(shp)
    abs_s_form = np.empty(shp)

    pos_rho_fail = np.zeros(shp, dtype=bool)
    pos_T_fail   = np.zeros(shp, dtype=bool)
    pos_KT_fail  = np.zeros(shp, dtype=bool)
    pos_cv_fail  = np.zeros(shp, dtype=bool)
    pos_cp_gt_cv_fail = np.zeros(shp, dtype=bool)

    tiny = 1e-30

    # Helper: central partial derivatives in (P,y) chart
    def _dP(step_P, P0):  # signed step, avoid P<0 nonsense if your table requires
        return max(abs(step_P), eps_abs_P) * (1.0 if P0 >= 0 else -1.0)

    def _dy(step_y, y0):
        return max(abs(step_y), eps_abs_y) * (1.0 if y0 >= 0 else -1.0)

    def _partials_at(P0, y0):
        dP = _dP(eps_rel_P * max(abs(P0), 1.0), P0)
        dy = _dy(eps_rel_y * max(abs(y0), 1.0), y0)

        # T and rho partials
        T_p = (get_Tliq_Py(P0 + dP, y0) - get_Tliq_Py(P0 - dP, y0)) / (2.0 * dP)
        T_y = (get_Tliq_Py(P0, y0 + dy) - get_Tliq_Py(P0, y0 - dy)) / (2.0 * dy)

        r_p = (get_rholiq_Py(P0 + dP, y0) - get_rholiq_Py(P0 - dP, y0)) / (2.0 * dP)
        r_y = (get_rholiq_Py(P0, y0 + dy) - get_rholiq_Py(P0, y0 - dy)) / (2.0 * dy)

        return T_p, T_y, r_p, r_y

    # Optional gamma provider
    _has_gamma = 'get_gamma_Py' in globals()

    # Loop points
    it = np.ndindex(shp)
    for idx in it:
        Pi, yi = float(P_arr[idx]), float(y_arr[idx])

        # State values
        Ti   = float(get_Tliq_Py(Pi, yi))
        rhoi = float(get_rholiq_Py(Pi, yi))
        cpi  = float(get_CPliq_Py(Pi, yi)) / J_K_kg_to_erg_K_g # SI units
        cvi  = float(get_CVliq_Py(Pi, yi)) / J_K_kg_to_erg_K_g # SI units
        ai   = float(get_alphaliq_Py(Pi, yi))
        KTi  = float(get_KT_liq(Pi, yi))

        # 1) cp-cv identity
        rhs_cc = Ti * ai * ai * KTi / max(rhoi, tiny)
        rel_cp_cv[idx] = abs((cpi - cvi) - rhs_cc) / max(abs(rhs_cc), tiny)

        # 2) Grüneisen identity
        gamma_calc = ai * KTi / max(rhoi * cvi, tiny)
        if _has_gamma:
            gamma_tab = float(get_gamma_Py(Pi, yi))
            rel_gamma[idx] = abs(gamma_calc - gamma_tab) / max(abs(gamma_tab), tiny)
        else:
            # Compare against itself to give zero; caller can still read gamma_calc via per_point
            rel_gamma[idx] = 0.0

        # 3) Maxwell: (∂P/∂T)_rho  ?= alpha K_T
        #    Use chain rule in (P,y): (∂T/∂P)_rho = T_P - T_y * (rho_P / rho_Y)
        T_P, T_Y, R_P, R_Y = _partials_at(Pi, yi)
        denom = T_P - T_Y * (R_P / max(R_Y, tiny))
        dP_dT_rho = 1.0 / denom if abs(denom) > tiny else np.nan
        rel_maxwell[idx] = abs(dP_dT_rho - ai * KTi) / max(abs(ai * KTi), tiny)

        # 4–6) Path-independence using our integrators (if present)
        try:
            u_val, u_mis = get_u_liq_from_Py(Pi, yi, P_ref, y_ref, u_ref, n1=n1, n2=n2, return_path_mismatch=True)
            abs_u_path[idx] = abs(u_mis)
        except NameError:
            abs_u_path[idx] = np.nan

        try:
            s_val, s_mis, s_form = get_s_liq_from_Py(Pi, yi, P_ref, y_ref, s_ref, n1=n1, n2=n2,
                                                 return_path_mismatch=True, use_cp_form_check=True)
            abs_s_path[idx] = abs(s_mis)
            abs_s_form[idx] = abs(s_form)
        except NameError:
            abs_s_path[idx] = np.nan
            abs_s_form[idx] = np.nan

        # 7) Positivity checks
        pos_rho_fail[idx]     = not (rhoi > 0.0)
        pos_T_fail[idx]       = not (Ti   > 0.0)
        pos_KT_fail[idx]      = not (KTi  > 0.0)
        pos_cv_fail[idx]      = not (cvi  > 0.0)
        pos_cp_gt_cv_fail[idx]= not (cpi  > cvi)

    # Summaries
    def _safe_stats(arr):
        a = arr[np.isfinite(arr)]
        if a.size == 0:
            return np.nan, np.nan
        return float(np.max(a)), float(np.sqrt(np.mean(a*a)))

    max_rel_cp_cv, rms_rel_cp_cv       = _safe_stats(rel_cp_cv)
    max_rel_gamma, rms_rel_gamma       = _safe_stats(rel_gamma)
    max_rel_maxwell, rms_rel_maxwell   = _safe_stats(rel_maxwell)
    max_abs_u_path, _                  = _safe_stats(abs_u_path)
    max_abs_s_path, _                  = _safe_stats(abs_s_path)
    max_abs_s_form, _                  = _safe_stats(abs_s_form)

    summary = {
        "max_rel": {
            "cp_minus_cv": max_rel_cp_cv,
            "gamma_identity": max_rel_gamma,
            "maxwell_dPdT_at_rho": max_rel_maxwell,
        },
        "rms_rel": {
            "cp_minus_cv": rms_rel_cp_cv,
            "gamma_identity": rms_rel_gamma,
            "maxwell_dPdT_at_rho": rms_rel_maxwell,
        },
        "max_abs": {
            "u_path_mismatch": max_abs_u_path,   # J/kg
            "s_path_mismatch": max_abs_s_path,   # J/(kg K)
            "s_cp_vs_cv_form": max_abs_s_form,   # J/(kg K)
        },
        "positivity_fail_frac": {
            "rho": float(np.mean(pos_rho_fail)),
            "T": float(np.mean(pos_T_fail)),
            "K_T": float(np.mean(pos_KT_fail)),
            "c_v": float(np.mean(pos_cv_fail)),
            "c_p_gt_c_v": float(np.mean(pos_cp_gt_cv_fail)),
        },
        "per_point": {
            "rel_cp_minus_cv": rel_cp_cv,
            "rel_gamma_identity": rel_gamma,
            "rel_maxwell": rel_maxwell,
            "abs_u_path_mismatch": abs_u_path,
            "abs_s_path_mismatch": abs_s_path,
            "abs_s_cp_vs_cv_form": abs_s_form,
        },
        "tolerances": {
            "cp_minus_cv_rel_tol": tol_rel_cp_cv,
            "gamma_identity_rel_tol": tol_rel_gamma,
            "maxwell_rel_tol": tol_rel_maxwell,
            "path_abs_tol": tol_abs_path
        },
        "pass_flags_overall": {
            "cp_minus_cv": (max_rel_cp_cv <= tol_rel_cp_cv),
            "gamma_identity": (max_rel_gamma <= tol_rel_gamma),
            "maxwell": (max_rel_maxwell <= tol_rel_maxwell),
            "u_path": (np.nan_to_num(max_abs_u_path, nan=0.0) <= tol_abs_path),
            "s_path": (np.nan_to_num(max_abs_s_path, nan=0.0) <= tol_abs_path),
            "s_form": (np.nan_to_num(max_abs_s_form, nan=0.0) <= tol_abs_path),
            "positivity": all(f == 0.0 for f in [
                float(np.mean(pos_rho_fail)),
                float(np.mean(pos_T_fail)),
                float(np.mean(pos_KT_fail)),
                float(np.mean(pos_cv_fail)),
            ])
        }
    }
    return summary

# to update temperature and density from pressure and y of the solid phase
def get_Tsol_en_Py(P, y):
    """
    Return mixture temperature T (K) at pressure P (Pa) and composition y for enstatite phase.
    """
    scalar = np.isscalar(P) and np.isscalar(y)
    P_arr = np.array(P, ndmin=1)
    y_arr = np.array(y, ndmin=1)
    if P_arr.shape != y_arr.shape:
        P_arr, y_arr = np.broadcast_arrays(P_arr, y_arr)
    pts = np.stack((P_arr.ravel(), y_arr.ravel()), axis=-1)
    vals = get_T_Py_sol_en(pts).reshape(P_arr.shape)
    return float(vals) if scalar else vals


def get_rhosol_en_Py(P, y):
    """
    Return mixture density rho (kg/m^3) at pressure P (Pa) and composition y for enstatite phase.
    """
    scalar = np.isscalar(P) and np.isscalar(y)
    P_arr = np.array(P, ndmin=1)
    y_arr = np.array(y, ndmin=1)
    if P_arr.shape != y_arr.shape:
        P_arr, y_arr = np.broadcast_arrays(P_arr, y_arr)
    pts = np.stack((P_arr.ravel(), y_arr.ravel()), axis=-1)
    vals = get_rho_Py_sol_en(pts).reshape(P_arr.shape)
    return float(vals) if scalar else vals


def get_alphasol_en_Py(P, y):
    """
    Return mixture thermal expansivity alpha (1/K) at pressure P (Pa) and composition y for enstatite phase.
    """
    scalar = np.isscalar(P) and np.isscalar(y)
    P_arr = np.array(P, ndmin=1)
    y_arr = np.array(y, ndmin=1)
    if P_arr.shape != y_arr.shape:
        P_arr, y_arr = np.broadcast_arrays(P_arr, y_arr)
    pts = np.stack((P_arr.ravel(), y_arr.ravel()), axis=-1)
    vals = get_alpha_Py_sol_en(pts).reshape(P_arr.shape)
    return float(vals) if scalar else vals

# to update temperature and density from pressure and y of the solid phase
def get_Tsol_pv_Py(P, y):
    """
    Return mixture temperature T (K) at pressure P (Pa) and composition y for enstatite phase.
    """
    scalar = np.isscalar(P) and np.isscalar(y)
    P_arr = np.array(P, ndmin=1)
    y_arr = np.array(y, ndmin=1)
    if P_arr.shape != y_arr.shape:
        P_arr, y_arr = np.broadcast_arrays(P_arr, y_arr)
    pts = np.stack((P_arr.ravel(), y_arr.ravel()), axis=-1)
    vals = get_T_Py_sol_pv(pts).reshape(P_arr.shape)
    return float(vals) if scalar else vals


def get_rhosol_pv_Py(P, y):
    """
    Return mixture density rho (kg/m^3) at pressure P (Pa) and composition y for enstatite phase.
    """
    scalar = np.isscalar(P) and np.isscalar(y)
    P_arr = np.array(P, ndmin=1)
    y_arr = np.array(y, ndmin=1)
    if P_arr.shape != y_arr.shape:
        P_arr, y_arr = np.broadcast_arrays(P_arr, y_arr)
    pts = np.stack((P_arr.ravel(), y_arr.ravel()), axis=-1)
    vals = get_rho_Py_sol_pv(pts).reshape(P_arr.shape)
    return float(vals) if scalar else vals


def get_alphasol_pv_Py(P, y):
    """
    Return mixture thermal expansivity alpha (1/K) at pressure P (Pa) and composition y for enstatite phase.
    """
    scalar = np.isscalar(P) and np.isscalar(y)
    P_arr = np.array(P, ndmin=1)
    y_arr = np.array(y, ndmin=1)
    if P_arr.shape != y_arr.shape:
        P_arr, y_arr = np.broadcast_arrays(P_arr, y_arr)
    pts = np.stack((P_arr.ravel(), y_arr.ravel()), axis=-1)
    vals = get_alpha_Py_sol_pv(pts).reshape(P_arr.shape)
    return float(vals) if scalar else vals

# to update temperature and density from pressure and y of the solid phase
def get_Tsol_ppv_Py(P, y):
    """
    Return mixture temperature T (K) at pressure P (Pa) and composition y for enstatite phase.
    """
    scalar = np.isscalar(P) and np.isscalar(y)
    P_arr = np.array(P, ndmin=1)
    y_arr = np.array(y, ndmin=1)
    if P_arr.shape != y_arr.shape:
        P_arr, y_arr = np.broadcast_arrays(P_arr, y_arr)
    pts = np.stack((P_arr.ravel(), y_arr.ravel()), axis=-1)
    vals = get_T_Py_sol_ppv(pts).reshape(P_arr.shape)
    return float(vals) if scalar else vals


def get_rhosol_ppv_Py(P, y):
    """
    Return mixture density rho (kg/m^3) at pressure P (Pa) and composition y for enstatite phase.
    """
    scalar = np.isscalar(P) and np.isscalar(y)
    P_arr = np.array(P, ndmin=1)
    y_arr = np.array(y, ndmin=1)
    if P_arr.shape != y_arr.shape:
        P_arr, y_arr = np.broadcast_arrays(P_arr, y_arr)
    pts = np.stack((P_arr.ravel(), y_arr.ravel()), axis=-1)
    vals = get_rho_Py_sol_ppv(pts).reshape(P_arr.shape)
    return float(vals) if scalar else vals


def get_alphasol_ppv_Py(P, y):
    """
    Return mixture thermal expansivity alpha (1/K) at pressure P (Pa) and composition y for enstatite phase.
    """
    scalar = np.isscalar(P) and np.isscalar(y)
    P_arr = np.array(P, ndmin=1)
    y_arr = np.array(y, ndmin=1)
    if P_arr.shape != y_arr.shape:
        P_arr, y_arr = np.broadcast_arrays(P_arr, y_arr)
    pts = np.stack((P_arr.ravel(), y_arr.ravel()), axis=-1)
    vals = get_alpha_Py_sol_ppv(pts).reshape(P_arr.shape)
    return float(vals) if scalar else vals

def get_Tsol_Py(P, y):
    """
    General mixture temperature: selects en, pv, or ppv interpolator based on P.
    """
    scalar = np.isscalar(P) and np.isscalar(y)
    P_arr = np.array(P, ndmin=1)
    y_arr = np.array(y, ndmin=1)
    if P_arr.shape != y_arr.shape:
        P_arr, y_arr = np.broadcast_arrays(P_arr, y_arr)

    T_out = np.empty_like(P_arr, dtype=float)
    mask_en = P_arr < _EN_BOUND
    mask_pv = (P_arr >= _EN_BOUND) & (P_arr < _PV_BOUND)
    mask_ppv = P_arr >= _PV_BOUND

    if np.any(mask_en):
        T_out[mask_en] = get_Tsol_en_Py(P_arr[mask_en], y_arr[mask_en])
    if np.any(mask_pv):
        T_out[mask_pv] = get_Tsol_pv_Py(P_arr[mask_pv], y_arr[mask_pv])
    if np.any(mask_ppv):
        T_out[mask_ppv] = get_Tsol_ppv_Py(P_arr[mask_ppv], y_arr[mask_ppv])

    return float(T_out) if scalar else T_out


def get_rhosol_Py(P, y):
    """
    General mixture density rho: selects en, pv, or ppv interpolator based on P.
    """
    scalar = np.isscalar(P) and np.isscalar(y)
    P_arr = np.array(P, ndmin=1)
    y_arr = np.array(y, ndmin=1)
    if P_arr.shape != y_arr.shape:
        P_arr, y_arr = np.broadcast_arrays(P_arr, y_arr)

    rho_out = np.empty_like(P_arr, dtype=float)
    mask_en = P_arr < _EN_BOUND
    mask_pv = (P_arr >= _EN_BOUND) & (P_arr < _PV_BOUND)
    mask_ppv = P_arr >= _PV_BOUND

    if np.any(mask_en):
        rho_out[mask_en] = get_rhosol_en_Py(P_arr[mask_en], y_arr[mask_en])
    if np.any(mask_pv):
        rho_out[mask_pv] = get_rhosol_pv_Py(P_arr[mask_pv], y_arr[mask_pv])
    if np.any(mask_ppv):
        rho_out[mask_ppv] = get_rhosol_ppv_Py(P_arr[mask_ppv], y_arr[mask_ppv])

    return float(rho_out) if scalar else rho_out

def get_CPsol_Py(P, y):
    return np.full_like(P, 1260 * J_K_kg_to_erg_K_g)  # constant specific heat for solid phase

def get_alphasol_Py(P, y):
    """
    General mixture thermal expansivity alpha: selects en, pv, or ppv interpolator based on P.
    """
    scalar = np.isscalar(P) and np.isscalar(y)
    P_arr = np.array(P, ndmin=1)
    y_arr = np.array(y, ndmin=1)
    if P_arr.shape != y_arr.shape:
        P_arr, y_arr = np.broadcast_arrays(P_arr, y_arr)

    alpha_out = np.empty_like(P_arr, dtype=float)
    mask_en = P_arr < _EN_BOUND
    mask_pv = (P_arr >= _EN_BOUND) & (P_arr < _PV_BOUND)
    mask_ppv = P_arr >= _PV_BOUND

    if np.any(mask_en):
        alpha_out[mask_en] = get_alphasol_en_Py(P_arr[mask_en], y_arr[mask_en])
    if np.any(mask_pv):
        alpha_out[mask_pv] = get_alphasol_pv_Py(P_arr[mask_pv], y_arr[mask_pv])
    if np.any(mask_ppv):
        alpha_out[mask_ppv] = get_alphasol_ppv_Py(P_arr[mask_ppv], y_arr[mask_ppv])

    return float(alpha_out) if scalar else alpha_out

def get_CVsol_Py(P, y):
    """
    General mixture CV: selects en, pv, or ppv interpolator based on P.
    """
    scalar = np.isscalar(P) and np.isscalar(y)
    P_arr = np.array(P, ndmin=1)
    y_arr = np.array(y, ndmin=1)
    if P_arr.shape != y_arr.shape:
        P_arr, y_arr = np.broadcast_arrays(P_arr, y_arr)

    CV_out = np.empty_like(P_arr, dtype=float)

    CP_sol = get_CPsol_Py(P_arr, y_arr)
    alpha_sol = get_alphasol_Py(P_arr, y_arr)
    T_sol = get_Tsol_Py(P_arr, y_arr)
    gamma = get_gamma_Py(P_arr, y_arr)
    CV_out = CP_sol / (1 + gamma * alpha_sol * T_sol)

    return float(CV_out) if scalar else CV_out

# ---- helpers ----
def _broadcast_PyS(P, y, S):
    """Broadcast P (Pa), y (composition), S (erg/K/g) to a common shape."""
    P_arr = np.array(P, ndmin=1)
    y_arr = np.array(y, ndmin=1)
    S_arr = np.array(S, ndmin=1)
    P_arr, y_arr, S_arr = np.broadcast_arrays(P_arr, y_arr, S_arr)
    scalar = np.isscalar(P) and np.isscalar(y) and np.isscalar(S)
    return P_arr, y_arr, S_arr, scalar

def _entropy_phase_masks(P_arr, S_arr):
    """Masks for liquid / mix / solid using S_liq(P) and S_sol(P)."""
    # interp1d objects accept arrays
    S_liq = s_melt_liquidus(P_arr)   # erg/K/g
    S_sol = s_melt_solidus(P_arr)    # erg/K/g

    m_liq = S_arr >  S_liq
    m_mix = (S_arr <= S_liq) & (S_arr > S_sol)
    m_sol = S_arr <= S_sol
    return m_liq, m_mix, m_sol

# --- module-level helpers (reusable) ---

def broadcast_SPy(S, P, y):
    """Broadcast S (erg/K/g), P (Pa), y to common shape and report scalarness."""
    scalar = np.isscalar(S) and np.isscalar(P) and np.isscalar(y)
    S_arr = np.array(S, ndmin=1)
    P_arr = np.array(P, ndmin=1)
    y_arr = np.array(y, ndmin=1)
    S_arr, P_arr, y_arr = np.broadcast_arrays(S_arr, P_arr, y_arr)
    return S_arr, P_arr, y_arr, scalar

def sigma01(x, halfwidth, min_width):
    """
    Smooth 0→1 transition centered at x=0 using tanh with given halfwidth.
    Both `halfwidth` and `min_width` may be arrays; they will be broadcast.
    """
    w = np.maximum(halfwidth, min_width)
    return 0.5 * (1.0 + np.tanh(x / w))

def get_T_SPy_smooth(S, P, y, blend_frac=0.025, min_width=1e-6, return_weights=False):
    """
    Temperature T(S,P,y) [K] with smooth entropy-based blending across solidus & liquidus.
    """
    S_arr, P_arr, y_arr, scalar = broadcast_SPy(S, P, y)

    S_liq = s_melt_liquidus(P_arr)   # erg/(K·g)
    S_sol = s_melt_solidus(P_arr)    # erg/(K·g)
    dS    = np.maximum(S_liq - S_sol, 0.0)
    wS    = np.maximum(blend_frac * dS, min_width)

    # Raw phase fields
    T_liq = get_Tliq_Py(P_arr, y_arr)
    T_mix = get_Tmix_Py(P_arr, y_arr)   # routes en/pv/ppv by P
    T_sol = get_Tsol_Py(P_arr, y_arr)   # routes en/pv/ppv by P

    # Blends in entropy space
    w_sm  = sigma01(S_arr - S_sol, wS, min_width)  # solid ↔ mix
    T_sm  = (1.0 - w_sm) * T_sol + w_sm * T_mix

    w_liq = sigma01(S_arr - S_liq, wS, min_width)  # (solid/mix) ↔ liquid
    T_out = (1.0 - w_liq) * T_sm + w_liq * T_liq

    if scalar:
        T_out = float(T_out)
        if return_weights:
            return T_out, float(w_sm), float(w_liq)
        return T_out
    if return_weights:
        return T_out, w_sm, w_liq
    return T_out

def get_rho_SPy_smooth(S, P, y, blend_frac=0.025, min_width=1e-6, return_weights=False):
    """
    Density rho(S,P,y) [kg/m³] with smooth entropy-based blending across solidus & liquidus.
    """
    S_arr, P_arr, y_arr, scalar = broadcast_SPy(S, P, y)

    S_liq = s_melt_liquidus(P_arr)   # erg/(K·g)
    S_sol = s_melt_solidus(P_arr)    # erg/(K·g)
    dS    = np.maximum(S_liq - S_sol, 0.0)
    wS    = np.maximum(blend_frac * dS, min_width)

    # Raw phase fields
    rho_liq = get_rholiq_Py(P_arr, y_arr)
    rho_mix = get_rhomix_Py(P_arr, y_arr)   # routes en/pv/ppv by P
    rho_sol = get_rhosol_Py(P_arr, y_arr)   # routes en/pv/ppv by P

    # Blends in entropy space
    w_sm  = sigma01(S_arr - S_sol, wS, min_width)  # solid ↔ mix
    rho_sm  = (1.0 - w_sm) * rho_sol + w_sm * rho_mix

    w_liq = sigma01(S_arr - S_liq, wS, min_width)  # (solid/mix) ↔ liquid
    rho_out = (1.0 - w_liq) * rho_sm + w_liq * rho_liq

    if scalar:
        rho_out = float(rho_out)
        if return_weights:
            return rho_out, float(w_sm), float(w_liq)
        return rho_out
    if return_weights:
        return rho_out, w_sm, w_liq
    return rho_out

def get_alpha_SPy_smooth(S, P, y, blend_frac=0.025, min_width=1e-6, return_weights=False):
    """
    Thermal expansivity α(S,P,y) [1/K] with smooth entropy-based blending across solidus & liquidus.
    """
    S_arr, P_arr, y_arr, scalar = broadcast_SPy(S, P, y)

    S_liq = s_melt_liquidus(P_arr)   # erg/(K·g)
    S_sol = s_melt_solidus(P_arr)    # erg/(K·g)
    dS    = np.maximum(S_liq - S_sol, 0.0)
    wS    = np.maximum(blend_frac * dS, min_width)

    # Raw phase fields
    alpha_liq = get_alphaliq_Py(P_arr, y_arr)
    alpha_mix = get_alphamix_Py(P_arr, y_arr)   # routes en/pv/ppv by P
    alpha_sol = get_alphasol_Py(P_arr, y_arr)   # routes en/pv/ppv by P

    # Blends in entropy space
    w_sm  = sigma01(S_arr - S_sol, wS, min_width)  # solid ↔ mix
    alpha_sm  = (1.0 - w_sm) * alpha_sol + w_sm * alpha_mix

    w_liq = sigma01(S_arr - S_liq, wS, min_width)  # (solid/mix) ↔ liquid
    alpha_out = (1.0 - w_liq) * alpha_sm + w_liq * alpha_liq

    if scalar:
        alpha_out = float(alpha_out)
        if return_weights:
            return alpha_out, float(w_sm), float(w_liq)
        return alpha_out
    if return_weights:
        return alpha_out, w_sm, w_liq
    return alpha_out

def get_CP_SPy_smooth(S, P, y, blend_frac=0.025, min_width=1e-6, return_weights=False):
    """
    Specific heat at constant pressure Cp(S,P,y) [erg/g/K] with smooth entropy-based blending
    across solidus & liquidus.
    """
    S_arr, P_arr, y_arr, scalar = broadcast_SPy(S, P, y)

    S_liq = s_melt_liquidus(P_arr)   # erg/(K·g)
    S_sol = s_melt_solidus(P_arr)    # erg/(K·g)
    dS    = np.maximum(S_liq - S_sol, 0.0)
    wS    = np.maximum(blend_frac * dS, min_width)

    # Raw phase fields
    CP_liq = get_CPliq_Py(P_arr, y_arr)
    CP_mix = get_CPmix_Py(P_arr, y_arr)   # routes en/pv/ppv by P
    CP_sol = get_CPsol_Py(P_arr, y_arr)   # routes en/pv/ppv by P

    # Blends in entropy space
    w_sm  = sigma01(S_arr - S_sol, wS, min_width)  # solid ↔ mix
    CP_sm  = (1.0 - w_sm) * CP_sol + w_sm * CP_mix

    w_liq = sigma01(S_arr - S_liq, wS, min_width)  # (solid/mix) ↔ liquid
    CP_out = (1.0 - w_liq) * CP_sm + w_liq * CP_liq

    if scalar:
        CP_out = float(CP_out)
        if return_weights:
            return CP_out, float(w_sm), float(w_liq)
        return CP_out
    if return_weights:
        return CP_out, w_sm, w_liq
    return CP_out

def get_CV_SPy_smooth(S, P, y, blend_frac=0.025, min_width=1e-6, return_weights=False):
    """
    Specific heat at constant volume Cv(S,P,y) [erg/g/K] with smooth entropy-based blending
    across solidus & liquidus.
    """
    S_arr, P_arr, y_arr, scalar = broadcast_SPy(S, P, y)

    S_liq = s_melt_liquidus(P_arr)   # erg/(K·g)
    S_sol = s_melt_solidus(P_arr)    # erg/(K·g)
    dS    = np.maximum(S_liq - S_sol, 0.0)
    wS    = np.maximum(blend_frac * dS, min_width)

    # Raw phase fields
    CV_liq = get_CVliq_Py(P_arr, y_arr)
    CV_mix = get_CVmix_Py(P_arr, y_arr)   # routes en/pv/ppv by P
    CV_sol = get_CVsol_Py(P_arr, y_arr)   # routes en/pv/ppv by P

    # Blends in entropy space
    w_sm  = sigma01(S_arr - S_sol, wS, min_width)  # solid ↔ mix
    CV_sm  = (1.0 - w_sm) * CV_sol + w_sm * CV_mix

    w_liq = sigma01(S_arr - S_liq, wS, min_width)  # (solid/mix) ↔ liquid
    CV_out = (1.0 - w_liq) * CV_sm + w_liq * CV_liq

    if scalar:
        CV_out = float(CV_out)
        if return_weights:
            return CV_out, float(w_sm), float(w_liq)
        return CV_out
    if return_weights:
        return CV_out, w_sm, w_liq
    return CV_out


# ---- temperature, density, alpha selected by entropy ----
def get_T_SPy(S, P, y):
    """
    Temperature T(P,S,y) [K], choosing phase by entropy:
      S > S_liq(P)          -> liquid
      S_sol(P) < S ≤ S_liq  -> mixture
      S ≤ S_sol(P)          -> solid
    """
    P_arr, y_arr, S_arr, scalar = _broadcast_PyS(P, y, S)
    T = np.full(P_arr.shape, np.nan, dtype=float)

    m_liq, m_mix, m_sol = _entropy_phase_masks(P_arr, S_arr)
    if np.any(m_liq): T[m_liq] = get_Tliq_Py(P_arr[m_liq], y_arr[m_liq])
    if np.any(m_mix): T[m_mix] = get_Tmix_Py(P_arr[m_mix], y_arr[m_mix])   # picks en/pv/ppv by P
    if np.any(m_sol): T[m_sol] = get_Tsol_Py(P_arr[m_sol], y_arr[m_sol])   # picks en/pv/ppv by P

    return float(T) if scalar else T

def get_rho_SPy(S, P, y):
    """
    Density rho(P,S,y) [kg/m^3], phase chosen by entropy as above.
    """
    P_arr, y_arr, S_arr, scalar = _broadcast_PyS(P, y, S)
    rho = np.full(P_arr.shape, np.nan, dtype=float)

    m_liq, m_mix, m_sol = _entropy_phase_masks(P_arr, S_arr)
    if np.any(m_liq): rho[m_liq] = get_rholiq_Py(P_arr[m_liq], y_arr[m_liq])
    if np.any(m_mix): rho[m_mix] = get_rhomix_Py(P_arr[m_mix], y_arr[m_mix])
    if np.any(m_sol): rho[m_sol] = get_rhosol_Py(P_arr[m_sol], y_arr[m_sol])

    return float(rho) if scalar else rho

def get_alpha_SPy(S, P, y):
    """
    Thermal expansivity α(P,S,y) [1/K], phase chosen by entropy as above.
    """
    P_arr, y_arr, S_arr, scalar = _broadcast_PyS(P, y, S)
    alpha = np.full(P_arr.shape, np.nan, dtype=float)

    m_liq, m_mix, m_sol = _entropy_phase_masks(P_arr, S_arr)
    if np.any(m_liq): alpha[m_liq] = get_alphaliq_Py(P_arr[m_liq], y_arr[m_liq])
    if np.any(m_mix): alpha[m_mix] = get_alphamix_Py(P_arr[m_mix], y_arr[m_mix])
    if np.any(m_sol): alpha[m_sol] = get_alphasol_Py(P_arr[m_sol], y_arr[m_sol])

    return float(alpha) if scalar else alpha

def get_CP_SPy(S, P, y):
    """
    Specific heat at constant pressure Cp(P,S,y) [erg/g/K], phase chosen by entropy as above.
    """
    P_arr, y_arr, S_arr, scalar = _broadcast_PyS(P, y, S)
    CP = np.full(P_arr.shape, np.nan, dtype=float)

    m_liq, m_mix, m_sol = _entropy_phase_masks(P_arr, S_arr)
    if np.any(m_liq): CP[m_liq] = get_CPliq_Py(P_arr[m_liq], y_arr[m_liq])
    if np.any(m_mix): CP[m_mix] = get_CPmix_Py(P_arr[m_mix], y_arr[m_mix])
    if np.any(m_sol): CP[m_sol] = get_CPsol_Py(P_arr[m_sol], y_arr[m_sol])

    return float(CP) if scalar else CP

def get_CV_SPy(S, P, y):
    """
    Specific heat at constant volume Cv(P,S,y) [erg/g/K], phase chosen by entropy as above.
    """
    P_arr, y_arr, S_arr, scalar = _broadcast_PyS(P, y, S)
    CV = np.full(P_arr.shape, np.nan, dtype=float)

    m_liq, m_mix, m_sol = _entropy_phase_masks(P_arr, S_arr)
    if np.any(m_liq): CV[m_liq] = get_CVliq_Py(P_arr[m_liq], y_arr[m_liq])
    if np.any(m_mix): CV[m_mix] = get_CVmix_Py(P_arr[m_mix], y_arr[m_mix])
    if np.any(m_sol): CV[m_sol] = get_CVsol_Py(P_arr[m_sol], y_arr[m_sol])

    return float(CV) if scalar else CV

def kappa_eddy(S, P, T, y, g, l, dsdr):

    """
    Eddy diffusivities for fully inviscid, viscous, and mixture of solid and liquid rock
    as described by Zhang et al. (2022), Equations 34a-35b.
    """

    z = 0.5 * (1 + np.tanh(y))
    alpha = get_alpha_SPy(S, P, y)  # 1/K
    CP = get_CP_SPy(S, P, y)  # erg/g/K
    rho = get_rho_SPy(S, P, y) * 1e-3  # g/cm^3
    eta_solid = get_eta_solid(P, T, out='poise')
    eta_liquid = 1000
    xcrit = 0.4
    x_melt = get_melt_fraction_x(S, P)  # melt fraction
    y_f = (x_melt - xcrit) / 0.15
    z_f = 0.5 * (1 + np.tanh(y_f))  # fraction of liquid in mixture

    eta_mix = 10 ** ((1 - z_f) * np.log10(eta_solid) + z_f * np.log10(eta_liquid))

    nu_solid = eta_solid / rho  # cm^2/s
    nu_mix = eta_mix / rho

    T_si_m = silicate_meltcurve(P / 1e9)  # convert P to GPa

    kappa_34a = (alpha * T * g * l ** 4 / CP / 32) ** 0.5
    kappa_34b = alpha * T * g * l ** 4 / CP / 32 / nu_solid
    kappa_35a = (alpha * T_si_m * g * l ** 4 / 32 / latent_heat_si) ** 0.5
    kappa_35b = alpha * T_si_m * g * l ** 4 / 32 / latent_heat_si / nu_mix

    k_h_mix_inviscid = z * kappa_34a + (1 - z) * kappa_35a
    k_h_mix_viscous = (1 - z) * kappa_34b + z * kappa_35b
    k_h_full_invscid = kappa_34a.copy()
    k_h_full_viscous = kappa_34b.copy()

    kappa = np.full(P.shape, np.nan, dtype=float)

    Re_34 = alpha * g * T * l ** 4 / CP / 18 / nu_mix ** 2
    Re_35 = alpha * g * T_si_m * l ** 4 / 18 / latent_heat_si / nu_mix ** 2

    m_liq, m_mix, m_sol = _entropy_phase_masks(P, S)

    if np.any(m_liq): kappa[m_liq] = k_h_full_invscid[m_liq]  # fully inviscid liquid
    if np.any(m_mix): kappa[m_mix] = k_h_mix_viscous[m_mix]  # viscous mixture of solid and liquid
    if np.any(m_sol): kappa[m_sol] = k_h_full_viscous[m_sol]  # fully viscous solid

    #pdb.set_trace()

    return kappa #(m_liq, m_mix, m_sol)#, eta_mix, x_melt, z_f, y_f, (m_liq, m_mix, m_sol)


def silicate_meltcurve(P_GPa):
    """
    Returns the silicate melting curve in GPa and K. In CMAPPER.
    """
    P_GPa = np.array(P_GPa)
    T_si0 = 1831 # K

    return T_si0 * (P_GPa / 4.6 + 1) ** 0.33 # Belonoshko et al. 2005

# Latent heat of silicate melting in erg/g (Hess 1990)
latent_heat_si = 7.2322e5 * (u.J/u.kg).to('erg/g')

def iron_meltcurve(P_GPa):
    """
    Returns the iron melting curve in GPa and K.
    """
    P_GPa = np.array(P_GPa)
    T_fe0 = 1900 # K

    return T_fe0 * (P_GPa / 31.3 + 1) ** (1/1.99) # Zhang et al. 2015

# Latent heat of iron melting in erg/g (Anderson & Duba 1997)
latent_heat_fe = 1.2e6 * (u.J/u.kg).to('erg/g')

def get_melt_fraction_x(sval, P):
    """
    Returns the melt fraction given the saturation and solidus entropy values.

    Parameters
    ----------
    sval : float or ndarray
        Entropy value at which to calculate the melt fraction.
    s_m : float
        S_liquidus(P_GPa).
    s_s : float
        S_solidus(P_GPa).

    Returns
    -------
    f_melt : float or ndarray
        Melt fraction.
    """
    #P_GPa = np.array(P) / 1e9  # Convert pressure to GPa
    sval = np.array(sval, ndmin=1)  # Ensure sval is an array
    s_l, s_s = s_melt_liquidus(P), s_melt_solidus(P)

    x = (sval - s_s) / (s_l - s_s)

    x[sval <= s_s] = 0.0  # No melt below solidus
    x[sval >= s_l] = 1.0  # Full melt above liquidus

    return x

############## Y(P, T) INVERSIONS ##############

def make_y_T_liq_interpolator(pgrid, y_grid, T_liq):
    """
    Return a function y = y_T_liq(P, T) that inverts T = T_liq(P,y)
    by:
      1) for each P, finding the two nearest pgrid rows
      2) inverting each row’s T_liq -> y via np.interp
      3) linearly interpolating in P

    pgrid : 1D array of pressures (same units as P/10 below)
    y_grid: 1D array of y-values (entropy grid)
    T_liq : 2D array with shape (len(pgrid), len(y_grid))
    """
    def y_T_liq(P, T):
        P = P * (u.dyn/(u.cm**2)).to('Pa') # in Pascals
        # match original: convert P→P_Pa
        P_arr = np.atleast_1d(P) / 10.0
        T_arr = np.atleast_1d(T)
        # broadcast shapes
        if P_arr.shape != T_arr.shape:
            P_arr, T_arr = np.broadcast_arrays(P_arr, T_arr)
        y_out = np.empty_like(P_arr, dtype=float)

        # find for each P the bracketing indices in pgrid
        idx = np.searchsorted(pgrid, P_arr)
        idx_lo = np.clip(idx - 1, 0, len(pgrid) - 2)
        idx_hi = idx_lo + 1

        P_lo = pgrid[idx_lo]
        P_hi = pgrid[idx_hi]

        # loop over each point (vectorized over P_arr size)
        for i, (Pl, Ph, Ti, lo, hi) in enumerate(zip(P_lo, P_hi, T_arr, idx_lo, idx_hi)):
            T_row_lo = T_liq[lo]
            T_row_hi = T_liq[hi]

            # invert T→y along each fixed-P slice
            y_lo = np.interp(Ti, T_row_lo, y_grid)
            y_hi = np.interp(Ti, T_row_hi, y_grid)

            # now interpolate in P
            if Ph != Pl:
                frac = (P_arr[i] - Pl) / (Ph - Pl)
                y_out[i] = y_lo + (y_hi - y_lo) * frac
            else:
                y_out[i] = y_lo

        # return scalar if inputs were scalars
        return y_out.item() if y_out.ndim == 0 else y_out

    return y_T_liq

# this will be used to update the entropy
#get_yliq_Py = make_y_T_liq_interpolator(p_grid, y_grid, T_liq)

# assume y_grid is already in your namespace from before
# _y_min, _y_max = float(y_grid.min()), float(y_grid.max())

_y_min, _y_max = 0.0, 10.0

def get_yliq_PT(P, Tliq, y_min=None, y_max=None, tol=1e-8, maxiter=150):
    """
    Invert get_Tliq_Py(P, y) to find the composition y that yields liquidus
    temperature Tliq at pressure P.

    Parameters
    ----------
    P : float or array_like
        Pressure(s) in Pa.
    Tmix : float or array_like
        Desired mixture temperature(s) in K.
    y_min, y_max : float, optional
        Bracketing bounds for y. Defaults to (y_grid.min(), y_grid.max()).
    tol : float, optional
        Tolerance for root finding (xtol in brentq).
    maxiter : int, optional
        Maximum iterations passed to brentq.

    Returns
    -------
    y_out : float or ndarray
        Composition(s) y such that get_Tmix_Py(P, y) ≈ Tmix.
        Returns NaN where no root is bracketed or on failure.
    """
    # set up arrays
    P_arr = np.array(P, ndmin=1)
    T_arr = np.array(Tliq, ndmin=1)
    if P_arr.shape != T_arr.shape:
        P_arr, T_arr = np.broadcast_arrays(P_arr, T_arr)

    y_low  = y_min if (y_min is not None) else _y_min
    y_high = y_max if (y_max is not None) else _y_max

    y_out = np.empty_like(P_arr, dtype=float)

    def f(y, P0, T0):
        return get_Tliq_Py(P0, y) - T0

    # loop over each element
    for idx in np.ndindex(P_arr.shape):
        P0 = P_arr[idx]
        T0 = T_arr[idx]
        try:
            f_lo = f(y_low,  P0, T0)
            f_hi = f(y_high, P0, T0)
        except Exception:
            y_out[idx] = np.nan
            continue

        # exact bound hits
        if np.isclose(f_lo, 0.0, atol=tol):
            y_out[idx] = y_low
        elif np.isclose(f_hi, 0.0, atol=tol):
            y_out[idx] = y_high
        # check bracket
        elif f_lo * f_hi > 0 or np.isnan(f_lo) or np.isnan(f_hi):
            y_out[idx] = np.nan
        else:
            # find root
            try:
                y_out[idx] = brentq(f, y_low, y_high,
                                    args=(P0, T0),
                                    xtol=tol,
                                    maxiter=maxiter)
            except ValueError:
                y_out[idx] = np.nan

    # return scalar if appropriate
    return float(y_out) if y_out.size == 1 else y_out

def get_ymix_PT(P, Tmix, y_min=None, y_max=None, tol=1e-8, maxiter=150):
    """
    Invert get_Tmix_Py(P, y) to find the composition y that yields mixture
    temperature Tmix at pressure P.

    Parameters
    ----------
    P : float or array_like
        Pressure(s) in Pa.
    Tmix : float or array_like
        Desired mixture temperature(s) in K.
    y_min, y_max : float, optional
        Bracketing bounds for y. Defaults to (y_grid.min(), y_grid.max()).
    tol : float, optional
        Tolerance for root finding (xtol in brentq).
    maxiter : int, optional
        Maximum iterations passed to brentq.

    Returns
    -------
    y_out : float or ndarray
        Composition(s) y such that get_Tmix_Py(P, y) ≈ Tmix.
        Returns NaN where no root is bracketed or on failure.
    """
    # set up arrays
    P_arr = np.array(P, ndmin=1)
    T_arr = np.array(Tmix, ndmin=1)
    if P_arr.shape != T_arr.shape:
        P_arr, T_arr = np.broadcast_arrays(P_arr, T_arr)

    y_low  = y_min if (y_min is not None) else _y_min
    y_high = y_max if (y_max is not None) else _y_max

    y_out = np.empty_like(P_arr, dtype=float)

    def f(y, P0, T0):
        return get_Tmix_Py(P0, y) - T0

    # loop over each element
    for idx in np.ndindex(P_arr.shape):
        P0 = P_arr[idx]
        T0 = T_arr[idx]
        try:
            f_lo = f(y_low,  P0, T0)
            f_hi = f(y_high, P0, T0)
        except Exception:
            y_out[idx] = np.nan
            continue

        # exact bound hits
        if np.isclose(f_lo, 0.0, atol=tol):
            y_out[idx] = y_low
        elif np.isclose(f_hi, 0.0, atol=tol):
            y_out[idx] = y_high
        # check bracket
        elif f_lo * f_hi > 0 or np.isnan(f_lo) or np.isnan(f_hi):
            y_out[idx] = np.nan
        else:
            # find root
            try:
                y_out[idx] = brentq(f, y_low, y_high,
                                    args=(P0, T0),
                                    xtol=tol,
                                    maxiter=maxiter)
            except ValueError:
                y_out[idx] = np.nan

    # return scalar if appropriate
    return float(y_out) if y_out.size == 1 else y_out

def get_ysol_PT(P, Tsol, y_min=None, y_max=None, tol=1e-8, maxiter=150):
    """
    Invert get_Tsol_Py(P, y) to find the composition y that yields solidus
    temperature Tsol at pressure P.

    Parameters
    ----------
    P : float or array_like
        Pressure(s) in Pa.
    Tmix : float or array_like
        Desired mixture temperature(s) in K.
    y_min, y_max : float, optional
        Bracketing bounds for y. Defaults to (y_grid.min(), y_grid.max()).
    tol : float, optional
        Tolerance for root finding (xtol in brentq).
    maxiter : int, optional
        Maximum iterations passed to brentq.

    Returns
    -------
    y_out : float or ndarray
        Composition(s) y such that get_Tmix_Py(P, y) ≈ Tmix.
        Returns NaN where no root is bracketed or on failure.
    """
    # set up arrays
    P_arr = np.array(P, ndmin=1)
    T_arr = np.array(Tsol, ndmin=1)
    if P_arr.shape != T_arr.shape:
        P_arr, T_arr = np.broadcast_arrays(P_arr, T_arr)

    y_low  = y_min if (y_min is not None) else _y_min
    y_high = y_max if (y_max is not None) else _y_max

    y_out = np.empty_like(P_arr, dtype=float)

    def f(y, P0, T0):
        return get_Tsol_Py(P0, y) - T0

    # loop over each element
    for idx in np.ndindex(P_arr.shape):
        P0 = P_arr[idx]
        T0 = T_arr[idx]
        try:
            f_lo = f(y_low,  P0, T0)
            f_hi = f(y_high, P0, T0)
        except Exception:
            y_out[idx] = np.nan
            continue

        # exact bound hits
        if np.isclose(f_lo, 0.0, atol=tol):
            y_out[idx] = y_low
        elif np.isclose(f_hi, 0.0, atol=tol):
            y_out[idx] = y_high
        # check bracket
        elif f_lo * f_hi > 0 or np.isnan(f_lo) or np.isnan(f_hi):
            y_out[idx] = np.nan
        else:
            # find root
            try:
                y_out[idx] = brentq(f, y_low, y_high,
                                    args=(P0, T0),
                                    xtol=tol,
                                    maxiter=maxiter)
            except ValueError:
                y_out[idx] = np.nan

    # return scalar if appropriate
    return float(y_out) if y_out.size == 1 else y_out

def get_y_SPT(S, P, T):
    """
    Invert get_S_Py(P, T) to find the composition y that yields entropy S at pressure P and temperature T.
    """
    P_arr, T_arr, S_arr, scalar = _broadcast_PyS(P, T, S)
    y_res = np.full(P_arr.shape, np.nan, dtype=float)

    m_liq, m_mix, m_sol = _entropy_phase_masks(P_arr, S_arr)
    if np.any(m_liq): y_res[m_liq] = get_yliq_PT(P_arr[m_liq], T_arr[m_liq])
    if np.any(m_mix): y_res[m_mix] = get_ymix_PT(P_arr[m_mix], T_arr[m_mix])   # picks en/pv/ppv by P
    if np.any(m_sol): y_res[m_sol] = get_ysol_PT(P_arr[m_sol], T_arr[m_sol])   # picks en/pv/ppv by P

    return float(y_res) if scalar else y_res
    #pdb.set_trace()
    #return y_res, (m_liq, m_mix, m_sol)  # return masks for debugging

def get_y_SPT_scalar(S, P, T):
    y_liq = get_yliq_PT(P, T)
    y_mix = get_ymix_PT(P, T)
    y_sol = get_ysol_PT(P, T)

    # check each one and return the one that is not nan
    if not np.isnan(y_liq):
        return y_liq, 'liquid'
    elif not np.isnan(y_mix):
        return y_mix, 'mixture'
    elif not np.isnan(y_sol):
        return y_sol, 'solid'
    else:
        return np.nan, 'none'

S_MAX = 5100.0 * J_K_kg_to_erg_K_g # S_max value in CMAPPER (erg K g⁻¹)

def get_y_SP(sval, P_Pa, width=0.02):

    P = P_Pa.copy()  # already in Pa
    # this will be used to update the y value
    s_m, s_s = s_melt_liquidus(P), s_melt_solidus(P)

    #sw = 0.02 * (s_l - s_s) # values used in CMAPPER?
    ym = (sval - s_m) / (S_MAX - s_m)
    ys = (sval - s_s) / (S_MAX - s_m)

    # ym = np.clip(ym, 0.0, 1.0)
    # ys = np.clip(ys, 0.0, 1.0)

    return ym, ys

def get_s_Py(P_Pa, y):
    """
    Returns the saturation entropy for a given pressure and y value.

    Parameters
    ----------
    P_Pa : float or ndarray
        Pressure in Pa.
    y : float or ndarray
        y value (entropy measure).

    Returns
    -------
    sval : float or ndarray
        Saturation entropy corresponding to the given pressure and y value.
    """
    s_m, s_s = s_melt_liquidus(P_Pa), s_melt_solidus(P_Pa) # already in erg/g/K

    sval_s = s_s + (S_MAX - s_s) * y

    sval_m = s_m + (S_MAX - s_m) * y

    # sval_s = s_s + 0.02 * (s_m - s_s) * y
    # sval_m = s_m + 0.02 * (s_m - s_s) * y

    return sval_s, sval_m


def get_y_SP_debug(sval, P_Pa):

    sval_s = sval / erg_to_kbbar

    ym = (sval_s - s_melt_liquidus(P_Pa)) / (S_MAX - s_melt_liquidus(P_Pa))
    ys = (sval_s - s_melt_solidus(P_Pa)) / (S_MAX - s_melt_solidus(P_Pa))

    return ys, ym


def get_t_sp(sval, P_Pa):
    """
    Returns temperature as a function of entropy (in kb/baryon) and pressure (in pascals)
    """

    ysolid, ymelt = get_y_SP_debug(sval, P_Pa)

    T_solid = get_Tsol_Py(P_Pa, ysolid)
    T_melt  = get_Tliq_Py(P_Pa, ymelt)

    return (T_solid, T_melt), (ysolid, ymelt)

def get_rho_sp(sval, P_Pa):
    """
    Returns density as a function of entropy (in kb/baryon and pressure (in pascals))
    """

    ysolid, ymelt = get_y_SP_debug(sval, P_Pa)

    rho_solid = get_rhosol_Py(P_Pa, ysolid)
    rho_melt  = get_rholiq_Py(P_Pa, ymelt)

    return (rho_solid, rho_melt), (ysolid, ymelt)

def get_cp_sp(sval, P_Pa):
    ysolid, ymelt = get_y_SP_debug(sval, P_Pa)

    CP_solid = get_CPsol_Py(P_Pa, ysolid)
    CP_melt = get_CPliq_Py(P_Pa, ymelt)

    return (CP_solid, CP_melt), (ysolid, ymelt)

def get_cv_sp(sval, P_Pa):

    ysolid, ymelt = get_y_SP_debug(sval, P_Pa)

    CV_solid = get_CVsol_Py(P_Pa, ysolid)
    CV_melt = get_CVliq_Py(P_Pa, ymelt)

    return (CV_solid, CV_melt), (ysolid, ymelt)

def get_alpha_sp(sval, P_Pa):

    ysolid, ymelt = get_y_SP_debug(sval, P_Pa)

    alpha_solid = get_alphasol_Py(P_Pa, ysolid)
    alpha_liq = get_alphaliq_Py(P_Pa, ymelt)

    return (alpha_solid, alpha_liq), (ysolid, ymelt)



#def get_z(sval, P_Pa, width=0.02):
def get_z(sval, P_Pa, T, width=0.02):
    """
    Returns the z value for the saturation entropy.

    Parameters
    ----------
    sval : float or ndarray
        Entropy value at which to calculate the z value.
    P_GPa : float or ndarray
        Pressure in GPa.
    width : float
        Width of the saturation region (not used here).

    Returns
    -------
    z : float or ndarray
        z value corresponding to sval.
    """
    # ym, ys = get_y_SP(sval, P_Pa, width)

    y = get_y_SPT(sval, P_Pa, T)

    # zm, zs = 0.5 * (1 + np.tanh(ym)), 0.5 * (1 + np.tanh(ys))

    # return zm, zs, ym, ys

    z = 0.5 * (1 + np.tanh(y))

    return z, y

####### OVERALL SILICATE VISCOSITY ######

_RG = 8.314462618      # J mol‑1 K‑1
# ---------- constant low‑P parameters (Ranalli 2001) ----------
_PARAMS = dict(
    olivine=dict(B=3.5e-15, n=3.0, E=4.3e5, V=1.0e-5),     # P < 23 GPa
    perovskite=dict(B=7.4e-17, n=3.5, E=5.0e5, V=1.0e-5),  # 23 ≤ P < 125 GPa
)

def _dislocation_creep_visc(P, T, strain_rate, *, B, n, E, V):
    """
    Dynamic viscosity (Pa s) for power‑law / dislocation creep.
    """
    pref = 0.5 * B ** (-1.0 / n)
    expo = np.exp((E + P * V) / (n * _RG * T))
    rate = strain_rate ** ((1.0 - n) / n)
    return pref * expo * rate


def silicate_viscosity(
    S_cgs,
    P,
    T,
    *,
    strain_rate=1.0e-15,
    ppv_eta_model=1, # 1 or 2 – see cython source
    eta_m=100.0,     # melt viscosity [Pa s]
    out="poise",      # 'Pa s', 'poise', or 'm2/s'|'kinematic'
):
    """
    Silicate viscosity following CMAPPER convention.

    Parameters
    ----------
    P_cgs, T        : pressure [cgs] and temperature [K] (scalar or ndarray)
    strain_rate : 2nd invariant of strain rate tensor [s⁻¹]
    x, width    : melt fraction & tanh‑blending width (see cython code)
    ppv_eta_model : 1 or 2 → choose post‑perovskite fit
    eta_m       : dynamic viscosity of the liquid silicate [Pa s]
    out         : 'Pa s' (default), 'poise', or 'm2/s'/'kinematic'

    Returns
    -------
    viscosity   : dynamic or kinematic viscosity, same shape as P, T
    """

    P = np.asarray(P, dtype=float)
    T = np.asarray(T, dtype=float)
    strain_rate = np.asarray(strain_rate, dtype=float)

    # ------------------------------------------------------------------ #
    # 1.  dislocation‑creep viscosity for olivine / perovskite
    # ------------------------------------------------------------------ #
    eta_dyn = np.empty_like(P)

    # masks
    mask_ol  =  P <  23.0e9              # < 23 GPa
    mask_pv  = (P >= 23.0e9) & (P < 125e9)
    mask_ppv =  P >= 125e9

    # low‑P olivine
    if mask_ol.any():
        prm = _PARAMS["olivine"]
        eta_dyn[mask_ol] = _dislocation_creep_visc(
            P[mask_ol], T[mask_ol], strain_rate,
            **prm
        )

    # mid‑P perovskite
    if mask_pv.any():
        prm = _PARAMS["perovskite"]
        eta_dyn[mask_pv] = _dislocation_creep_visc(
            P[mask_pv], T[mask_pv], strain_rate,
            **prm
        )

    # ------------------------------------------------------------------ #
    # 2.  post‑perovskite high‑P viscosity
    # ------------------------------------------------------------------ #
    if mask_ppv.any():
        if ppv_eta_model == 1:
            eta0     = 1.05e34    # Pa s
            E        = 7.8e5      # J mol⁻¹
            p_decay  = 1100e9     # Pa
            V        = 1.7e-6 * np.exp(-P[mask_ppv] / p_decay)
        elif ppv_eta_model == 2:
            eta0     = 1.9e21
            E        = 1.62e5
            p_decay  = 1610e9
            V        = 1.4e-6 * np.exp(-P[mask_ppv] / p_decay)
        else:
            raise ValueError("`ppv_eta_model` must be 1 or 2")

        eta_dyn[mask_ppv] = eta0 * np.exp((E + P[mask_ppv] * V) / (_RG * T[mask_ppv])
                                          - E / (_RG * 1600.0))

    # ------------------------------------------------------------------ #
    # 3.  melt–solid blending (log‑linear in log10 space)
    # ------------------------------------------------------------------ #
    # x = get_melt_fraction_x(sval, P)
    # y = (x - 0.4) / width
    # y_m, y_s = get_y_SP(sval, P * 1e-9, width=width)  # convert P to GPa
    # z = 0.5 * (1.0 + np.tanh(y_s))          # 0 → solid, 1 → melt
    # zm, zs, ym, ys = get_z(sval, P, width=width)  # convert P to GPa
    zs, ys = get_z(S_cgs, P, T)  # z ∈ [0,1], y ∈ [0,1] (melt fraction)

    # ensure broadcast shapes
    eta_dyn, z = np.broadcast_arrays(eta_dyn, zs)

    eta_liq_dyn = eta_m                   # Pa s
    eta_mix_dyn = 10.0 ** ((1.0 - z) * np.log10(eta_liq_dyn) +
                           z * np.log10(eta_dyn))

    # ------------------------------------------------------------------ #
    # 4.  unit conversion / kinematic option
    # ------------------------------------------------------------------ #
    if out.lower() in {"pa s", "pas", "pa-s"}:
        result = eta_mix_dyn
    elif out.lower() in {"poise"}:
        result = 10.0 * eta_mix_dyn       # 1 Pa s = 10 poise
    elif out.lower() in {"m2/s", "kinematic", "nu"}:
        result = eta_mix_dyn
    else:
        raise ValueError("`out` must be 'Pa s', 'poise', or 'm2/s'/'kinematic'")

    return result, z, ys

def get_eta_solid(P, T, strain_rate=1.0e-15, ppv_eta_model=1, out="cgs"):

    P = np.asarray(P, dtype=float)
    T = np.asarray(T, dtype=float)
    strain_rate = np.asarray(strain_rate, dtype=float)

    # ------------------------------------------------------------------ #
    # 1.  dislocation‑creep viscosity for olivine / perovskite
    # ------------------------------------------------------------------ #
    eta_dyn = np.empty_like(P)

    # masks
    mask_ol  =  P <  23.0e9              # < 23 GPa
    mask_pv  = (P >= 23.0e9) & (P < 125e9)
    mask_ppv =  P >= 125e9

    # low‑P olivine
    if mask_ol.any():
        prm = _PARAMS["olivine"]
        eta_dyn[mask_ol] = _dislocation_creep_visc(
            P[mask_ol], T[mask_ol], strain_rate,
            **prm
        )

    # mid‑P perovskite
    if mask_pv.any():
        prm = _PARAMS["perovskite"]
        eta_dyn[mask_pv] = _dislocation_creep_visc(
            P[mask_pv], T[mask_pv], strain_rate,
            **prm
        )

    # ------------------------------------------------------------------ #
    # 2.  post‑perovskite high‑P viscosity
    # ------------------------------------------------------------------ #
    if mask_ppv.any():
        if ppv_eta_model == 1:
            eta0     = 1.05e34    # Pa s
            E        = 7.8e5      # J mol⁻¹
            p_decay  = 1100e9     # Pa
            V        = 1.7e-6 * np.exp(-P[mask_ppv] / p_decay)
        elif ppv_eta_model == 2:
            eta0     = 1.9e21
            E        = 1.62e5
            p_decay  = 1610e9
            V        = 1.4e-6 * np.exp(-P[mask_ppv] / p_decay)
        else:
            raise ValueError("`ppv_eta_model` must be 1 or 2")

        eta_dyn[mask_ppv] = eta0 * np.exp((E + P[mask_ppv] * V) / (_RG * T[mask_ppv])
                                          - E / (_RG * 1600.0))

    # ------------------------------------------------------------------ #
    # 4.  unit conversion / kinematic option
    # ------------------------------------------------------------------ #
    if out.lower() in {"pa s", "pas", "pa-s"}:
        result = eta_dyn
    elif out.lower() in {"poise"}:
        result = 10.0 * eta_dyn       # 1 Pa s = 10 poise
    elif out.lower() in {"m2/s", "kinematic", "nu"}:
        result = eta_dyn
    else:
        raise ValueError("`out` must be 'Pa s', 'poise', or 'm2/s'/'kinematic'")

    return result



###### RADIOGENIC HEATING (EARTH MANTLE) ######

_SEC_IN_GYR = u.Gyr.to('s')
SI_to_cgs = (u.W/u.kg).to('erg/(s * g)')

# McDonough & Sun (1995) present-day specific heat-production rates (W kg⁻¹)
_ISOTOPES = ('K40', 'Th232', 'U235', 'U238')
_Q0       = np.array([8.69e-13, 2.24e-12, 8.48e-14, 1.97e-12]) * SI_to_cgs # erg s⁻¹ kg⁻¹
_TAU_GYR  = np.array([1.25,     14.0,      0.704,    4.47   ])

def radiogenic_heat(t_sec, *, weights=None, t_now_gyr=4.567):
    """
    Radiogenic mantle heat production per unit mass from ⁴⁰K, ²³²Th, ²³⁵U, ²³⁸U.

    Parameters
    ----------
    t_sec : float or array-like
        Planetary age since formation, **in seconds**.
    weights : dict | sequence | None, optional
        Relative inventory factors for each isotope.
          – None  → Earth-like (all 1.0)
          – dict  → keys 'K40','Th232','U235','U238' with scale factors
          – list/tuple/ndarray → length-4 array in the isotope order above
    t_now_gyr : float, default 4.567
        Present-day age used for q₀ (Earth ≈ 4.567 Gyr).

    Returns
    -------
    H_total : ndarray
        Total heat production (erg s⁻¹ g⁻¹) at the supplied age(s).
    H_isotopes : dict
        Individual isotope contributions (erg s⁻¹ g⁻¹).
    """
    # Build weight vector ---------------------------------------------------
    if weights is None:
        w = np.ones_like(_Q0)
    elif isinstance(weights, dict):
        w = np.array([weights.get(k, 1.0) for k in _ISOTOPES], dtype=float)
    else:
        w = np.asarray(weights, dtype=float)
        if w.shape != _Q0.shape:
            raise ValueError("weights must have length 4")

    # Convert constants to seconds -----------------------------------------
    tau_sec = _TAU_GYR * _SEC_IN_GYR
    t_now_sec = t_now_gyr * _SEC_IN_GYR

    # Vectorised evaluation of Eq. (43) ------------------------------------
    t = np.asarray(t_sec, dtype=float)
    H_each = w * _Q0 * np.exp(np.log(2.0) * (t_now_sec - t) / tau_sec)
    H_tot  = H_each.sum(axis=0)

    return H_tot, dict(zip(_ISOTOPES, H_each))



########## Thermal conductivities from Stamenkovic et al. (20211) ##########

Tref = 2000.0 # K
k_phonon_ref = 0.71  # W m⁻¹ K⁻¹ at Tref = 2000 K, ρref = 4000 kg m⁻³
rho_ref = 3870.0    # kg m⁻³
kel_ref = 7.1e2 # W m⁻¹ K⁻¹ at Tref = 2000 K, ρref = 4000 kg m⁻³
k_B_SI = 1.3806503e-23 # m^2 kgs⁻² K⁻¹
h_SI = 6.62606896e-34 # J s
E_g = 8.0109e-19 # J
c = 299.792458e6 # m/s

def get_gamma_rhoy(rho, y, dlnrho = 0.05):
    """
    Smooth, vectorized Grüneisen parameter γ(P,y):
       • Olivine   for P < 23 GPa
       • Perovskite for 23 GPa ≤ P < 125 GPa
       • Post‑pv   for P ≥ 125 GPa

    Smooth blending (tanh) over ±5% of each boundary.
    """
    # 1) broadcast inputs
    scalar = np.isscalar(rho) and np.isscalar(y)
    rho_arr = np.array(rho, ndmin=1)
    lnrho = np.log(rho_arr)
    lnrho_plus = lnrho*(1.0 + dlnrho)
    lnrho_minus = lnrho*(1.0 - dlnrho)
    y_arr = np.array(y, ndmin=1)
    if rho_arr.shape != y_arr.shape:
        rho_arr, y_arr = np.broadcast_arrays(rho_arr, y_arr)

    p = _PPV_PARAMS
    x_pp    = rho / p['rho0']
    x_pp_plus = np.exp(lnrho_plus) / p['rho0']
    x_pp_minus = np.exp(lnrho_minus) / p['rho0']

    gamma_pp = p['gamma_inf'] + (p['gamma0'] - p['gamma_inf']) * x_pp**(-p['lambda'])
    gamma_pp_plus = p['gamma_inf'] + (p['gamma0'] - p['gamma_inf']) * x_pp_plus**(-p['lambda'])
    gamma_pp_minus = p['gamma_inf'] + (p['gamma0'] - p['gamma_inf']) * x_pp_minus**(-p['lambda'])

    dlngamma_dlnrho_pp = -(np.log(gamma_pp_plus) - np.log(gamma_pp_minus)) / (lnrho_plus - lnrho_minus)

    gamma    = gamma_pp

    return gamma, dlngamma_dlnrho_pp

def get_k_phonon(P_arr, T_arr, y_arr):
    """
    Phonon thermal conductivity k_ph(P,y) from Stamenkovic et al. (2011).
    P in Pa, k_ph in W m⁻¹ K⁻¹.
    """
    # 1) broadcast inputs
    scalar = np.isscalar(P_arr) and np.isscalar(y_arr)
    P_arr = np.array(P_arr, ndmin=1)
    T_arr = np.array(T_arr, ndmin=1)
    y_arr = np.array(y_arr, ndmin=1)
    if P_arr.shape != y_arr.shape:
        P_arr, y_arr = np.broadcast_arrays(P_arr, y_arr)

    # 2) get density and Grüneisen parameter
    #rho_arr = get_rhomix_pv_Py(P_arr, y_arr)  # kg m⁻³
    rho_arr = get_rholiq_Py(P_arr, y_arr)  # kg m⁻³
    gamma, dlngamma_dlnrho = get_gamma_rhoy(rho_arr, y_arr)

    # 3) compute k_ph

    psi = 3 * gamma + 2 * dlngamma_dlnrho - 1/3

    k_phonon = k_phonon_ref * (rho_arr / rho_ref)**psi * (Tref / T_arr)

    return float(k_phonon) if scalar else k_phonon

def get_k_rad(T):
    Chi = 8.5e-11 #W / m / K^4
    return Chi * T ** 3

# Equations 33 - 36 in Stamenkovic et al. (2011)

def get_k_el(T):
    return kel_ref * (T / 1e4) ** 0.25 * np.exp(-E_g / (2 * k_B_SI * T))

def get_nu_p(T):
    k_el = get_k_el(T)
    return (5.9281e-19 / h_SI) * (T / 1e4) ** 0.25 * (k_el/kel_ref) ** 0.5

def get_I_nu(T):
    nu = get_nu_p(T)
    return 2 * h_SI / c ** 2 * (nu ** 3 / np.exp(h_SI * nu / (k_B_SI * T)) - 1)

def _planck_kernel(x):
    # robust near x=0
    return x**3 / np.expm1(x)

def get_Xi(T):
    """
    Eq. (36) in Stamenković et al. (2011).
    Returns dimensionless Xi(T) in [0, 1].
    """
    T = np.atleast_1d(T).astype(float)
    out = np.empty_like(T, dtype=float)

    for i, Ti in enumerate(T):
        nu_p = get_nu_p(Ti)                  # Eq. (33)
        x_p  = h_SI * nu_p / (k_B_SI * Ti)   # upper limit in dimensionless form
        val, _ = quad(_planck_kernel, 0.0, x_p, epsabs=1e-10, epsrel=1e-8, limit=200)
        out[i] = (15.0 / np.pi**4) * val
        # numerical safety
        out[i] = float(np.clip(out[i], 0.0, 1.0))

    return out[0] if out.size == 1 else out

def get_keff_rad(T):
    return get_k_rad(T) * (1.0 - get_Xi(T))

def get_k_total(P, T, y):
    """
    Total thermal conductivity k_total(P,y) from Stamenkovic et al. (2011).
    P in Pa, k_total in W m⁻¹ K⁻¹.
    """
    k_ph = get_k_phonon(P, T, y)
    k_rad = get_keff_rad(T)
    k_el = get_k_el(T)
    return k_ph + k_rad + k_el


## Liquid lattice thermal conductivity (Deng & Stixrude 2021) ##

# constants from the paper
V0 = 38.9             # cm^3 / mol (reference)
k0 = 1.16             # W m^-1 K^-1
T0_ref = 3000.0       # K
g = 1.75
a = -0.09

M_molar = 100.387e-3  # kg / mol for MgSiO3

def get_k_lat_liq(P, T, y):
    P = np.asarray(P, dtype=float)
    T = np.asarray(T, dtype=float)
    y = np.asarray(y, dtype=float)

    # 1) get density (must be kg/m^3)
    rho = get_rholiq_Py(P, y)   # <-- must return kg/m^3

    # 2) molar volume in cm^3 / mol
    V_cm3_per_mol = (M_molar / rho) * 1e6   # (m^3/mol -> cm^3/mol)

    # 3) thermal conductivity
    k_lat_liq = k0 * (V0 / V_cm3_per_mol)**g * (T0_ref / T)**a

    return k_lat_liq

sigma0 = 1.99e5    # S/m
deltaE = 75.94     # kJ/mol
deltaV = -0.061    # cm^3/mol
R_gas = 8.314      # J/(mol*K)

# convert constants once
deltaE_Jpermol = deltaE * 1e3         # J/mol
deltaV_m3permol = deltaV * 1e-6       # m^3/mol

def sigma_elec_liq(P, T, y=None):
    """
    Eq. (7) from Holmström et al. (2018): sigma = sigma0 * exp(-(DeltaE + P*DeltaV)/(R*T))
    P: pressure in Pa (scalar or array)
    T: temperature in K (scalar or array)
    y: composition (unused here; keep for future combination with Eq.6)
    returns: sigma in S/m
    """
    P = np.asarray(P, dtype=float)
    T = np.asarray(T, dtype=float)

    if np.any(T <= 0):
        raise ValueError("Temperature must be > 0 K")

    # compute energy term (J/mol)
    energy_term = deltaE_Jpermol + P * deltaV_m3permol

    exponent = - energy_term / (R_gas * T)

    # numeric safety (optional): clip exponent to avoid extreme under/overflow
    exponent = np.clip(exponent, -700, 700)

    sigma = sigma0 * np.exp(exponent)
    return sigma

def get_k_elec_liq(P, T, y=None):
    """
    Electrical thermal conductivity of liquid silicate from the Wiedemann-Franz law.
    k_elec = L * sigma * T
    P: pressure in Pa (scalar or array)
    T: temperature in K (scalar or array)
    y: composition (unused here; keep for future combination with Eq.6)
    returns: k_elec in W/(m*K)
    """
    L = 2.7e-8  # W Ω K⁻², Lorenz number

    sigma = sigma_elec_liq(P, T, y)  # S/m

    k_elec = L * sigma * T
    return k_elec

####### Liquid and Iron-Alloy EOS calls ########

FeSi_alloy_eos = np.load('eos/zhang_eos/zhang_multiphase/Fe16Si_fischer.npz')

P_grid_Fe = FeSi_alloy_eos['P_grid_Pa']
T_grid_Fe = FeSi_alloy_eos['T_grid_K']
rho_grid_Fe = FeSi_alloy_eos['rho_PT_grid_kg_m3']
s_grid = FeSi_alloy_eos['s_PT_grid_J_K_kg'] * J_K_kg_to_erg_K_g
u_grid_Fe = FeSi_alloy_eos['eth_PT_grid_J_kg'] * (u.J/u.kg).to('erg/g')
CP_grid_Fe = FeSi_alloy_eos['cP_PT_grid_J_K_kg'] * J_K_kg_to_erg_K_g
CV_grid_Fe = FeSi_alloy_eos['cV_PT_grid_J_K_kg'] * J_K_kg_to_erg_K_g
alpha_grid_Fe = FeSi_alloy_eos['alpha_PT_grid__K']


get_rho_FeSi = RGI((P_grid_Fe, T_grid_Fe), rho_grid_Fe, **rgi_kwargs)
get_s_FeSi = RGI((P_grid_Fe, T_grid_Fe), s_grid, **rgi_kwargs)
get_u_FeSi = RGI((P_grid_Fe, T_grid_Fe), u_grid_Fe, **rgi_kwargs)

get_CP_FeSi = RGI((P_grid_Fe, T_grid_Fe), CP_grid_Fe, **rgi_kwargs)
get_CV_FeSi = RGI((P_grid_Fe, T_grid_Fe), CV_grid_Fe, **rgi_kwargs)
get_alpha_FeSi = RGI((P_grid_Fe, T_grid_Fe), alpha_grid_Fe, **rgi_kwargs)

def get_rho_FeSi_pt(P, T):
    """
    Get the density of Fe-Si alloy at pressure P and temperature T.
    P in Pa, T in K.
    """
    scalar = np.isscalar(P) and np.isscalar(T)
    P_arr = np.array(P, ndmin=1)
    T_arr = np.array(T, ndmin=1)
    if P_arr.shape != T_arr.shape:
        P_arr, T_arr = np.broadcast_arrays(P_arr, T_arr)
    pts = np.stack((P_arr.ravel(), T_arr.ravel()), axis=-1)
    vals = get_rho_FeSi(pts).reshape(P_arr.shape)
    return float(vals) if scalar else vals

def get_s_FeSi_pt(P, T):
    """
    Get the entropy of Fe-Si alloy at pressure P and temperature T.
    P in Pa, T in K.
    """
    scalar = np.isscalar(P) and np.isscalar(T)
    P_arr = np.array(P, ndmin=1)
    T_arr = np.array(T, ndmin=1)
    if P_arr.shape != T_arr.shape:
        P_arr, T_arr = np.broadcast_arrays(P_arr, T_arr)
    pts = np.stack((P_arr.ravel(), T_arr.ravel()), axis=-1)
    vals = get_s_FeSi(pts).reshape(P_arr.shape)
    return float(vals) if scalar else vals

def get_u_FeSi_pt(P, T):
    """
    Get the internal energy of Fe-Si alloy at pressure P and temperature T.
    P in Pa, T in K.
    """
    scalar = np.isscalar(P) and np.isscalar(T)
    P_arr = np.array(P, ndmin=1)
    T_arr = np.array(T, ndmin=1)
    if P_arr.shape != T_arr.shape:
        P_arr, T_arr = np.broadcast_arrays(P_arr, T_arr)
    pts = np.stack((P_arr.ravel(), T_arr.ravel()), axis=-1)
    vals = get_u_FeSi(pts).reshape(P_arr.shape)
    return float(vals) if scalar else vals

def get_CP_FeSi_pt(P, T):
    """
    Get the isobaric heat capacity of Fe-Si alloy at pressure P and temperature T.
    P in Pa, T in K.
    """
    scalar = np.isscalar(P) and np.isscalar(T)
    P_arr = np.array(P, ndmin=1)
    T_arr = np.array(T, ndmin=1)
    if P_arr.shape != T_arr.shape:
        P_arr, T_arr = np.broadcast_arrays(P_arr, T_arr)
    pts = np.stack((P_arr.ravel(), T_arr.ravel()), axis=-1)
    vals = get_CP_FeSi(pts).reshape(P_arr.shape)
    return float(vals) if scalar else vals

def get_CV_FeSi_pt(P, T):
    """
    Get the isochoric heat capacity of Fe-Si alloy at pressure P and temperature T.
    P in Pa, T in K.
    """
    scalar = np.isscalar(P) and np.isscalar(T)
    P_arr = np.array(P, ndmin=1)
    T_arr = np.array(T, ndmin=1)
    if P_arr.shape != T_arr.shape:
        P_arr, T_arr = np.broadcast_arrays(P_arr, T_arr)
    pts = np.stack((P_arr.ravel(), T_arr.ravel()), axis=-1)
    vals = get_CV_FeSi(pts).reshape(P_arr.shape)
    return float(vals) if scalar else vals

def get_alpha_FeSi_pt(P, T):
    """
    Get the thermal expansion coefficient of Fe-Si alloy at pressure P and temperature T.
    P in Pa, T in K.
    """
    scalar = np.isscalar(P) and np.isscalar(T)
    P_arr = np.array(P, ndmin=1)
    T_arr = np.array(T, ndmin=1)
    if P_arr.shape != T_arr.shape:
        P_arr, T_arr = np.broadcast_arrays(P_arr, T_arr)
    pts = np.stack((P_arr.ravel(), T_arr.ravel()), axis=-1)
    vals = get_alpha_FeSi(pts).reshape(P_arr.shape)
    return float(vals) if scalar else vals

# these come from your loaded table
#   T_grid_Fe is the 1D array of tabulated temperatures (in K)
T_min_Fe = 0
T_max_Fe = 200000

def get_T_sp_inv(_s, _P, xtol=1e-8, maxiter=500):
    """
    Invert s(P, T) → T, i.e. find T such that get_s_FeSi_pt(P, T) == s.

    Parameters
    ----------
    _s : float or array_like
        Entropy value(s) in kB/baryon.
    _P : float or array_like
        Pressure value(s) in Pa.
    xtol : float, optional
        Tolerance on the temperature root (passed to brentq).
    maxiter : int, optional
        Maximum number of iterations for brentq.

    Returns
    -------
    T_sol : float or ndarray
        Temperature(s) in K.  If any root‐finding fails, the corresponding
        entry is set to np.nan.
    """
    s_arr = np.atleast_1d(_s)
    P_arr = np.atleast_1d(_P)
    s_arr, P_arr = np.broadcast_arrays(s_arr, P_arr)

    def _find_T(s_val, P_val):
        # The function whose root in T we seek:
        def err(T_val):
            return get_s_FeSi_pt(P_val, T_val) * erg_to_kbbar / s_val - 1

        try:
            return brentq(err, T_min_Fe, T_max_Fe, xtol=xtol, maxiter=maxiter)
        except ValueError:
            # e.g. f(T_min)*f(T_max) ≥ 0 or NaN → no bracket
            return np.nan

    # vectorize over the pair (s_val, P_val)
    T_roots = np.vectorize(_find_T)(s_arr, P_arr)

    # return scalar if inputs were scalars
    if T_roots.size == 1:
        return float(T_roots)
    return T_roots

###### S, P basis ######

sp_data_FeSi = np.load('eos/zhang_eos/zhang_multiphase/Fe16Si_fischer_sp.npz')

svals_sp_FeSi = sp_data_FeSi['s_vals'] # kb/baryon
pvals_sp_FeSi = sp_data_FeSi['P_grid_Pa'] # log K

rho_grid_sp_FeSi = sp_data_FeSi['rho_SP_grid'] # in g/cm^3
t_grid_sp_FeSi = sp_data_FeSi['T_SP_grid_K'] # in K
u_grid_sp_FeSi = sp_data_FeSi['u_SP_grid'] # in erg/g

t_rgi_sp_FeSi = RGI((svals_sp_FeSi, pvals_sp_FeSi), t_grid_sp_FeSi, method='linear', \
            bounds_error=False, fill_value=None)
rho_rgi_sp_FeSi = RGI((svals_sp_FeSi, pvals_sp_FeSi), rho_grid_sp_FeSi, method='linear', \
            bounds_error=False, fill_value=None)
u_rgi_sp_FeSi = RGI((svals_sp_FeSi, pvals_sp_FeSi), u_grid_sp_FeSi, method='linear', \
            bounds_error=False, fill_value=None)

def get_rho_FeSi_sp(S, P):
    """
    Get the density of liquid FeSi at entropy S and pressure P.
    S in kb/baryon, P in Pa.
    """

    scalar = np.isscalar(S) and np.isscalar(P)
    S_arr = np.array(S, ndmin=1)
    P_arr = np.array(P, ndmin=1)
    if S_arr.shape != P_arr.shape:
        S_arr, P_arr = np.broadcast_arrays(S_arr, P_arr)
    pts = np.stack((S_arr.ravel(), P_arr.ravel()), axis=-1)
    vals = rho_rgi_sp_FeSi(pts).reshape(S_arr.shape)
    return float(vals) if scalar else vals

def get_T_FeSi_sp(S, P):
    """
    Get the temperature of liquid Fe at entropy S and pressure P.
    S in kb/baryon, P in Pa.
    """

    scalar = np.isscalar(S) and np.isscalar(P)
    S_arr = np.array(S, ndmin=1)
    P_arr = np.array(P, ndmin=1)
    if S_arr.shape != P_arr.shape:
        S_arr, P_arr = np.broadcast_arrays(S_arr, P_arr)
    pts = np.stack((S_arr.ravel(), P_arr.ravel()), axis=-1)
    vals = t_rgi_sp_FeSi(pts).reshape(S_arr.shape) # K
    return float(vals) if scalar else vals

def get_u_FeSi_sp(S, P):
    """
    Get the internal energy of liquid Fe at entropy S and pressure P.
    S in kb/baryon, P in Pa.
    """

    scalar = np.isscalar(S) and np.isscalar(P)
    S_arr = np.array(S, ndmin=1)
    P_arr = np.array(P, ndmin=1)
    if S_arr.shape != P_arr.shape:
        S_arr, P_arr = np.broadcast_arrays(S_arr, P_arr)
    pts = np.stack((S_arr.ravel(), P_arr.ravel()), axis=-1)
    vals = u_rgi_sp_FeSi(pts).reshape(S_arr.shape) # erg/g
    return float(vals) if scalar else vals


####### Liquid and Iron EOS calls ########

P_grid_Fe_l = np.loadtxt('eos/zhang_eos/zhang_multiphase/iron2/Pgrid_Fel.txt')
T_grid_Fe_l = np.loadtxt('eos/zhang_eos/zhang_multiphase/iron2/Tgrid_Fel.txt')

s_grid_Fe_l = np.loadtxt('eos/zhang_eos/zhang_multiphase/iron2/s_Fel.txt') * J_K_kg_to_erg_K_g
rho_grid_Fe_l = np.loadtxt('eos/zhang_eos/zhang_multiphase/iron2/rho_Fel.txt')
u_grid_Fe_l = np.loadtxt('eos/zhang_eos/zhang_multiphase/iron2/Eth_Fel.txt')* (u.J/u.kg).to('erg/g')
CP_grid_Fe_l = np.loadtxt('eos/zhang_eos/zhang_multiphase/iron2/cP_Fel.txt') * J_K_kg_to_erg_K_g
CV_grid_Fe_l = np.loadtxt('eos/zhang_eos/zhang_multiphase/iron2/cV_Fel.txt') * J_K_kg_to_erg_K_g
alpha_grid_Fe_l = np.loadtxt('eos/zhang_eos/zhang_multiphase/iron2/alpha_Fel.txt')

get_rho_Fel = RGI((P_grid_Fe_l, T_grid_Fe_l), rho_grid_Fe_l, **rgi_kwargs)
get_s_Fel = RGI((P_grid_Fe_l, T_grid_Fe_l), s_grid_Fe_l, **rgi_kwargs)
get_u_Fel = RGI((P_grid_Fe_l, T_grid_Fe_l), u_grid_Fe_l, **rgi_kwargs)

get_CP_Fel = RGI((P_grid_Fe_l, T_grid_Fe_l), CP_grid_Fe_l, **rgi_kwargs)
get_CV_Fel = RGI((P_grid_Fe_l, T_grid_Fe_l), CV_grid_Fe_l, **rgi_kwargs)
get_alpha_Fel = RGI((P_grid_Fe_l, T_grid_Fe_l), alpha_grid_Fe_l, **rgi_kwargs)

def get_rho_Fel_pt(P, T):
    """
    Get the density of Fe-Si alloy at pressure P and temperature T.
    P in Pa, T in K.
    """
    scalar = np.isscalar(P) and np.isscalar(T)
    P_arr = np.array(P, ndmin=1)
    T_arr = np.array(T, ndmin=1)
    if P_arr.shape != T_arr.shape:
        P_arr, T_arr = np.broadcast_arrays(P_arr, T_arr)
    pts = np.stack((P_arr.ravel(), T_arr.ravel()), axis=-1)
    vals = get_rho_Fel(pts).reshape(P_arr.shape)
    return float(vals) if scalar else vals

def get_s_Fel_pt(P, T):
    """
    Get the entropy of Fe-Si alloy at pressure P and temperature T.
    P in Pa, T in K.
    """
    scalar = np.isscalar(P) and np.isscalar(T)
    P_arr = np.array(P, ndmin=1)
    T_arr = np.array(T, ndmin=1)
    if P_arr.shape != T_arr.shape:
        P_arr, T_arr = np.broadcast_arrays(P_arr, T_arr)
    pts = np.stack((P_arr.ravel(), T_arr.ravel()), axis=-1)
    vals = get_s_Fel(pts).reshape(P_arr.shape)
    return float(vals) if scalar else vals

def get_u_Fel_pt(P, T):
    """
    Get the internal energy of Fe-Si alloy at pressure P and temperature T.
    P in Pa, T in K.
    """
    scalar = np.isscalar(P) and np.isscalar(T)
    P_arr = np.array(P, ndmin=1)
    T_arr = np.array(T, ndmin=1)
    if P_arr.shape != T_arr.shape:
        P_arr, T_arr = np.broadcast_arrays(P_arr, T_arr)
    pts = np.stack((P_arr.ravel(), T_arr.ravel()), axis=-1)
    vals = get_u_Fel(pts).reshape(P_arr.shape)
    return float(vals) if scalar else vals

def get_CP_Fel_pt(P, T):
    """
    Get the isobaric heat capacity of Fe-Si alloy at pressure P and temperature T.
    P in Pa, T in K.
    """
    scalar = np.isscalar(P) and np.isscalar(T)
    P_arr = np.array(P, ndmin=1)
    T_arr = np.array(T, ndmin=1)
    if P_arr.shape != T_arr.shape:
        P_arr, T_arr = np.broadcast_arrays(P_arr, T_arr)
    pts = np.stack((P_arr.ravel(), T_arr.ravel()), axis=-1)
    vals = get_CP_Fel(pts).reshape(P_arr.shape)
    return float(vals) if scalar else vals

def get_CV_Fel_pt(P, T):
    """
    Get the isochoric heat capacity of Fe-Si alloy at pressure P and temperature T.
    P in Pa, T in K.
    """
    scalar = np.isscalar(P) and np.isscalar(T)
    P_arr = np.array(P, ndmin=1)
    T_arr = np.array(T, ndmin=1)
    if P_arr.shape != T_arr.shape:
        P_arr, T_arr = np.broadcast_arrays(P_arr, T_arr)
    pts = np.stack((P_arr.ravel(), T_arr.ravel()), axis=-1)
    vals = get_CV_Fel(pts).reshape(P_arr.shape)
    return float(vals) if scalar else vals

def get_alpha_Fel_pt(P, T):
    """
    Get the thermal expansion coefficient of Fe-Si alloy at pressure P and temperature T.
    P in Pa, T in K.
    """
    scalar = np.isscalar(P) and np.isscalar(T)
    P_arr = np.array(P, ndmin=1)
    T_arr = np.array(T, ndmin=1)
    if P_arr.shape != T_arr.shape:
        P_arr, T_arr = np.broadcast_arrays(P_arr, T_arr)
    pts = np.stack((P_arr.ravel(), T_arr.ravel()), axis=-1)
    vals = get_alpha_Fel(pts).reshape(P_arr.shape)
    return float(vals) if scalar else vals


###### S, P basis ######

sp_data_Fel = np.load('eos/zhang_eos/zhang_multiphase/iron2_sp.npz')

svals_sp_Fel = sp_data_Fel['s_vals'] # kb/baryon
logpvals_sp_Fel = sp_data_Fel['logpvals'] # log K

logrho_grid_sp_Fel = sp_data_Fel['logrho_sp'] # in g/cm^3
logt_grid_sp_Fel = sp_data_Fel['logt_sp'] # in K
logu_grid_sp_Fel = sp_data_Fel['logu_sp'] # in erg/g

logt_rgi_sp_Fel = RGI((svals_sp_Fel, logpvals_sp_Fel), logt_grid_sp_Fel, method='linear', \
            bounds_error=False, fill_value=None)
logrho_rgi_sp_Fel = RGI((svals_sp_Fel, logpvals_sp_Fel), logrho_grid_sp_Fel, method='linear', \
            bounds_error=False, fill_value=None)
logu_rgi_sp_Fel = RGI((svals_sp_Fel, logpvals_sp_Fel), logu_grid_sp_Fel, method='linear', \
            bounds_error=False, fill_value=None)

def get_rho_Fel_sp(S, P):
    """
    Get the density of liquid Fe at entropy S and pressure P.
    S in kb/baryon, P in Pa.
    """

    logP_cgs = np.log10(P / dyn_to_Pa)
    scalar = np.isscalar(S) and np.isscalar(P)
    S_arr = np.array(S, ndmin=1)
    logP_arr = np.array(logP_cgs, ndmin=1)
    if S_arr.shape != logP_arr.shape:
        S_arr, logP_arr = np.broadcast_arrays(S_arr, logP_arr)
    pts = np.stack((S_arr.ravel(), logP_arr.ravel()), axis=-1)
    vals = 10 ** (logrho_rgi_sp_Fel(pts).reshape(S_arr.shape) + 3) # kg/m^3
    return float(vals) if scalar else vals

def get_T_Fel_sp(S, P):
    """
    Get the temperature of liquid Fe at entropy S and pressure P.
    S in kb/baryon, P in Pa.
    """

    logP_cgs = np.log10(P / dyn_to_Pa)
    scalar = np.isscalar(S) and np.isscalar(P)
    S_arr = np.array(S, ndmin=1)
    logP_arr = np.array(logP_cgs, ndmin=1)
    if S_arr.shape != logP_arr.shape:
        S_arr, logP_arr = np.broadcast_arrays(S_arr, logP_arr)
    pts = np.stack((S_arr.ravel(), logP_arr.ravel()), axis=-1)
    vals = 10 ** logt_rgi_sp_Fel(pts).reshape(S_arr.shape) # K
    return float(vals) if scalar else vals

def get_u_Fel_sp(S, P):
    """
    Get the internal energy of liquid Fe at entropy S and pressure P.
    S in kb/baryon, P in Pa.
    """

    logP_cgs = np.log10(P / dyn_to_Pa)
    scalar = np.isscalar(S) and np.isscalar(P)
    S_arr = np.array(S, ndmin=1)
    logP_arr = np.array(logP_cgs, ndmin=1)
    if S_arr.shape != logP_arr.shape:
        S_arr, logP_arr = np.broadcast_arrays(S_arr, logP_arr)
    pts = np.stack((S_arr.ravel(), logP_arr.ravel()), axis=-1)
    vals = 10 ** logu_rgi_sp_Fel(pts).reshape(S_arr.shape) # erg/g
    return float(vals) if scalar else vals


######## wrapper functions to select between Fe_alloy and Fe_liquid ########
