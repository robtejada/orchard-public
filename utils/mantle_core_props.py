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

J_K_kg_to_erg_K_g = (u.J / (u.kg * u.K)).to('erg/(K*g)') # specific entropy conversion
J_kg_to_erg_g = (u.J / u.kg).to('erg/g') # specific energy conversion
dyn_to_Pa = (u.dyn/u.cm**2).to('Pa') # dyn/cm² to Pa conversion
kb = k_B.to('erg/K') # ergs/K
erg_to_kbbar = (u.erg/u.Kelvin/u.gram).to(k_B/amu)

# Liquid lattice thermal conductivity (Deng & Stixrude 2021) ##

# constants from the paper
# [Deng & Stixrude (2021), Peng & Deng (2025)]
V0 = [38.9, 27.559]             # cm^3 / mol (reference)
k0 = [1.16, 2.10617913]             # W m^-1 K^-1
T0_ref = 3000.0       # K
g = [1.75, 1.22955707]
a = [0.09, 0.17726994]

M_molar = 100.387e-3  # kg / mol for MgSiO3

def get_k_lat_liq(rho, T, cond_type='PD25'):
    # Deng & Stixrude (2021)
    # Peng & Deng (2025)
    rho_cgs = np.asarray(rho, dtype=float)
    T = np.asarray(T, dtype=float)

    rho_SI = rho_cgs * 1e3  # convert to kg/m^3

    if np.any(T <= 0):
        raise ValueError("Temperature must be > 0 K")

    # 2) molar volume in cm^3 / mol
    V_cm3_per_mol = (M_molar / rho_SI) * 1e6   # (m^3/mol -> cm^3/mol)

    # 3) thermal conductivity
    V_ref = V0[0] if cond_type == 'SD21' else V0[1]
    a_exp = a[0] if cond_type == 'SD21' else a[1]
    g_exp = g[0] if cond_type == 'SD21' else g[1]
    k_const = k0[0] if cond_type == 'SD21' else k0[1]

    k_lat_liq = k_const * (V_ref / V_cm3_per_mol)**g_exp * (T0_ref / T)**a_exp

    return k_lat_liq

sigma0 = 1.99e5    # S/m
deltaE = 75.94     # kJ/mol
deltaV = -0.061    # cm^3/mol
R_gas = 8.314      # J/(mol*K)

# convert constants once
deltaE_Jpermol = deltaE * 1e3         # J/mol
deltaV_m3permol = deltaV * 1e-6       # m^3/mol

def sigma_elec_liq(P, T):
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

def get_k_elec_liq(P, T):
    """
    Electrical thermal conductivity of liquid silicate from the Wiedemann-Franz law.
    k_elec = L * sigma * T
    P: pressure in Pa (scalar or array)
    T: temperature in K (scalar or array)
    y: composition (unused here; keep for future combination with Eq.6)
    returns: k_elec in W/(m*K)
    """
    L = 2.7e-8  # W Ω K⁻², Lorenz number

    sigma = sigma_elec_liq(P, T)  # S/m

    k_elec = L * sigma * T
    return k_elec

Tref = 2000.0 # K
k_phonon_ref = 0.71  # W m⁻¹ K⁻¹ at Tref = 2000 K, ρref = 4000 kg m⁻³
rho_ref = 3870.0    # kg m⁻³
kel_ref = 7.1e2 # W m⁻¹ K⁻¹ at Tref = 2000 K, ρref = 4000 kg m⁻³
k_B_SI = 1.3806503e-23 # m^2 kgs⁻² K⁻¹
h_SI = 6.62606896e-34 # J s
E_g = 8.0109e-19 # J
c = 299.792458e6 # m/s

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

def get_eta_solid(P, T, strain_rate=1.0e-15, ppv_eta_model=1, out="cgs", material='mg2sio4'):

    P = np.asarray(P, dtype=float)
    T = np.asarray(T, dtype=float)
    strain_rate = np.asarray(strain_rate, dtype=float)

    # ------------------------------------------------------------------ #
    # 1.  dislocation‑creep viscosity for olivine / perovskite
    # ------------------------------------------------------------------ #
    eta_dyn = np.empty_like(P)

    # masks
    # mask_ol  =  P <  23.0e9              # < 23 GPa
    # mask_pv  = (P >= 23.0e9) & (P < 125e9)
    # mask_ppv =  P >= 125e9

    # # low‑P olivine
    # if mask_ol.any():
    #     prm = _PARAMS["olivine"]
    #     eta_dyn[mask_ol] = _dislocation_creep_visc(
    #         P[mask_ol], T[mask_ol], strain_rate,
    #         **prm
    #     )

    # mid‑P perovskite
    # if mask_pv.any():
    if material.lower() == 'mg2sio4':
        prm = _PARAMS["olivine"]
    elif material.lower() == 'mgsio3':
        prm = _PARAMS["perovskite"]
    eta_dyn = _dislocation_creep_visc(
        P, T, strain_rate,
        **prm
    )

    # ------------------------------------------------------------------ #
    # 2.  post‑perovskite high‑P viscosity
    # ------------------------------------------------------------------ #
    # if mask_ppv.any():
    if ppv_eta_model == 1:
        eta0     = 1.05e34    # Pa s
        E        = 7.8e5      # J mol⁻¹
        p_decay  = 1100e9     # Pa
        V        = 1.7e-6 * np.exp(-P / p_decay)
    elif ppv_eta_model == 2:
        eta0     = 1.9e21
        E        = 1.62e5
        p_decay  = 1610e9
        V        = 1.4e-6 * np.exp(-P / p_decay)
    else:
        raise ValueError("`ppv_eta_model` must be 1 or 2")

    eta_dyn = eta0 * np.exp((E + P * V) / (_RG * T)
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
_Q0       = np.array([8.69e-13, 2.24e-12, 8.48e-14, 1.97e-12]) * SI_to_cgs # erg s⁻¹ g⁻¹
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