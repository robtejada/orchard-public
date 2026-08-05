"""
Physical constants used by Orchard/APPLE.

Unit system used in this module:
- Base numeric constants are exported as plain floats in CGS:
  - mass: g
  - length: cm
  - time: s
  - temperature: K
  - pressure: dyn / cm^2
  - energy: erg
- Conversion factors are dimensionless multipliers between explicit unit systems.

Implementation notes:
- Values are sourced from `astropy.constants` wherever available.
- For adopted planetary values not provided by astropy (e.g., Saturn/Uranus/Neptune
  reference radii/masses used by this project), `astropy.constants.Constant` is used
  with explicit units and references.
- Constants are exported as floats for backward compatibility with existing solver code.
"""

from astropy import units as u
from astropy.constants import Constant
from astropy.constants import (
    G as G_ast,
    L_sun as L_sun_ast,
    M_earth as M_earth_ast,
    M_jup as M_jup_ast,
    M_sun as M_sun_ast,
    N_A as N_A_ast,
    R_earth as R_earth_ast,
    R_jup as R_jup_ast,
    R_sun as R_sun_ast,
    au as au_ast,
    c as c_ast,
    e as e_ast,
    h as h_ast,
    k_B as k_B_ast,
    m_e as m_e_ast,
    m_n as m_n_ast,
    m_p as m_p_ast,
    sigma_sb as sigma_sb_ast,
    u as amu_ast,
)


def _adopted_constant(abbrev, name, value, unit, reference):
    """Create a project-adopted astropy constant with explicit units."""
    return Constant(abbrev, name, value, unit, 0.0, reference, system="cgs")


# ---------------------------------------------------------------------
# Thermodynamic conversion factors
# ---------------------------------------------------------------------
# k_B / baryon in CGS: erg / (K g)
erg_to_kbbar = (u.erg / u.K / u.g).to(k_B_ast / amu_ast)
MJ_to_kbbar = (u.MJ / u.K / u.kg).to(k_B_ast / amu_ast)
kbbar_to_erg = 1.0 / erg_to_kbbar
erg_to_MJ = (u.erg / u.K / u.g).to(u.MJ / u.K / u.kg)

# Pressure conversion
dyn_to_bar = (u.dyne / u.cm**2).to(u.bar)
dyn_to_GPa = (u.dyne / u.cm**2).to(u.GPa)
dyn_to_Pa = (u.dyne / u.cm**2).to(u.Pa)
dyn_to_Mbar = (u.dyne / u.cm**2).to(u.bar) * 1e6


# ---------------------------------------------------------------------
# Canonical Solar System constants from astropy (exported as CGS floats)
# ---------------------------------------------------------------------
R_jup = R_jup_ast.to_value(u.cm)  # cm
M_jup = M_jup_ast.to_value(u.g)  # g

msun = M_sun_ast.to_value(u.g)  # g
rsun = R_sun_ast.to_value(u.cm)  # cm
lsun = L_sun_ast.to_value(u.erg / u.s)  # erg / s

mearth = M_earth_ast.to_value(u.g)  # g
rearth = R_earth_ast.to_value(u.cm)  # cm
mjup = M_jup  # g (legacy alias)


# ---------------------------------------------------------------------
# Radiation / EOS helpers (CGS)
# ---------------------------------------------------------------------
# Radiation density constant a = 4 sigma / c in erg / (cm^3 K^4)
a = (4.0 * sigma_sb_ast / c_ast).to_value(u.erg / (u.cm**3 * u.K**4))


# ---------------------------------------------------------------------
# Adopted giant-planet figure constants (Seidelmann+2007 / project defaults)
# ---------------------------------------------------------------------
_PLANET_REF = "Adopted in APPLE/Orchard; Seidelmann et al. 2007 where applicable"

# Jupiter reference radii (cm)
_R_JUP_MEAN = _adopted_constant("R_jup_mean", "Jupiter volumetric mean radius", 6.9911e9, "cm", _PLANET_REF)
_R_JUP_EQ = _adopted_constant("R_jup_eq", "Jupiter equatorial radius", 7.1492e9, "cm", _PLANET_REF)
_R_JUP_POL = _adopted_constant("R_jup_pol", "Jupiter polar radius", 6.6854e9, "cm", _PLANET_REF)
_DR_JUP_MEAN = _adopted_constant("dR_jup_mean", "Jupiter mean-radius uncertainty", 6.0e5, "cm", _PLANET_REF)
_DR_JUP_EQ = _adopted_constant("dR_jup_eq", "Jupiter equatorial-radius uncertainty", 4.0e5, "cm", _PLANET_REF)
_DR_JUP_POL = _adopted_constant("dR_jup_pol", "Jupiter polar-radius uncertainty", 1.0e6, "cm", _PLANET_REF)

rjup = _R_JUP_MEAN.cgs.value
drjup = _DR_JUP_MEAN.cgs.value
drjup_eq = _DR_JUP_EQ.cgs.value
rj_eq = rjup_eq = _R_JUP_EQ.cgs.value
rj_pol = rjup_pol = _R_JUP_POL.cgs.value
drjup_pol = _DR_JUP_POL.cgs.value

# Saturn adopted mass/radii (g, cm)
_M_SAT = _adopted_constant("M_sat", "Saturn mass", 568.34e27, "g", _PLANET_REF)
_R_SAT_MEAN = _adopted_constant("R_sat_mean", "Saturn volumetric mean radius", 58232e5, "cm", _PLANET_REF)
_R_SAT_EQ = _adopted_constant("R_sat_eq", "Saturn equatorial radius", 60268e5, "cm", _PLANET_REF)
_R_SAT_POL = _adopted_constant("R_sat_pol", "Saturn polar radius", 54364e5, "cm", _PLANET_REF)
_DR_SAT_MEAN = _adopted_constant("dR_sat_mean", "Saturn mean-radius uncertainty", 6.0e5, "cm", _PLANET_REF)
_DR_SAT_EQ = _adopted_constant("dR_sat_eq", "Saturn equatorial-radius uncertainty", 4.0e5, "cm", _PLANET_REF)
_DR_SAT_POL = _adopted_constant("dR_sat_pol", "Saturn polar-radius uncertainty", 1.0e6, "cm", _PLANET_REF)

msat = _M_SAT.cgs.value
rsat = _R_SAT_MEAN.cgs.value
drsat = _DR_SAT_MEAN.cgs.value
rs_eq = rsat_eq = _R_SAT_EQ.cgs.value
drsat_eq = _DR_SAT_EQ.cgs.value
rs_pol = rsat_pol = _R_SAT_POL.cgs.value
drsat_pol = _DR_SAT_POL.cgs.value

# Uranus / Neptune adopted masses and volumetric radii (g, cm)
_M_URA = _adopted_constant("M_ura", "Uranus mass", 8.6811e28, "g", _PLANET_REF)
_R_URA = _adopted_constant("R_ura", "Uranus volumetric mean radius", 2.5362e9, "cm", _PLANET_REF)
_M_NEP = _adopted_constant("M_nep", "Neptune mass", 1.02413e29, "g", _PLANET_REF)
_R_NEP = _adopted_constant("R_nep", "Neptune volumetric mean radius", 2.4622e9, "cm", _PLANET_REF)

mura = _M_URA.cgs.value
rura = _R_URA.cgs.value
mnep = _M_NEP.cgs.value
rnep = _R_NEP.cgs.value


# ---------------------------------------------------------------------
# Fundamental constants (CGS floats)
# ---------------------------------------------------------------------
G = G_ast.cgs.value  # cm^3 / (g s^2)
h = h_ast.cgs.value  # erg s
qe = e_ast.esu.value  # statC (Fr)
c = clight = c_ast.cgs.value  # cm / s
k = kb = k_B_ast.cgs.value  # erg / K
kb_ev = k_B_ast.to_value(u.eV / u.K)  # eV / K
avogadro = N_A_ast.value  # 1 / mol
rgas = cgas = k * avogadro  # erg / (K mol)

# Particle masses (g)
amu = amu_ast.cgs.value
mn = m_n_ast.cgs.value
mp = m_p_ast.cgs.value
me = m_e_ast.cgs.value

# Mean atomic masses (g)
mh = 1.00794 * amu
mhe = 4.002602 * amu

boltz_sigma = sigma_sb = sigma_sb_ast.cgs.value  # erg / (cm^2 s K^4)
arad = crad = boltz_sigma * 4.0 / c  # erg / (cm^3 K^4)
au = au_ast.cgs.value  # cm


# ---------------------------------------------------------------------
# Time conversion factors
# ---------------------------------------------------------------------
s_to_yr = (1.0 * u.s).to_value(u.yr)
s_to_Myr = (1.0 * u.s).to_value(u.Myr)
s_to_Gyr = (1.0 * u.s).to_value(u.Gyr)
Myr_to_s = (1.0 * u.Myr).to_value(u.s)
Gyr_to_s = (1.0 * u.Gyr).to_value(u.s)


# ---------------------------------------------------------------------
# Metallicity reference constants (dimensionless)
# ---------------------------------------------------------------------
# Keep all Z_solar choices centralized so atmosphere/opacity conversions
# are consistent across modules and easy to audit.
Z_SOLAR_BAHCALL06 = 0.01884
Z_SOLAR_CHEN23 = 0.017
Z_SOLAR_FORTNEY07 = 0.0156
