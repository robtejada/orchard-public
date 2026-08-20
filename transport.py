### ======================================================================================
"""
Module: transport.py — Coupled heat + composition transport via global Newton-Raphson.

Solves the time-dependent entropy, helium, and metal evolution equations on the
1-D Lagrangian mass grid. This is the physics core of ORCHARD, called every
timestep by evolution.py before the hydrostatic equilibrium re-solve.

Main function:
    update_SYZ() returns a 32-tuple of updated profiles and diagnostics.

Equations solved (3N coupled, simultaneously):
    - N entropy equations:
      dS/dt = -(1/T)(dF/dm) + (L/T)(dchi/dt) + Q_radio/T
              - (1/T)(dU/dY * dY/dt + dU/dZ * dZ/dt)

      Where fluxes F = F_conv + F_rad + F_cond + F_semi:
        Convective (MLT):  F_conv = -phi * |dS_eff/dr|^(3/2)
        Radiative:         F_rad  = -Lambda_rad * dT/dr
        Conductive:        F_cond = -Lambda_cond * dT/dr
        Semiconvective:    active where Schwarzschild-unstable but Ledoux-stable

      The effective entropy gradient includes Ledoux composition terms:
        dS_eff/dr = -dS/dr + (dS/dY)|_{rho,P} * dY/dr + (dS/dZ)|_{rho,P} * dZ/dr

    - N helium equations:
      dY/dt from diffusion + rain advection (gated by He miscibility curve)

    - N metal equations:
      dZ/dt from diffusion + rain advection (gated by metal miscibility curve)

Solver:
    Global Newton-Raphson on the 3N x 3N system [S, Y, Z] with:
    - Trust-region damped steps (safe_newton_step) preventing non-physical states
    - Auto-damping: reduces weight if oscillating, increases if steadily converging
    - Sparse LU factorization via scipy.sparse.linalg.spsolve
    - Stalled acceptance after 30+ iterations if residual is small
    - Up to 3 full restarts if NR fails (halves weight, resets to old state)

Heat transport:
    Envelope:  French/Becker H-He conductivity + Z*water conductivity
    Mantle:    lattice + electronic (Wiedemann-Franz) + optional radiative
    Core:      constant k (configurable)
    Radiative: Rosseland mean opacity with temporal ramp at early times

Composition physics:
    He rain:    enforced when Y > Y_misc(P,T) from misc/ miscibility curves
    Metal rain: enforced when Z > Z_misc(P,T) from Gilmore or Gupta curves
    Diffusion:  microscopic D + MLT-enhanced D_conv; suppressed in rain zones

Mantle-specific physics (sub-Neptunes/super-Earths):
    Viscous convection: blends inviscid liquid and solid dislocation creep via
                        melt fraction chi; four eddy diffusivity branches
    Latent heat:        smooth tanh melt fraction chi(T,P) with relaxation filter
    Radiogenic heating: K-40, Th-232, U-235, U-238 applied to mantle cells

Developers:
    - Roberto Tejada Arevalo (APPLE+ORCHARD)
    - Ankan Sur (APPLE)
    - Yubo Su (APPLE)

Affiliation:
    Department of Astrophysical Sciences, Princeton University
"""
### ======================================================================================


import numpy as np
import os as _os_env
# --- On-demand FD Jacobian probe (debug affordance; env-gated) ---
# ORCHARD_FD_PROBE_GYR=x : run the finite-difference Jacobian check once,
#                          at the first Newton iteration after age x Gyr
#                          (complements the automatic max-iter FD check).
# ORCHARD_FD_CELL=k      : center the probe on cell k.
_FD_PROBE_GYR = float(_os_env.environ.get('ORCHARD_FD_PROBE_GYR', '0'))
_FD_CELL = int(_os_env.environ.get('ORCHARD_FD_CELL', '-1'))
if _FD_PROBE_GYR > 0:
    print(f"*** FD PROBE ARMED: t>={_FD_PROBE_GYR} Gyr, cell {_FD_CELL} ***",
          flush=True)
from scipy.sparse.linalg import spsolve
from scipy.sparse import csc_array, coo_array
from pathlib import Path
from typing import NamedTuple
from utils import const
from utils.const import *
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
import atm_bc
from misc import misc
from misc import gilmore_misc as zmisc
norm = kbbar_to_erg
from initial import mixtures_eos, mantle_eos, core_eos
from scipy.interpolate import RegularGridInterpolator as RGI
import astropy.units as u
from utils.common import *
from conductivities.water_cond import (
    Lambda_e,
    Lambda_p,
    set_french19_extrapolation_mode,
)
from conductivities.adiabat_cond import get_Lambda_cond, get_Z_ref_for_adiabats, get_D_H, get_D_He
from utils import mantle_core_props as mantle_core

# ------------------ GENERAL PARAMETERS & CONFIG --------------------- #
N = int(config['general']['N'])
high_precision = config['general'].getboolean('high_precision')

# ------------------ INITIAL PARAMETERS ------------------------------ #
M_factor = float(config['initial'].get('M_Mearth', config['initial'].get('M_factor', '317.907')))
M_factor_MJ = float(config['initial'].get('M_MJup', config['initial'].get('M_factor_MJ', '1.0')))
M_planet = float(config['initial'].get('M_planet_unit_cgs', config['initial'].get('M_planet', '5.972167867791379e27'))) * M_factor
if M_factor_MJ != 1.0:
    M_planet *= M_factor_MJ

# ------------------ BOUNDARY CONDITIONS ----------------------------- #
BC_atm = config['boundary_condition']['bc_atm']
planet = config['boundary_condition']['planet']
Y_atm_bc = float(config['boundary_condition']['Y_atm'])
# Use shared solar reference for C23-style atmosphere metallicity conversions.
Z_atm_bc = float(config['boundary_condition']['Z_atm']) * const.Z_SOLAR_CHEN23
cloud = config['boundary_condition'].getboolean('cloud')
irradiation = config['boundary_condition'].getboolean('irradiation')
y_atm_fixed = config['boundary_condition'].getboolean('y_atm_fixed')
z_atm_fixed = config['boundary_condition'].getboolean('z_atm_fixed')
T_eq = float(config['boundary_condition']['T_eq'])
bond_albedo = float(config['boundary_condition']['bond_albedo'])
rock_gray_kappa_ir = float(config['boundary_condition'].get('rock_gray_kappa_ir', 1e-3))
rock_gray_p_match_bar = float(config['boundary_condition'].get('rock_gray_p_match_bar', 1.0))
rock_gray_p_match = rock_gray_p_match_bar * 1e6

# ------------------ EXTRAPOLATED ENDPOINT (stellar irradiation penetration) ----- #
extrap_endpoint = config['boundary_condition'].getboolean('extrap_endpoint', fallback=False)
ee_kappa_v = float(config['boundary_condition'].get('ee_kappa_v', 0.01))
ee_mu = float(config['boundary_condition'].get('ee_mu', 0.5))
ee_F_inc_override = float(config['boundary_condition'].get('ee_F_inc_override', -1.0))
ee_apply_theta = config['boundary_condition'].getboolean('ee_apply_theta', fallback=True)

# ------------------ CORE PARAMETERS --------------------------------- #
mc = float(config['core']['mass_core']) # in earth masses
mantle_comp = str(config['core']['mantle_comp']) # either mgsio3, mg2sio4, or h2o
try:
    mantle_thermal_conductivity = float(config['core']['mantle_thermal_conductivity'])
except (KeyError, ValueError):
    mantle_thermal_conductivity = None
core_thermal_conductivity = float(config['core']['core_thermal_conductivity'])
viscous_mantle_convection = config['core'].getboolean('viscous_mantle_convection')
mantle_convection = config['core'].getboolean('mantle_convection')
iron_core_convection = config['core'].getboolean('iron_core_convection')

# ------------------ TRANSPORT PARAMETERS ---------------------------- #
convection = config['transport'].getboolean('convection')
semiconvection = config['transport'].getboolean('semiconvection')
# eps_S accounting fix (2026-07, opt-in): if True, the heat terms of the
# entropy residual (flux divergence + composition dU terms) are metered at the
# time-centered temperature T_old + (dt/2)*(dT/dt)_prev — the midpoint
# predicted from the previous accepted step's actual temperature rate —
# instead of the legacy old-state T. Second-order heat metering in time;
# removes the transport closure defect eps_S (~ +1.6e-3 * E_rad
# over-extraction during cooling).
# Default False keeps the historical behavior bit-identical. Changes results
# slightly when True (cooling history shifts at the 1.6e-3 * E_rad level).
# See writeups/energy_conservation/energy_conservation_methods.tex, Sec. 6.
time_centered_heat_factor = config['transport'].getboolean(
    'time_centered_heat_factor', fallback=False)
if semiconvection:
    logger.warning(
        "WARNING: Semiconvection is still experimental in ORCHARD and the code will likely crash. "
        "It is advised to not use semiconvection for now."
    )
sc_zones_min = int(config['transport'].get('sc_zones_min', 3))
radiation = config['transport'].getboolean('radiation')
# Early-time radiative ramp timescale [Gyr]: Lambda_rad is scaled by
# (2/pi)*arctan(t/rad_ramp_gyr) to keep radiation from dominating hot
# early envelopes.  <= 0 disables the ramp (full-strength radiation from
# t=0) -- required for cold, stably-stratified outer envelopes (e.g.
# inhomogeneous U/N models), where a throttled Lambda_rad leaves the
# surface with no heat channel once convection dies and forces the solver
# into unphysical inverted-S states.
rad_ramp_gyr = float(config['transport'].get('rad_ramp_gyr', 0.5))
opac = config['transport'].get('opac', fallback='default')
radioactive = config['transport'].getboolean('radioactive')
losses = config['transport'].getboolean('losses')
tol = float(config['transport']['tol_NR'])
debug_transport_NR = config['transport'].getboolean('debug_transport_NR', fallback=False)
R_rho = get_diffusion_param('R_rho')
Nu_T_val = float(config['transport']['Nu_T'])
Nu_X_val = float(config['transport']['Nu_X'])
Pr_val = float(config['transport']['Pr'])
tau_val = float(config['transport']['tau'])
DY_micro_val = get_diffusion_param('Dmicro_y')
DZ_micro_val = get_diffusion_param('Dmicro_z')
DZ_micro_val_in = get_diffusion_param('Dmicro_z_in')
# High-Z sampling cap for EOS composition derivatives (see the Zold_d block
# in update_SYZ): dS/dZ|_PT diverges beyond Z ~ 0.98; derivatives are
# evaluated at min(Z, cap).  Set to 1.0 to disable.
eos_deriv_z_cap = config['transport'].getfloat('eos_deriv_z_cap', fallback=0.97)

_jsp_offsets = None   # cached diagonal offsets of the banded transport J


def _csc_from_banded(J):
    """Bit-identical fast path for csc_array(J) on the banded transport
    Jacobian.  csc_array(dense) scans all (3N)^2 entries every Newton
    iteration (~7 ms at N=600, ~30% of a step); the J is banded, so
    extracting the cached nonzero diagonals gives the same canonical CSC
    (same entries, same values, tocsc canonical ordering -> identical
    spsolve result) in ~1 ms.  A per-call count check detects any pattern
    drift (e.g. physics terms switching on) and falls back to the full
    scan, re-deriving the offset cache -- so the result is ALWAYS exactly
    the canonical sparse form of J."""
    global _jsp_offsets
    total = np.count_nonzero(J)
    if _jsp_offsets is not None:
        rows_l, cols_l, data_l, cnt = [], [], [], 0
        for off in _jsp_offsets:
            d = np.diagonal(J, off)
            nz = np.flatnonzero(d)
            if nz.size:
                r = nz if off >= 0 else nz - off
                rows_l.append(r)
                cols_l.append(r + off)
                data_l.append(d[nz])
                cnt += nz.size
        if cnt == total:
            rows = np.concatenate(rows_l) if rows_l else np.zeros(0, dtype=np.intp)
            cols = np.concatenate(cols_l) if cols_l else np.zeros(0, dtype=np.intp)
            data = np.concatenate(data_l) if data_l else np.zeros(0)
            return csc_array(coo_array((data, (rows, cols)), shape=J.shape))
    r, c = np.nonzero(J)
    _jsp_offsets = np.unique(c - r)
    return csc_array(coo_array((J[r, c], (r, c)), shape=J.shape))


pressure_dependent_Dmicro = getbool_diffusion_param(
    'pressure_dependent_Dmicro', fallback=False
)
y_dependent_conductivity = getbool_diffusion_param(
    'y_dependent_conductivity', fallback=False
)
# f_rock-weighted envelope metal conductivity: blend the water proxy with the
# silicate mantle model so the metal conduction channel recovers rocky
# conductivities in the f_rock -> 1 limit.  Default False = legacy water-only
# metal term (bit-identical).
env_metal_cond_frock = getbool_diffusion_param(
    'env_metal_cond_frock', fallback=False
)
# Dimensionless enclosed-mass threshold for stronger interior Z diffusion.
# Using m/M avoids unit bugs from comparing grams to dimensionless numbers.
z_diff_inner_mfrac = get_diffusion_param('z_diff_inner_mfrac', default=0.7)
hinge_smooth = config['transport'].getboolean('hinge_smooth')
smooth_func = str(config['transport']['hinge_func'])
hinge_k = float(config['transport'].get('hinge_k', 1e7))
flux_tolerance = float(config['transport']['tol_flux'])  # tolerance for flux convergence if High Precision is enabled
latent_heat_effect = config['transport'].getboolean('latent_heat_effect')
latent_heat_factor = float(config['transport']['latent_heat_factor'])
latent_delta_t = float(config['transport'].get('latent_delta_t', 300.0))
t_chi_relax = float(config['transport'].get('t_chi_relax', config['diffusion'].get('t_Db_relax', 0.0))) * const.Myr_to_s
chi_transport_relax = config['transport'].getboolean('chi_transport_relax', fallback=True)
implicit_viscous_transition = config['transport'].getboolean('implicit_viscous_transition', fallback=True)
eddy_overshoot = config['transport'].getboolean('eddy_overshoot', fallback=False)
eddy_overshoot_factor = float(config['transport'].get('eddy_overshoot_factor', 0.1))
eddy_overshoot_fhp = float(config['transport'].get('eddy_overshoot_fhp', 1.0))
eddy_overshoot_lambda_cap = float(config['transport'].get('eddy_overshoot_lambda_cap', 10.0))
smooth_ledoux = getbool_diffusion_param('smooth_ledoux', fallback=False)
smooth_ledoux_n = get_diffusion_param('smooth_ledoux_n', default=2.0)

# --- Rayleigh-Taylor guard (default False = exact no-op) -------------------
# At envelope faces where the REALIZED density profile is inverted (deeper
# cell LESS dense than the cell above it), composition gradients cannot
# physically stabilize the stratification: a real fluid overturns on a
# dynamical timescale regardless of what the linearized Ledoux bracket says.
# Discovered on the sat3 Saturn cold-start board (2026-07-21): Y ~ 0.6
# He-rain shells parked above hot Z-deposit shelves held RT-inverted
# density profiles for Gyr while the saved N^2 stayed >= 0 (the bracket's
# dS/dY, dS/dZ terms read the composition gradients as stabilizing) and
# D sat at the microscopic floor.  With rt_guard = True the composition
# derivatives dSdY/dSdZ (both Ledoux rho-P and Schwarzschild P-T variants)
# are zeroed at RT-inverted faces, reducing the bracket to its thermal
# drive there, so MLT convection/mixing turns on and overturns the layer.
# The mask is built once per update_SYZ call from the step-start rho
# (frozen through the NR solve -- the lagged-coefficient pattern, no new
# Jacobian terms), and zeroing the shared derivative arrays keeps the
# residual, Jacobian, and R0_inv diagnostics mutually consistent.
rt_guard = config['transport'].getboolean('rt_guard', fallback=False)
_rt_guard_ncalls = 0   # throttled-logging counter
# Treat the surface-BC Jacobian coupling (dTint/dS, dTint/dY, dTint/dZ) as
# lagged/Picard: the surface-loss residual uses the step-start T_int (explicit
# within each transport solve), so the consistent Jacobian contribution is
# zero. The coupling is a normally-helpful semi-implicit acceleration, but it
# can steer the Newton trajectory into unrecoverable surface states when the
# outer zone thermally decouples (e.g. radiation = False ice giants at the
# surface-convection shutoff). False (default) keeps the accelerated implicit
# coupling; True removes it (exact residual, zero surface J coupling), which
# also reproduces the solver behavior of pre-2026-07-22 runs, where a
# pressure-interpolation bug zeroed the coupling unintentionally.
lag_bc_jacobian = config['transport'].getboolean('lag_bc_jacobian', fallback=False)
# Smooth convective criterion: blends D_mlt and D_micro via a sigmoid of the
# raw (pre-softhinge) bracket, eliminating the hard Schwarzschild/Ledoux step
# that creates composition-staircase artifacts.  See smooth_conv_indicator()
# helper and parameter_descriptions.md for the full formula.
smooth_convection_criterion = config['transport'].getboolean('smooth_convection_criterion', fallback=False)
smooth_conv_k = float(config['transport'].get('smooth_conv_k', 10.0))
smooth_conv_scale = float(config['transport'].get('smooth_conv_scale', 0.0))
# Gaussian-smoothing width (cells) applied to bracket_raw_b before the sigmoid is taken,
# so adjacent cells get similar sigma_conv values and the staircase pattern is broken.
# 0 disables spatial smoothing.
smooth_conv_bracket_sigma = float(config['transport'].get('smooth_conv_bracket_sigma', 2.0))
# Spatial smoothing window for D_b to suppress grid-scale oscillations.
# 0 = off; positive integer = half-width of running geometric mean (e.g., 3
# means a 7-point window: 3 neighbors on each side).
smooth_D_spatial = int(config['transport'].get('smooth_D_spatial', 0)) # RTA: For future debugging use (already applied in other branches)
# Use local pressure scale height (P/rho/g) instead of constant Hr for the
# mixing-length and rain advection scale height in the envelope.
use_local_Hp = config['transport'].getboolean('use_local_Hp', fallback=False)
# Gaussian smoothing of dS/dY, dS/dZ EOS derivatives to suppress grid-scale noise
smooth_eos_derivs = getbool_diffusion_param('smooth_eos_derivs', fallback=False)
smooth_eos_sigma = get_diffusion_param('smooth_eos_sigma', default=3.0)  # sigma in grid cells
# Convective flux time-relaxation in rain region to damp onset oscillations
flux_relax = config['transport'].getboolean('flux_relax', fallback=False)
flux_relax_tau = float(config['transport'].get('flux_relax_tau', 2.0))  # relaxation time in units of t_Db_relax
semi_shoot = config['transport'].getboolean('semi_shoot', fallback=False)
semi_shoot_len_env = int(config['transport'].get('semi_shoot_len_env', 4))
semi_shoot_len_mantle = int(config['transport'].get('semi_shoot_len_mantle', 0))
semi_shoot_fhp = float(config['transport'].get('semi_shoot_fhp', 1.0))
semi_shoot_cap_frac = float(config['transport'].get('semi_shoot_cap_frac', 1.0))

# ---- Semiconvection closure model selection (gated by the same master
# `semiconvection` boolean).  'nusselt' = legacy constant-Nu chi ramp (exact
# pre-existing behavior); 'spruit' = Spruit (2013) / Fuentes et al. (2026)
# layered-convection closure: Nu_T = 1 + tau^(-1/2)/R_rho and
# D_eff = kappa_T * tau^(1/2)/R_rho in the window 1 < R_rho < tau^(-1/2)
# (R_rho is the code's R0_inv; no layered state exists above the ceiling).
semi_model = config['transport'].get('semi_model', fallback='nusselt').strip().lower()
if semi_model not in ('nusselt', 'spruit'):
    raise ValueError(
        f"[transport] semi_model must be 'nusselt' or 'spruit', got '{semi_model}'")
spruit_edge_frac = float(config['transport'].get('spruit_edge_frac', 0.1))
spruit_tau_local = config['transport'].getboolean('spruit_tau_local', fallback=False)
spruit_nu_x_max = float(config['transport'].get('spruit_nu_x_max', 1e10))
if semiconvection and semi_shoot and semi_model == 'spruit':
    raise ValueError(
        "semi_shoot=True is incompatible with semi_model='spruit' (the semi_shoot "
        "branch would silently bypass the Spruit targets); use semi_model='nusselt' "
        "with semi_shoot.")
tab = config['equation_of_state'].getboolean('eos_tab')

# ------------------ DIFFUSION PARAMETERS ---------------------------- #
helium_rain = config['diffusion'].getboolean('helium_rain')
metal_rain = config['diffusion'].getboolean('metal_rain')
# Kind of Z (metal) rain. zmisc_hhe = True (default) uses the MODIFIED H-He
# miscibility diagram as a stand-in for H-Z miscibility (the only metal-rain
# mode currently available — a numerical test, not a physical curve). When
# False, the physical curve selected by misc_kind is used: 'water' (Gupta)
# or 'mgsio3' (Gilmore/silicate). The zmisc_deltaT/zmisc_deltaP offsets only
# apply in the H-He stand-in mode.
zmisc_hhe = config['diffusion'].getboolean('zmisc_hhe', fallback=True)
misc_kind = str(config['diffusion']['misc_kind'])
if zmisc_hhe:
    misc_kind = 'hhe'
if metal_rain and misc_kind != 'hhe':
    raise RuntimeError(
        "metal_rain with a physical Z miscibility curve (water/silicate) is an "
        "experimental feature and is not yet available in this version of ORCHARD. "
        "Tejada Arevalo et al. (2026b) included hydrogen-rock phase separation only. "
        "Future versions of ORCHARD will include both hydrogen-rock and hydrogen-water miscibility. "
        "For testing, set zmisc_hhe = True (the default) to use the H-He miscibility curve "
        "as a stand-in for H-Z miscibility."
    )
composition_change = config['diffusion'].getboolean('composition_change')
evol_Z = config['diffusion'].getboolean('evol_Z')
D_MLT = config['diffusion'].getboolean('D_MLT')
delta_T = float(config['diffusion']['misc_deltaT'])
# Pressure offset for the miscibility curve [Mbar]: shifts the curve to
# higher pressure (deeper in the planet), mirroring the misc_deltaT
# temperature offset. Stored internally in dyn/cm^2.
delta_P = float(config['diffusion'].get('misc_deltaP', 0.0)) * 1e12
# Separate offsets for the metal-rain channel when misc_kind='hhe' uses the
# H-He curve as the H-Z miscibility stand-in. They default to the He-channel
# values, but must be set independently when helium rain and the Z stand-in
# run together (He needs its physical calibration, e.g. deltaT~410, deltaP=0,
# while the Z stand-in needs deltaT~12000-14000, deltaP~4-6 Mbar).
z_delta_T = float(config['diffusion'].get('zmisc_deltaT', delta_T))
z_delta_P = float(config['diffusion'].get('zmisc_deltaP', delta_P / 1e12)) * 1e12
# Deep-pressure cap on the Z-rain activation window [Mbar]. 0 = off (legacy:
# window extends to P[innermisc_index]). When set, cells with P > zmisc_p_max
# are rain-inert for the Z channel: the stand-in miscibility curve is not
# meant to apply at envelope-base depths, and capping keeps envelope-base
# depths out of the activation window.
z_p_max = float(config['diffusion'].get('zmisc_p_max', 0.0)) * 1e12
Hr = float(config['diffusion']['Hr_y'])
# Diffusion suppression strength in rain zone.
# Higher values suppress convective composition diffusion more aggressively
# where Y exceeds the miscibility curve, letting advective sedimentation
# dominate. Default 0.0 = suppression off (backward-compatible for runs
# that do not set this parameter).
rain_conv_supp_k = float(config['diffusion'].get('rain_conv_supp_k', 0.0))
alpha_herain = float(config['diffusion'].get('alpha_herain', 1.0))
misc_threshold = float(config['diffusion']['innermisc_thesh'])
misc_curve = str(config['diffusion']['misc_curve'])
# misc_kind is resolved above (overridden to 'hhe' when zmisc_hhe = True).
pmisc_min = float(config['diffusion']['pmisc_min']) # minimum pressure to cap misc zone in GPa
weight = float(config['diffusion']['under_relax'])
t_Db_relax = float(config['diffusion']['t_Db_relax']) * const.Myr_to_s
# Opt-in hull clamp for the Mobius D relaxation (see relax_D_safely):
# result-changing pole-artifact fix.
relax_hull_clamp = config['diffusion'].getboolean('relax_hull_clamp',
                                                  fallback=False)
max_Db_change = float(config['diffusion'].get('max_Db_change', 1e6))
ym_t_lag = float(config['diffusion']['t_Db_relax']) * const.Myr_to_s
zm_t_lag = ym_t_lag
# Raw t_Db_relax in Myr: the flux_relax branch rebuilds tau_relax_s as
# flux_relax_tau * t_Db_relax_myr * Myr_to_s -- keep that exact multiply order
# (substituting the pre-scaled t_Db_relax above would regroup the product and
# perturb the float).
t_Db_relax_myr = float(config['diffusion']['t_Db_relax'])
# Rain activation smoothing constants (hoisted out of the Newton loop --
# previously re-parsed on every Jacobian evaluation).
rain_softplus_beta = float(config['diffusion'].get('rain_softplus_beta', 1e6))
rain_sign_eps = float(config['diffusion'].get('rain_sign_eps', 1e-12))
lam_factor = float(config['diffusion']['lam_factor'])
water_cond_extrapolation_mode = int(config['diffusion'].get('water_cond_extrapolation_mode', 1))
set_french19_extrapolation_mode(water_cond_extrapolation_mode)

# NOTE (2026-07, deliberate): the misc_kind == 'water' path below is a
# RESERVED CAPABILITY for future water phase separation, not dead code.
# In this ORCHARD version zmisc_hhe is always True, which forces
# misc_kind = 'hhe', so the water arms (here and in the Jacobian's
# z_advection_local branches) are unreachable by any released config.
# They are intentionally kept and must not be removed in cleanups
# (author decision: DEFER, prep for the future).
if misc_kind == 'water':
    from misc import gupta_misc_lowT as zmisc
elif misc_kind == 'hhe':
    # Test mode: the H-He miscibility curve (selected by misc_curve, e.g.
    # Lorenzen) stands in for H-Z miscibility. Z1m takes the role of Ym and
    # Z2m is pinned at 1, so metals rain wherever Z exceeds the H-He curve.
    # No separate zmisc module is needed.
    zmisc = None
else:
    from misc import gilmore_misc as zmisc

# Planet-specific constants
# if BC_atm in ('rock_gray', 'ideal_gray') and planet != 'Super_Earth':
#     raise ValueError("BC_atm='rock_gray' and 'ideal_gray' are only implemented for planet='Super_Earth'.")

if planet=="Jupiter":
    M0 = mjup
    R0 = R_jup
elif planet=="Saturn":
    M0 = msat
    R0 = rsat

elif planet=="Uranus":
    M0 = 14.57*const.mearth
    R0 = 4.007*const.rearth

elif planet=="Neptune":
    M0 = 17.15*const.mearth
    R0 = 3.883*const.rearth

elif planet=="Sub_Neptune":
    M0 = 5.0*const.mearth
    R0 = 3.0*const.rearth

elif planet=="Super_Earth":
    M0 = 3.0*const.mearth
    R0 = 1.5*const.rearth

elif planet=='Super_Jupiter':
    M0 = M_planet
    R0 = R_jup * (M_planet / mjup)**(1/3)

S0 = norm   # scaling factor for entropy
T0 = 1000   # typical temperature scale
D_b_first = None


# ------------------ Initialize BC (Atmosphere) Object --------------- #
# One module-level object for the selected BC (see atm_bc.create_atm); the
# BC-derivative dispatch inside update_SYZ reads it directly.
atm = atm_bc.create_atm(
    BC_atm, planet,
    cloud=cloud, irradiation=irradiation,
    T_eq=T_eq, bond_albedo=bond_albedo,
    rock_gray_kappa_ir=rock_gray_kappa_ir,
    rock_gray_p_match=rock_gray_p_match,
)

def get_Lambda_cond_water(rho_val, T_val):
    return (Lambda_e(rho_val, T_val) + Lambda_p(rho_val, T_val))*(u.W/u.m/u.K).to('erg/(s * cm * K)') # in CGS units...



# Radioactive decay of rock
#radioactive_power = (2.7e14*u.J/u.s/u.Mearth).to('erg/(s*g)').value # from Nettelmann et al. (2011)

# Envelope opacity table(s) for radiative transport (selected by [transport] opac).
if radiation:
    if opac == 'default':
        # 100x100 (logRho, logT) -> log10(kappa_R) at 3.16x solar metallicity.
        opa = np.genfromtxt('opacities/ross.3.16.default.60')
        xgrid = opa[:, 0].reshape(100, 100)
        ygrid = opa[:, 1].reshape(100, 100)
        zgrid = opa[:, 2].reshape(100, 100)
        xgrid_arr = np.log10(xgrid[:, 0])
        ygrid_arr = np.log10(ygrid[0, :])
        k_interp = RGI(
            (xgrid_arr, ygrid_arr), np.log10(zgrid),
            method='linear', bounds_error=False, fill_value=None,
        )
    elif opac == 'water_rich':
        # water_rich_opac is lazy-imported at the use site; nothing to preload.
        pass
    else:  # opac == 'metal_rich'
        try:
            OPACITY_DIR = Path(__file__).resolve().parent / 'opacities'
            data_30zsol = np.load(OPACITY_DIR / 'opac_means_30zsol.npz', allow_pickle=True)
            logrho_30zsol = data_30zsol['logRho']           # shape (nrho,)
            logt_30zsol   = data_30zsol['logT']             # shape (ntemp,)
            opac_30zsol   = data_30zsol['rosseland_log10']  # shape (nrho, ntemp)

            zsol_grid = [5, 10, 30, 50, 100]

            opac_grid = []
            for zsol_f in zsol_grid:
                data = np.load(OPACITY_DIR / f'opac_means_{zsol_f}zsol.npz', allow_pickle=True)
                opac_grid.append(data['rosseland_log10'])

            interp = RGI((zsol_grid, logrho_30zsol, logt_30zsol), opac_grid, method='linear', bounds_error=False, fill_value=None)
        except FileNotFoundError:
            raise FileNotFoundError(
                "Metal-rich opacity files not found in opacities/. "
                "Either provide opac_means_*zsol.npz or set [transport] opac = default."
            )

def get_rosseland_opacity(rho, T, fzsol):
    logk = interp((fzsol, np.log10(rho), np.log10(T)))
    return 10.0**logk

max_itr = 150

# Flag to only run FD Jacobian check once
_fd_jac_check_done = False
_ddc_log_prev = None    # (raw_count, active_count) at last DDC WINDOW print


class TransportResult(NamedTuple):
    """Converged state and diagnostics returned by update_SYZ for one timestep.

    A NamedTuple IS a tuple, so positional unpacking of the historical
    32-element return keeps working; evolution.py reads fields by name.
    Four fields are produced but not consumed by evolution.py and are kept
    for debugging/API stability: latent_flux (currently a zeros
    placeholder), phi, J (the full 3N x 3N Jacobian), and B (the residual).
    """
    S: object                # entropy [kb/baryon], cell-centered (N)
    Y: object                # helium mass fraction (N)
    Z: object                # heavy-element mass fraction (N)
    f_rock: object           # rock fraction of Z (N) — pass-through
    loss_energy: object      # energy radiated across the top boundary [erg]
    D2: object               # convective (MLT) diffusivity diagnostic (N+1)
    D_b: object              # He diffusion coefficient at boundaries (N+1)
    DZ_b: object             # Z diffusion coefficient at boundaries (N+1)
    total_flux: object       # total heat flux (N+1), cgs
    conv_flux: object
    rad_flux: object
    radiogenic_flux: object
    semi_flux: object
    latent_flux: object      # zeros placeholder — unused downstream
    Lambda: object           # total conductivity at boundaries
    phi: object              # MLT flux prefactor — unused downstream
    J: object                # 3N x 3N Newton Jacobian — unused downstream
    B: object                # 3N Newton residual — unused downstream
    niter: object            # Newton iterations used
    R0_inv_b: object         # inverse density ratio (semiconvection)
    Lambda_cond_i: object
    Lambda_rad_i: object
    bracket_dsdr_b: object   # convective bracket (Ledoux) at boundaries
    Ym_i: object             # lagged He-miscibility profile
    Z1m_i: object            # lagged Z-miscibility profiles
    Z2m_i: object
    innermisc_index: object  # innermost miscibility-active zone index
    Cp_i: object             # heat capacity at interior boundaries
    chi_full: object         # melt fraction (mantle cells)
    y_rain_gap: object       # (P_outer, P_inner) of the He rain gap, or None
    Nu_T_eff_i: object       # relaxed semiconvective Nusselt numbers
    Nu_X_eff_i: object

def safe_newton_step(zold, dZ, base_weight, max_backtrack=10,
                     S_floor=1e-4, max_rel_dS=0.25,
                     max_rel_dY=0.5, max_rel_dZ=1.0):
    """
    Try a damped Newton step znew = zold - step * dZ with trust-region
    guards that prevent:
      1. Non-finite values (NaN / inf),
      2. Negative or near-zero entropy (S entries at indices ``::3``),
      3. Excessively large relative changes in S (which drive violent
         chi / viscosity swings near the melt front),
      4. Unphysical helium fraction (Y < 0 or Y > 1),
      5. Negative heavy-element fraction (Z < 0),
      6. Excessively large relative changes in Y or Z (which destabilize
         the solver during intense helium/metal rain).

    If any guard is violated the step size is halved and retried (up to
    *max_backtrack* times).

    Parameters
    ----------
    zold : ndarray
        Current state vector [S0, Y0, Z0, S1, Y1, Z1, ...].
    dZ : ndarray
        Newton correction vector (same layout as *zold*).
    base_weight : float
        Starting damping factor (0 < base_weight <= 1).
    max_backtrack : int
        Maximum number of halvings before giving up.
    S_floor : float
        Minimum acceptable value for any entropy entry.
    max_rel_dS : float
        Maximum allowed ``|ΔS_i| / max(|S_i|, S_floor)`` across all
        cells in a single Newton update.  Exceeding this triggers a
        backtrack.  A value of 0.25 limits entropy changes to ~25 % per
        Newton iterate, which keeps trial temperatures (and therefore
        chi) from jumping wildly.
    max_rel_dY : float
        Maximum allowed relative change in Y per iterate (default 0.5).
    max_rel_dZ : float
        Maximum allowed relative change in Z per iterate (default 1.0).

    Returns
    -------
    (znew, effective_step, applied_step_vector)
    """
    Y_floor = 1e-6
    Z_floor = 1e-10

    step = base_weight
    _last_guard = None
    for _bt in range(max_backtrack):
        step_solve = step * dZ
        zcandidate = zold - step_solve

        # --- Guard 1: finite-ness ---
        if not np.all(np.isfinite(zcandidate)):
            _last_guard = 1; step *= 0.5; continue

        # --- Guard 2: positive entropy ---
        S_candidate = zcandidate[::3]
        if np.any(S_candidate < S_floor):
            _last_guard = 2; step *= 0.5; continue

        # --- Guard 3: relative entropy change cap ---
        S_old = zold[::3]
        dS = step_solve[::3]
        denom_S = np.maximum(np.abs(S_old), S_floor)
        _rel_dS = np.abs(dS) / denom_S
        if np.max(_rel_dS) > max_rel_dS:
            _last_guard = 3; step *= 0.5; continue

        # --- Guard 4: physical Y bounds ---
        Y_candidate = zcandidate[1::3]
        if np.any(Y_candidate < 0.0) or np.any(Y_candidate > 1.0):
            _last_guard = 4; step *= 0.5; continue

        # --- Guard 5: non-negative Z ---
        # Allow tiny negative values within numerical noise (Z_floor),
        # then clip to zero.  The sparse LU solve can introduce O(1e-12)
        # fill-in that pushes Z slightly below zero when Z_old ≈ 0.
        Z_candidate = zcandidate[2::3]
        if np.any(Z_candidate < -Z_floor):
            _last_guard = 5; step *= 0.5; continue
        zcandidate[2::3] = np.maximum(Z_candidate, 0.0)

        # --- Guard 6: relative Y change cap ---
        Y_old = zold[1::3]
        dY = step_solve[1::3]
        denom_Y = np.maximum(np.abs(Y_old), Y_floor)
        _rel_dY = np.abs(dY) / denom_Y
        if np.max(_rel_dY) > max_rel_dY:
            _last_guard = 6; step *= 0.5; continue

        # --- Guard 7: relative Z change cap ---
        Z_old = zold[2::3]
        dZc = step_solve[2::3]
        denom_Zc = np.maximum(np.abs(Z_old), Z_floor)
        _rel_dZc = np.abs(dZc) / denom_Zc
        if np.max(_rel_dZc) > max_rel_dZ:
            _last_guard = 7; step *= 0.5; continue

        return zcandidate, step, step_solve

    # --- Diagnostic: which guard killed us and where ---
    _gnames = {1:'non-finite', 2:'S<floor', 3:'rel_dS',
               4:'Y bounds', 5:'Z<0', 6:'rel_dY', 7:'rel_dZ'}
    _msg = (f"safe_newton_step: guards not satisfied after backtracking. "
            f"Last guard={_last_guard} ({_gnames.get(_last_guard,'?')}), "
            f"final step={step:.3e}")
    if _last_guard == 3:
        _w = int(np.argmax(_rel_dS))
        _msg += f", worst S zone={_w}, rel_dS={_rel_dS[_w]:.4e}, S_old={S_old[_w]:.6e}, dS={dS[_w]:.6e}"
    elif _last_guard == 5:
        _neg = np.where(Z_candidate < 0)[0]
        _msg += f", Z<0 at zones {_neg[:5].tolist()}, min_Z={np.min(Z_candidate):.6e}"
    elif _last_guard == 7:
        _w = int(np.argmax(_rel_dZc))
        _msg += f", worst Z zone={_w}, rel_dZ={_rel_dZc[_w]:.4e}, Z_old={Z_old[_w]:.6e}, dZc={dZc[_w]:.6e}"
    elif _last_guard == 6:
        _w = int(np.argmax(_rel_dY))
        _msg += f", worst Y zone={_w}, rel_dY={_rel_dY[_w]:.4e}"
    elif _last_guard == 2:
        _neg_S = np.where(S_candidate < S_floor)[0]
        _msg += f", S<floor at zones {_neg_S[:5].tolist()}, min_S={np.min(S_candidate):.6e}"
    elif _last_guard == 1:
        _bad = np.where(~np.isfinite(zcandidate))[0]
        _msg += f", non-finite at raw indices {_bad[:5].tolist()}"
    elif _last_guard == 4:
        _bad_Y = np.where((Y_candidate < 0) | (Y_candidate > 1))[0]
        _msg += f", Y out of bounds at zones {_bad_Y[:5].tolist()}"
    print(_msg)
    raise RuntimeError(_msg)

def relax_D_safely(D_y0, D_A, D_B, exp_arg, dt, tau, D_floor=0.0,
                   apply_hull=False):
    """
    Robust update for the 'logistic/Mobius' closed-form relaxation:
        w = ((D - A)/(D - B))
        w(t+dt) = w(t) * exp(exp_arg)
        D = (B*w - A)/(w - 1)

    BUGFIXES:
      - protects divisions by ~0
      - clamps exp() to avoid overflow
      - avoids catastrophic cancellation when w ~ 1
      - falls back to stable exponential blending when ill-conditioned
    """
    # Ensure arrays (works for scalars too)
    D_y0 = np.asarray(D_y0, dtype=float)
    D_A  = np.asarray(D_A,  dtype=float)
    D_B  = np.asarray(D_B,  dtype=float)

    # --- BUGFIX: stable fallback (always safe) ---
    # Blend from old -> target (choose which one is your "target"; here we assume D_B is the target)
    # If your target is D_A instead, just swap D_B -> D_A in the blend.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        blend = D_B + (D_y0 - D_B) * np.exp(-dt / max(tau, 1e-99))

    # --- BUGFIX: protect (D_y0 - D_B) in the w0 definition ---
    eps = 1e-30
    denom0 = D_y0 - D_B
    denom0 = np.where(np.abs(denom0) < eps * (1.0 + np.abs(D_B)), np.sign(denom0) * eps, denom0)

    w0 = (D_y0 - D_A) / denom0

    # --- BUGFIX: clamp exp argument to avoid overflow ---
    # exp(±700) is near the largest finite float64 range before inf.
    exp_arg_clamped = np.clip(exp_arg, -700.0, 700.0)
    ew = np.exp(exp_arg_clamped)

    w = w0 * ew

    # --- BUGFIX: avoid division by ~0 when w ~ 1 ---
    denom1 = w - 1.0
    denom1_small = np.abs(denom1) < 1e-12  # threshold can be tuned

    # Closed-form candidate
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        D_closed = (D_B * w - D_A) / denom1

    # --- BUGFIX: replace bad points with fallback blend ---
    bad = (~np.isfinite(D_closed)) | denom1_small | (~np.isfinite(w)) | (~np.isfinite(w0))
    D_out = np.where(bad, blend, D_closed)

    # --- Hull clamp (flag-gated; [diffusion] relax_hull_clamp): the
    # underlying logistic ODE interpolates between the old value and the
    # A/B fixed points and can never leave their hull -- any excursion is
    # the closed form's pole artifact (|w - 1| small but far above the
    # 1e-12 fallback window; with a log-spread of ~38 between a dead face
    # at the floor and a live 1e15 target, D_closed reaches ~+4e4 in log
    # space and the downstream exp() overflows to inf).
    # Gated because finite out-of-hull excursions DO occur on healthy
    # production paths (bitcheck-verified): the clamp is a result-changing
    # correction, opt-in to preserve champion bit-reproducibility.
    if apply_hull:
        _hull_lo = np.minimum(np.minimum(D_y0, D_A), D_B)
        _hull_hi = np.maximum(np.maximum(D_y0, D_A), D_B)
        D_out = np.clip(D_out, _hull_lo, _hull_hi)

    # Enforce floor (diffusivity should not be negative)
    if D_floor is not None:
        D_out = np.maximum(D_out, D_floor)

    return D_out


def fill_nonfinite_linear(arr, valid_slice):
    """
    Fill non-finite values in ``arr[valid_slice]`` using:
      - linear interpolation for interior gaps
      - linear extrapolation for edge gaps

    The helper intentionally leaves arrays untouched when there are no finite
    anchor points in the segment so existing safety assertions can still fail.
    """
    segment = np.asarray(arr[valid_slice], dtype=float).copy()
    bad = ~np.isfinite(segment)
    if not np.any(bad):
        return

    good_idx = np.flatnonzero(np.isfinite(segment))
    if good_idx.size == 0:
        return

    if good_idx.size == 1:
        segment[bad] = segment[good_idx[0]]
    else:
        interp_fn = interp1d(
            good_idx.astype(float),
            segment[good_idx],
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
            assume_sorted=True,
        )
        bad_idx = np.flatnonzero(bad).astype(float)
        segment[bad] = interp_fn(bad_idx)

    arr[valid_slice] = segment


def smooth_conv_indicator(bracket_raw_b, scale=None, k=10.0):
    """
    Smooth Schwarzschild/Ledoux convective indicator on zone boundaries.

        sigma_conv_b = 1 / (1 + exp(-k * bracket_raw_b / scale))

    Used to BLEND D_mlt and D_micro continuously instead of the hard
    "if bracket > 0 then D_mlt else D_micro" step the rest of the code
    performs.  Operates on the RAW (pre-softhinge) bracket so the indicator
    spans the full negative-to-positive range:
      - bracket_raw << -scale: sigma -> 0 (fully Ledoux-stable, D = D_micro)
      - bracket_raw =  0     : sigma = 0.5 (marginal, half D_mlt + half D_micro)
      - bracket_raw >>  scale: sigma -> 1 (fully convective, D = D_mlt + D_micro)
    In our standard transport.py block, D_b ends up being
        D_b = sigma_conv_b * D_mlt_b + D_micro_b
    so cells in the staircase-gradient region get graded mixing rather than
    binary on/off.

    Parameters
    ----------
    bracket_raw_b : ndarray, shape (N+1,)
        The PRE-softhinge bracket (can be negative in stable regions).
    scale : float or None
        Bracket scale that sets the sigmoid width.  If None or <= 0,
        auto-calibrated as the 25th percentile of the positive bracket
        distribution (per call).
    k : float
        Sharpness of the sigmoid.  Large k recovers a hard step; small k is
        very gradual.  Default 10.

    Returns
    -------
    sigma_conv_b : ndarray, shape (N+1,)
        Smooth conv indicator in [0, 1].
    dsigma_db_b : ndarray, shape (N+1,)
        Derivative d sigma_conv / d bracket_raw, useful for the Newton-Raphson
        Jacobian chain rule.  Shape (N+1,).
    """
    b = np.asarray(bracket_raw_b, dtype=float)
    if scale is None or scale <= 0.0:
        positive = b[b > 0.0]
        if positive.size > 0:
            sc = np.percentile(positive, 25)
            sc = max(sc, 1e-30)
        else:
            sc = 1.0
    else:
        sc = float(scale)

    # Numerically stable sigmoid: clip the exponent argument to avoid
    # overflow far from the transition (where the answer is 0 or 1 anyway).
    arg = np.clip(k * b / sc, -50.0, 50.0)
    sigma = 1.0 / (1.0 + np.exp(-arg))
    # Derivative: d sigma / d b = (k / sc) * sigma * (1 - sigma)
    dsigma_db = (k / sc) * sigma * (1.0 - sigma)
    return sigma, dsigma_db


def update_SYZ(
    T,
    r_b,
    P,
    rho,
    m_b,
    dm,
    T_int,
    dt,
    t,
    Yold,
    Zold,
    f_rock_old,
    Sold,
    kcore,
    kcore_fe,
    D_b_old,
    DZ_b_old,
    Ym_old_i,
    Z1m_old_i,
    Z2m_old_i,
    innermisc_index,
    chi_prev_full,
    y_rain_gap_old,
    Nu_T_eff_old,
    Nu_X_eff_old,
    omega_old=0.0,  # unused; retained for call-signature stability
    dTdt_prev=None,
    ):

    """
    Updates S, Y, Z profiles via a global Newton-Raphson approach that accounts
    for:
      - He-rain and metal-rain regions (miscibility constraints)
      - Advection and diffusion of Helium/Metals
      - Radiative, convective, and semiconvective heat transport
      - Mantle and Fe-core conduction (kcore, kcore_fe indexing)
      - Latent heat effects and radiogenic heating
      - Bare-rock models (kcore==0, no H/He envelope)

    Args:
        T (array): Temperature in each of the N zones (cell-centered).
        r_b (array): Radial boundaries (length N+1).
        P (array): Pressure in each zone (length N).
        rho (array): Density in each zone (length N).
        m_b (array): Mass array for zone boundaries (length N+1).
        dm (array): Mass in each cell (length N).
        T_int (float): Internal boundary condition temperature from the atmosphere.
        dt (float): Current timestep [s].
        t (float): Current global time [s].
        Yold (array): Previous Helium fraction in each zone (length N).
        Zold (array): Previous Metal fraction in each zone (length N).
        f_rock_old (array): Rock fraction in each zone (length N).
        Sold (array): Previous entropy in each zone (length N).
        kcore (int): Index separating envelope from mantle (cells [0, kcore) are envelope).
        kcore_fe (int): Index separating mantle from Fe core (cells [kcore, kcore_fe) are mantle).
        D_b_old (array): Old He diffusion coefficients across boundaries (length N+1).
        DZ_b_old (array): Old metal diffusion coefficients across boundaries (length N+1).
        Ym_old_i (array): Old He-rain saturation profile (averaged) for the
                         interior shells (length N-1).
        Z1m_old_i (array): Old metal-rain saturation profile 1 for the
                          interior shells (length N-1).
        Z2m_old_i (array): Old metal-rain saturation profile 2 for the
                          interior shells (length N-1).
        innermisc_index (int): Index for the inner boundary of miscibility region
                               inside the structure.
        chi_prev_full (array): Melt fraction from the previous timestep (length N).
        y_rain_gap_old (array or None): Previous-timestep He rain gap bounds in pressure.
        Nu_T_eff_old (array): Time-lagged thermal Nusselt number from the previous
                              timestep (length N-1).  Initialized to ones.
        Nu_X_eff_old (array): Time-lagged compositional Nusselt number from the
                              previous timestep (length N-1).  Initialized to ones.

    Returns:
        Tuple of updated arrays:
        (S_new, Y_new, Z_new, f_rock_old, loss_energy, D2_div, D_b, DZ_b,
         total_flux, conv_flux_b, rad_flux_b, radiogenic_flux_b, semi_flux_b,
         latent_flux_b, Lambda_b, phi_b, J, B, itr,
         R0_inv_b, Lambda_cond_i, Lambda_rad_i, bracket_dsdr_b,
         Ym_lag_i, Z1m_lag_i, Z2m_lag_i, innermisc_index,
         Cp_i, chi_curr_full, y_rain_gap_old_out,
         Nu_T_eff_i, Nu_X_eff_i)

        - S_new (array): Updated entropy (length N).
        - Y_new (array): Updated helium fraction (length N).
        - Z_new (array): Updated metal fraction (length N).
        - f_rock_old (array): Rock fraction, unchanged, returned for continuity (length N).
        - loss_energy (float): Surface luminosity [erg/s] from the transport solve.
        - D2_div (array): Advection/diffusion ratio for He.
        - D_b (array): New He diffusion coefficients at boundaries (length N+1).
        - DZ_b (array): New metal diffusion coefficients at boundaries (length N+1).
        - total_flux (array): Combined flux at boundaries (length N+1).
        - conv_flux_b (array): Convective flux at boundaries (length N+1).
        - rad_flux_b (array): Radiative flux at boundaries (length N+1).
        - radiogenic_flux_b (array): Radiogenic heating flux at boundaries (length N+1).
        - semi_flux_b (array): Semiconvective flux at boundaries (length N+1).
        - latent_flux_b (array): Latent heat flux at boundaries (length N+1).
        - Lambda_b (array): Conductivity + radiative conduction array at boundaries (length N+1).
        - phi_b (array): Mixing length factor at boundaries (length N+1).
        - J (2D array): Final Jacobian from the global Newton-Raphson (3N x 3N).
        - B (1D array): Final residual vector from the global Newton-Raphson (length 3N).
        - itr (int): Total number of global iterations used in the solver (includes pre-iterations and restarts).
        - R0_inv_b (array): 1 / R0 proxy for distinguishing semiconvective from stable layers (length N+1).
        - Lambda_cond_i (array): Conductive opacity at interior boundaries (length N-1).
        - Lambda_rad_i (array): Radiative opacity at interior boundaries (length N-1).
        - bracket_dsdr_b (array): Entropy gradient bracket at boundaries (length N+1).
        - Ym_lag_i (array): Time-lagged He miscibility in the interior (length N-1).
        - Z1m_lag_i (array): Time-lagged metal miscibility profile 1 (length N-1).
        - Z2m_lag_i (array): Time-lagged metal miscibility profile 2 (length N-1).
        - innermisc_index (int): Possibly updated index for the inner boundary
                                 of immiscibility region.
        - Cp_i (array): Heat capacity at interior boundaries (length N-1).
        - chi_curr_full (array): Updated melt fraction (length N).
        - y_rain_gap_old_out (array or None): Carried-forward He-rain gap bounds from the
          latest Newton evaluation. Two-element pressure bounds [P_outer, P_inner]
          where Y_i > Ym_lag_i, or None if no region is present.

    Notes:
        1) The function constructs a global system of equations combining
           heat transport (rad+conv+semi-conv) and composition transport
           (He + metals) into a single Newton-Raphson solve.
        2) The updated solution is returned to evolution.py, which calls
           hydrostatic_equilibrium for the next iteration.
        3) Variables are scaled to dimensionless forms inside the solver
           (e.g., T/T0, S*S0, etc.) for numerical stability.
        4) If 'composition_change' in config is False, S is updated ignoring
           composition effects on internal energy. If True, the chemical
           potentials (dUdY, dUdZ) are included.
        5) Mantle conduction uses mantle_eos; Fe-core conduction uses core_eos.
           The three-region structure (envelope / mantle / Fe core) is indexed
           by kcore and kcore_fe.
        6) For bare-rock models (kcore==0), all composition transport and
           envelope-specific physics are disabled; only mantle/core conduction
           and surface cooling are active.

        Variable convention: Three kinds: cell-centered, boundary-centered, and interior-boundary-centered
        cell-centered have N entries, no suffix
        boundary-centered values have N+1 entries, suffix "_b"
        interior-boundary-centered values have N-1 entries, suffix "_i"

        We want the first kcore cells to be envelope, so the first kcore-1 interior
        boundaries have convection (and so the first (kcore) entries in `_b` arrays).

    """
    # Sentinel normalization:
    #   - kcore=-1    => coreless (all cells are envelope)
    #   - kcore_fe=-1 => no Fe core (mantle extends to the center)
    # Convert to slice-safe indices locally.
    kcore_raw = int(kcore)
    kcore_fe_raw = int(kcore_fe)
    if kcore_raw < 0:
        kcore = N
        kcore_fe = N
    else:
        kcore = int(np.clip(kcore_raw, 0, N))
        if kcore_fe_raw < 0:
            kcore_fe = N
        else:
            kcore_fe = int(np.clip(kcore_fe_raw, kcore, N))
    if len(P) > 0:
        innermisc_index = int(np.clip(int(innermisc_index), 0, len(P) - 1))

    # --- safety check: catch NaNs before we even start NR ---
    try:
        assert_all_not_nan(Sold, Yold, Zold, T, P, rho)
    except AssertionError as e:
        raise RuntimeError(f"NaNs entering update_SYZ in previous step: {e}")

    # Carry-forward diagnostic state (updated each Jacobian evaluation).
    y_rain_gap_old_out = y_rain_gap_old

    # ------------------------------------------------------------------
    # Bare-rock support (kcore == 0)
    # ------------------------------------------------------------------
    # APPLE was originally written for an H/He envelope over a rocky/iron core.
    # In that original design "envelope" cells are [0, ..., kcore-1] and many
    # slices use the pattern `:kcore-1` for *envelope interior boundaries*.
    #
    # When kcore == 0 (true bare rock; no envelope), the Python slice `:kcore-1`
    # becomes `:-1` which silently selects "almost the whole array". That applies
    # H/He envelope physics (mixtures_eos, opacities, miscibility) to the mantle
    # and Fe core and can crash the Newton solve or, worse, produce unphysical
    # results without an obvious exception.
    #
    # The intent for bare rock is:
    #   - The surface is the top of the mantle (cell 0) and cools directly.
    #   - There is no Y/Z evolution (no helium or "trace metals" in a mantle).
    #   - mixtures_eos calls must be *envelope-only*.
    # ------------------------------------------------------------------
    has_envelope = (kcore > 0)
    is_bare_rock = (kcore == 0)

    # Enforce a consistent boundary condition choice for bare rock.
    if is_bare_rock and BC_atm not in ('bare', 'gray', 'ideal_gray'):
        raise ValueError(
            "Bare-rock structure detected (kcore==0 => no H/He envelope), "
            f"but bc_atm='{BC_atm}'. Set [boundary_condition]/bc_atm='bare', 'gray', or 'ideal_gray' "
            "to cool the surface directly."
        )
    if (not is_bare_rock) and BC_atm in ('bare'):
        # This is almost certainly a misconfiguration, and it can hide bugs by
        # treating an envelope planet as if it were airless at the surface.
        raise ValueError(
            "bc_atm='bare', 'gray', and 'ideal_gray' are only valid when kcore==0 (true bare-rock model). "
            "Choose a tabulated atmosphere BC (c23/f11/b97/g75/f07, ...) when kcore>0."
        )

    # Local toggles: for bare rock we disable all composition transport/coupling.
    # We keep these as *local* variables so we don't mutate module-level config.
    composition_change_local = (composition_change and has_envelope)
    advection_local = (helium_rain and has_envelope)
    z_advection_local = (metal_rain and has_envelope)
    evol_Z_local = (evol_Z and has_envelope)
    semiconvection_local = (semiconvection and has_envelope)

    # Safe, explicit slices/indices for envelope vs mantle/core.
    # - Cell-centered arrays have length N.
    # - Interior-boundary arrays (suffix _i) have length N-1 and correspond to
    #   interfaces between adjacent cells.
    env_cells = slice(0, kcore)            # envelope cells
    env_int_end = max(kcore - 1, 0)        # number of envelope interior boundaries
    env_int = slice(0, env_int_end)        # envelope interior-boundary slice

    global _fd_jac_check_done, D_b_first, DZ_b_first

    mantle_cells = slice(kcore, kcore_fe)  # mantle cells (becomes slice(0,kcore_fe) for bare rock)

    # For envelope planets, the "top of mantle" boundary is the envelope-mantle
    # interface at interior-boundary index (kcore-1). For bare rock there is no
    # envelope-mantle interface; start at 0 (first interior boundary between
    # mantle cells) to avoid negative indexing.
    mantle_int_start = 0 if is_bare_rock else max(kcore - 1, 0)
    mantle_int_end = max(kcore_fe - 1, 0)  # clamp to avoid slice(0,-1) when kcore_fe==0
    mantle_int = slice(mantle_int_start, mantle_int_end) if mantle_int_end > mantle_int_start else slice(0, 0)

    # For many "core boundary" quantities, we want the first interior boundary
    # within the core region. With an envelope, that's (kcore-1); for bare rock
    # the surface boundary is not part of _i arrays, so the first usable index is 0.
    core_i_start = 0 if is_bare_rock else max(kcore - 1, 0)

    # Fe-core interior boundaries start at (kcore_fe-1) when kcore_fe>0, else 0.
    fe_int_start = max(kcore_fe - 1, 0)

    # --- Helper arrays to obtain geometric and arithmetic means for interface/boundary calculations --- #
    T_i = get_geom_mean(T)
    r_i = r_b[1:-1]
    r = get_geom_mean(r_b)
    P_i = get_geom_mean(P)
    rho_i = get_geom_mean(rho)
    Yold_i = get_arith_mean(Yold)
    Zold_i = get_arith_mean(Zold)
    f_rock_old_i = get_arith_mean(f_rock_old)
    Sold_i = get_arith_mean(Sold)
    global D_b_first, DZ_b_first
    D_b_first = None
    DZ_b_first = None
    # chi_prev_full = None

    # --- Dimensionless scaling for solver stability --- #
    S_old_prime = Sold * norm / S0  # Entropy
    dm_prime = dm / M0
    r_prime = r / R0
    dr_prime_i = np.diff(r_prime)
    r_prime_b = r_b / R0

    dr = np.diff(r_b)  # cell widths in radius

    g = G * m_b[0] / r[0] ** 2
    g_i = G * m_b[1:-1] / r_i ** 2
    # Precompute interior-Z diffusion mask in enclosed-mass fraction coordinates.
    # This keeps units consistent between evolution.py and transport.py.
    mfrac_b = m_b / m_b[0]
    inner_z_diff_mask_b = mfrac_b < z_diff_inner_mfrac

    pbase1 = 10  # bars
    pbase2 = 30  # bars
    cond = ((P >= pbase1 * 1e6) & (P <= pbase2 * 1e6))

    # atmosphere boundary condition expect single values for Y, Z at the atmosphere
    # take mean over the range of pressures corresponding to the atmosphere
    S_atm = np.mean(Sold[cond]) if len(Sold[cond]) > 0 else Sold[0] 
    Y_atm = np.mean(Yold[cond]) if not y_atm_fixed else Y_atm_bc
    Z_atm = np.mean(Zold[cond]) if not z_atm_fixed else Z_atm_bc

    Y_prime = Y_atm/(1 - Z_atm)

    # 1) Evaluate boundary-condition derivatives based on user-specified BC_atm.
    # For non-bare BCs this is dTint/dS, dTint/dY, dTint/dZ.
    # For bare BC we also build a direct surface-flux derivative dFsurf/dS.
    dFsurf_dS = 0.0

    temp_func = interp1d(P_i, T_i,fill_value='extrapolate')
    S_func = interp1d(P_i, Sold_i,fill_value='extrapolate')
    Y_func = interp1d(P_i, Yold_i,fill_value='extrapolate')
    Z_func = interp1d(P_i, Zold_i,fill_value='extrapolate')
    frock_func = interp1d(P_i, f_rock_old_i,fill_value='extrapolate')

    # The profile interpolants above are built on *linear* pressure
    # [dyn/cm^2]; evaluate them at 10**logP, not logP.  Passing the log
    # value asks for extrapolation ~1e6 dyn/cm^2 below the surface, which
    # yields a garbage (typically negative) T1 -> dTdS = 0, silently
    # removing the surface-cooling feedback from J[0,0] and destabilizing
    # the Newton solve during fast-cooling phases (g75/f07-class BCs).
    target_logP_1bar = 6
    T1        = float(temp_func(10.0**target_logP_1bar))
    S1        = float(S_func(10.0**target_logP_1bar))
    Yold_t1   = float(Y_func(10.0**target_logP_1bar))
    Zold_t1   = float(Z_func(10.0**target_logP_1bar))
    frock_t1  = float(frock_func(10.0**target_logP_1bar))

    target_logP_10bar = 7
    T10        = float(temp_func(10.0**target_logP_10bar))
    S10        = float(S_func(10.0**target_logP_10bar))
    Yold_t10   = float(Y_func(10.0**target_logP_10bar))
    Zold_t10   = float(Z_func(10.0**target_logP_10bar))
    frock_t10  = float(frock_func(10.0**target_logP_10bar))

    # 1) Evaluate partial derivatives dTint/dS, dTint/dY, dTint/dZ based on user-specified BC_atm.
    #    This ensures T_int is updated if S, Y, or Z in the top layer changes.
    if BC_atm == 'c23' and y_atm_fixed:
        dTdS = atm.get_dtintds(np.log10(g), T_int, Y_atm, Z_atm) # DOUBLE CHECK UNITS OF DTINT_DS--MAKE SURE CONSISTENT WITH OUR S0 NORMALIZATION.
        dTint_dY = 0.0
        dTint_dZ = 0.0

    elif BC_atm == 'c23' and not y_atm_fixed:
        # if Yprime > 0.25: Yprime = 0.25
        dTdS = atm.get_dtintds(np.log10(g), T_int, Y_prime, Z_atm)
        dTint_dY = atm.get_dtintdy(S_atm, np.log10(g), Y_prime, Z_atm)
        dTint_dZ = 0.0

    elif BC_atm == 'c26':
        dTdS = atm.get_dtintds(np.log10(g), T_int, Y_atm, Z_atm)
        if not y_atm_fixed and not z_atm_fixed:
            dTint_dY = atm.get_dtintdy(S_atm, np.log10(g), Y_atm, Z_atm)
            dTint_dZ = atm.get_dtintdz(S_atm, np.log10(g), Y_atm, Z_atm)
        elif y_atm_fixed and not z_atm_fixed:
            dTint_dY = 0.0
            dTint_dZ = atm.get_dtintdz(S_atm, np.log10(g), Y_atm, Z_atm)
        elif not y_atm_fixed and z_atm_fixed:
            dTint_dY = atm.get_dtintdy(S_atm, np.log10(g), Y_atm, Z_atm)
            dTint_dZ = 0.0
        else:  # both fixed
            dTint_dY = 0.0
            dTint_dZ = 0.0

    elif BC_atm == 'm21':

        Z_atm_solar = Z_atm / const.Z_SOLAR_CHEN23
        #dT10dS = mixtures_eos.get_dtds_sp(S10, 7, Yold[t10_id], Zold[t10_id], f_rock_old[t10_id])
        dT10dS = mixtures_eos.get_dtds_sp(S10, target_logP_10bar, Yold_t10, Zold_t10, frock_t10)
        dTint_dT10 = atm.get_dtintdt10(np.log10(g), T10, Z_atm=Z_atm_solar)
        dTdS = dTint_dT10 * dT10dS * norm
        dTint_dY = 0.0
        if not z_atm_fixed:
            dTint_dZ = atm.get_dtintdz(np.log10(g), T10, Z_atm=Z_atm_solar)
        else:
            dTint_dZ = 0.0

    elif BC_atm == 'p20':
        dT10dS = mixtures_eos.get_dtds_sp(S10, target_logP_10bar, Yold_t10, Zold_t10, frock_t10)
        dTint_dT10 = atm.get_dtintdt10(np.log10(g), T10)
        dTdS = dTint_dT10 * dT10dS * norm
        dTint_dY = 0.0
        dTint_dZ = 0.0

    elif BC_atm == 'f11':
        dT10dS = mixtures_eos.get_dtds_sp(S10, target_logP_10bar, Yold_t10, Zold_t10, frock_t10)
        dTint_dT10 = atm.get_dtintdt10(np.log10(g), T10)
        dTdS = dTint_dT10 * dT10dS * norm
        dTint_dY = 0.0
        dTint_dZ = 0.0

    elif BC_atm == 'b97':
        dT10dS = mixtures_eos.get_dtds_sp(S10, target_logP_10bar, Yold_t10, Zold_t10, frock_t10)
        dTint_dT10 = atm.get_dtintdt10(np.log10(g), T10)
        dTdS = dTint_dT10 * dT10dS * norm
        dTint_dY = 0.0
        dTint_dZ = 0.0

    elif BC_atm == 'g75':
        dT10dS = mixtures_eos.get_dtds_sp(S1, target_logP_1bar, Yold_t1, Zold_t1, frock_t1)
        dTint_dT1 = atm.get_dtintdt1(np.log10(g), T1)
        dTdS = dTint_dT1 * dT10dS * norm
        dTint_dY = 0.0
        dTint_dZ = 0.0

    elif BC_atm == 'f07':
        target_logP = 9  # F07 grid goes down to 1 kbar = 1e9 dyn/cm^2
        logP_core = np.log10(P[kcore]) if kcore < N else np.log10(P[-1])
        if target_logP > logP_core:
            target_logP = np.log10(P[kcore - 1]) if kcore > 0 else logP_core
        #target_logP = np.log10(1e6)
        # Interpolants are on linear pressure; evaluate at 10**logP (see
        # the 1-bar/10-bar extraction above).
        T1000       = float(temp_func(10.0**target_logP))
        S1000      = float(S_func(10.0**target_logP))
        Yold_t1000   = float(Y_func(10.0**target_logP))
        Zold_t1000   = float(Z_func(10.0**target_logP))
        frock_t1000  = float(frock_func(10.0**target_logP))
        #dT10dS = mixtures_eos.get_dtds_sp(S10, 7, Yold[t10_id], Zold[t10_id], f_rock_old[t10_id])
        dT1000dS = mixtures_eos.get_dtds_sp(S1000, target_logP, Yold_t1000, Zold_t1000, frock_t1000)
        # Fortney+07 tables use their own solar reference. Cap Z at 0.94 before
        # building the metallicity factor: the f07 lookup becomes unreliable
        # above that, so clipping here keeps the derivative consistent with
        # the BC evaluation in evolution.py.
        f_Z = z_to_metallicity_factor(np.clip(Zold_t1000, 0.0, 0.94), const.Z_SOLAR_FORTNEY07)
        dTint_dT1000 = atm.get_dtintdt1000(np.log10(g), np.log10(T1000), f_Z)
        dTdS = dTint_dT1000 * dT1000dS * norm
        dTint_dY = 0.0
        dTint_dZ = 0.0

    elif BC_atm == 'bare':
        # Bare (airless) boundary condition.
        #
        # In "true bare rock" mode we enforce kcore==0 above. That means the surface
        # layer is cell 0 (mantle if kcore_fe>0, otherwise Fe core). The emitting
        # temperature for the bare-atm model is therefore the *surface cell* temperature.
        #
        # We need dTint/dS for the Newton solve. We use the chain rule:
        #   dTint/dS = (dTint/dTsurf) * (dTsurf/dS)
        # and the thermodynamic identity at fixed pressure:
        #   (∂S/∂T)_P = Cp/T  =>  (∂T/∂S)_P = T/Cp
        Tsurf = float(T[0])

        if kcore_fe > 0:
            # Surface is rocky mantle
            Cp_surf = mantle_eos.get_cp_pt(P[kcore] * mantle_eos.dyn_to_GPa, Tsurf)  # erg/g/K
        else:
            # Surface is iron (no mantle shell)
            Cp_surf = core_eos.get_CP_pt(P[kcore] * core_eos.dyn_to_GPa, Tsurf)  # erg/g/K

        # EOS helpers may return numpy scalars or 1-element arrays; .item() handles both.
        Cp_surf = np.asarray(Cp_surf).item()
        dTsurf_dS_cgs = Tsurf / Cp_surf  # K per (erg/g/K)

        # The module-level `atm` IS the bare_atm object here: create_atm built
        # it under the same BC_atm == 'bare' condition that guards this branch.
        # (The old globals().get("atm_bare", <fresh instance>) fallback was
        # unreachable — and its default was constructed eagerly on every call.)
        # Smooth the Jacobian derivative across T_surf = T_eq (same
        # sigmoid treatment as the gray case — see note there).
        _delta_T = max(0.01 * T_eq, 1.0)
        _x = (Tsurf - T_eq) / _delta_T
        _sigmoid = 1.0 / (1.0 + np.exp(np.clip(-_x, -500, 500)))
        dFsurf_dTsurf = 4.0 * const.sigma_sb * Tsurf**3 * _sigmoid
        dFsurf_dS = dFsurf_dTsurf * dTsurf_dS_cgs * norm                  # erg/(s cm^2) per solver-S
        # Bare losses are applied directly from F_int(T_surf) in the Jacobian section below.
        dTdS = 0.0

        dTint_dY = 0.0
        dTint_dZ = 0.0

    elif BC_atm in ('gray', 'ideal_gray'):
        # Semi-gray rocky atmosphere variants: use intrinsic surface flux from T_surf and g.
        # Jacobian uses dFsurf/dS through T_surf(S) at the top cell.
        Tsurf = float(T[0])

        if kcore_fe > 0:
            Cp_surf = mantle_eos.get_cp_pt(P[kcore] * mantle_eos.dyn_to_GPa, Tsurf)  # erg/g/K
        else:
            Cp_surf = core_eos.get_CP_pt(P[kcore] * core_eos.dyn_to_GPa, Tsurf)  # erg/g/K

        Cp_surf = np.asarray(Cp_surf).item()
        dTsurf_dS_cgs = Tsurf / Cp_surf  # K per (erg/g/K)

        # The module-level `atm` is the rock_gray/ideal_gray object here
        # (create_atm built it under the same BC_atm guard as this branch).
        if BC_atm == 'gray':
            # Smooth the Jacobian derivative across T_surf = T_eq.
            # The physical flux uses max(T^4 - T_eq^4, 0) (exact), but
            # its Heaviside derivative makes Newton oscillate when
            # T_surf ≈ T_eq.  A sigmoid transition keeps the Jacobian
            # well-conditioned without changing the residual.
            _delta_T = max(0.01 * T_eq, 1.0)
            _x = (Tsurf - T_eq) / _delta_T
            _sigmoid = 1.0 / (1.0 + np.exp(np.clip(-_x, -500, 500)))
            dFsurf_dTsurf = 4.0 * const.sigma_sb * Tsurf**3 * _sigmoid
        else:
            dFsurf_dTsurf = atm.get_dintrinsic_flux_dtsurf(Tsurf, g)

        dFsurf_dTsurf = float(np.asarray(dFsurf_dTsurf).item())
        dFsurf_dS = dFsurf_dTsurf * dTsurf_dS_cgs * norm
        dTdS = 0.0
        dTint_dY = 0.0
        dTint_dZ = 0.0

    else:
        raise Exception(
            "Invalid atmospheric boundary condition name."
        )

    # Guard: these enter the Jacobian only (J[0,0..2]); a non-finite value
    # from a table-edge lookup would poison the sparse solve and abort the
    # run.  Dropping the term degrades Newton conditioning, never the
    # converged solution.
    if not np.isfinite(dTdS):
        logger.warning("Non-finite dTdS from BC '%s'; zeroing surface J term.", BC_atm)
        dTdS = 0.0
    if not np.isfinite(dTint_dY):
        dTint_dY = 0.0
    if not np.isfinite(dTint_dZ):
        dTint_dZ = 0.0

    # Optional lagged/Picard treatment of the surface-BC Jacobian coupling
    # (see lag_bc_jacobian comment at the config parse): the residual's
    # surface loss uses the step-start T_int, so zeroing these terms is the
    # consistent inexact-Newton choice and never changes a converged solution.
    if lag_bc_jacobian:
        dTdS = 0.0
        dTint_dY = 0.0
        dTint_dZ = 0.0

    # Compute relevant scale heights, conduction/radiation
    Hpress_i = P_i / rho_i / g_i

    if use_local_Hp and has_envelope:
        # Use the local pressure scale height in the envelope for both the
        # mixing-length velocity and the rain advection velocity.  Fall back
        # to the constant Hr in the mantle/core where Hpress_i can be noisy.
        Hp_i = np.where(
            np.arange(N - 1) < kcore,
            Hpress_i,
            Hr,
        )
        # Near the center of coreless models P/(rho*g_i) diverges as
        # g_i -> 0.  Cap at the local radius: a convective eddy cannot
        # travel farther than its distance from the center, so the
        # mixing length is bounded by r_i (cell-centred) or r_b
        # (boundary-centred).  This is a local, position-dependent
        # ceiling that only activates in the innermost cells.
        Hp_i = np.minimum(Hp_i, r_i)
        # Boundary-centred version: Hpress_i is already interior-centred
        # (N-1 values, same count as Hp_b[1:N]).  Use it directly in the
        # envelope; fall back to constant Hr in the compact interior.
        Hp_b = Hr + np.zeros(N + 1)
        Hp_b[1:N] = np.where(
            np.arange(N - 1) < kcore,
            Hpress_i,
            Hr,
        )
        Hp_b[1:N] = np.minimum(Hp_b[1:N], r_b[1:N])
        # Guard: Hp_b should never be smaller than a sensible floor
        Hp_b = np.maximum(Hp_b, 1e3)
    else:
        Hp_i = Hr + np.zeros(N - 1)
        Hp_b = Hr + np.zeros(N + 1)
    Hp_rain_b = alpha_herain * Hp_b

    # Specific heats for MLT (convective) transport
    Cv = np.zeros_like(Yold)
    Cp_i = np.zeros_like(Yold_i)

    # Envelope EOS is only valid when kcore>0. For bare rock, these remain zero
    # and we later fill mantle/core Cv/Cp from mantle_eos/core_eos.
    if has_envelope:
        Cv[env_cells] = mixtures_eos.get_cv_rhot(
            np.log10(rho[env_cells]),
            np.log10(T[env_cells]),
            Yold[env_cells],
            Zold[env_cells],
            f_rock_old[env_cells],
            dt=0.1,
            tab=tab,
            ideal_guess=False,
            arr_guess=np.log10(P[env_cells]),
        )

    if env_int_end > 0:
        Cp_i[env_int] = mixtures_eos.get_cp_pt(
            np.log10(P_i[env_int]),
            np.log10(T_i[env_int]),
            Yold_i[env_int],
            Zold_i[env_int],
            f_rock_old_i[env_int],
            dt=0.1,
        )

    # Radiative diffusion is an envelope-only transport term (requires an opacity model
    # for H/He). In bare-rock mode we keep it identically zero to avoid Z/(1-Z) blowups
    # when Z~1 in the mantle/core.
    Lambda_rad_i = np.zeros_like(T_i)
    if radiation and env_int_end > 0:
        c = const.c
        if opac == 'water_rich':
            import water_rich_opac as h2o_opac
            ross_k_i = 10 ** h2o_opac.get_kappa_R(np.log10(P_i[env_int]), np.log10(T_i[env_int]), np.log10(Zold_i[env_int]))
        elif opac == 'default':
            ross_k_i = 10 ** k_interp((np.log10(T_i[env_int]), np.log10(rho_i[env_int])))
        else:  # opac == 'metal_rich'
            f_Z = z_to_metallicity_factor(np.clip(Zold_i[env_int], 0.0, 1.0 - 1e-12), const.Z_SOLAR_FORTNEY07)
            ross_k_i = get_rosseland_opacity(rho_i[env_int], T_i[env_int], f_Z)


        if rad_ramp_gyr > 0.0:
            _theta_rad = np.arctan(t / (rad_ramp_gyr * const.Gyr_to_s)) * 2 / np.pi
        else:
            _theta_rad = 1.0
        Lambda_rad_i[env_int] = (
            4 * a * c * T_i[env_int] ** 3 / ross_k_i / 3.0 / rho_i[env_int]
            * _theta_rad
        )

    # Track conductive and radiative contributions separately for diagnostics.
    # Lambda_cond_i includes all non-radiative diffusion terms actually used
    # (envelope conduction, core conductivity overrides, and overshoot additions).
    Lambda_cond_i = np.zeros_like(T_i)
    if env_int_end > 0:
        # Anchored composition correction:
        # `get_Lambda_cond(P)` is a pressure-only baseline built from French/Becker tabulated
        # conductivities along adiabats whose structure includes nonzero heavy-element mass
        # fractions. We therefore preserve that baseline at Z = Z_ref(P) and only adjust away
        # from it using the French (2019) water proxy conductivity.
        z_mix = np.clip(Zold_i[env_int], 0.0, 1.0 - 1e-12)
        _cond_Y = Yold_i[env_int] if y_dependent_conductivity else None
        lambda_ref = get_Lambda_cond(P_i[env_int], Y=_cond_Y)
        lambda_w = get_Lambda_cond_water(rho_i[env_int], T_i[env_int])
        z_ref = get_Z_ref_for_adiabats(P_i[env_int], Y=_cond_Y)

        denom = np.maximum(1.0 - z_ref, 1e-12)
        lambda_hhe_proxy = (lambda_ref - z_ref * lambda_w) / denom
        # Guard numerical pathologies if the water proxy exceeds the reference baseline enough
        # to make the inferred H/He proxy slightly negative.
        lambda_hhe_proxy = np.maximum(lambda_hhe_proxy, np.finfo(float).tiny)

        if env_metal_cond_frock:
            # f_rock-weighted metal conductivity: water/silicate blend so the
            # Z -> 1, f_rock -> 1 limit recovers the mantle (k_el + k_lat)
            # model.  The H-He proxy reconstruction above keeps lambda_w: the
            # French/Becker baseline's Z_ref content is watery by construction.
            _lam_sil = (
                mantle_core.get_k_el(T_i[env_int])
                + mantle_core.get_k_lat_liq(rho_i[env_int], T_i[env_int])
            ) * (u.W / u.m / u.K).to('erg/(s * cm * K)')
            _fr_env = np.clip(f_rock_old_i[env_int], 0.0, 1.0)
            lambda_z_metal = (1.0 - _fr_env) * lambda_w + _fr_env * _lam_sil
        else:
            lambda_z_metal = lambda_w

        Lambda_cond_i[env_int] = lam_factor * (
            (1.0 - z_mix) * lambda_hhe_proxy + z_mix * lambda_z_metal
        )
    Lambda_i = Lambda_cond_i.copy()
    Lambda_i += Lambda_rad_i

    # ============================================================
    # EXTRAPOLATED ENDPOINT METHOD: stellar irradiation penetration
    # ============================================================
    # Per-cell heating source Q_cell_irr(k) [erg/s] for envelope
    # cells, decaying exponentially with optical depth from the
    # surface.  Plugged into the entropy residual below the way
    # radiogenic heating and latent heat enter (frozen across NR
    # iterations within a step; only diagonal Jacobian via 1/T').
    #
    # tau_v(k) ≈ kappa_v * P[k] / g[k]   (hydrostatic estimate)
    # Q_cell_irr(k) = (F_inc / mu) * kappa_v * exp(-tau_v/mu) * dm
    Q_cell_irr = np.zeros(N)
    if extrap_endpoint and kcore > 0:
        if ee_F_inc_override > 0.0:
            F_inc_ee = ee_F_inc_override
        else:
            F_inc_ee = float(const.sigma_sb) * T_eq**4

        # Cell-centered gravity g[k] = G * m_enc / r_cell^2.
        # m_b is descending (surface to center), so m_b[k] is the mass
        # enclosed at the OUTER boundary of cell k.  Use mid-cell.
        m_enc_cell = 0.5 * (m_b[:-1] + m_b[1:])
        r_cell_loc = 0.5 * (r_b[:-1] + r_b[1:])
        g_cell = const.G * m_enc_cell / np.maximum(r_cell_loc, 1e-30)**2

        tau_v_cell = ee_kappa_v * P / np.maximum(g_cell, 1e-30)
        inv_mu = 1.0 / max(ee_mu, 1e-6)
        q_spec = (F_inc_ee * inv_mu) * ee_kappa_v * np.exp(-tau_v_cell * inv_mu)
        Q_cell_irr[:kcore] = q_spec[:kcore] * dm[:kcore]

        if ee_apply_theta:
            Q_cell_irr *= np.arctan(t / (0.5 * const.Gyr_to_s)) * 2.0 / np.pi

    if mc>0:
        # Setting thermal conductivities in the mantle and the core regions. 
        if mantle_int_end > mantle_int_start:
            if mantle_thermal_conductivity is None:
                if mantle_comp == 'h2o':
                    Lambda_i[mantle_int] = get_Lambda_cond_water(rho_i[mantle_int], T_i[mantle_int])
                    Lambda_cond_i[mantle_int] = Lambda_i[mantle_int]
                else:
                    # electronic contribution to the thermal conductivity
                    Lambda_i[mantle_int] = mantle_core.get_k_el(T_i[mantle_int]) * (u.W/u.m/u.K).to('erg/(s * cm * K)')
                    # lattice contribution to the thermal conductivity
                    Lambda_i[mantle_int] += mantle_core.get_k_lat_liq(rho_i[mantle_int], T_i[mantle_int]) * (u.W/u.m/u.K).to('erg/(s * cm * K)')
                    Lambda_cond_i[mantle_int] = Lambda_i[mantle_int]
            else:
                # User-specified constant mantle thermal conductivity.
                Lambda_i[mantle_int] = mantle_thermal_conductivity * (u.W/u.m/u.K).to('erg/(s * cm * K)')
                Lambda_cond_i[mantle_int] = Lambda_i[mantle_int]

        core_cond_val = core_thermal_conductivity * (u.W/u.m/u.K).to('erg/(s * cm * K)')
        Lambda_i[fe_int_start:] = core_cond_val
        Lambda_cond_i[fe_int_start:] = core_cond_val


    # ============================================================
    # LATENT HEAT IMPLEMENTATION: LHS latent term via smooth χ(T,P)
    # ============================================================

    # Mantle index ranges (already defined above; shown here for clarity)
    # Use the precomputed safe slices so kcore==0 doesn't create negative indexing.
    mantle_slice_cells = mantle_cells      # cell-centered mantle
    mantle_slice_int   = mantle_int        # mantle-related interior boundaries

    # ---------- (keep this) freeze-mask for your viscosity logic ----------
    P_mantle = P[mantle_slice_cells]
    T_mantle = T[mantle_slice_cells]
    Tm_mantle = mantle_eos.get_T_melt(P_mantle * mantle_eos.dyn_to_GPa)  # K
    L_lat_mantle = mantle_eos.L * latent_heat_factor  # erg/g
    L_lat_core = core_eos.L #* latent_heat_factor  # erg/g

    freeze_mask_cells = (T_mantle <= Tm_mantle)  # used later for viscous convection options
    # ----------------------------------------------------------------------

    # --------------- smooth liquid fraction χ(T,P) ---------------
    # Width of the χ transition in K; configurable for stability studies.
    DELTA_T_LAT = max(float(latent_delta_t), 1.0)

    # χ drives both latent source terms and viscous/inviscid transport blending.
    use_chi_for_transport = bool(viscous_mantle_convection)
    use_chi_for_latent = bool(latent_heat_effect)

    def _chi_relax_theta(cap_to_dt=True):
        if t_chi_relax <= 0.0:
            return 1.0
        # For transport we avoid tau=min(2*dt, tau), which otherwise keeps
        # theta ~ 0.39 even at tiny dt and can chatter near phase boundaries.
        tau_eff = min(2.0 * dt, t_chi_relax) if cap_to_dt else t_chi_relax
        tau_eff = max(tau_eff, 1e-30)
        return 1.0 - np.exp(-dt / tau_eff)

    theta_chi_latent = _chi_relax_theta(cap_to_dt=True)
    theta_chi_transport = _chi_relax_theta(cap_to_dt=False)

    if L_lat_core > 0.0 and kcore_fe > 0:
        P_core = P[kcore_fe:]
        Tm_core = core_eos.get_T_melt(P_core * core_eos.dyn_to_GPa)  # K
    else:
        Tm_core = None
        L_lat_core = 0.0

    def _relaxed_chi_from_temperature(
        temp_cells,
        chi_prev_state,
        apply_relax=True,
        theta_relax=1.0,
    ):
        """Build χ(T), optionally applying timestep relaxation."""
        chi_target_full = np.zeros(N, dtype=float)
        dchi_dT_target_full = np.zeros(N, dtype=float)

        idx_mantle = np.arange(kcore, kcore_fe)
        if idx_mantle.size > 0:
            x_m = np.clip((temp_cells[idx_mantle] - Tm_mantle) / DELTA_T_LAT, -40.0, 40.0)
            chi_target_full[idx_mantle] = 0.5 * (1.0 + np.tanh(x_m))
            dchi_dT_target_full[idx_mantle] = 0.5 * (1.0 / np.cosh(x_m) ** 2) / DELTA_T_LAT

        if L_lat_core > 0.0 and Tm_core is not None:
            idx_core = np.arange(kcore_fe, N)
            if idx_core.size > 0:
                x_c = np.clip((temp_cells[idx_core] - Tm_core) / DELTA_T_LAT, -40.0, 40.0)
                chi_target_full[idx_core] = 0.5 * (1.0 + np.tanh(x_c))
                dchi_dT_target_full[idx_core] = 0.5 * (1.0 / np.cosh(x_c) ** 2) / DELTA_T_LAT

        chi_prev_arr = np.asarray(chi_prev_state, dtype=float)
        if (
            chi_prev_arr.shape[0] != N
            or np.any(~np.isfinite(chi_prev_arr))
            or np.any(chi_prev_arr < 0.0)
        ):
            # First step or bad carry-over: initialize from the target χ.
            chi_prev_arr = np.copy(chi_target_full)
        else:
            chi_prev_arr = np.clip(chi_prev_arr, 0.0, 1.0)

        if apply_relax:
            chi_out = np.clip(
                chi_prev_arr + theta_relax * (chi_target_full - chi_prev_arr),
                0.0,
                1.0,
            )
            dchi_dT_out = theta_relax * dchi_dT_target_full
        else:
            chi_out = np.copy(chi_target_full)
            dchi_dT_out = dchi_dT_target_full
        return chi_out, dchi_dT_out, chi_prev_arr

    transport_chi_apply_relax = bool(use_chi_for_transport and chi_transport_relax)

    if use_chi_for_transport or use_chi_for_latent:
        # Transport χ can optionally be relaxed; latent always uses relaxed χ
        # inside the Jacobian via apply_relax=True in the latent block.
        chi_curr_transport_full, _, chi_prev_relax_full = _relaxed_chi_from_temperature(
            T,
            chi_prev_full,
            apply_relax=transport_chi_apply_relax,
            theta_relax=theta_chi_transport,
        )
    else:
        chi_prev_arr = np.asarray(chi_prev_full, dtype=float)
        if (
            chi_prev_arr.shape[0] != N
            or np.any(~np.isfinite(chi_prev_arr))
            or np.any(chi_prev_arr < 0.0)
        ):
            chi_prev_arr = np.zeros(N, dtype=float)
        else:
            chi_prev_arr = np.clip(chi_prev_arr, 0.0, 1.0)
        chi_curr_transport_full = np.copy(chi_prev_arr)
        chi_prev_relax_full = chi_prev_arr

    # ============================================================
    # CONVECTION AND COMPOSITION DERIVATIVES
    # ============================================================

    # Thermodynamic derivative delta = - dlogrho/dlogT at constant P
    delta_i = np.zeros_like(Yold_i)
    if env_int_end > 0:
        # Envelope-only: mixtures_eos derivatives are undefined in the mantle/core.
        delta_i[env_int] = -mixtures_eos.get_dlogrho_dlogt_py(
            np.log10(P_i[env_int]),
            np.log10(T_i[env_int]),
            Yold_i[env_int],
            Zold_i[env_int],
            f_rock_old_i[env_int],
            dt=0.1,
        )

    assert_all_not_nan([delta_i, Cp_i, Hpress_i])

    # Helium & metal partial derivatives for composition coupling (envelope only).
    # We allocate full-length arrays and fill only the envelope pieces so that later
    # code can safely reference them without worrying about kcore==0 edge cases.
    dy_i = np.zeros_like(Yold_i)
    dz_i = np.zeros_like(Zold_i)
    dy = np.zeros_like(Yold)
    dz = np.zeros_like(Zold)

    # High-Z sampling cap for EOS COMPOSITION DERIVATIVES only (the state
    # itself is never modified).  Measured (2026-07-17, deep Jupiter
    # conditions): dS/dY|_PT is smooth to Z = 0.999, but dS/dZ|_PT flips
    # sign and diverges ~100x beyond Z ~ 0.98 -- the ideal-mixing entropy
    # log tail amplified by table-edge finite differences.  Evaluating the
    # derivatives at min(Z, cap) retains their saturated physical values;
    # the neglected piece is O((1-Z)ln(1-Z)) because every composition
    # derivative multiplies gradients/changes of the vanishing H-He
    # minority.  Production envelopes (Z <~ 0.5) sample below the cap and
    # are unaffected.  CAUTION: inhomogeneous ice-giant profiles can build
    # deep-envelope Z ~ 0.98 ABOVE the cap; freezing their Ledoux/Jacobian
    # derivatives at the cap value makes the Newton solve dramatically
    # stiffer through convective-overturn epochs (timestep collapse).  Such
    # models should set [transport] eos_deriv_z_cap = 1.0 (cap disabled) --
    # see the Uranus/Neptune inhomogeneous example configs.
    Zold_d = np.minimum(Zold, eos_deriv_z_cap)
    Zold_d_i = np.minimum(Zold_i, eos_deriv_z_cap)

    if has_envelope:
        dy[env_cells] = mixtures_eos.adaptive_dx(Yold[env_cells])
        dz[env_cells] = mixtures_eos.adaptive_dx(Zold_d[env_cells])
    if env_int_end > 0:
        dy_i[env_int] = mixtures_eos.adaptive_dx(Yold_i[env_int])
        dz_i[env_int] = mixtures_eos.adaptive_dx(Zold_d_i[env_int])
    # dT/dY, dT/dZ, plus chemical potentials dU/dY, dU/dZ
    dTdY = np.zeros_like(Yold)
    dTdZ = np.zeros_like(Yold)
    dUdY = np.zeros_like(Yold)
    dUdZ = np.zeros_like(Yold)

    # Envelope-only: mantle/core do not have meaningful Y/Z in this formulation.
    if has_envelope:
        dTdY[env_cells] = mixtures_eos.get_dtdy_srho(
            Sold[env_cells],
            np.log10(rho[env_cells]),
            Yold[env_cells],
            Zold_d[env_cells],
            f_rock_old[env_cells],
            dy=dy[env_cells],
            tab=tab,
            ideal_guess=False,
            arr_guess=np.log10(T[env_cells]),
        )
        dTdZ[env_cells] = mixtures_eos.get_dtdz_srho(
            Sold[env_cells],
            np.log10(rho[env_cells]),
            Yold[env_cells],
            Zold_d[env_cells],
            f_rock_old[env_cells],
            dz=dz[env_cells],
            tab=tab,
            ideal_guess=False,
            arr_guess=np.log10(T[env_cells]),
        )
        dUdY[env_cells] = mixtures_eos.get_dudy_srho(
            Sold[env_cells],
            np.log10(rho[env_cells]),
            Yold[env_cells],
            Zold_d[env_cells],
            f_rock_old[env_cells],
            dy=dy[env_cells],
            tab=tab,
            ideal_guess=False,
            arr_p_guess=np.log10(P[env_cells]),
            arr_t_guess=np.log10(T[env_cells]),
        )
        dUdZ[env_cells] = mixtures_eos.get_dudz_srho(
            Sold[env_cells],
            np.log10(rho[env_cells]),
            Yold[env_cells],
            Zold_d[env_cells],
            f_rock_old[env_cells],
            dz=dz[env_cells],
            tab=tab,
            ideal_guess=False,
            arr_p_guess=np.log10(P[env_cells]),
            arr_t_guess=np.log10(T[env_cells]),
        )

        # Guard EOS derivative noise: interpolate isolated NaN/inf in the
        # envelope and linearly extrapolate if gaps sit on segment edges.
        for _arr in (dTdY, dTdZ, dUdY, dUdZ):
            fill_nonfinite_linear(_arr, env_cells)

    # For Ledoux vs Schwarzschild gradients
    dSdY_rhop_i = np.zeros_like(Yold_i)
    dSdZ_rhop_i = np.zeros_like(Yold_i)
    dSdY_pt_i = np.zeros_like(Yold_i)
    dSdZ_pt_i = np.zeros_like(Yold_i)

    if env_int_end > 0:
        dSdY_rhop_i[env_int] = mixtures_eos.get_dsdy_rhop_srho(
            Sold_i[env_int],
            np.log10(rho_i[env_int]),
            Yold_i[env_int],
            Zold_d_i[env_int],
            f_rock_old_i[env_int],
            dy=dy_i[env_int],
            ds=0.1,
            tab=tab,
            ideal_guess=False,
            arr_guess=np.log10(P_i[env_int]),
        )
        dSdZ_rhop_i[env_int] = mixtures_eos.get_dsdz_rhop_srho(
            Sold_i[env_int],
            np.log10(rho_i[env_int]),
            Yold_i[env_int],
            Zold_d_i[env_int],
            f_rock_old_i[env_int],
            dz=dz_i[env_int],
            ds=0.1,
            tab=tab,
            ideal_guess=False,
            arr_guess=np.log10(P_i[env_int]),
        )

    dSdZ_rhop_b = np.concatenate(([0], dSdZ_rhop_i, [0]))

    if env_int_end > 0:
        dSdY_pt_i[env_int] = mixtures_eos.get_dsdy_pt(
            np.log10(P_i[env_int]),
            np.log10(T_i[env_int]),
            Yold_i[env_int],
            Zold_d_i[env_int],
            f_rock_old_i[env_int],
            dy=dy_i[env_int],
        )
        dSdZ_pt_i[env_int] = mixtures_eos.get_dsdz_pt(
            np.log10(P_i[env_int]),
            np.log10(T_i[env_int]),
            Yold_i[env_int],
            Zold_d_i[env_int],
            f_rock_old_i[env_int],
            dz=dz_i[env_int],
        )

        for _arr in (dSdY_rhop_i, dSdZ_rhop_i, dSdY_pt_i, dSdZ_pt_i):
            fill_nonfinite_linear(_arr, env_int)

        # Glitch-targeted Gaussian smoothing of EOS derivatives.
        # Detects zone-to-zone oscillations (sign changes in second differences)
        # and applies local Gaussian smoothing only to glitchy regions, preserving
        # real EOS features in monotonic regions.
        if smooth_eos_derivs and env_int_end > 5:
            sigma = smooth_eos_sigma
            hw = max(int(3 * sigma), 2)  # half-width of smoothing window
            for _arr in (dSdY_rhop_i, dSdZ_rhop_i, dSdY_pt_i, dSdZ_pt_i):
                vals = _arr[env_int].copy()
                n = len(vals)
                if n < 5:
                    continue
                # Second differences detect curvature oscillations
                d2 = np.diff(vals, n=2)
                # Glitch = sign change in second difference (oscillatory curvature)
                sign_changes = np.zeros(n, dtype=bool)
                for j in range(len(d2) - 1):
                    if d2[j] * d2[j + 1] < 0:
                        # Mark the zone and its neighbors
                        lo = max(0, j)
                        hi = min(n, j + 4)
                        sign_changes[lo:hi] = True
                # Expand glitch mask by hw zones on each side
                glitch_mask = np.zeros(n, dtype=bool)
                for j in range(n):
                    if sign_changes[j]:
                        lo = max(0, j - hw)
                        hi = min(n, j + hw + 1)
                        glitch_mask[lo:hi] = True
                if np.any(glitch_mask):
                    smoothed = gaussian_filter1d(vals, sigma=sigma, mode='nearest')
                    # Blend: use smoothed values at glitch zones, original elsewhere
                    vals[glitch_mask] = smoothed[glitch_mask]
                    _arr[env_int] = vals

    dSdY_pt_b = np.concatenate(([0], dSdY_pt_i, [0]))
    dSdZ_pt_b = np.concatenate(([0], dSdZ_pt_i, [0]))

    # Weighted mix for semiconvection transitions
    dSdY_i = R_rho * dSdY_rhop_i + (1 - R_rho) * dSdY_pt_i
    dSdZ_i = R_rho * dSdZ_rhop_i + (1 - R_rho) * dSdZ_pt_i
    dSdY_b = np.concatenate(([0], dSdY_i, [0]))
    dSdZ_b = np.concatenate(([0], dSdZ_i, [0]))

    # --- Rayleigh-Taylor guard: see module-level comment at rt_guard -------
    # Face b sits between cell b-1 (above) and cell b (below); surface->center
    # ordering means stability requires rho[b] >= rho[b-1].  Mask envelope
    # faces (both cells < kcore) where the step-start profile is inverted and
    # zero every composition-stabilization derivative there.  Self-limiting:
    # once the overturn removes the inversion the mask empties.
    if rt_guard and kcore > 2:
        global _rt_guard_ncalls
        _rho_step = np.asarray(rho, dtype=float)
        rt_mask_b = np.zeros(N + 1, dtype=bool)
        _inv_faces = _rho_step[1:kcore] < _rho_step[:kcore - 1] * (1.0 - 1e-9)
        rt_mask_b[1:kcore][_inv_faces] = True
        n_rt = int(rt_mask_b.sum())
        if n_rt > 0:
            dSdY_b[rt_mask_b] = 0.0
            dSdZ_b[rt_mask_b] = 0.0
            dSdY_pt_b[rt_mask_b] = 0.0
            dSdZ_pt_b[rt_mask_b] = 0.0
            if _rt_guard_ncalls == 0 or _rt_guard_ncalls % 200 == 0:
                _kf = np.where(rt_mask_b)[0]
                print(f"RT-GUARD active: {n_rt} inverted envelope faces "
                      f"(k {_kf.min()}-{_kf.max()}), composition "
                      f"stabilization zeroed", flush=True)
            _rt_guard_ncalls += 1

    # Dimensionless derivatives
    dTdY_prime = dTdY / T0
    dTdZ_prime = dTdZ / T0
    dSdY_prime_b = dSdY_b / S0
    dSdZ_prime_b = dSdZ_b / S0
    dUdY_prime = dUdY / S0
    dUdZ_prime = dUdZ / S0

    assert_all_not_nan([dSdY_rhop_i, dSdY_pt_i, dSdZ_rhop_i, dSdZ_pt_i])
    assert_all_not_nan([dTdY, dTdZ, dUdY, dUdZ])

    # In bare-rock mode we force composition_change_local=False so the Newton solve
    # does not try to couple entropy evolution to (non-existent) Y/Z evolution.
    if composition_change_local:
        dS_mult = 1
    else:
        dS_mult = 0

    # MLT velocity scale factor
    mantle_boundary_idx = np.array([], dtype=int)
    n_mantle_bnds = 0
    mantle_transport_prefactor = np.zeros(0)
    mantle_kappa_34a = np.zeros(0)
    mantle_kappa_34b = np.zeros(0)
    mantle_kappa_35a = np.zeros(0)
    mantle_kappa_35b = np.zeros(0)
    mantle_kappa_floor = 1e-60
    # Maximum log10 ratio between inviscid and viscous kappa branches.
    # Caps the dynamic range of the blend to prevent catastrophic sensitivity
    # to small chi changes during solidification (bare-rock stability fix).
    MAX_LOG_KAPPA_RATIO = 6.0

    if convection:

        phi_i = np.zeros(N - 1)
        # Default convective geometry factor from the current phi_i state.
        # Some paths (e.g. coreless/all-envelope runs) can skip the deeper
        # mantle/Fe finalization block, so initialize phi_b here to ensure the
        # nested Jacobian always captures a defined array.
        phi_b = np.concatenate(([0.0], phi_i, [0.0]))

        # Envelope / H-He region above the rocky mantle (interior boundaries only).
        # Use env_int so kcore==0 does not silently become ":-1".
        if env_int_end > 0:
            phi_i[env_int] = (
                np.sqrt(delta_i[env_int])
                * 4 * np.pi * r_i[env_int] ** 2
                * rho_i[env_int]
                * T_i[env_int]
                * (g_i[env_int] * Hp_i[env_int] ** 4 / Cp_i[env_int] / 32) ** 0.5
            )
            phi_b = np.concatenate(([0.0], phi_i, [0.0]))

        # default convective flux exponent
        flux_bracket_power_dsdr = np.zeros(N + 1) + 1.5  # 3/2 base scaling

        # store old entropy gradient for diagnostics if needed
        denom = np.diff(r)  # shape (N-1,)
        bracket_dsdr_old = np.zeros(N + 1)
        bracket_dsdr_old[1:N] = (-np.diff(Sold) / denom * norm)


        # Thermodynamic properties of the rocky/icy mantle region.
        #
        # Important bare-rock detail:
        #   - For envelope planets, mantle boundary arrays include the envelope-mantle interface.
        #   - For bare rock, the surface boundary is not part of _i arrays, so mantle boundary
        #     arrays begin at interior boundary index 0 (between mantle cells 0 and 1).
        # Using `mantle_int` ensures both cases are handled safely.
        if mantle_int_end > mantle_int_start:
            Cp_i[mantle_int] = mantle_eos.get_cp_pt(
                P_i[mantle_int] * mantle_eos.dyn_to_GPa,
                T_i[mantle_int],
            )

        if mantle_convection and mantle_int_end > mantle_int_start:
            alpha_mantle = mantle_eos.get_alpha_pt(
                P_i[mantle_int] * mantle_eos.dyn_to_GPa,
                T_i[mantle_int],
            )
            # dynamic viscosity (solid) in poise; we'll blend with a liquid ref. in log10-space
            dyn_viscosity_solid = mantle_core.get_eta_solid(
                P_i[mantle_int] * mantle_core.dyn_to_Pa,
                T_i[mantle_int],
                out='poise',
            )
        else:
            alpha_mantle = np.zeros(0)
            dyn_viscosity_solid = np.zeros(0)

        if (kcore_fe - kcore) > 0:
            Cv[mantle_slice_cells] = mantle_eos.get_cv_pt(
                P[mantle_slice_cells] * mantle_eos.dyn_to_GPa,
                T[mantle_slice_cells],
            )

        # --- BEGIN smooth χ-blend + log-viscosity interpolation ----------------
        # Mantle convection block: skip entirely when mantle_convection is off,
        # leaving phi_i at zero for mantle boundaries (conduction only).
        if mantle_convection:
            # liquid dynamic viscosity: 100 Pa*s => 1000 poise (cgs)
            ETA_LIQ_PA_S   = 100.0
            ETA_LIQ_POISE  = ETA_LIQ_PA_S * 10.0

            # pick χ at mantle *cells* (fallback to freeze-mask if latent is off)
            n_mantle_cells = kcore_fe - kcore
            mantle_boundary_idx = np.array([], dtype=int)
            n_mantle_bnds = 0
            if n_mantle_cells > 0:
                # For MLT transport, use smooth χ whenever viscous-core convection is enabled.
                if viscous_mantle_convection and (chi_curr_transport_full is not None):
                    chi_cells = chi_curr_transport_full[mantle_slice_cells].astype(float)
                else:
                    chi_cells = (~freeze_mask_cells).astype(float)

                # Decide which *interior* boundaries we treat as mantle boundaries for MLT.
                # For envelope planets, this includes the envelope-mantle interface at (kcore-1).
                # For bare rock, the surface boundary is not part of _i arrays, so mantle
                # boundaries begin at interior index 0 (between mantle cells 0 and 1).
                mantle_boundary_idx = np.arange(mantle_int_start, mantle_int_end, dtype=int)
                n_mantle_bnds = mantle_boundary_idx.size

                # Only proceed if we actually have mantle interior boundaries.
                # (E.g., a 1-zone mantle has n_mantle_bnds==0.)
                if n_mantle_bnds > 0:
                    # Map chi (cells) -> chi_b (interior boundaries).
                    # - Envelope case: first boundary touches envelope+mantle, so use mantle cell(0).
                    # - Bare-rock case: boundaries are between mantle cells, so use averages only.
                    if has_envelope:
                        chi_b_local = np.empty(n_mantle_bnds, dtype=float)
                        chi_b_local[0] = chi_cells[0]
                        if n_mantle_bnds > 1:
                            chi_b_local[1:] = 0.5 * (chi_cells[1:] + chi_cells[:-1])

                        # Preserve historical behavior for envelope planets: the mantle-boundary
                        # grid here is constructed to align one-to-one with mantle cells, so using
                        # the cell-centered melt curve (Tm_mantle) is consistent and avoids subtle
                        # regressions in previous gas-giant setups.
                        Tm_b = Tm_mantle
                    else:
                        # Bare rock: no "top boundary" inside _i arrays.
                        chi_b_local = 0.5 * (chi_cells[1:] + chi_cells[:-1])
                        Tm_b = 0.5 * (Tm_mantle[1:] + Tm_mantle[:-1])

                    # Viscosity-limited branch uses chi-blended dynamic viscosity at boundaries
                    rho_b = np.clip(rho_i[mantle_boundary_idx], 1e-20, None)
                    cp_b = np.maximum(Cp_i[mantle_boundary_idx], 1e-30)
                    hp_b = np.maximum(Hp_i[mantle_boundary_idx], 1e-30)
                    g_b = np.maximum(g_i[mantle_boundary_idx], 0.0)
                    t_b = np.maximum(T_i[mantle_boundary_idx], 1e-30)
                    t_melt_b = np.maximum(Tm_b, 1e-30)
                    alpha_mantle_pos = np.maximum(alpha_mantle, 0.0)

                    log_nu_liq = np.log10(ETA_LIQ_POISE / rho_b)
                    log_nu_solid = np.log10(np.clip(dyn_viscosity_solid / rho_b, 1e-50, None))
                    log_nu_eff = chi_b_local * log_nu_liq + (1.0 - chi_b_local) * log_nu_solid
                    nu_eff = 10.0 ** log_nu_eff
                    nu_eff = np.clip(nu_eff, 1e-4, None)  # cm^2/s

                    # 34a) liquid-limited eddy diffusivity
                    kappa_34a_arg = (
                        alpha_mantle_pos
                        * t_b
                        * g_b
                        * hp_b ** 4
                        / cp_b
                        / 32.0
                    )
                    kappa_34a = np.sqrt(np.maximum(kappa_34a_arg, 0.0))  # cm^2/s

                    # 34b) viscosity-limited eddy diffusivity
                    kappa_34b = (
                        alpha_mantle_pos
                        * t_b
                        * g_b
                        * hp_b ** 4
                        / cp_b
                        / np.clip(nu_eff, 1e-30, None)
                        / 32.0
                    )  # cm^2/s

                    # Smooth blend of transport between liquid and viscosity-limited
                    if viscous_mantle_convection:
                        alpha_x = np.maximum(np.abs(
                            mantle_eos.get_alpha_x(
                                P_i[mantle_boundary_idx] * mantle_eos.dyn_to_GPa,
                                T_i[mantle_boundary_idx],
                                rho_i[mantle_boundary_idx],
                                dT_melt = DELTA_T_LAT,
                            )
                        ), 0.0)

                        # 35a/35b: mixture analogs evaluated on the melt curve at boundaries
                        L_lat = max(float(L_lat_mantle), 1e-10)  # erg/g guard

                        kappa_35a_arg = (
                            alpha_x
                            * t_melt_b
                            * g_b
                            * hp_b ** 4
                            / (32.0 * L_lat)
                        )
                        kappa_35a = np.sqrt(np.maximum(kappa_35a_arg, 0.0))

                        kappa_35b = (
                            alpha_x
                            * t_melt_b
                            * g_b
                            * hp_b ** 4
                            / (32.0 * L_lat * np.clip(nu_eff, 1e-30, None))
                        )

                        # Zhang 38a/38b analog with z ≡ chi_b
                        kappa_inv_combo = np.maximum(
                            chi_b_local * kappa_34a + (1.0 - chi_b_local) * kappa_35a,
                            mantle_kappa_floor,
                        )
                        kappa_vis_combo = np.maximum(
                            (1.0 - chi_b_local) * kappa_34b + chi_b_local * kappa_35b,
                            mantle_kappa_floor,
                        )

                        # --- Cap the dynamic range of the inviscid/viscous
                        # kappa ratio (see MAX_LOG_KAPPA_RATIO definition above).
                        log_inv = np.log10(kappa_inv_combo)
                        log_vis = np.log10(kappa_vis_combo)
                        log_ratio = log_inv - log_vis
                        excess = np.maximum(log_ratio - MAX_LOG_KAPPA_RATIO, 0.0)
                        # Bring inviscid branch down and viscous branch up by
                        # half the excess each, preserving the geometric mean.
                        log_inv_capped = log_inv - 0.5 * excess
                        log_vis_capped = log_vis + 0.5 * excess
                        kappa_inv_combo = 10.0 ** log_inv_capped
                        kappa_vis_combo = 10.0 ** log_vis_capped

                        # Primary blend (log-space to keep positivity and avoid kinks)
                        log_kappa_eff = (
                            chi_b_local * log_inv_capped
                            + (1.0 - chi_b_local) * log_vis_capped
                        )
                        kappa_eff_local = 10 ** log_kappa_eff
                    else:
                        kappa_eff_local = kappa_34a

                    # Cache mantle transport ingredients so the Jacobian can
                    # rebuild the viscous/inviscid transition from trial χ.
                    if viscous_mantle_convection:
                        mantle_kappa_34a = np.maximum(kappa_34a, mantle_kappa_floor)
                        mantle_kappa_34b = np.maximum(kappa_34b, mantle_kappa_floor)
                        mantle_kappa_35a = np.maximum(kappa_35a, mantle_kappa_floor)
                        mantle_kappa_35b = np.maximum(kappa_35b, mantle_kappa_floor)

                    mantle_transport_prefactor = (
                        4.0
                        * np.pi
                        * r_i[mantle_boundary_idx] ** 2
                        * rho_i[mantle_boundary_idx]
                        * T_i[mantle_boundary_idx]
                    )

                    # Write phi_i on mantle boundaries with blended kappa
                    phi_i[mantle_boundary_idx] = mantle_transport_prefactor * kappa_eff_local

                    # OPTIONAL exponent smoothing:
                    exponent_b = 1.5 + 0.5 * (1.0 - chi_b_local)
                    flux_bracket_power_dsdr[mantle_boundary_idx + 1] = exponent_b

                # if t >= 3e-9 * const.Gyr_to_s:
                # --- END smooth χ-blend + log-viscosity interpolation ------------------

                # NOTE: previous "frozen_boundary_mask_local" hard switch and
                #       phi_i overwrite were REMOVED in favor of the smooth blend.

        # inner metallic / iron region
        # fe_int_start guards kcore_fe==0 (pure-iron planet) so we don't slice with -1.
        Cp_i[fe_int_start:] = core_eos.get_CP_pt(
            P_i[fe_int_start:] * core_eos.dyn_to_GPa,
            T_i[fe_int_start:]
        )
        Cv[kcore_fe:] = core_eos.get_CV_pt(
            P[kcore_fe:] * core_eos.dyn_to_GPa,
            T[kcore_fe:]
        )
        # Iron core convection: skip when iron_core_convection is off,
        # leaving phi_i at zero for core boundaries (conduction only).
        if iron_core_convection:
            alpha_core = core_eos.get_alpha_pt(
                P_i[fe_int_start:] * core_eos.dyn_to_GPa,
                T_i[fe_int_start:]
            )

            # Guard: alpha_core can go negative near the iron melt
            # curve because the liquid EOS finite-difference stencil
            # (T ± dT) dips into the solid regime where density is
            # unphysical.  Negative alpha means no buoyancy-driven
            # convection, so clamp the MLT argument to zero.
            kappa_eddy_fe = np.sqrt(np.maximum(
                alpha_core
                * T_i[fe_int_start:]
                * g_i[fe_int_start:]
                * Hp_i[fe_int_start:] ** 4
                / Cp_i[fe_int_start:]
                / 32,
                0.0,
            ))
            phi_i[fe_int_start:] = (
                4 * np.pi
                * r_i[fe_int_start:] ** 2
                * rho_i[fe_int_start:]
                * T_i[fe_int_start:]
                * kappa_eddy_fe
            )
            flux_bracket_power_dsdr[kcore_fe:] = np.minimum(
                flux_bracket_power_dsdr[kcore_fe:], 1.5
            )

            # finalize phi_b

            n = T_i.size
            k = int(kcore_fe-1)

            # boolean mask: True only for zones inside the iron-rich core
            in_core = np.arange(n) > k

            # melting temperature of iron at each zone pressure (make sure units match what get_T_melt expects)
            T_melt_fe = core_eos.get_T_melt(P_i * core_eos.dyn_to_GPa)

            # # frozen zones: only where (inside core) AND (below iron melt temp)
            solid_core_mask = in_core & (T_i <= T_melt_fe)

            phi_i[solid_core_mask] = 0.0

            # x_fe_melt = (T_i - T_melt_fe) / DELTA_T_LAT
            # chi_b = 0.5*(1 + np.tanh(x_fe_melt))          # 0 solid, 1 liquid
            # phi_i[in_core] *= chi_b[in_core]      # smoothly shut off convection as it freezes


        phi_b = np.concatenate(([0.0], phi_i, [0.0]))


        # no convective MLT deeper than kcore_fe
        #phi_b[kcore_fe:] = 0.0

    else:
        phi_b = np.zeros(N + 1)
        flux_bracket_power_dsdr = np.zeros(N + 1) + 1.5

    phi_i_base = np.copy(phi_b[1:-1])

    def _map_mantle_chi_to_boundaries(chi_cells_local):
        """Map mantle cell χ to mantle interior-boundary χ."""
        if n_mantle_bnds <= 0:
            return np.zeros(0, dtype=float)
        if has_envelope:
            chi_b_local = np.empty(n_mantle_bnds, dtype=float)
            chi_b_local[0] = chi_cells_local[0]
            if n_mantle_bnds > 1:
                chi_b_local[1:] = 0.5 * (chi_cells_local[1:] + chi_cells_local[:-1])
            return chi_b_local
        return 0.5 * (chi_cells_local[1:] + chi_cells_local[:-1])

    def _transport_state_from_chi(chi_full):
        """
        Build transport geometry/exponent for a given χ profile.
        This lets the Newton residual see inviscid<->viscous transition updates.
        """
        phi_i_eval = np.copy(phi_i_base)
        flux_power_eval = np.copy(flux_bracket_power_dsdr)
        if (
            (not convection)
            or (not mantle_convection)
            or (not viscous_mantle_convection)
            or (n_mantle_bnds <= 0)
            or (mantle_transport_prefactor.size != n_mantle_bnds)
            or (mantle_kappa_34a.size != n_mantle_bnds)
            or (mantle_kappa_34b.size != n_mantle_bnds)
            or (mantle_kappa_35a.size != n_mantle_bnds)
            or (mantle_kappa_35b.size != n_mantle_bnds)
        ):
            return np.concatenate(([0.0], phi_i_eval, [0.0])), flux_power_eval

        chi_cells_local = np.asarray(chi_full[mantle_slice_cells], dtype=float)
        if chi_cells_local.size != (kcore_fe - kcore):
            return np.concatenate(([0.0], phi_i_eval, [0.0])), flux_power_eval
        chi_cells_local = np.clip(chi_cells_local, 0.0, 1.0)

        chi_b_local = _map_mantle_chi_to_boundaries(chi_cells_local)
        if chi_b_local.size != n_mantle_bnds:
            return np.concatenate(([0.0], phi_i_eval, [0.0])), flux_power_eval

        kappa_inv_combo = np.maximum(
            chi_b_local * mantle_kappa_34a + (1.0 - chi_b_local) * mantle_kappa_35a,
            mantle_kappa_floor,
        )
        kappa_vis_combo = np.maximum(
            (1.0 - chi_b_local) * mantle_kappa_34b + chi_b_local * mantle_kappa_35b,
            mantle_kappa_floor,
        )
        # Apply the same dynamic-range cap as the main transport block.
        log_inv = np.log10(kappa_inv_combo)
        log_vis = np.log10(kappa_vis_combo)
        log_ratio = log_inv - log_vis
        excess = np.maximum(log_ratio - MAX_LOG_KAPPA_RATIO, 0.0)
        log_inv = log_inv - 0.5 * excess
        log_vis = log_vis + 0.5 * excess

        log_kappa_eff = (
            chi_b_local * log_inv
            + (1.0 - chi_b_local) * log_vis
        )
        kappa_eff_local = 10.0 ** log_kappa_eff

        phi_i_eval[mantle_boundary_idx] = mantle_transport_prefactor * kappa_eff_local
        flux_power_eval[mantle_boundary_idx + 1] = 1.5 + 0.5 * (1.0 - chi_b_local)
        return np.concatenate(([0.0], phi_i_eval, [0.0])), flux_power_eval

    # Map each mantle transport boundary to contributing mantle cells so we can
    # build implicit χ->phi/kappa Jacobian terms.
    chi_map_cell_a = np.full(n_mantle_bnds, -1, dtype=int)
    chi_map_cell_b = np.full(n_mantle_bnds, -1, dtype=int)
    chi_map_w_a = np.zeros(n_mantle_bnds, dtype=float)
    chi_map_w_b = np.zeros(n_mantle_bnds, dtype=float)
    if n_mantle_bnds > 0:
        if has_envelope:
            # EMB boundary uses only the top mantle cell.
            chi_map_cell_a[0] = kcore
            chi_map_w_a[0] = 1.0
            for lb in range(1, n_mantle_bnds):
                chi_map_cell_a[lb] = kcore + lb - 1
                chi_map_cell_b[lb] = kcore + lb
                chi_map_w_a[lb] = 0.5
                chi_map_w_b[lb] = 0.5
        else:
            for lb in range(n_mantle_bnds):
                chi_map_cell_a[lb] = kcore + lb
                chi_map_cell_b[lb] = kcore + lb + 1
                chi_map_w_a[lb] = 0.5
                chi_map_w_b[lb] = 0.5

    # Snapshot of Lambda BEFORE the eddy-overshoot addition below: the purely
    # microscopic (conductive + radiative) conductivity, used by the Spruit DDC
    # closure so its kappa_T never inherits macroscopic overshoot transport.
    Lambda_micro_i = Lambda_i.copy()

    # Optional artificial overshoot transport across the envelope-mantle interface.
    # This is evaluated *after* Cp_i is fully initialized in mantle/core convection setup,
    # and uses a mantle-side convective velocity scale (strongest positive old bracket in
    # the mantle interior) to avoid vanishing when the EMB itself is Ledoux-stable.
    if eddy_overshoot and convection and has_envelope and (mc > 0) and (kcore > 0) and (kcore_fe > kcore):
        emb_i = kcore - 1  # interior-boundary index of envelope-mantle interface
        if 0 <= emb_i < (N - 1):
            # Old-time convective bracket on interior boundaries (index-aligned with i-arrays).
            bracket_old_i = np.maximum(bracket_dsdr_old[1:N], 0.0)

            # Pick a mantle-side anchor boundary with the largest positive bracket.
            # Fully mantle boundaries are [kcore, ..., kcore_fe-2].
            m_start = kcore
            m_stop = max(kcore_fe - 1, kcore)
            if m_stop > m_start:
                mantle_cand = np.arange(m_start, m_stop, dtype=int)
                anchor_i = int(mantle_cand[np.argmax(bracket_old_i[mantle_cand])])
            else:
                anchor_i = emb_i

            bracket_anchor = float(bracket_old_i[anchor_i])
            cp_anchor = float(np.clip(Cp_i[anchor_i], 1e-30, None))
            hp_anchor = float(np.clip(Hp_i[anchor_i], 1e-30, None))
            g_anchor = float(np.clip(g_i[anchor_i], 1e-30, None))
            hp_emb = float(np.clip(Hp_i[emb_i], 1e-30, None))

            if bracket_anchor > 0.0:
                v_mlt_anchor = np.sqrt(g_anchor * hp_anchor**2 / cp_anchor / 8.0 * bracket_anchor)
                kappa_ov0 = max(float(eddy_overshoot_factor), 0.0) * v_mlt_anchor * hp_anchor
                H_ov = max(float(eddy_overshoot_fhp), 0.0) * hp_emb

                if (kappa_ov0 > 0.0) and (H_ov > 0.0):
                    # Apply overshoot transport from EMB outward across envelope boundaries.
                    env_bnd_idx = np.arange(emb_i + 1, dtype=int)
                    delta_r = np.abs(r_i[env_bnd_idx] - r_i[emb_i])
                    kappa_ov_i = kappa_ov0 * np.exp(-delta_r / H_ov)
                    lambda_ov_i = rho_i[env_bnd_idx] * Cp_i[env_bnd_idx] * kappa_ov_i

                    cap = float(eddy_overshoot_lambda_cap)
                    if cap > 0.0:
                        lambda_cap_i = cap * np.maximum(Lambda_i[env_bnd_idx], 1e-30)
                        lambda_ov_i = np.minimum(lambda_ov_i, lambda_cap_i)

                    Lambda_i[env_bnd_idx] += lambda_ov_i
                    Lambda_cond_i[env_bnd_idx] += lambda_ov_i

    # Boundary array with dimensionless scaling
    Lambda_b = np.concatenate(([0], Lambda_i, [0]))
    Lambda_prime_b = 4 * np.pi * r_prime_b ** 2 * Lambda_b * R0 / S0 / M0 # normalized conduction + radiation

    if semiconvection_local and semi_model == 'spruit':
        # Microscopic thermal diffusivity kappa_T = Lambda/(rho*Cp) on interior
        # boundaries (frozen for this update_SYZ call, like Lambda_i itself).
        # Includes Lambda_rad with its early-time Theta(t) ramp deliberately:
        # the closure must see the same diffusivity the heat equation uses,
        # and the early-time error direction (smaller D_eff) is conservative.
        kappa_T_micro_i = Lambda_micro_i / np.maximum(rho_i * Cp_i, 1e-30)
        # Consistency ratio: semi_phi multiplies the FULL Lambda_i (which may
        # include the eddy-overshoot addition); scaling Nu_T - 1 by
        # Lambda_micro/Lambda makes the delivered kappa_T_eff enhancement
        # exactly Spruit's regardless (ratio = 1 unless overshoot is active).
        lam_ratio_i = Lambda_micro_i / np.maximum(Lambda_i, 1e-30)


    # ------------------------------------------------------------------
    # H/He (and H/water) immiscibility curves used for He-rain / metal-rain.
    # This is strictly envelope physics. For bare rock (kcore==0) there is no
    # envelope region, so we skip these calculations and return neutral defaults.
    # ------------------------------------------------------------------
    if env_int_end > 0:
        # Allowed Hydrogen-Helium immiscibility pressure range.
        # The floor rises with the misc_deltaP curve shift: below
        # 1 Mbar + delta_P the shifted curve is fully miscible.
        misc_idx_Y = (P_i[:kcore-1] > 1e12 + delta_P) & (P_i[:kcore-1] < P[innermisc_index])

        # Allowed Hydrogen-Water immiscibility temperature and pressure ranges
        if misc_kind == 'water':
            misc_idx_Z_T = np.where((T_i[:kcore-1] >= 300) & (T_i[:kcore-1] <= 6000))[0]
        elif misc_kind == 'hhe':
            # H-He stand-in for H-Z: use the same pressure window as the
            # He-rain miscibility region above (with the Z-channel offsets).
            misc_idx_Z_T = np.where((P_i[:kcore-1] > 1e12 + z_delta_P) & (P_i[:kcore-1] < P[innermisc_index]))[0]
        else:
            misc_idx_Z_T = np.where((P_i[:kcore-1] >= 1 * 1e10) & (P_i[:kcore-1] <= 200 * 1e10))[0]

        # Deep cap on the Z-rain window (all misc kinds): drop cells below
        # zmisc_p_max so envelope-base cells never activate rain.
        # Z1m_i defaults to 1 (fully miscible) outside the window, so capped
        # cells get zero supersaturation and zero curve derivatives.
        if z_p_max > 0.0:
            misc_idx_Z_T = misc_idx_Z_T[P_i[:kcore-1][misc_idx_Z_T] < z_p_max]

        # Hydrogen-Helium immiscibility implicit expansion derivative along temperature

        if advection_local:
            dYm_dT_temp_i = misc.get_dydt_misc(
                np.log10(P_i[:kcore - 1][misc_idx_Y]),
                np.log10(T_i[:kcore - 1][misc_idx_Y]),
                delta_T,
                tab=True,
                misc_curve=misc_curve,
                deltaP=delta_P
            )

            Ym_temp_i = misc.get_y_misc_pt(
                np.log10(P_i[:kcore - 1][misc_idx_Y]),
                np.log10(T_i[:kcore - 1][misc_idx_Y]),
                delta_T,
                tab=True,
                misc_curve=misc_curve,
                deltaP=delta_P
            )

            dYm_dP_temp_i = misc.get_dydp_misc(
                np.log10(P_i[:kcore - 1][misc_idx_Y]),
                np.log10(T_i[:kcore - 1][misc_idx_Y]),
                delta_T,
                tab=True,
                misc_curve=misc_curve,
                deltaP=delta_P
            )
        else:
            dYm_dT_temp_i = np.zeros(len(P_i[:kcore-1][misc_idx_Y]))
            dYm_dP_temp_i = np.zeros(len(P_i[:kcore-1][misc_idx_Y]))
            Ym_temp_i = np.ones(len(P_i[:kcore-1][misc_idx_Y]))

        if z_advection_local and misc_kind == 'hhe':
            # Test mode: evaluate the H-He miscibility curve (e.g. Lorenzen)
            # at the local (P, T) and use it as the H-Z saturation curve.
            # Z1m plays the role of Ym; Z2m = 1 with zero derivatives so the
            # metal-rich branch never gates the rain.
            logp_z_i = np.log10(P_i[:kcore-1][misc_idx_Z_T])
            logt_z_i = np.log10(T_i[:kcore-1][misc_idx_Z_T])

            Z1m_temp_i = misc.get_y_misc_pt(
                logp_z_i, logt_z_i, z_delta_T, tab=True, misc_curve=misc_curve,
                deltaP=z_delta_P
            )
            dZ1m_dT_temp_i = misc.get_dydt_misc(
                logp_z_i, logt_z_i, z_delta_T, tab=True, misc_curve=misc_curve,
                deltaP=z_delta_P
            )
            dZ1m_dP_temp_i = misc.get_dydp_misc(
                logp_z_i, logt_z_i, z_delta_T, tab=True, misc_curve=misc_curve,
                deltaP=z_delta_P
            )

            Z2m_temp_i = np.ones_like(Z1m_temp_i)
            dZ2m_dT_temp_i = np.zeros_like(Z1m_temp_i)
            dZ2m_dP_temp_i = np.zeros_like(Z1m_temp_i)
        elif z_advection_local:
            dZ1m_dT_temp_i, dZ2m_dT_temp_i = zmisc.get_dzdt_misc(
                P_i[:kcore-1][misc_idx_Z_T]*1e-10,  # GPa
                T_i[:kcore-1][misc_idx_Z_T],
                np.full_like(T_i[:kcore-1][misc_idx_Z_T], 0.0),  # adaptive dx
                tab=True
            )

            dZ1m_dP_temp_i, dZ2m_dP_temp_i = zmisc.get_dzdp_misc(
                P_i[:kcore-1][misc_idx_Z_T]*1e-10,  # GPa
                T_i[:kcore-1][misc_idx_Z_T],
                np.full_like(T_i[:kcore-1][misc_idx_Z_T], 0.0),
                tab=True
            )

            Z1m_temp_i, Z2m_temp_i = zmisc.get_zmisc(
                P_i[:kcore-1][misc_idx_Z_T]*1e-10,  # GPa
                T_i[:kcore-1][misc_idx_Z_T],
                np.full_like(T_i[:kcore-1][misc_idx_Z_T], 0.0),
                tab=True
            )
        else:
            dZ1m_dT_temp_i, dZ2m_dT_temp_i = np.zeros(len(P_i[:kcore-1][misc_idx_Z_T])), np.zeros(len(P_i[:kcore-1][misc_idx_Z_T]))
            dZ1m_dP_temp_i, dZ2m_dP_temp_i = np.zeros(len(P_i[:kcore-1][misc_idx_Z_T])), np.zeros(len(P_i[:kcore-1][misc_idx_Z_T]))
            Z1m_temp_i, Z2m_temp_i = np.ones(len(P_i[:kcore-1][misc_idx_Z_T])), np.zeros(len(P_i[:kcore-1][misc_idx_Z_T]))

        # IMPORTANT: avoid chained indexing like `arr[:k][mask] = ...` which
        # assigns into a temporary copy and silently leaves `arr` unchanged.
        # That would keep Ym_i/Z1m_i/Z2m_i at their defaults (often 1s) and can
        # completely suppress rain even when the miscibility curve indicates an
        # immiscibility region.

        Ym_i = np.ones(N - 1)
        dYm_dT_i = np.zeros(N - 1)
        dYm_dP_i = np.zeros(N - 1)

        ym_mask_i = np.zeros(N - 1, dtype=bool)
        ym_mask_i[:kcore - 1] = misc_idx_Y
        Ym_i[ym_mask_i] = Ym_temp_i
        dYm_dT_i[ym_mask_i] = dYm_dT_temp_i
        dYm_dP_i[ym_mask_i] = dYm_dP_temp_i

        Z1m_i = np.ones(N - 1)
        Z2m_i = np.ones(N - 1)
        dZ1m_dT_i = np.zeros(N - 1)
        dZ2m_dT_i = np.zeros(N - 1)
        dZ1m_dP_i = np.zeros(N - 1)
        dZ2m_dP_i = np.zeros(N - 1)

        z_mask_i = np.zeros(N - 1, dtype=bool)
        z_mask_i[misc_idx_Z_T] = True
        Z1m_i[z_mask_i] = Z1m_temp_i
        Z2m_i[z_mask_i] = Z2m_temp_i
        dZ1m_dT_i[z_mask_i] = dZ1m_dT_temp_i
        dZ2m_dT_i[z_mask_i] = dZ2m_dT_temp_i
        dZ1m_dP_i[z_mask_i] = dZ1m_dP_temp_i
        dZ2m_dP_i[z_mask_i] = dZ2m_dP_temp_i

        if ym_t_lag > 0:
            dYm_dT_i *= (1 / ym_t_lag) / (1 / dt + 1 / ym_t_lag)

        if zm_t_lag > 0:
            dZ1m_dT_i *= (1/zm_t_lag)/(1/dt + 1/zm_t_lag)
            dZ2m_dT_i *= (1/zm_t_lag)/(1/dt + 1/zm_t_lag)

            dZ1m_dP_i *= (1/zm_t_lag)/(1/dt + 1/zm_t_lag)
            dZ2m_dP_i *= (1/zm_t_lag)/(1/dt + 1/zm_t_lag)
    else:
        # Neutral defaults: no immiscibility constraints (used for bare rock and very thin envelopes).
        Ym_i = np.ones(N - 1)
        Z1m_i = np.ones(N - 1)
        Z2m_i = np.ones(N - 1)
        dYm_dT_i = np.zeros(N - 1)
        dYm_dP_i = np.zeros(N - 1)
        dZ1m_dT_i = np.zeros(N - 1)
        dZ2m_dT_i = np.zeros(N - 1)
        dZ1m_dP_i = np.zeros(N - 1)
        dZ2m_dP_i = np.zeros(N - 1)

    assert_all_not_nan([Ym_i, Z1m_i, Z2m_i])

    assert_all_not_nan([dYm_dT_i, dZ1m_dT_i, dZ2m_dT_i, dZ1m_dP_i, dZ2m_dP_i])

    def _trial_temperature_state(S, Y, Z, T, Cv):
        """Trial thermodynamic state at the current Newton iterate.

        Linearizes temperature about the previous accepted hydrostatic state:
            T_trial' = T' + (T'/Cv) * S0 * (S - S_old')
                          + dT/dY * (Y - Yold) + dT/dZ * (Z - Zold)

        Returns (T_prime, Cv_safe, deltaT_prime, T_trial_prime, T_trial).
        Shared by the chi(T)-relaxation block and the latent-heat block inside
        Jacobian; it is CALLED at both sites (not computed once and reused),
        so the operation count and order match the previously duplicated
        inline code exactly — bit-identical by construction.
        """
        T_prime = T / T0
        Cv_safe = np.maximum(Cv, 1e-99)
        deltaT_prime = (
            (T_prime / Cv_safe) * S0 * (S - S_old_prime)
            + dTdY_prime * (Y - Yold)
            + dTdZ_prime * (Z - Zold)
        )
        T_trial_prime = T_prime + deltaT_prime
        T_trial = T_trial_prime * T0
        return T_prime, Cv_safe, deltaT_prime, T_trial_prime, T_trial

    def Jacobian(S, Y, Z, T_int, kcore_fe, chi_prev_full):

        # D_b_first and DZ_b_first are module-level globals used across calls.
        # chi_prev_full is a parameter of the enclosing function `update_SYZ`
        # and therefore should not be declared global here. Removing it from
        # the global declaration allows the nested function to access the
        # outer-scope parameter via closure (or use `nonlocal` if assignment
        # to the outer variable is required).
        global D_b_first, DZ_b_first
        nonlocal phi_b

        """
        Builds the global system:
        - J (3N x 3N) for derivatives of residual equations wrt S, Y, Z,
        - B (3N) for residual vector,
        - flux arrays (conv, rad, semi) for diagnosing solution,
        - updated diffusion coefficients D_b, D2_b,
        - and the total energy loss across the top boundary (loss_energy).
        """

        tiny = 1e-99
        (T_prime, Cv_safe, deltaT_prime,
         T_trial_prime, T_trial) = _trial_temperature_state(S, Y, Z, T, Cv)

        chi_transport_eval_full = chi_curr_transport_full
        dchi_dT_transport_full = np.zeros(N, dtype=float)
        if implicit_viscous_transition and use_chi_for_transport and (n_mantle_bnds > 0):
            chi_transport_eval_full, dchi_dT_transport_full, _ = _relaxed_chi_from_temperature(
                T_trial,
                chi_prev_relax_full,
                apply_relax=transport_chi_apply_relax,
                theta_relax=theta_chi_transport,
            )
            phi_b_eval, flux_bracket_power_eval = _transport_state_from_chi(chi_transport_eval_full)
        else:
            phi_b_eval = phi_b
            flux_bracket_power_eval = flux_bracket_power_dsdr

        # Keep the latest transport geometry for diagnostics/output.
        phi_b = phi_b_eval
        phi_prime_b = phi_b_eval * np.power(S0, flux_bracket_power_eval - 1) / (
            np.power(R0, flux_bracket_power_eval) * M0 * T0
        )

        conv_flux_b = np.zeros(N + 1)
        rad_flux_b = np.zeros(N + 1)
        semi_flux_b = np.zeros(N + 1)

        bracket_dsdr_b = np.zeros(N + 1)# flux [k] -> [k-1] cell
        R0_inv_b = np.zeros(N + 1)


        # CONVECTIVE FLUX BRACKET TERM
        # Evaluate local gradient -dS/dr + dS/dY*dY/dr + dS/dZ*dZ/dr.
        denom = np.diff(r)  # shape: (N-1,)
        bracket_dsdr_b[1:N] = (
            (-np.diff(S) / denom) * norm +
            ((dSdY_b[1:N] * np.diff(Y)) / denom) * dS_mult +
            ((dSdZ_b[1:N] * np.diff(Z)) / denom) * dS_mult
        )

        # --- Flux time-relaxation of composition bracket terms ---
        # Relax only the Ledoux composition terms (dSdY*dY/dr + dSdZ*dZ/dr)
        # between timesteps, keeping the entropy gradient (-dS/dr) fully
        # responsive to NR updates.  This damps oscillatory bracket sign
        # changes at He rain onset without fighting the Newton solver.
        if flux_relax and dS_mult > 0:
            comp_term_new = np.zeros(N + 1)
            comp_term_new[1:N] = (
                ((dSdY_b[1:N] * np.diff(Y)) / denom) * dS_mult +
                ((dSdZ_b[1:N] * np.diff(Z)) / denom) * dS_mult
            )
            comp_term_old = np.zeros(N + 1)
            comp_term_old[1:N] = (
                ((dSdY_b[1:N] * np.diff(Yold)) / denom) * dS_mult +
                ((dSdZ_b[1:N] * np.diff(Zold)) / denom) * dS_mult
            )
            tau_relax_s = flux_relax_tau * t_Db_relax_myr * const.Myr_to_s
            if tau_relax_s > 0 and dt > 0:
                alpha_relax = dt / (dt + tau_relax_s)
                comp_term_relaxed = alpha_relax * comp_term_new + (1 - alpha_relax) * comp_term_old
                # Rebuild bracket: entropy gradient (current) + relaxed composition terms
                entropy_grad_b = np.zeros(N + 1)
                entropy_grad_b[1:N] = (-np.diff(S) / denom) * norm
                bracket_dsdr_b = entropy_grad_b + comp_term_relaxed

        # --- Schwarzschild / Ledoux bracket decomposition ---
        # Save the raw Ledoux bracket before softhinge clamping, and compute
        # the Schwarzschild bracket (P,T derivatives only, no density stabilization).
        # Used by smooth_ledoux (composition D transition) and by the continuous
        # semiconvection model (Nu_T ramp, D_semi, semi heat flux).
        if smooth_ledoux or semiconvection_local:
            bracket_ledoux_raw_b = bracket_dsdr_b.copy()
            bracket_schwarz_b = np.zeros(N + 1)
            bracket_schwarz_b[1:N] = (
                (-np.diff(S) / denom) * norm +
                ((dSdY_pt_b[1:N] * np.diff(Y)) / denom) * dS_mult +
                ((dSdZ_pt_b[1:N] * np.diff(Z)) / denom) * dS_mult
            )

            # Old-state brackets (frozen during NR iteration) for stable
            # semiconvection detection that doesn't flicker between iterations.
            bracket_ledoux_old_b = np.zeros(N + 1)
            bracket_ledoux_old_b[1:N] = (
                (-np.diff(Sold) / denom) * norm +
                ((dSdY_b[1:N] * np.diff(Yold)) / denom) * dS_mult +
                ((dSdZ_b[1:N] * np.diff(Zold)) / denom) * dS_mult
            )
            bracket_schwarz_old_b = np.zeros(N + 1)
            bracket_schwarz_old_b[1:N] = (
                (-np.diff(Sold) / denom) * norm +
                ((dSdY_pt_b[1:N] * np.diff(Yold)) / denom) * dS_mult +
                ((dSdZ_pt_b[1:N] * np.diff(Zold)) / denom) * dS_mult
            )

        # calculate composition flux
        Y_i = get_arith_mean(Y)
        Y_b = np.concatenate(([0], Y_i, [0]))
        S_i = get_arith_mean(S)
        S_old_prime_i = get_arith_mean(S_old_prime)
        Z_i = get_arith_mean(Z)
        Z_b = np.concatenate(([0], Z_i, [0]))

        def softhinge(x, x0=1e-7, scale=1e-6, k=hinge_k, smooth_func='classic'):
            if hinge_smooth:
                if smooth_func == 'classic':
                    x = np.maximum(0, x)
                    return x ** 3 / (scale ** 2 + x ** 2)
                elif smooth_func == 'softplus':
                    z = k * x
                    smooth_bracket = (np.maximum(z, 0) + np.log1p(np.exp(-np.abs(z)))) / k
                else:
                    raise ValueError("Invalid smooth_func. Choose 'classic', 'logexp', 'softplus' or 'arctan'.")
                return smooth_bracket
            else:
                return np.maximum(0, x)

        def ds_dx(x, x0=1e-7, scale=1e-6, k=hinge_k, smooth_func='classic'):
            """Derivative of softhinge w.r.t. original (pre-transformed) bracket."""
            if hinge_smooth:
                if smooth_func == 'classic':
                    x_pos = np.maximum(0, x)
                    return x_pos ** 2 * (3 * scale ** 2 + x_pos ** 2) / ((scale ** 2 + x_pos ** 2) ** 2)
                elif smooth_func == 'softplus':
                    return 1 / (1 + np.exp(-x * k))
            else:
                return np.ones(len(x))

        # Evaluate derivative at ORIGINAL bracket before softhinge transforms it
        # Save the RAW (pre-softhinge) bracket so smooth_convection_criterion
        # can use the full negative-to-positive range for its sigmoid blend.
        # In stable regions bracket_dsdr_b post-softhinge is exactly 0; pre-softhinge
        # it is negative, which is what we need for sigma_conv_b -> 0 there.
        bracket_raw_b = bracket_dsdr_b.copy()
        d_bracket_dsdr_b = ds_dx(bracket_dsdr_b, smooth_func=smooth_func)
        bracket_dsdr_b = softhinge(bracket_dsdr_b, smooth_func=smooth_func)

        flux_dsdr_conv_b = np.power(bracket_dsdr_b * R0 / norm, flux_bracket_power_eval)

        if hinge_smooth:
            d_flux_dS = flux_bracket_power_eval * np.power(
                np.power(flux_dsdr_conv_b, (1 / flux_bracket_power_eval)),
                (flux_bracket_power_eval - 1),
            ) * d_bracket_dsdr_b
        else:
            d_flux_dS = flux_bracket_power_eval * np.power(
                np.power(flux_dsdr_conv_b, (1 / flux_bracket_power_eval)),
                (flux_bracket_power_eval - 1),
            )

        # Additional Jacobian terms from χ-dependent mantle transport geometry:
        # convective flux includes φ(χ) and exponent n(χ), so we add
        # dF_conv/dχ * dχ/d(S,Y,Z) contributions explicitly.
        melt_flux_jac_terms = []
        if (
            implicit_viscous_transition
            and use_chi_for_transport
            and (n_mantle_bnds > 0)
            and (mantle_boundary_idx.size == n_mantle_bnds)
            and (mantle_transport_prefactor.size == n_mantle_bnds)
        ):
            chi_cells_local = np.asarray(chi_transport_eval_full[mantle_slice_cells], dtype=float)
            if chi_cells_local.size == (kcore_fe - kcore):
                chi_b_local = _map_mantle_chi_to_boundaries(np.clip(chi_cells_local, 0.0, 1.0))
            else:
                chi_b_local = np.zeros(0)

            if chi_b_local.size == n_mantle_bnds:
                dT_dSprime_cell = T0 * (T_prime / Cv_safe) * S0
                dT_dYprime_cell = T0 * dTdY_prime
                dT_dZprime_cell = T0 * dTdZ_prime

                def _local_conv_flux_from_chi(chi_val, lb, bidx):
                    chi_eff = float(np.clip(chi_val, 0.0, 1.0))
                    kappa_inv = np.maximum(
                        chi_eff * mantle_kappa_34a[lb] + (1.0 - chi_eff) * mantle_kappa_35a[lb],
                        mantle_kappa_floor,
                    )
                    kappa_vis = np.maximum(
                        (1.0 - chi_eff) * mantle_kappa_34b[lb] + chi_eff * mantle_kappa_35b[lb],
                        mantle_kappa_floor,
                    )
                    log_kappa_eff = (
                        chi_eff * np.log10(kappa_inv)
                        + (1.0 - chi_eff) * np.log10(kappa_vis)
                    )
                    kappa_eff = 10.0 ** log_kappa_eff
                    exponent_eff = 1.5 + 0.5 * (1.0 - chi_eff)
                    phi_local = mantle_transport_prefactor[lb] * kappa_eff
                    phi_prime_local = (
                        phi_local
                        * np.power(S0, exponent_eff - 1.0)
                        / (np.power(R0, exponent_eff) * M0 * T0)
                    )
                    bracket_local = np.maximum(bracket_dsdr_b[bidx] * R0 / norm, 0.0)
                    flux_local = np.power(bracket_local, exponent_eff)
                    return -phi_prime_local * flux_local

                for lb in range(n_mantle_bnds):
                    bint = int(mantle_boundary_idx[lb])      # interior-boundary index (0..N-2)
                    bidx = bint + 1                          # boundary index in _b arrays (1..N-1)
                    if not (1 <= bidx <= N - 1):
                        continue
                    if not np.isfinite(chi_b_local[lb]):
                        continue

                    chi0 = float(np.clip(chi_b_local[lb], 0.0, 1.0))
                    eps_chi = 1e-6
                    chi_p = min(1.0, chi0 + eps_chi)
                    chi_m = max(0.0, chi0 - eps_chi)
                    if chi_p == chi_m:
                        dF_dchi = 0.0
                    else:
                        flux_p = _local_conv_flux_from_chi(chi_p, lb, bidx)
                        flux_m = _local_conv_flux_from_chi(chi_m, lb, bidx)
                        dF_dchi = (flux_p - flux_m) / (chi_p - chi_m)
                    if not np.isfinite(dF_dchi) or abs(dF_dchi) == 0.0:
                        continue

                    # Boundary bidx contributes with + sign to left-cell residual
                    # and - sign to right-cell residual through dflux = F[k+1]-F[k].
                    k_left_cell = bidx - 1
                    k_right_cell = bidx
                    coef_left = dt / dm_prime[k_left_cell] / T_prime[k_left_cell]
                    coef_right = dt / dm_prime[k_right_cell] / T_prime[k_right_cell]

                    for q_cell, q_w in (
                        (int(chi_map_cell_a[lb]), float(chi_map_w_a[lb])),
                        (int(chi_map_cell_b[lb]), float(chi_map_w_b[lb])),
                    ):
                        if q_cell < 0 or q_w == 0.0:
                            continue
                        dchi_dS_q = q_w * dchi_dT_transport_full[q_cell] * dT_dSprime_cell[q_cell]
                        dchi_dY_q = q_w * dchi_dT_transport_full[q_cell] * dT_dYprime_cell[q_cell]
                        dchi_dZ_q = q_w * dchi_dT_transport_full[q_cell] * dT_dZprime_cell[q_cell]

                        if np.isfinite(dchi_dS_q) and dchi_dS_q != 0.0:
                            dF_dS_q = dF_dchi * dchi_dS_q
                            melt_flux_jac_terms.append((3 * k_left_cell, 3 * q_cell, coef_left * dF_dS_q))
                            melt_flux_jac_terms.append((3 * k_right_cell, 3 * q_cell, -coef_right * dF_dS_q))
                        if np.isfinite(dchi_dY_q) and dchi_dY_q != 0.0:
                            dF_dY_q = dF_dchi * dchi_dY_q
                            melt_flux_jac_terms.append((3 * k_left_cell, 3 * q_cell + 1, coef_left * dF_dY_q))
                            melt_flux_jac_terms.append((3 * k_right_cell, 3 * q_cell + 1, -coef_right * dF_dY_q))
                        if np.isfinite(dchi_dZ_q) and dchi_dZ_q != 0.0:
                            dF_dZ_q = dF_dchi * dchi_dZ_q
                            melt_flux_jac_terms.append((3 * k_left_cell, 3 * q_cell + 2, coef_left * dF_dZ_q))
                            melt_flux_jac_terms.append((3 * k_right_cell, 3 * q_cell + 2, -coef_right * dF_dZ_q))


        # ------------------- DIFFUSION COEFFICIENTS ----------------------- #
        if pressure_dependent_Dmicro:
            # Pressure-dependent diffusion from French (2012) + Preising (2023) ab initio data.
            # Cap at the configured constant to avoid overly large D at low surface
            # pressures, which can destabilize the Newton solver.
            _dmicro_Y = Yold_i if y_dependent_conductivity else None
            D_micro_b = np.zeros(N + 1)
            D_micro_b[1:-1] = np.minimum(get_D_He(P_i, Y=_dmicro_Y), DY_micro_val)
            DZ_micro_b = np.zeros(N + 1)
            DZ_micro_b[1:-1] = np.minimum(get_D_H(P_i, Y=_dmicro_Y), DZ_micro_val)
        else:
            D_micro_b = np.zeros(N + 1) + DY_micro_val
            DZ_micro_b = np.zeros(N + 1) + DZ_micro_val
        # Use enclosed-mass-fraction mask (unit-safe) for interior Z diffusion boost.
        DZ_micro_b[inner_z_diff_mask_b] = DZ_micro_val_in

        if D_MLT:
            # local MLT approach
            v_MLT_i = np.sqrt(g_i * Hp_i ** 2 / Cp_i / 8 * bracket_dsdr_b[1:-1])
            D_mlt_b = np.zeros(N + 1)
            D_mlt_b[1:-1] = (v_MLT_i * Hp_i / 3.0)
            D_b_new = D_mlt_b + D_micro_b
            DZ_b_new = D_mlt_b + DZ_micro_b
        else:
            D_b_new = D_micro_b

        # Smooth convective criterion: replace the hard "if bracket > 0 then
        # D_mlt else D_micro" gate with a sigmoid blend
        #     D_b = sigma_conv * D_mlt + D_micro
        # where sigma_conv is built from the RAW (pre-softhinge) bracket.
        # This eliminates the staircase artifact in composition gradients,
        # where alternating cells get D ~ 1e10 vs D_micro = 0.05 because the
        # bracket flips just-above / just-at zero.  When disabled (default),
        # sigma_conv_b is left at 1.0 (no effect on existing physics, since
        # D_mlt_b is already 0 in stable regions via softhinge).
        sigma_conv_b = np.ones(N + 1)
        dsigma_conv_db_b = np.zeros(N + 1)
        # D_mlt_smooth_b holds the |raw_bracket|-derived MLT diffusivity used
        # by the smooth_conv blend; mirrors D_mlt_b when smooth_conv is off,
        # and provides D_mlt_b / sigma_conv_b when on (used by the Jacobian).
        D_mlt_smooth_b = D_mlt_b.copy() if D_MLT else np.zeros(N + 1)
        if smooth_convection_criterion and D_MLT:
            scale = smooth_conv_scale if smooth_conv_scale > 0.0 else None
            # Spatially smooth bracket_raw before applying the sigmoid -- this is the
            # key step that breaks the staircase pattern.  The Newton solver lands
            # adjacent cells alternately on bracket > 0 and bracket = 0 in the
            # gradient region; a Gaussian filter over ~2 cells averages those out so
            # adjacent cells get similar sigma_conv (~0.5) rather than 0/1 alternation.
            if smooth_conv_bracket_sigma > 0.0:
                # Smooth in the interior; keep boundaries (idx 0 and N) untouched.
                bracket_for_sigma = bracket_raw_b.copy()
                bracket_for_sigma[1:-1] = gaussian_filter1d(
                    bracket_raw_b[1:-1], sigma=smooth_conv_bracket_sigma, mode='nearest',
                )
            else:
                bracket_for_sigma = bracket_raw_b
            sigma_conv_b, dsigma_conv_db_b = smooth_conv_indicator(
                bracket_for_sigma, scale=scale, k=smooth_conv_k,
            )
            # Re-build D_b_new and DZ_b_new with the smooth blend.  We use the
            # SMOOTHED bracket (bracket_for_sigma) in v_MLT so D_mlt_smooth_b
            # is non-zero and continuous across cells in the gradient region.
            # If we used bracket_raw_b directly, D_mlt_smooth would alternate
            # because bracket_raw alternates between small-positive (in
            # convective cells) and 0 / negative (in stable cells).
            v_MLT_smooth_i = np.sqrt(
                g_i * Hp_i ** 2 / Cp_i / 8 * np.abs(bracket_for_sigma[1:-1])
            )
            D_mlt_smooth_b = np.zeros(N + 1)
            D_mlt_smooth_b[1:-1] = v_MLT_smooth_i * Hp_i / 3.0
            D_b_new = sigma_conv_b * D_mlt_smooth_b + D_micro_b
            DZ_b_new = sigma_conv_b * D_mlt_smooth_b + DZ_micro_b
            # Keep D_mlt_b consistent for downstream code (Jacobian etc.).
            D_mlt_b = sigma_conv_b * D_mlt_smooth_b

        # --- Smooth Ledoux transition: add transitional diffusivity ---
        # In zones that are Schwarzschild-unstable but Ledoux-stable, smoothly
        # ramp composition diffusivity from D_mlt(Schwarzschild) down to zero,
        # instead of the hard cutoff at the Ledoux boundary.
        # This reduces artificial staircase artifacts at convective/stable interfaces.
        if smooth_ledoux and D_MLT:
            # Schwarzschild-based MLT diffusivity (what D_mlt would be without
            # composition density stabilization).
            bracket_schwarz_pos_i = np.maximum(bracket_schwarz_b[1:-1], 0.0)
            v_MLT_schwarz_i = np.sqrt(
                g_i * Hp_i ** 2 / Cp_i / 8 * bracket_schwarz_pos_i
            )
            D_schwarz_i = v_MLT_schwarz_i * Hp_i / 3.0

            # Transition zone: Schwarzschild > 0 (thermally unstable) AND
            #                   Ledoux <= 0 (composition-stabilized).
            schwarz_pos_i = bracket_schwarz_b[1:-1] > 0
            ledoux_neg_i = bracket_ledoux_raw_b[1:-1] <= 0
            transition_i = schwarz_pos_i & ledoux_neg_i

            # Stability parameter chi: 0 at the Schwarzschild boundary (barely
            # Ledoux-stable), 1 deep in the Ledoux-stable zone.
            # chi = |bracket_ledoux| / (bracket_schwarz + |bracket_ledoux|)
            denom_stab_i = (
                bracket_schwarz_b[1:-1] - bracket_ledoux_raw_b[1:-1]
            )
            denom_stab_i = np.maximum(denom_stab_i, 1e-30)

            chi_i = np.zeros(N - 1)
            chi_i[transition_i] = np.clip(
                -bracket_ledoux_raw_b[1:-1][transition_i]
                / denom_stab_i[transition_i],
                0.0,
                1.0,
            )

            # Smooth ramp: f = (1 - chi)^n
            f_transition_i = np.zeros(N - 1)
            f_transition_i[transition_i] = (
                (1.0 - chi_i[transition_i]) ** smooth_ledoux_n
            )

            # Add transitional diffusivity (only in transition zones)
            D_b_new[1:-1] += f_transition_i * D_schwarz_i
            DZ_b_new[1:-1] += f_transition_i * D_schwarz_i

        t_Db_relax_curr = min(2 * dt, t_Db_relax)

        if t_Db_relax_curr > 0:         # Limit changes in D_b for numerical stability

            # -----------------------------
            # D (Y diffusion) relaxation
            # -----------------------------
            # BUGFIX: avoid log(0) / log(neg)
            D_floor = 1e-99  # choose a floor appropriate for your units
            Dnew = np.maximum(D_b_new[:kcore], D_floor)
            Dmic = np.maximum(0.99 * D_micro_b[:kcore], D_floor)
            Dold = np.maximum(D_b_old[:kcore], D_floor)

            D_A  = np.log(Dnew)
            D_B  = np.log(Dmic)
            D_y0 = np.log(Dold)

            # Same exp_arg form you already use, but made safe
            # BUGFIX: protect denom from ~0
            denom = np.abs(D_A * D_B)
            denom = np.maximum(denom, 1e-30)

            exp_arg = -dt / t_Db_relax_curr * (D_A - D_B) / denom

            # BUGFIX: robust relaxation (no NaNs/Infs)
            D_y = relax_D_safely(
                D_y0=D_y0,
                D_A=D_A,
                D_B=D_B,
                exp_arg=exp_arg,
                dt=dt,
                tau=t_Db_relax_curr,
                D_floor=None,   # D_y is log-space; don't floor it here
                apply_hull=relax_hull_clamp,
            )

            # Back to linear space and stitch with outer region
            D_b = np.concatenate([np.exp(D_y), D_b_new[kcore:]])

            # Jacobian damping factor of this relaxation operator (consumed
            # by the bracket-sensitivity block below).  The Mobius form pulls
            # ln D toward the fresh target D_A = ln(D_b_new) at rate
            # lam = |exp_arg|, so to leading order
            #     d(D_used)/d(D_new) ~ (D_used/D_new) * (1 - exp(-lam)).
            # Without this factor the Jacobian uses the undamped sensitivity:
            # harmless when dt >> tau (lam >~ 4, factor ~ 1) but ~5x too
            # large at rain-era dt = 0.5 Myr with tau = 1 Myr (lam ~ 0.2),
            # driving Newton overshoot/oscillation in flickering rain cells
            # (the stalled-accept/max-iter exits this block's comment warns
            # about).  Clipped to [0, 1]: an under-stated sensitivity only
            # slows Newton; an over-stated one destabilizes it.
            w_relax_y_b = np.ones(N + 1)
            w_relax_y_b[:kcore] = np.clip(
                (D_b[:kcore] / np.maximum(D_b_new[:kcore], D_floor))
                * (1.0 - np.exp(-np.abs(exp_arg))),
                0.0, 1.0,
            )


            # -----------------------------
            # DZ (Z diffusion) relaxation
            # -----------------------------
            # BUGFIX: avoid log(0) / log(neg)
            DZnew = np.maximum(DZ_b_new[:kcore], D_floor)
            DZmic = np.maximum(0.99 * DZ_micro_b[:kcore], D_floor)
            DZold = np.maximum(DZ_b_old[:kcore], D_floor)

            DZ_A = np.log(DZnew)
            DZ_B = np.log(DZmic)
            D_z0 = np.log(DZold)

            # Same exp_arg form, protected
            denom_z = np.abs(DZ_A * DZ_B)
            denom_z = np.maximum(denom_z, 1e-30)

            exp_arg_z = -dt / t_Db_relax_curr * (DZ_A - DZ_B) / denom_z

            # BUGFIX: robust relaxation (no NaNs/Infs)
            D_z = relax_D_safely(
                D_y0=D_z0,
                D_A=DZ_A,
                D_B=DZ_B,
                exp_arg=exp_arg_z,
                dt=dt,
                tau=t_Db_relax_curr,
                D_floor=None,   # log-space
                apply_hull=relax_hull_clamp,
            )

            DZ_b = np.concatenate([np.exp(D_z), DZ_b_new[kcore:]])

            # Z-channel relaxation damping factor (same form as w_relax_y_b).
            w_relax_z_b = np.ones(N + 1)
            w_relax_z_b[:kcore] = np.clip(
                (DZ_b[:kcore] / np.maximum(DZ_b_new[:kcore], D_floor))
                * (1.0 - np.exp(-np.abs(exp_arg_z))),
                0.0, 1.0,
            )

        else:
            D_b = D_b_new
            DZ_b = DZ_b_new
            # No relaxation applied -> no Jacobian damping.
            w_relax_y_b = np.ones(N + 1)
            w_relax_z_b = np.ones(N + 1)

        #if np.all(D_b < 0):
        assert np.all(D_b >= 0)
        assert np.all(DZ_b >= 0)

        # Ratio clamp vs the first NR iteration.  Guarded against faces that
        # were DEAD (D == 0) at the first iteration and ignite mid-solve: the
        # unguarded ratio is inf (x/0) or nan (0/0) and poisons the Jacobian
        # (can occur when a dead face ignites mid-solve).  A dead
        # first-iteration face is held dead for this step -- identical to the
        # old finite x/0 arithmetic (0 * clipped) -- and ignites on the next
        # call, whose first iteration sees the new state.
        if D_b_first is None:
            D_b_first = np.copy(D_b)
        else:
            _first_live = D_b_first > 0.0
            fractional_change = D_b / np.maximum(D_b_first, 1e-300)
            new_change = np.clip(fractional_change, 1/max_Db_change, max_Db_change)
            D_b = np.where(_first_live, D_b_first * new_change, 0.0)

        if DZ_b_first is None:
            DZ_b_first = np.copy(DZ_b)
        else:
            _first_live = DZ_b_first > 0.0
            fractional_change = DZ_b / np.maximum(DZ_b_first, 1e-300)
            new_change = np.clip(fractional_change, 1/max_Db_change, max_Db_change)
            DZ_b = np.where(_first_live, DZ_b_first * new_change, 0.0)

        # ------------------------------------------------------------------
        # Bracket sensitivity of the diffusion coefficients D_b and DZ_b.
        #
        # The MLT diffusion coefficient D_mlt ∝ sqrt(bracket_dsdr) depends
        # on S, Y, Z through the convective bracket:
        #   bracket = (-dS/dr)*norm + (dSdY*dY/dr + dSdZ*dZ/dr)*dS_mult
        #
        # Chain rule: d(D_b)/dX = d(D_mlt)/d(bracket) * d(bracket)/dX
        #   where d(D_mlt)/d(bracket) = D_mlt / (2*bracket)  [from sqrt]
        #
        # These derivatives enter the Y and Z equation Jacobians through
        # the diffusion terms: sigma1p * diff(Y) and sigma1p_z * diff(Z).
        # Without these terms, the Jacobian treats D_b as constant, causing
        # NR non-convergence at the Ledoux boundary where D_b is highly
        # sensitive to composition changes.
        #
        # Note: the t_Db_relax relaxation and D_b_first clamping applied
        # above slightly damp d(D_b)/d(bracket) relative to the raw
        # d(D_mlt)/d(bracket).  We use the undamped value as an
        # approximation — it overestimates the sensitivity but is much
        # better than zero for Newton convergence.
        # ------------------------------------------------------------------
        if D_MLT:
            tiny_bracket = 1e-30
            dDmlt_dBracket_b = np.zeros(N + 1)
            pos_bracket = bracket_dsdr_b[1:-1] > tiny_bracket
            dDmlt_dBracket_b[1:-1] = np.where(
                pos_bracket,
                D_mlt_b[1:-1] / (2.0 * np.maximum(bracket_dsdr_b[1:-1], tiny_bracket))
                * d_bracket_dsdr_b[1:-1],
                0.0,
            )
            # Smooth-conv chain-rule term: D_b = sigma_conv * D_mlt_smooth + D_micro
            # contributes  d sigma / d bracket * D_mlt_smooth  in addition to the
            # standard  sigma * d D_mlt_smooth / d bracket  already included via
            # the formula above.  Identity (zero contribution) when smooth_conv
            # is disabled because dsigma_conv_db_b is all zero.
            if smooth_convection_criterion:
                dDmlt_dBracket_b += dsigma_conv_db_b * D_mlt_smooth_b
            # Cap the fractional sensitivity to prevent blowup near bracket = 0
            max_frac_sensitivity = 10.0
            dDmlt_dBracket_b[1:-1] = np.minimum(
                dDmlt_dBracket_b[1:-1],
                max_frac_sensitivity * np.maximum(D_b[1:-1], 1e-99)
            )

            # Account for the t_Db_relax time-lag operator (per channel):
            # differentiate the D_b/DZ_b actually used in the residual, not
            # the raw pre-relaxation value.  See w_relax_*_b above.
            dDmlt_dBracket_z_base = dDmlt_dBracket_b * w_relax_z_b
            dDmlt_dBracket_b = dDmlt_dBracket_b * w_relax_y_b

            # Where D_b_first clamping is active, attenuate (not zero) the
            # bracket sensitivity.  Zeroing removes crucial Jacobian
            # feedback at the Ledoux transition.  Instead, scale by the
            # ratio of clamped to unclamped D_b.
            if D_b_first is not None:
                frac_y = D_b / np.maximum(D_b_first, 1e-99)
                clamped_y = (frac_y < 1.0 / 1000) | (frac_y > 1000)
                D_mlt_unclamped = D_mlt_b if D_MLT else D_b
                atten_y = np.where(
                    D_mlt_unclamped > 1e-99,
                    D_b / np.maximum(D_mlt_unclamped, 1e-99),
                    0.0,
                )
                atten_y = np.clip(atten_y, 0.0, 1.0)
                dDmlt_dBracket_b = np.where(
                    clamped_y, dDmlt_dBracket_b * atten_y, dDmlt_dBracket_b
                )
            if DZ_b_first is not None:
                frac_z = DZ_b / np.maximum(DZ_b_first, 1e-99)
                clamped_z = (frac_z < 1.0 / 1000) | (frac_z > 1000)
                D_mlt_unclamped_z = D_mlt_b if D_MLT else DZ_b
                atten_z = np.where(
                    D_mlt_unclamped_z > 1e-99,
                    DZ_b / np.maximum(D_mlt_unclamped_z, 1e-99),
                    0.0,
                )
                atten_z = np.clip(atten_z, 0.0, 1.0)
                dDmlt_dBracket_z_b = np.where(
                    clamped_z,
                    dDmlt_dBracket_z_base * atten_z,
                    dDmlt_dBracket_z_base,
                )
            else:
                dDmlt_dBracket_z_b = dDmlt_dBracket_z_base.copy()

            # Bracket sensitivities to NR variables at each boundary.
            # denom_dr = np.diff(r) — recomputed because outer `denom` may
            # have been overwritten by the D_b relaxation block.
            denom_dr = np.diff(r)

            dBr_dS_R = np.zeros(N + 1)
            dBr_dS_L = np.zeros(N + 1)
            dBr_dS_R[1:N] = -norm / denom_dr
            dBr_dS_L[1:N] = +norm / denom_dr

            dBr_dY_R = np.zeros(N + 1)
            dBr_dY_L = np.zeros(N + 1)
            dBr_dY_R[1:N] = +dSdY_b[1:N] * dS_mult / denom_dr
            dBr_dY_L[1:N] = -dSdY_b[1:N] * dS_mult / denom_dr

            dBr_dZ_R = np.zeros(N + 1)
            dBr_dZ_L = np.zeros(N + 1)
            dBr_dZ_R[1:N] = +dSdZ_b[1:N] * dS_mult / denom_dr
            dBr_dZ_L[1:N] = -dSdZ_b[1:N] * dS_mult / denom_dr

            # Fractional sensitivity: d(sigma1p)/dX = sigma1p * dSigma1_frac * dBr_dX
            dSigma1_frac_b = np.zeros(N + 1)
            dSigma1_frac_b[1:-1] = np.where(
                D_b[1:-1] > 1e-30,
                dDmlt_dBracket_b[1:-1] / D_b[1:-1],
                0.0,
            )
            dSigma1z_frac_b = np.zeros(N + 1)
            dSigma1z_frac_b[1:-1] = np.where(
                DZ_b[1:-1] > 1e-30,
                dDmlt_dBracket_z_b[1:-1] / DZ_b[1:-1],
                0.0,
            )
        else:
            dSigma1_frac_b = np.zeros(N + 1)
            dSigma1z_frac_b = np.zeros(N + 1)
            dBr_dS_R = np.zeros(N + 1)
            dBr_dS_L = np.zeros(N + 1)
            dBr_dY_R = np.zeros(N + 1)
            dBr_dY_L = np.zeros(N + 1)
            dBr_dZ_R = np.zeros(N + 1)
            dBr_dZ_L = np.zeros(N + 1)

        # Time-lag the He-rain saturation (Ym) to avoid large jumps
        if ym_t_lag > 0:
            Ym_lag_i = (Ym_old_i / dt + Ym_i / ym_t_lag) / (1 / dt + 1 / ym_t_lag)
        else:
            Ym_lag_i = Ym_i

        if zm_t_lag > 0:
            Z1m_lag_i = (Z1m_old_i/dt + Z1m_i/zm_t_lag)/(1/dt + 1/zm_t_lag)
            Z2m_lag_i = (Z2m_old_i/dt + Z2m_i/zm_t_lag)/(1/dt + 1/zm_t_lag)
        else:
            Z1m_lag_i = Z1m_i
            Z2m_lag_i = Z2m_i

        # Rain gap (pressure bounds) based on supersaturation. He rain opens
        # it where Y_i > Ym_lag_i*(1-Z_i); metal rain mirrors this where
        # Z_i > Z1m_lag_i, so the rain-aware CFL timestep cap in evolution.py
        # sees Z rain exactly as it sees He rain.
        y_rain_gap_new = None
        y_rain_mask_i = (Y_i > Ym_lag_i * (1 - Z_i))
        rain_mask_any_i = y_rain_mask_i.copy()
        if z_advection_local:
            rain_mask_any_i |= (Z_i > Z1m_lag_i)
        if np.any(rain_mask_any_i):
            idx = np.flatnonzero(rain_mask_any_i)
            y_rain_gap_new = np.asarray([P_i[idx[0]], P_i[idx[-1]]], dtype=float)

        D2_b = np.zeros(N + 1)
        D2Z_b = np.zeros(N + 1)

        has_rain_gap = (y_rain_gap_new is not None)

        if advection_local and has_rain_gap:
            D2_b[1:-1] = D_b[1:-1]
        if z_advection_local:
            D2Z_b[1:-1] = DZ_b[1:-1] #* 10.0

        # No diffusion in the compact core.
        D2_b[kcore:] = 0
        D2Z_b[kcore:] = 0
        D_b[kcore:] = 0
        DZ_b[kcore:] = 0

        if evol_Z_local:
            DZ_b = DZ_b.copy()
        else:
            DZ_b = np.zeros(N + 1)

        if z_advection_local:
            D2Z_b = DZ_b.copy() #* 10.0

        try:
            assert_all_not_nan([D_b, D2_b, DZ_b, D2Z_b])
        except AssertionError:
            print("NaN detected in diffusion coefficients:")
            print("D_b:", D_b)
            print("D2_b:", D2_b)
            print("DZ_b:", DZ_b)
            print("D2Z_b:", D2Z_b)
            raise


        dsdr_i = np.diff(S) / np.diff(r) * norm
        dydr_i = np.diff(Y) / np.diff(r) * dS_mult
        dzdr_i = np.diff(Z) / np.diff(r) * dS_mult

        # Evaluate Schwarzschild vs Ledoux
        schwarz_i = (-dsdr_i + dSdY_pt_i * dydr_i + dSdZ_pt_i * dzdr_i)
        ledoux_i = (-dsdr_i + dSdY_rhop_i * dydr_i + dSdZ_rhop_i * dzdr_i)
        schwarz_safe = np.where(np.abs(schwarz_i) < 1e-20, 1e-20, schwarz_i)
        R0_inv_b[1:-1] = (schwarz_safe - ledoux_i) / schwarz_safe

        # Critical inverse density ratio for double-diffusive semiconvection:
        #   R_crit_inv = (Pr + 1) / (Pr + tau)
        # Semiconvection is active where 1 < 1/R0 < R_crit_inv.
        R_crit_inv = (Pr_val + 1.0) / (Pr_val + tau_val)

        # Semiconvective detection
        Nu_T_i = np.ones(N - 1)
        if semiconvection_local:
            if semi_shoot:
                # In semi_shoot mode, freeze bulk semiconvection detection to the
                # old state for this timestep to reduce Newton-iteration flicker.
                denom_old = np.diff(r)
                dsdr_old_i = np.diff(Sold) / denom_old * norm
                dydr_old_i = np.diff(Yold) / denom_old * dS_mult
                dzdr_old_i = np.diff(Zold) / denom_old * dS_mult
                schwarz_old_i = (-dsdr_old_i + dSdY_pt_i * dydr_old_i + dSdZ_pt_i * dzdr_old_i)
                ledoux_old_i = (-dsdr_old_i + dSdY_rhop_i * dydr_old_i + dSdZ_rhop_i * dzdr_old_i)
                R0_inv_old_i = (schwarz_old_i - ledoux_old_i + ZERO) / (schwarz_old_i + ZERO)
                sc_idx = np.where((R0_inv_old_i > 1.0) & (R0_inv_old_i < R_crit_inv))[0]
            else:
                sc_idx = np.where((R0_inv_b[1:-1] > 1.0) & (R0_inv_b[1:-1] < R_crit_inv))[0]
        else:
            sc_idx = np.array([], dtype=int)

        # Identify consecutive semiconv. zones (avoid single-zone noise)
        def get_consecutive_idxs(idxs, len_consec=sc_zones_min):
            # Guard the len(idxs) <= 1 corner: the loop below never runs, so
            # the trailing block would reference an unbound loop variable
            # (audit #66 -- UnboundLocalError killed whole runs whenever
            # sc_zones_min <= 1 selected a single zone).
            if len(idxs) == 0:
                return np.array([], dtype=int)
            if len(idxs) == 1:
                return (np.asarray(idxs, dtype=int) if len_consec <= 1
                        else np.array([], dtype=int))
            idxs_ret = []
            running_len = 1
            for i in range(1, len(idxs)):
                if idxs[i] - idxs[i - 1] == 1:
                    running_len += 1
                else:
                    if running_len >= len_consec:
                        idxs_ret.extend(idxs[i - running_len : i])
                    running_len = 1
            if running_len >= len_consec:
                idxs_ret.extend(idxs[i - running_len + 1 : i + 1])
            return np.array(idxs_ret, dtype=int)

        sc_idx = get_consecutive_idxs(sc_idx)

        # Additional semiconvection flux
        semi_bracket_b = np.zeros(N + 1)

        # Initialize time-lagged Nusselt numbers from previous timestep.
        # Updated in the continuous semiconvection path; kept at old values
        # (or snapped to 1) in all other paths.
        Nu_T_eff_i = Nu_T_eff_old.copy()
        Nu_X_eff_i = Nu_X_eff_old.copy()
        if semi_shoot and semiconvection_local and has_envelope and (kcore > 0):
            # Keep the original bulk semiconvection selection, then add a local
            # EMB-centered overshoot extension with exponential taper.
            if sc_idx.size > 0:
                Nu_T_i[sc_idx - 1] = Nu_T_val

            w_ov_i = np.zeros(N - 1)
            emb_i = kcore - 1  # interior-boundary index at envelope-mantle interface
            if 0 <= emb_i < (N - 1):
                ov_env = max(int(semi_shoot_len_env), 0)
                ov_mantle = max(int(semi_shoot_len_mantle), 0)
                i_start = max(0, emb_i - ov_env)             # outward (envelope) side
                i_stop = min(N - 2, emb_i + ov_mantle)       # inward (mantle) side
                if i_stop >= i_start:
                    ov_idx_i = np.arange(i_start, i_stop + 1, dtype=int)
                    hp_emb = float(np.clip(Hp_i[emb_i], 1e-30, None))
                    H_ov = max(float(semi_shoot_fhp), 0.0) * hp_emb
                    if H_ov > 0.0:
                        delta_r = np.abs(r_i[ov_idx_i] - r_i[emb_i])
                        w_ov_i[ov_idx_i] = np.exp(-delta_r / H_ov)

            # Cap the overshoot enhancement fraction to keep Jacobian updates tame.
            cap_frac = max(float(semi_shoot_cap_frac), 0.0)
            nu_t_ov_i = 1.0 + cap_frac * w_ov_i * (Nu_T_val - 1.0)
            Nu_T_i = np.maximum(Nu_T_i, nu_t_ov_i)

            semi_phi_b = np.concatenate((
                [0],
                4 * np.pi * r_i ** 2 * Lambda_i * (Nu_T_i - 1) * T_i / Cp_i,
                [0],
            ))
            semi_phi_prime_b = semi_phi_b / (M0 * T0)

            # Keep historical bulk Nu_X behavior unchanged.
            if sc_idx.size > 0:
                D_b[sc_idx] = D_micro_b[sc_idx] * Nu_X_val
                DZ_b[sc_idx] = D_micro_b[sc_idx] * Nu_X_val

            # Apply tapered Nu_X extension only on envelope boundaries (no mantle/core composition bridge).
            w_ov_b = np.zeros(N + 1)
            w_ov_b[1:N] = w_ov_i
            b_all = np.arange(N + 1)
            ov_env_b = b_all[(w_ov_b > 0.0) & (b_all > 0) & (b_all < kcore)]
            if ov_env_b.size > 0:
                nu_x_ov = 1.0 + cap_frac * w_ov_b[ov_env_b] * (Nu_X_val - 1.0)
                D_b[ov_env_b] = np.maximum(D_b[ov_env_b], D_micro_b[ov_env_b] * nu_x_ov)
                DZ_b[ov_env_b] = np.maximum(DZ_b[ov_env_b], D_micro_b[ov_env_b] * nu_x_ov)

            # Only allow semi transport where Nu_T enhancement is active; clamp bracket
            # to positive values to avoid sign flips in marginally stable overshoot zones.
            semi_active_i = Nu_T_i > (1.0 + 1e-12)
            semi_bracket_i = np.zeros(N - 1)
            semi_bracket_i[semi_active_i] = np.maximum(schwarz_i[semi_active_i], 0.0)
            semi_bracket_b[1:-1] = semi_bracket_i
        else:
            # --- Continuous semiconvection model (envelope, semi_shoot=False) ---
            # Replaces the binary on/off detection with a smooth stability-
            # parameter (chi) approach.  Nu_T and D_semi ramp continuously
            # from their enhanced values (at the Schwarzschild boundary) to
            # their background values (deep in the Ledoux-stable interior).
            #
            # Old-state chi is used for detection so that Nu_T (and therefore
            # semi_phi) is frozen during the Newton iteration, giving the
            # Jacobian an accurate linear model.
            if semiconvection_local:
                # Double-diffusive semiconvection criterion using old-state
                # 1/R0 (frozen during NR to avoid flicker).
                # Semiconvection is active where 1 < 1/R0 < R_crit_inv.
                R0_inv_old_sc_i = (
                    (bracket_schwarz_old_b[1:-1] - bracket_ledoux_old_b[1:-1] + ZERO)
                    / (bracket_schwarz_old_b[1:-1] + ZERO)
                )
                if semi_model == 'spruit':
                    # Spruit (2013): no layered state exists above the density-
                    # ratio ceiling R_rho = tau^(-1/2), which is NARROWER than
                    # the instability window's R_crit_inv.  Narrowing at
                    # detection (not post-filter) keeps the consecutive-zone
                    # filter semantics coherent: zones above the ceiling may
                    # not vouch for the contiguity of a layered patch.
                    if spruit_tau_local:
                        tau_loc_i = np.clip(
                            D_micro_b[1:-1]
                            / np.maximum(kappa_T_micro_i, 1e-30),
                            1e-12, 1.0)
                    else:
                        tau_loc_i = np.full(N - 1, tau_val)
                    Rrho_ceiling_i = tau_loc_i ** -0.5
                    sc_transition_raw = ((R0_inv_old_sc_i > 1.0)
                                         & (R0_inv_old_sc_i < Rrho_ceiling_i))
                    _sc_win_raw = sc_transition_raw.copy()  # pre-filter window
                    # (2026-07-28 de-kluge: the deposit-band and He-misc
                    # window exclusions of 2026-07-18 are REMOVED.  They
                    # were stall band-aids whose root causes are now fixed
                    # (spurious dead-face semi Jacobian couplings deleted;
                    # deposit-band T/rho error exemption).  The un-excluded
                    # eroding champion completes 4.57 Gyr with DDC active
                    # at the deep staircase from 1.4 Gyr -- the exclusions
                    # were suppressing real transport (T_eff shifts 24 K).
                    # The sc_zones_min consecutive filter remains the sole,
                    # physical single-face-noise gate.)
                else:
                    sc_transition_raw = (R0_inv_old_sc_i > 1.0) & (R0_inv_old_sc_i < R_crit_inv)
                    _sc_win_raw = sc_transition_raw.copy()

                # Apply consecutive-zone filter: require at least sc_zones_min
                # adjacent semiconvective zones to suppress single-zone noise.
                sc_raw_idx = np.where(sc_transition_raw)[0]
                sc_filt_idx = get_consecutive_idxs(sc_raw_idx)
                sc_transition_i = np.zeros(N - 1, dtype=bool)
                if sc_filt_idx.size > 0:
                    sc_transition_i[sc_filt_idx] = True
                # DDC-window visibility (change-only, gate-print precedent):
                # WHERE the double-diffusive window sits, how much of it the
                # stall-guard exclusions + consecutive-zone filter remove.
                # Face i (interior index) = boundary b = i+1; m/M from m_b.
                global _ddc_log_prev
                _n_win = int(np.sum(_sc_win_raw))
                _n_eff = int(np.sum(sc_transition_i))
                if _ddc_log_prev != (_n_win, _n_eff):
                    if _n_win > 0:
                        _iw = np.where(_sc_win_raw)[0]
                        _mmw = (m_b[_iw[0] + 1] / m_b[0],
                                m_b[_iw[-1] + 1] / m_b[0])
                        _wtxt = (f"{_n_win} faces b[{_iw[0]+1}..{_iw[-1]+1}] "
                                 f"m/M[{_mmw[1]:.3f}..{_mmw[0]:.3f}]")
                    else:
                        _wtxt = "0 faces"
                    if _n_eff > 0:
                        _ie = np.where(sc_transition_i)[0]
                        _etxt = f"{_n_eff} faces b[{_ie[0]+1}..{_ie[-1]+1}]"
                    else:
                        _etxt = "0 faces"
                    print("DDC WINDOW: t=%.4f Gyr raw=%s -> active=%s "
                          "(excl+filter removed %d)"
                          % (t * const.s_to_Gyr, _wtxt, _etxt,
                             _n_win - _n_eff), flush=True)
                    _ddc_log_prev = (_n_win, _n_eff)

                if semi_model == 'spruit':
                    # --- Spruit (2013) / Fuentes et al. (2026) targets ---
                    # Thermal:      Nu_T = 1 + tau^(-1/2)/R_rho
                    #               (kappa_T_eff = kappa_T * tau^(-1/2)/R_rho)
                    # Composition:  D_eff = kappa_T * tau^(1/2)/R_rho, same for
                    #               He and Z, expressed as a Nusselt multiplier
                    #               Nu_X = D_eff/D_micro so the existing lag ->
                    #               D_semi = D_micro*Nu_X_eff machinery
                    #               reproduces D_eff unchanged.
                    # Targets are functions of OLD-state quantities only, so
                    # the ceiling discontinuity never enters a Newton solve.
                    Rrho_safe_i = np.maximum(R0_inv_old_sc_i, 1.0 + 1e-12)
                    Nu_T_tgt_i = np.ones(N - 1)
                    Nu_X_tgt_i = np.ones(N - 1)
                    _m_sp = sc_transition_i
                    if np.any(_m_sp):
                        Nu_T_tgt_i[_m_sp] = 1.0 + (
                            tau_loc_i[_m_sp] ** -0.5 / Rrho_safe_i[_m_sp]
                        ) * lam_ratio_i[_m_sp]
                        D_eff_sp = (kappa_T_micro_i[_m_sp]
                                    * tau_loc_i[_m_sp] ** 0.5
                                    / Rrho_safe_i[_m_sp])
                        Nu_X_tgt_i[_m_sp] = np.clip(
                            D_eff_sp
                            / np.maximum(D_micro_b[1:-1][_m_sp], 1e-12),
                            1.0, spruit_nu_x_max)
                        if spruit_edge_frac > 0.0:
                            # Smoothstep taper over the top spruit_edge_frac
                            # of the window, softening Spruit's discontinuous
                            # shutoff at the ceiling for the solver.
                            _w_edge = spruit_edge_frac * (
                                Rrho_ceiling_i[_m_sp] - 1.0)
                            _s_edge = np.clip(
                                (Rrho_ceiling_i[_m_sp] - Rrho_safe_i[_m_sp])
                                / np.maximum(_w_edge, 1e-30), 0.0, 1.0)
                            _f_edge = _s_edge * _s_edge * (3.0 - 2.0 * _s_edge)
                            Nu_T_tgt_i[_m_sp] = 1.0 + (
                                Nu_T_tgt_i[_m_sp] - 1.0) * _f_edge
                            Nu_X_tgt_i[_m_sp] = 1.0 + (
                                Nu_X_tgt_i[_m_sp] - 1.0) * _f_edge
                else:
                    # Stability parameter chi_sc: maps the double-diffusive window
                    # [1, R_crit_inv] onto [0, 1].
                    #   chi = 0  at 1/R0 = 1      (Schwarzschild boundary, full SC)
                    #   chi = 1  at 1/R0 = R_crit  (critical boundary, SC shuts off)
                    chi_sc_i = np.zeros(N - 1)
                    chi_sc_i[sc_transition_i] = np.clip(
                        (R0_inv_old_sc_i[sc_transition_i] - 1.0) / (R_crit_inv - 1.0),
                        0.0, 1.0,
                    )

                    # Smooth ramp:  f = (1 - chi)^n
                    #   f = 1  at Schwarzschild boundary  (full semiconvection)
                    #   f = 0  at R_crit boundary          (no semiconvection)
                    f_sc_i = np.zeros(N - 1)
                    f_sc_i[sc_transition_i] = (
                        (1.0 - chi_sc_i[sc_transition_i]) ** smooth_ledoux_n
                    )

                    # Continuous Nu_T / Nu_X targets from chi ramp.
                    Nu_T_tgt_i = 1.0 + (Nu_T_val - 1.0) * f_sc_i
                    Nu_X_tgt_i = 1.0 + (Nu_X_val - 1.0) * f_sc_i

                # Time-lag Nu_T and Nu_X: exponential relaxation toward target.
                # If no semiconvective zones exist this timestep, snap to 1
                # (no lag applied) so that stale enhancements don't linger.
                # SPRUIT-scoped fix (audit #142): the D_b timescale
                # t_Db_relax_curr = min(2*dt, t_Db_relax) pins
                # theta_nu ~ 0.39 whenever dt < t_Db_relax/2, so the
                # configured relaxation time was IGNORED and the lag
                # completed in ~2.5 steps regardless.  spruit uses the
                # full configured t_Db_relax; the legacy nusselt path
                # keeps the historical (pinned) behavior bit-for-bit.
                _t_nu_relax = (t_Db_relax if semi_model == 'spruit'
                               else t_Db_relax_curr)
                if _t_nu_relax > 0 and np.any(sc_transition_i):
                    theta_nu = 1.0 - np.exp(-dt / _t_nu_relax)
                    Nu_T_eff_i = Nu_T_eff_old + theta_nu * (Nu_T_tgt_i - Nu_T_eff_old)
                    Nu_X_eff_i = Nu_X_eff_old + theta_nu * (Nu_X_tgt_i - Nu_X_eff_old)
                else:
                    # No relaxation timescale or no SC zones: use instantaneous target.
                    Nu_T_eff_i = Nu_T_tgt_i.copy()
                    Nu_X_eff_i = Nu_X_tgt_i.copy()

                # Clamp from below at 1 (Nusselt number cannot be sub-unity).
                Nu_T_eff_i = np.maximum(Nu_T_eff_i, 1.0)
                Nu_X_eff_i = np.maximum(Nu_X_eff_i, 1.0)

                Nu_T_i = Nu_T_eff_i

                # Semi_phi: semiconvective heat transport coefficient.
                # Naturally zero outside the transition zone (Nu_T = 1 there).
                semi_phi_b = np.concatenate((
                    [0],
                    4 * np.pi * r_i ** 2 * Lambda_i * (Nu_T_i - 1) * T_i / Cp_i,
                    [0],
                ))
                semi_phi_prime_b = semi_phi_b / (M0 * T0)

                # Continuous D_semi (compositional Nusselt enhancement).
                # Use max() blending so smooth_ledoux D_transition is preserved.
                # ENVELOPE FACES ONLY (audit #13): the core masking
                # D_b[kcore:] = 0 ran earlier, so an unmasked blend would
                # re-populate core faces with D >= D_micro and DRAIN the
                # envelope base against composition-pinned core cells --
                # He/Z mass destroyed, not transported (measured -1.3e-5
                # He drift per 0.05 Gyr).  Interior face b = i+1, so faces
                # >= kcore are i-indices >= kcore-1.
                D_semi_i = D_micro_b[1:-1] * Nu_X_eff_i
                if 0 < kcore <= N - 1:
                    D_semi_i[kcore - 1:] = 0.0
                D_b[1:-1] = np.maximum(D_b[1:-1], D_semi_i)
                DZ_b[1:-1] = np.maximum(DZ_b[1:-1], D_semi_i)

                # Semi bracket: current-state Schwarzschild gradient, clamped
                # to positive.  semi_phi naturally zeroes non-SC zones.
                semi_bracket_b[1:-1] = np.maximum(schwarz_i, 0.0)
            else:
                semi_phi_prime_b = np.zeros(N + 1)
                # No semiconvection: snap Nusselt numbers to baseline.
                Nu_T_eff_i = np.ones(N - 1)
                Nu_X_eff_i = np.ones(N - 1)

            if mc > 0 and kcore > 0:
                # Semiconvection is an envelope phenomenon; do not apply the
                # semi flux "bridge" into the compact rocky/iron region.
                # Zero the ENVELOPE-MANTLE face and everything deeper: the
                # EMB face is b-index kcore (between cell kcore-1 and cell
                # kcore).  The previous form zeroed b-index kcore-1 -- an
                # intra-envelope face -- and left the actual EMB flux live
                # (audit #62).
                semi_phi_prime_b[kcore:] = 0

        flux_dsdr_semi_b = semi_bracket_b / norm

        # Smooth helpers used by rain activation products and diffusion
        # suppression.  Defined here (before flux setup) so they are
        # available in the diffusion-suppression block below.
        # (rain_softplus_beta / rain_sign_eps are module-level constants.)

        def smooth_positive(x, beta):
            # Numerically stable softplus(beta*x)/beta.
            z = beta * x
            return (np.maximum(z, 0.0) + np.log1p(np.exp(-np.abs(z)))) / beta

        def smooth_positive_deriv(x, beta):
            # Derivative of smooth_positive w.r.t. x: sigmoid(beta*x).
            z = beta * x
            return np.where(z >= 0,
                            1.0 / (1.0 + np.exp(-z)),
                            np.exp(z) / (1.0 + np.exp(z)))

        def smooth_sign(x, eps):
            # Differentiable sign surrogate in [-1, 1].
            return x / np.sqrt(x * x + eps * eps)

        # Preallocate the flux arrays (length N+1)
        Yflux1_b = np.zeros(N + 1)
        Yflux2_b = np.zeros(N + 1)
        Zflux1_b = np.zeros(N + 1)
        Zflux2_b = np.zeros(N + 1)

        common_area = 4 * np.pi * r_i[:N-1]**2 * rho_i[:N-1]
        # Also, use dr computed from r_i differences. Assuming:
        #    dr = r_i[1:] - r_i[:-1]  -> shape (N)
        # We need dr for indices 0 to N-2:

        # Now assign the fluxes over indices 1..N-1 (total of N-1 elements)
        Yflux1_b[1:N] = common_area * D_b[1:N] / dr[:N-1]
        Zflux1_b[1:N] = common_area * DZ_b[1:N] / dr[:N-1]

        # ------------------------------------------------------------------
        # Advective (sedimentation) flux coefficient: Yflux2_b, Zflux2_b
        #
        # Historically two rain models existed ('legacy' and 'bottom_up',
        # Helled, Müller & Knierim 2025, A&A, eq. 8); they were unified into
        # this single implicit scheme.  The advection velocity is D2/Hp,
        # where D2 is the convective diffusion coefficient and Hp is a
        # helium rain scale-height parameter.  The resulting velocity is
        # very small (~1e-8 cm/s) and the flux is modulated by the
        # Yflux_new / Zflux_new activation products (smooth functions of
        # supersaturation) in the Jacobian and residual; sedimentation is
        # handled implicitly in the Newton solver.
        #
        # Convective composition diffusion is smoothly suppressed in the
        # rain zone (controlled by rain_conv_supp_k) so that advective
        # sedimentation can dominate over mixing.  Setting
        # rain_conv_supp_k = 0 (the default) disables the suppression.
        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # Advection velocity coefficients (Yflux2_b, Zflux2_b).
        #
        # The scheme uses v = D2/(Hp * misc_threshold) for Y and
        # v_z = D2Z/Hp for Z (same alpha_herain * Hp scale height).
        # Coefficients are set everywhere; the
        # Yflux_new / Zflux_new activation products naturally gate
        # the flux to the supersaturation region.
        #
        # Convective diffusion is smoothly suppressed in the rain zone
        # via rain_conv_supp_k (applied below).
        # ------------------------------------------------------------------

        # Set advection coefficients (unmasked – same for both models).
        Yflux2_b[1:N] = common_area * (D2_b[1:N] / Hp_rain_b[1:N] / misc_threshold)
        # Z rain advection mirrors Y: v_z = D2Z / (alpha_herain * Hp), using
        # the local pressure scale height when use_local_Hp is enabled.
        Zflux2_b[1:N] = common_area * (D2Z_b[1:N] / Hp_rain_b[1:N])

        # Build the boundary-centered rain mask from the cell-centered
        # supersaturation mask y_rain_mask_i (shape N-1).
        # A boundary is flagged if EITHER adjacent cell is
        # supersaturated.  Retained for diagnostics.
        y_rain_mask_b = np.zeros(N + 1, dtype=bool)
        y_rain_mask_b[1:N] = y_rain_mask_i
        y_rain_mask_b[:N-1] |= y_rain_mask_i

        # Smooth diffusion suppression in the rain zone.
        #
        # Where the composition exceeds the miscibility curve
        # (supersaturated), gradually reduce convective diffusion so the
        # advective sedimentation flux dominates.  The suppression factor
        #
        #     supp = 1 / (1 + alpha * [softplus(Y - Ym*(1-Z))
        #                              + softplus(Z - Z1m)])
        #
        # is 1 outside the rain zone (no supersaturation) and decays
        # smoothly toward 0 inside it, controlled by rain_conv_supp_k.
        # The metal-rain term enters on equal footing: with metal rain off,
        # Z1m_lag_i = 1 so delta_Z_i < 0 and the term vanishes (backward
        # compatible with He-only runs).  Without it, the deep deposit
        # gradient diffuses back up through the rain shell and refills the
        # drained outer envelope.
        # When rain_conv_supp_k = 0 (the default) this block is a no-op:
        # supp ≡ 1 everywhere, preserving the pre-change legacy/bottom_up
        # behaviour bit-for-bit.  This avoids the hard diffusion cut that
        # destabilised Newton in earlier revisions.
        delta_Y_i = Y_i - Ym_lag_i * (1 - Z_i)
        delta_Z_i = Z_i - Z1m_lag_i
        supp_i = 1.0 / (1.0 + rain_conv_supp_k
                        * (smooth_positive(delta_Y_i, rain_softplus_beta)
                           + smooth_positive(delta_Z_i, rain_softplus_beta)))
        # supp_i is already a boundary-centred quantity (length N-1)
        # because Y_i, Z_i, Ym_lag_i are arithmetic means aligned with
        # interior boundaries 1..N-1.  Embed it directly in supp_b at
        # those slots; endpoints 0 and N stay at 1.0 (no interior flux
        # contribution).  This keeps the Y/Z Jacobian strictly in-band
        # (Stage 2 derivative terms below depend on Y[k], Y[k+1] only).
        supp_b = np.ones(N + 1)
        supp_b[1:N] = supp_i
        # Save pre-suppression flux coefficients for the Jacobian
        # chain-rule corrections in the Y- and Z-equation blocks below.
        Yflux1_b_pre = Yflux1_b.copy()
        Zflux1_b_pre = Zflux1_b.copy()
        Yflux1_b *= supp_b
        Zflux1_b *= supp_b
        # Chain-rule derivatives of supp_i w.r.t. Y_i, Z_i (the arith-
        # mean composition at the boundary).  Used below to add the
        # d(supp_b)/dY and d(supp_b)/dZ contributions to the Y- and
        # Z-equation Jacobian blocks.  Each cell-centered derivative
        # picks up an extra factor of 1/2 from d(Y_i)/dY[k] = 1/2 and
        # d(Z_i)/dZ[k] = 1/2 (applied at the insertion site).  When
        # rain_conv_supp_k = 0 these are identically zero and the
        # Jacobian additions become no-ops, preserving Stage 1 behaviour.
        if rain_conv_supp_k != 0.0:
            _dspY_i = smooth_positive_deriv(delta_Y_i, rain_softplus_beta)
            _dspZ_i = smooth_positive_deriv(delta_Z_i, rain_softplus_beta)
            # supp = 1/(1 + k*[sp(delta_Y) + sp(delta_Z)]) with
            # delta_Y = Y - Ym*(1-Z), delta_Z = Z - Z1m, so
            #   d(supp)/d(Y_i) = -k * sigmoid(beta*delta_Y) * supp^2
            #   d(supp)/d(Z_i) = -k * (sigmoid(beta*delta_Y) * Ym_lag
            #                          + sigmoid(beta*delta_Z)) * supp^2
            # (d delta_Y/dZ = +Ym_lag from -Ym*(1-Z); d delta_Z/dZ = 1.
            #  The previous +Ym_lag sign here was an error, dormant while
            #  rain_conv_supp_k defaulted to 0.)
            dsupp_dYi_i = -rain_conv_supp_k * _dspY_i * supp_i ** 2
            dsupp_dZi_i = (
                -rain_conv_supp_k * (_dspY_i * Ym_lag_i + _dspZ_i) * supp_i ** 2
            )
        else:
            dsupp_dYi_i = np.zeros(N - 1)
            dsupp_dZi_i = np.zeros(N - 1)

        dP_dT_i = np.zeros(N - 1)
        # this derivative leads to nan convergence... dunno why yet
        # dP_dT_i[:kcore-1] = mixtures_eos.get_dpdt_rhot_rhoy(
        #                     np.log10(rho_i[:kcore-1]),
        #                     np.log10(T_i[:kcore-1]),
        #                     Yold_i[:kcore-1],
        #                     Zold_i[:kcore-1],
        #                     )

        # (smooth_positive, smooth_sign, rain_softplus_beta, rain_sign_eps
        #  are defined above, before the bottom_up diffusion suppression block.)

        # Implementation of Y flux from difference to local Ymisc (smoothed ReLU)
        Yflux1_new_i = smooth_positive(
            Y_i - Ym_lag_i * (1 - Z_i)
            - (dYm_dT_i + dYm_dP_i*dP_dT_i)* T_i / Cp_i * (S_i - S_old_prime_i) * S0 * (1 - Z_i),
            rain_softplus_beta,
        )
        Yflux2_new_i = smooth_positive(misc_threshold - Y_i, rain_softplus_beta)
        Yflux_new_i = Yflux1_new_i * Yflux2_new_i
        Yflux_new_b = np.concatenate(([0], Yflux_new_i, [0]))

        # Jacobian chain-rule derivatives: sigmoid(beta * arg) for each
        # activation factor.  These replace the incorrect sgn_y_flux proxy
        # that was previously used, which caused NR oscillation at the
        # miscibility boundary where sigmoid ≈ 0.5 but sgn ≈ 1.
        dact1_Y = smooth_positive_deriv(
            Y_i - Ym_lag_i * (1 - Z_i)
            - (dYm_dT_i + dYm_dP_i*dP_dT_i) * T_i / Cp_i * (S_i - S_old_prime_i) * S0 * (1 - Z_i),
            rain_softplus_beta,
        )
        dact2_Y = smooth_positive_deriv(misc_threshold - Y_i, rain_softplus_beta)

        # ==========================
        # Z-rain: fluxes + active rain mask
        # ==========================

        # updating Z flux due to Hydrogen–Water immiscibility:
        Zflux1_new_i = smooth_positive(
            Z_i - Z1m_lag_i
            - (dZ1m_dT_i + dZ1m_dP_i * dP_dT_i) * T_i / Cp_i
              * (S_i - S_old_prime_i) * S0,
            rain_softplus_beta,
        )

        # advecting water (Z2m) flux toward Z2m_lag_i
        # (alternative: Zflux2_new_i = np.maximum(0.0, misc_threshold - Z_i))
        Zflux2_new_i = smooth_positive(
            Z2m_lag_i - Z_i
            + (dZ2m_dT_i + dZ2m_dP_i * dP_dT_i) * T_i / Cp_i
              * (S_i - S_old_prime_i) * S0,
            rain_softplus_beta,
        )

        # Rain flux: activation product, mirroring the Y-rain construction
        Zflux_new_i = Zflux1_new_i * Zflux2_new_i
        Zflux_new_b = np.concatenate(([0.0], Zflux_new_i, [0.0]))

        # Jacobian chain-rule derivatives for Z activation factors
        dact1_Z = smooth_positive_deriv(
            Z_i - Z1m_lag_i
            - (dZ1m_dT_i + dZ1m_dP_i * dP_dT_i) * T_i / Cp_i
              * (S_i - S_old_prime_i) * S0,
            rain_softplus_beta,
        )
        dact2_Z = smooth_positive_deriv(
            Z2m_lag_i - Z_i
            + (dZ2m_dT_i + dZ2m_dP_i * dP_dT_i) * T_i / Cp_i
              * (S_i - S_old_prime_i) * S0,
            rain_softplus_beta,
        )

        # NOTE: Z diffusion is smoothly suppressed
        # via the same supp_b factor applied to Yflux1_b and Zflux1_b
        # upstream (based on Y supersaturation).  When rain_conv_supp_k = 0
        # the suppression is a no-op.

        # --------------- BUILDING JACOBIAN AND B RESIDUAL MATRIX --------------- #

        # Idiomatic usage notes:
        # when calculating fluxes for N cells, there are (N - 1) right and left
        # fluxes
        # _i's take their own indicies, _b's take [1:-1], and cell-centered
        # values must match the mask (e.g. S_flux[mask_left] = (...  [k_left]))

        # Build the Jacobian (3N x 3N) and residual vector B (3N)
        J = np.zeros((3 * N, 3 * N))
        B = np.zeros(3 * N)

        # Trial thermodynamic state used by the Newton solve.
        # This makes latent heat depend on the current iterate, not just the
        # previous accepted hydrostatic state.
        (T_prime, Cv_safe, deltaT_prime,
         T_trial_prime, T_trial) = _trial_temperature_state(S, Y, Z, T, Cv)

        if latent_heat_effect:

            # Recompute melt fraction from the trial state and relax it toward
            # the previous accepted χ so latent/viscous transitions cannot jump
            # discontinuously in a single step.
            chi_curr_full, dchi_dT_full, chi_prev_eff = _relaxed_chi_from_temperature(
                T_trial,
                chi_prev_relax_full,
                apply_relax=True,
                theta_relax=theta_chi_latent,
            )
            dchi_full = chi_curr_full - chi_prev_eff

            # Build a consistent latent residual/Jacobian pair:
            #   R_lat = (L/(T0*S0)) * (Δχ / T')
            tiny = 1e-99
            latent_Lprime = np.zeros(N)  # L/(T0*S0)
            idx_mantle = np.arange(kcore, kcore_fe)
            if idx_mantle.size > 0:
                latent_Lprime[idx_mantle] = L_lat_mantle / T0 / S0
            if L_lat_core > 0.0:
                idx_core = np.arange(kcore_fe, N)
                if idx_core.size > 0:
                    latent_Lprime[idx_core] = L_lat_core / T0 / S0

            invT_trial = 1.0 / np.maximum(T_trial_prime, tiny)
            latent_coeff_prime = latent_Lprime * invT_trial

            # Chain-rule Jacobian contributions for S/Y/Z in the S-equation row.
            dTprime_dSprime = (T_prime / Cv_safe) * S0
            dTprime_dYprime = dTdY_prime
            dTprime_dZprime = dTdZ_prime
            common_lat = dchi_dT_full * T0 * invT_trial - dchi_full * invT_trial * invT_trial

            J_lat_S_diag = latent_Lprime * dTprime_dSprime * common_lat
            J_lat_Y_diag = latent_Lprime * dTprime_dYprime * common_lat
            J_lat_Z_diag = latent_Lprime * dTprime_dZprime * common_lat
        else:
            # Keep return contract unchanged when latent is disabled.
            chi_curr_full = chi_transport_eval_full
            J_lat_S_diag = np.zeros(N)
            J_lat_Y_diag = np.zeros(N)
            J_lat_Z_diag = np.zeros(N)

        # (Optional small correction; usually not needed for stability)
        # If you want to include derivative of the 1/T' factor wrt S, uncomment:
        #   dTprime_dSprime = (S0 / norm) * (T / Cp) / T0  # per cell
        #   J[row_S, col_S] -= latent_coeff_prime[k] * dTprime_dSprime[k] * dchi_full[k] / T_prime[k]



        conv_flux_b = np.zeros(N + 1)
        rad_flux_b = np.zeros(N + 1)
        radiogenic_flux_b = np.zeros(N + 1)
        pos_flux = flux_dsdr_conv_b[1]

        # We'll fill J and B row.
        # S eqn -> row i
        # Y eqn -> row i+1
        # Z eqn -> row i+2

        # PART 1: S DERIVATIVES IN J
        # PART 2: Y DERIVATIVES IN J
        # PART 3: Z DERIVATIVES IN J

        def _comp_offdiag_flux_rows(dSd_prime_b, dSd_pt_b, dTd_prime,
                                    col_kp1_off, col_km1_off):
            """S-equation off-diagonal composition derivatives: the [k+1] and
            [k-1] columns for one composition channel.

            Emits EXACTLY the operand chains of the historical PART 2B/2C
            (helium: dSdY*, columns +4/-2) and PART 3B/3C (metal: dSdZ*,
            columns +5/-1), which a mechanical substitution diff proved
            identical. Note the deliberate historical asymmetry WITHIN each
            channel: the [k+1] chains divide by dr_prime_i (and S0 for the
            semiconvective term) EARLY, the [k-1] chains LATE. Do not
            normalize the order — it sets the last bit of the Jacobian.
            (The [k] diagonal blocks, PART 2A/3A, remain inline because their
            operand orders genuinely differ between the channels.)
            """
            conv_val_kp1 = np.zeros(N)
            conv_val_kp1[k_right] = (
                -dt
                / dm_prime[k_right]
                / T_prime[k_right]
                / dr_prime_i
                * phi_prime_b[1:-1]
                * dSd_prime_b[1:-1]
                * d_flux_dS[1:-1]
            )

            rad_val_kp1 = np.zeros(N)
            rad_val_kp1[k_right] = (
                dt
                / dm_prime[k_right]
                / T_prime[k_right]
                / dr_prime_i
                * dTd_prime[k_right]
                * Lambda_prime_b[1:-1]
            )

            # Semiconvective flux: NO Jacobian contribution (2026-07 DDC
            # rework).  The former semi_val_kp1/km1 entries were provably
            # wrong (audit #18: missing 1/dr, flipped sign) and corrupted
            # Newton whenever semiconvection was on.  With semi_phi frozen
            # at old-state Nu, the semi flux is treated as a lagged/Picard
            # term (exact residual + zero Jacobian = consistent inexact
            # Newton).
            combined_flux_kp1 = conv_val_kp1 + rad_val_kp1
            J[i_right, i_right + col_kp1_off] = combined_flux_kp1[mask_right]

            conv_val_km1 = np.zeros(N)
            conv_val_km1[k_left] = (
                -dt
                / dm_prime[k_left]
                / T_prime[k_left]
                * phi_prime_b[1:-1]
                * dSd_prime_b[1:-1]
                * d_flux_dS[1:-1]
                / dr_prime_i
            )

            rad_val_km1 = np.zeros(N)
            rad_val_km1[k_left] = (
                dt
                / dm_prime[k_left]
                / T_prime[k_left]
                * dTd_prime[k_left]
                * Lambda_prime_b[1:-1]
                / dr_prime_i
            )

            # (semi flux: Picard -- no Jacobian term; see note above.)
            combined_flux_km1 = conv_val_km1 + rad_val_km1
            J[i_left, i_left + col_km1_off] = combined_flux_km1[mask_left]

        def _supp_stage2_rows(row_off, flux1_pre, diff_comp):
            """Diffusion-suppression Stage-2 chain-rule terms for one
            composition equation (row i + row_off).

            Proven substitution-identical between the Y (row_off=1) and Z
            (row_off=2) channels. The base factors use the channel's
            diffusive-flux prefactor and composition difference:
              right boundary: +dt/dm[k_right] * F_pre[k+1] * (1/2) * diff[k]
              left  boundary: -dt/dm[k_left]  * F_pre[k]   * (1/2) * diff[k-1]
            The column TARGETS (+1/+2 diagonals, +4/+5 and -2/-1 neighbor
            cells) are the same for both channels because the suppression
            factor depends on BOTH compositions.
            """
            supp_r_base = np.zeros(N)
            supp_r_base[k_right] = (
                dt / dm[k_right] * flux1_pre[1:-1] * 0.5 * diff_comp
            )
            supp_l_base = np.zeros(N)
            supp_l_base[k_left] = (
                -dt / dm[k_left] * flux1_pre[1:-1] * 0.5 * diff_comp
            )

            # Right-boundary contributions
            supp_Y_diag_r = np.zeros(N)
            supp_Y_diag_r[k_right] = supp_r_base[k_right] * dsupp_dYi_i
            supp_Z_diag_r = np.zeros(N)
            supp_Z_diag_r[k_right] = supp_r_base[k_right] * dsupp_dZi_i
            J[i_right + row_off, i_right + 4] += (
                supp_r_base[mask_right] * dsupp_dYi_i
            )
            J[i_right + row_off, i_right + 5] += (
                supp_r_base[mask_right] * dsupp_dZi_i
            )

            # Left-boundary contributions
            supp_Y_diag_l = np.zeros(N)
            supp_Y_diag_l[k_left] = supp_l_base[k_left] * dsupp_dYi_i
            supp_Z_diag_l = np.zeros(N)
            supp_Z_diag_l[k_left] = supp_l_base[k_left] * dsupp_dZi_i
            J[i_left + row_off, i_left - 2] += (
                supp_l_base[mask_left] * dsupp_dYi_i
            )
            J[i_left + row_off, i_left - 1] += (
                supp_l_base[mask_left] * dsupp_dZi_i
            )

            # Combine diagonals from both boundaries
            J[i_array + row_off, i_array + 1] += supp_Y_diag_r + supp_Y_diag_l
            J[i_array + row_off, i_array + 2] += supp_Z_diag_r + supp_Z_diag_l

        def _comp_residual(row_off, C, Cold_arr, s1p, s1m, s2p, s2m, flux_new_b):
            """Composition-equation residual (implicit form): (C - Cold) plus
            the diffusive and advective flux divergences, in the exact
            historical operand order (identical between the two channels).
            """
            res = np.zeros(N)
            res += (C - Cold_arr)
            res[mask_right] += dt/dm[k_right] * s1p * np.diff(C)
            res[mask_left] -= dt/dm[k_left] * s1m * np.diff(C)
            res[mask_right] += dt/dm[k_right] * s2p * flux_new_b[1:-1]
            res[mask_left] -= dt/dm[k_left] * s2m * flux_new_b[1:-1]
            B[i_array + row_off] = res

        # ----------------------------------------------------------------
        # PART 1A: "S[k]" => The Diagonal (J[i, i])
        # ----------------------------------------------------------------

        # We'll build an array val_s_diag that is (N,) and eventually assign it
        # to J[i_array, i_array]. We skip separate code for the first/last zones
        # by using partial contributions for left and right references.

        k_array = np.arange(N)           # 0..N-1
        i_array = 3 * k_array


        mask_left = (k_array > 0)        # True for k=1...N-1, False for k=0
        mask_right = (k_array < N - 1)   # True for k=0...N-2, False for k=N-1
        k_left = k_array[mask_left]
        k_right = k_array[mask_right]
        i_left = i_array[mask_left]
        i_right = i_array[mask_right]

        val_s_diag = np.ones(N)  # Start at "1" for each zone
                             # (like past code that does "1 - conv_km1 - conv_kp1 - ...")

        if latent_heat_effect:
            val_s_diag += J_lat_S_diag

        # e.g. Sub off the "conv_km1" if k>0
        conv_km1 = np.zeros(N)

        d_flux_term = (
            phi_prime_b[1:-1]          # geometric factor
            * d_flux_dS[1:-1]    # full chain rule
            / dr_prime_i          # dr_prime_i = r[k] - r[k-1] for k>0
        )

        conv_km1[mask_left] = - d_flux_term

        # e.g. Sub off the "conv_kp1" if k<N-1
        conv_kp1 = np.zeros(N)
        conv_kp1[mask_right] = - d_flux_term

        val_s_diag += (
            dt
            / dm_prime
            / T_prime
            * (conv_km1 + conv_kp1)
        )

        rad_km1 = np.zeros(N)
        rad_km1[mask_left] = -(
            Lambda_prime_b[1:-1]
            * (T_prime[k_left] / Cv[k_left])
            / dr_prime_i
        )

        rad_kp1 = np.zeros(N)
        rad_kp1[mask_right] = -(
            Lambda_prime_b[1:-1]
            * (T_prime[k_right] / Cv[k_right])
            / dr_prime_i
        )
        val_s_diag += (
            dt
            / dm_prime
            / T_prime
            * (rad_km1 + rad_kp1)
            * S0
        )

        # Semiconvective flux: NO S-diagonal Jacobian contribution (2026-07
        # DDC rework).  FD-probe verdict (on-demand probe at a LIVE 3-face
        # SC band, 1.40 Gyr): the former entries were units-CORRECT at
        # active faces but carried no Heaviside gate on the
        # max(schwarz, 0) clamp -- at clamped faces (the common, marginal
        # case) the residual has zero semi dependence while J claimed full
        # coupling, so Newton pushed on dead faces (the audit-#17 probe
        # measured exactly that mismatch).  Treated as a lagged/Picard
        # term instead: exact residual, zero Jacobian -- consistent
        # inexact Newton; gate-run
        # cost was niter 17 vs 13 control.  If Newton cost ever becomes
        # limiting, the exact alternative is: old entries multiplied by
        # the current-state clamp indicator (schwarz_i > 0).

        J[i_array, i_array] = val_s_diag

        # ----------------------------------------------------------------
        # PART 1B: "S[k-1]" => J[i, i-3], only valid for k>0
        # ----------------------------------------------------------------

        # fill only for k>0: k_left
        val_s_km1 = np.zeros(N)
        val_s_km1[mask_left] = (
            # convective k-1 term
            dt
            / dm_prime[k_left]
            / T_prime[k_left]
            * phi_prime_b[1:-1]
            / dr_prime_i
            * d_flux_dS[1:-1]

            +

            # radiative/conductive k-1 term
            dt
            / dm_prime[k_left]
            / T_prime[k_left]
            * Lambda_prime_b[1:-1]
            / dr_prime_i
            * (T_prime[k_left - 1] / Cv[k_left - 1])
            * S0
            # (semi flux: Picard -- no S[k-1] Jacobian term; audit #17)
        )

        J[i_left, i_left - 3] = val_s_km1[mask_left]

        # ----------------------------------------------------------------
        # PART 1C: "S[k+1]" => J[i, i+3], only valid for k<N-1
        # ----------------------------------------------------------------

        # fill only for k<N-1: k_array[mask_right]
        val_s_kp1 = np.zeros(N)
        val_s_kp1[mask_right] = (
            # convective k+1 term
            dt
            / dm_prime[k_right]
            / T_prime[k_right]
            * phi_prime_b[1:-1]
            / dr_prime_i
            * d_flux_dS[1:-1]

            +

            # radiative/conductive k+1 term
            dt
            / dm_prime[k_right]
            / T_prime[k_right]
            * Lambda_prime_b[1:-1]
            / dr_prime_i
            * (T_prime[k_right + 1] / Cv[k_right + 1])
            * S0
            # (semi flux: Picard -- no S[k+1] Jacobian term; audit #17)
        )

        J[i_right, i_right + 3] = val_s_kp1[mask_right]


        # ----------------------------------------------------------------
        # PART 2A: "Y[k]" => The Diagonal (J[i, i+1])
        # ----------------------------------------------------------------

        y_diag_conv_kp1 = np.zeros(N)
        y_diag_conv_kp1[k_right] = (
            dt
            / dm_prime[k_right]
            / T_prime[k_right]
            / dr_prime_i
            * phi_prime_b[1:-1]
            * dSdY_prime_b[1:-1]
            * d_flux_dS[1:-1]
        )

        y_diag_conv_km1 = np.zeros(N)
        y_diag_conv_km1[k_left] = (
            dt
            / dm_prime[k_left]
            / T_prime[k_left]
            / dr_prime_i
            * phi_prime_b[1:-1]
            * dSdY_prime_b[1:-1]
            * d_flux_dS[1:-1]
        )

        y_diag_rad_kp1 = np.zeros(N)
        y_diag_rad_kp1[k_right] = (
            - dt
            / dm_prime[k_right]
            / T_prime[k_right]
            * dTdY_prime[k_right]
            * Lambda_prime_b[1:-1]
            / dr_prime_i
        )

        y_diag_rad_km1 = np.zeros(N)
        y_diag_rad_km1[k_left] = (
            - dt
            / dm_prime[k_left]
            / T_prime[k_left]
            * dTdY_prime[k_left]
            * Lambda_prime_b[1:-1]
            / dr_prime_i
        )

        # (semi flux: Picard -- no Y-diagonal Jacobian terms; the former
        #  y_diag_semi_* entries were missing 1/dr and sign-flipped, audit
        #  #18.)
        J[i_array, i_array + 1] = (
            y_diag_conv_km1
            + y_diag_rad_km1
            + y_diag_conv_kp1
            + y_diag_rad_kp1
        )
        if latent_heat_effect:
            J[i_array, i_array + 1] += J_lat_Y_diag

        # ----------------------------------------------------------------
        # PART 2B + 2C: "Y[k+1]" / "Y[k-1]" columns (J[i, i+4] / J[i, i-2])
        # ----------------------------------------------------------------
        _comp_offdiag_flux_rows(dSdY_prime_b, dSdY_pt_b, dTdY_prime,
                                col_kp1_off=4, col_km1_off=-2)

        # ----------------------------------------------------------------
        # PART 3A: "Z[k]" => The Diagonal (J[i, i+2])
        # ----------------------------------------------------------------

        z_diag_conv_kp1 = np.zeros(N)
        z_diag_conv_kp1[k_right] = (
            dt
            / dm_prime[k_right]
            / T_prime[k_right]
            / dr_prime_i
            * phi_prime_b[1:-1]
            * dSdZ_prime_b[1:-1]
            * d_flux_dS[1:-1]
        )

        z_diag_conv_km1 = np.zeros(N)
        z_diag_conv_km1[k_left] = (
            dt
            / dm_prime[k_left]
            / T_prime[k_left]
            / dr_prime_i
            * phi_prime_b[1:-1]
            * dSdZ_prime_b[1:-1]
            * d_flux_dS[1:-1]
        )

        z_diag_rad_kp1 = np.zeros(N)
        z_diag_rad_kp1[k_right] = (
            - dt
            / dm_prime[k_right]
            / T_prime[k_right]
            / dr_prime_i
            * dTdZ_prime[k_right]
            * Lambda_prime_b[1:-1]
        )

        z_diag_rad_km1 = np.zeros(N)
        z_diag_rad_km1[k_left] = (
            - dt
            / dm_prime[k_left]
            / T_prime[k_left]
            / dr_prime_i
            * dTdZ_prime[k_left]
            * Lambda_prime_b[1:-1]
        )

        # (semi flux: Picard -- no Z-diagonal Jacobian terms; audit #18.)
        J[i_array, i_array + 2] = (
            z_diag_conv_km1
            + z_diag_rad_km1
            + z_diag_conv_kp1
            + z_diag_rad_kp1
        )
        if latent_heat_effect:
            J[i_array, i_array + 2] += J_lat_Z_diag

        # ----------------------------------------------------------------
        # PART 3B + 3C: "Z[k+1]" / "Z[k-1]" columns (J[i, i+5] / J[i, i-1])
        # ----------------------------------------------------------------
        _comp_offdiag_flux_rows(dSdZ_prime_b, dSdZ_pt_b, dTdZ_prime,
                                col_kp1_off=5, col_km1_off=-1)

        if melt_flux_jac_terms:
            for row_i, col_i, val_i in melt_flux_jac_terms:
                J[row_i, col_i] += val_i

        # -------------------- Surface Fluxes Special Treatment -------------------- #

        # (A) Geometry factor for the emitting surface.
        # Use the *outer boundary radius* directly so bare rock (kcore==0) does not
        # accidentally reference kcore-1, and so we consistently compute
        # A_surf = 4*pi*R^2*sigma regardless of envelope/core partitioning.
        r_surf = r_prime_b[0]  # dimensionless (r_b[0] / R0)
        A_geom = 4 * np.pi * (R0 * r_surf) ** 2
        A_surf = A_geom * const.sigma_sb

        # --- Composition effects in internal energy (bulk) ---
        if composition_change_local:
            J[i_array, i_array+1] += dUdY_prime / T_prime / T0
            J[i_array, i_array+2] += dUdZ_prime / T_prime / T0

        # --- Surface radiative losses (fully implicit) ---
        if losses:
            if BC_atm in ('bare', 'gray', 'ideal_gray'):
                # Surface-coupled rocky BCs: use intrinsic surface flux from top-cell temperature.
                # This keeps the top cooling term tied to the same variable evolved by transport.
                top = 0
                Tsurf_trial = T_trial[top]
                Tsurf_trial = np.maximum(Tsurf_trial, 1.0)

                if BC_atm == 'bare':
                    F_int = atm.get_intrinsic_flux(Tsurf_trial)  # erg/(s cm^2)
                elif BC_atm == 'gray':
                    Tint_trial, _ = atm.get_tint_teff(T_surf=Tsurf_trial, g=g)
                    F_int = const.sigma_sb * np.maximum(Tint_trial, 0.0) ** 4
                else:
                    F_int = atm.get_intrinsic_flux(Tsurf_trial, g)
                surface_flux = A_geom * F_int                            # erg/s
                rad_flux_b[0] = -surface_flux / (M0 * T0 * S0)

                # Jacobian contribution through S at the top cell.
                # d(surface_flux)/dS = A_geom * dFsurf_dS
                pref = (dt / dm_prime[0] / T_prime[0]) * A_geom / (M0 * T0 * S0)
                J[0, 0] += pref * dFsurf_dS
            else:
                # Residual uses the *nonlinear* flux only (no Taylor term)
                surface_flux = A_surf * T_int**4
                rad_flux_b[0] = -surface_flux / (M0 * T0 * S0)

                # Jacobian: d(surface_flux)/dT_int = A_surf * 4 T_int^3
                dL_dTint = A_surf * 4.0 * T_int**3

                # Common prefactor coming from your discretization of the surface entropy equation
                pref = (dt / dm_prime[0] / T_prime[0]) * dL_dTint / (M0 * T0 * S0)

                # S sensitivity (only if you are treating T_int as dependent on S in this solve)
                J[0, 0] += pref * dTdS

                # Y/Z sensitivities only if composition coupling is active
                if composition_change_local:
                    J[0, 1] += pref * dTint_dY
                    J[0, 2] += pref * dTint_dZ

        else:
            surface_flux = 0.0
            rad_flux_b[0] = 0.0


        # deltaT2 => T difference from cell k to k+1
        deltaT2_i = np.diff(T_trial_prime)

        # Update flux arrays using deltaT2 for the boundary between k and k+1
        conv_flux_b[1:-1] = -phi_prime_b[1:-1] * flux_dsdr_conv_b[1:-1] #** 1.5
        rad_flux_b[1:-1] = Lambda_prime_b[1:-1] / dr_prime_i * deltaT2_i
        semi_flux_b[1:-1] = -semi_phi_prime_b[1:-1] * flux_dsdr_semi_b[1:-1]

        latent_flux_b = np.zeros(N +1) # dummy flux

        # build combined_flux
        combined_flux_b = conv_flux_b + rad_flux_b + semi_flux_b + radiogenic_flux_b #+ latent_flux_b  # shape (N+1,)

        # We define fluxTerm for each k in 0..N-1
        dflux = np.diff(combined_flux_b)

        # --- radiogenic heating as a cell source (erg s^-1 g^-1) ---
        if radioactive:
            # H_spec(t) in erg s^-1 g^-1 (Earth-like specific power unless you pass weights)
            H_spec = float(mantle_core.radiogenic_heat(t)[0])
        else:
            H_spec = 0.0

        # Mask mantle cells: kcore ... kcore_fe-1 (no heating in H/He envelope or Fe core)
        mask_mantle = np.zeros(N, dtype=bool)
        mask_mantle[kcore:kcore_fe] = True

        # Per-cell total radiogenic power (erg/s)
        Q_cell = np.zeros(N)
        Q_cell[mask_mantle] = H_spec * dm[mask_mantle]

        # Convert to solver units consistent with your flux scaling
        Q_prime = Q_cell / (M0 * T0 * S0)


        # Now we combine
        if time_centered_heat_factor and dTdt_prev is not None:
            # eps_S fix: meter the drained heat at the time-centered
            # temperature T_half = T_old + (dt/2)*(dT/dt)_prev, built from the
            # previous accepted step's ACTUAL temperature rate (lagged
            # predictor). A within-step linearized estimate was tried first
            # and rejected: the (T/Cv)*dS constant-structure form
            # overestimates the realized |dT| by ~2.7x (the true step is
            # closer to constant-P and partially offset by compression
            # heating), overshooting the midpoint. The lagged rate is
            # second-order for smooth cooling, independent of the Newton
            # unknowns (so the analytic Jacobian's old-state 1/T factor stays
            # consistent to O(dT/T) ~ 1e-4 — the FD-Jacobian debug check will
            # show that small lag), and falls back to the legacy weighting on
            # the first step and right after a regrid (dTdt_prev=None). The
            # floor at T_old/2 guards pathological rates only.
            _T_heat_prime = np.maximum(T + 0.5 * dt * dTdt_prev, 0.5 * T) / T0
        else:
            # Legacy old-state weighting, bit-identical with all prior results.
            _T_heat_prime = T_prime
        B_s = (S - S_old_prime) + dt / dm_prime / _T_heat_prime * dflux

        # composition changes
        if composition_change_local:
            B_s += (
                dUdY_prime / _T_heat_prime * (Y - Yold) / T0
                + dUdZ_prime / _T_heat_prime * (Z - Zold) / T0
            )

        if latent_heat_effect:
            # latent_coeff_prime already includes mantle/core pieces:
            #   mantle: (L_lat_mantle / T0 / S0) / T_prime
            #   core:   (L_lat_core   / T0 / S0) / T_prime
            # dchi_full = chi_curr_full - chi_prev_full
            B_s += latent_coeff_prime * dchi_full

        if radioactive:
            # Radiogenic heating enters the residual with a minus sign so that
            # positive Q drives entropy upward in the Newton update z_new = z_old - J^{-1} B.
            B_s -= dt / dm_prime / T_prime * Q_prime

            tiny = 1e-99
            # Consistent with the trial-state relation used above:
            #   T' = T'_old + (T'/Cv) * S0 * (S - S_old') + ...
            dTprime_dSprime_q = (T_prime / np.maximum(Cv, tiny)) * S0

            # Implicit derivative of [- dt/dm' * Q'/T'] w.r.t. S.
            J[i_array, i_array] += (
                (dt / dm_prime)
                * (Q_prime / np.maximum(T_prime, tiny) ** 2)
                * dTprime_dSprime_q
            )


    
        if extrap_endpoint and kcore > 0:
            # Stellar irradiation deposited per cell.  Q_cell_irr is precomputed
            # in the outer scope from the trial-state opacity proxy and is held
            # fixed across NR iterations within the timestep — same recipe as
            # radiogenic heating above.
            Q_prime_irr = Q_cell_irr / (M0 * T0 * S0)
            B_s -= dt / dm_prime / T_prime * Q_prime_irr

            tiny_ee = 1e-99
            dTprime_dSprime_ee = (T_prime / np.maximum(Cv, tiny_ee)) * S0
            J[i_array, i_array] += (
                (dt / dm_prime)
                * (Q_prime_irr / np.maximum(T_prime, tiny_ee) ** 2)
                * dTprime_dSprime_ee
            )

        # assign
        B[i_array] = B_s


        # ================== Y-Equation (Advection-Diffusion) ================== #
        #
        # The diffusive part (sigma1 * diff(Y)) is identical for both rain
        # models.  The advective (sedimentation) part differs structurally:
        #
        # ---- Legacy model ----
        # Advective flux  =  Yflux2_b * sgn(Yflux_new) * Yflux_new
        # where Yflux_new = softplus(Y - Y_misc) * softplus(misc_thresh - Y)
        # is a smooth activation measuring supersaturation.  The Jacobian
        # includes chain-rule derivatives of Yflux_new w.r.t. S, Y, Z
        # through the miscibility curve.  The pseudo-velocity D2/Hp is
        # very small (~1e-8 cm/s), so this product stays well-conditioned.
        # ----------------------------------------------------------------

        sigma1p = Yflux1_b[1:-1]
        sigma1m = Yflux1_b[1:-1]

        # ---- Activation-product advection (unified for legacy and bottom_up) ----
        #
        # Both rain models use the same Yflux_new activation-product flux
        # formula and the same Jacobian structure.  Yflux1_b (diffusion) and
        # Yflux2_b (advection) are set identically; the activation product
        # Yflux_new naturally gates the advective flux to the supersaturated
        # region.  The bottom_up model's rain mask (computed upstream) is
        # available for diagnostics but does not modify flux coefficients.

        # Use smooth sign to prevent on/off chatter from hard switching.
        sgn_y_flux = smooth_sign(Yflux_new_b[1:-1], rain_sign_eps)
        sigma2p = Yflux2_b[1:-1] * sgn_y_flux
        sigma2m = Yflux2_b[1:-1] * sgn_y_flux

        # Fill the Jacobian blocks for Y: J[i+1][...]
        # These include chain-rule derivatives of the Yflux_new
        # activation product w.r.t. S, Y, Z through the miscibility
        # curve (dYm_dT, dYm_dP, etc.).

        y_adv_flux_im3 = np.zeros(N)
        y_adv_flux_im3[k_left] = (
            dt
            / dm[k_left]
            * (
                sigma2m
                * (
                    (dYm_dT_i + dYm_dP_i*dP_dT_i)
                    * T_i
                    / Cp_i
                )
                * S0
                * (1 - Z_i)
                / 2.0
                * dact1_Y * Yflux2_new_i
            )
        )
        J[i_array + 1, i_array - 3] = y_adv_flux_im3

        y_adv_flux_im2 = np.zeros(N)
        y_adv_flux_im2[k_left] = (
            dt
            / dm[k_left]
            * (
                sigma1m
                - sigma2m
                * dact1_Y
                / 2
                * Yflux2_new_i
                + sigma2m
                * dact2_Y
                / 2
                * Yflux1_new_i
            )
        )
        J[i_array + 1, i_array - 2] = y_adv_flux_im2

        # Z at left neighbor (k-1): dB_Y[k]/dZ[k-1] through left boundary Yflux_new
        # Left boundary residual: -dt/dm * sigma2m * Yflux_new_i[k-1]
        # d(Yflux_new_i[k-1])/dZ[k-1] via dZ_i[k-1]/dZ[k-1]=1/2
        # d(arg1)/dZ_i = Ym_lag_i + (dYm_dT + dYm_dP*dP_dT)*T/Cp*(S_i - S_old_prime)*S0
        y_adv_flux_im1_z = np.zeros(N)
        y_adv_flux_im1_z[k_left] = (
            dt / dm[k_left] * (
                -sigma2m * dact1_Y * Yflux2_new_i * (
                    Ym_lag_i
                    + (dYm_dT_i + dYm_dP_i * dP_dT_i) * T_i / Cp_i
                    * (S_i - S_old_prime_i) * S0
                ) / 2.0
            )
        )
        J[i_array + 1, i_array - 1] = y_adv_flux_im1_z

        y_adv_flux_i_p = np.zeros(N)
        y_adv_flux_i_p[k_right] = (
            dt
            / dm[k_right]
            * ( - sigma2p
                * (dYm_dT_i
                + dYm_dP_i*dP_dT_i)
                * T_i
                / Cp_i
                * S0
                * (1 - Z_i)
                / 2.0
               * dact1_Y * Yflux2_new_i

            )
        )

        y_adv_flux_i_n = np.zeros(N)
        y_adv_flux_i_n[k_left] = (
            dt
            / dm[k_left]
            * (
                +sigma2m
                * (dYm_dT_i
                + dYm_dP_i*dP_dT_i)
                * T_i
                / Cp_i
                * S0
                * (1 - Z_i)
                / 2.0
                * dact1_Y * Yflux2_new_i

            )
        )
        J[i_array + 1, i_array] = y_adv_flux_i_p + y_adv_flux_i_n

        y_adv_flux_ip1_kp1 = np.zeros(N)
        y_adv_flux_ip1_kp1[k_right] = (
            dt
            / dm[k_right]
            * (
                - sigma1p
                + sigma2p
                * dact1_Y
                / 2
                * Yflux2_new_i

                - sigma2p
                * dact2_Y
                / 2
                * Yflux1_new_i
                )
        )

        y_adv_flux_ip1_km1 = np.zeros(N)
        y_adv_flux_ip1_km1[k_left] = (
            dt
            / dm[k_left]
            * (
                - sigma1m
                - sigma2m
                * dact1_Y
                / 2
                * Yflux2_new_i
                + sigma2m
                * dact2_Y
                / 2
                * Yflux1_new_i
            )
        )
        J[i_array + 1, i_array + 1] = 1 + y_adv_flux_ip1_kp1 + y_adv_flux_ip1_km1

        # Z at current cell (k): dB_Y[k]/dZ[k] through Yflux_new at both boundaries
        # Right boundary: +dt/dm * sigma2p * d(Yflux_new_i[k])/dZ[k]
        # Left boundary:  -dt/dm * sigma2m * d(Yflux_new_i[k-1])/dZ[k]
        y_adv_z_right = np.zeros(N)
        y_adv_z_right[k_right] = (
            dt / dm[k_right] * (
                sigma2p * dact1_Y * Yflux2_new_i * (
                    Ym_lag_i
                    + (dYm_dT_i + dYm_dP_i * dP_dT_i) * T_i / Cp_i
                    * (S_i - S_old_prime_i) * S0
                ) / 2.0
            )
        )
        y_adv_z_left = np.zeros(N)
        y_adv_z_left[k_left] = (
            dt / dm[k_left] * (
                -sigma2m * dact1_Y * Yflux2_new_i * (
                    Ym_lag_i
                    + (dYm_dT_i + dYm_dP_i * dP_dT_i) * T_i / Cp_i
                    * (S_i - S_old_prime_i) * S0
                ) / 2.0
            )
        )
        J[i_array + 1, i_array + 2] = y_adv_z_right + y_adv_z_left

        y_adv_flux_ip3 = np.zeros(N)
        y_adv_flux_ip3[k_right] = (
            dt
            / dm[k_right]
            * (
                -sigma2p
                * (dYm_dT_i
                + dYm_dP_i*dP_dT_i)
                * T_i
                / Cp_i
                * S0
                * (1 - Z_i)
                / 2.0
                * dact1_Y * Yflux2_new_i
            )
        )

        J[i_right + 1, i_right + 3] = y_adv_flux_ip3[mask_right]

        y_adv_flux_ip4 = np.zeros(N)
        y_adv_flux_ip4[k_right] = (
            dt
            / dm[k_right]
            * (
                sigma1p
                + sigma2p
                * dact1_Y
                / 2
                * Yflux2_new_i
                - sigma2p
                * dact2_Y
                / 2
                * Yflux1_new_i
            )
        )
        J[i_right + 1, i_right + 4] = y_adv_flux_ip4[mask_right]

        # Z at right neighbor (k+1): dB_Y[k]/dZ[k+1] through right boundary Yflux_new
        # Right boundary: +dt/dm * sigma2p * d(Yflux_new_i[k])/dZ[k+1]
        y_adv_flux_ip5_z = np.zeros(N)
        y_adv_flux_ip5_z[k_right] = (
            dt / dm[k_right] * (
                sigma2p * dact1_Y * Yflux2_new_i * (
                    Ym_lag_i
                    + (dYm_dT_i + dYm_dP_i * dP_dT_i) * T_i / Cp_i
                    * (S_i - S_old_prime_i) * S0
                ) / 2.0
            )
        )
        J[i_right + 1, i_right + 5] = y_adv_flux_ip5_z[mask_right]

        # ------------------------------------------------------------------
        # Bracket-through-D_b corrections to the Y-equation Jacobian.
        #
        # sigma1p = Yflux1_b ∝ D_b, and D_b depends on bracket_dsdr_b
        # via D_mlt ∝ sqrt(bracket).  The correction for each entry is:
        #   ±dt/dm * sigma1p * dSigma1_frac_b * dBr_dX_{L/R} * diff(Y)
        #
        # See bracket sensitivity precomputation above for derivation.
        # ------------------------------------------------------------------
        diffY = np.diff(Y)

        # Right boundary base: +dt/dm * sigma1p * dSigma1_frac * diffY
        y_bracket_right_base = np.zeros(N)
        y_bracket_right_base[k_right] = (
            dt / dm[k_right] * sigma1p * dSigma1_frac_b[1:-1] * diffY
        )
        # Left boundary base: -dt/dm * sigma1m * dSigma1_frac * diffY
        y_bracket_left_base = np.zeros(N)
        y_bracket_left_base[k_left] = (
            -dt / dm[k_left] * sigma1m * dSigma1_frac_b[1:-1] * diffY
        )

        _dBr_S_L_i = dBr_dS_L[1:-1]  # N-1 interior boundary values
        _dBr_S_R_i = dBr_dS_R[1:-1]
        _dBr_Y_L_i = dBr_dY_L[1:-1]
        _dBr_Y_R_i = dBr_dY_R[1:-1]
        _dBr_Z_L_i = dBr_dZ_L[1:-1]
        _dBr_Z_R_i = dBr_dZ_R[1:-1]

        # Right boundary: cell k is LEFT cell → use dBr_X_L
        y_br_S_diag_r = np.zeros(N)
        y_br_S_diag_r[k_right] = y_bracket_right_base[k_right] * _dBr_S_L_i
        y_br_Y_diag_r = np.zeros(N)
        y_br_Y_diag_r[k_right] = y_bracket_right_base[k_right] * _dBr_Y_L_i
        y_br_Z_diag_r = np.zeros(N)
        y_br_Z_diag_r[k_right] = y_bracket_right_base[k_right] * _dBr_Z_L_i

        # Right boundary off-diagonal (k+1 is RIGHT cell → dBr_X_R)
        J[i_right + 1, i_right + 3] += y_bracket_right_base[mask_right] * _dBr_S_R_i
        J[i_right + 1, i_right + 4] += y_bracket_right_base[mask_right] * _dBr_Y_R_i
        J[i_right + 1, i_right + 5] += y_bracket_right_base[mask_right] * _dBr_Z_R_i

        # Left boundary: cell k is RIGHT cell → use dBr_X_R
        y_br_S_diag_l = np.zeros(N)
        y_br_S_diag_l[k_left] = y_bracket_left_base[k_left] * _dBr_S_R_i
        y_br_Y_diag_l = np.zeros(N)
        y_br_Y_diag_l[k_left] = y_bracket_left_base[k_left] * _dBr_Y_R_i
        y_br_Z_diag_l = np.zeros(N)
        y_br_Z_diag_l[k_left] = y_bracket_left_base[k_left] * _dBr_Z_R_i

        # Left boundary off-diagonal (k-1 is LEFT cell → dBr_X_L)
        y_br_S_off_l = np.zeros(N)
        y_br_S_off_l[k_left] = y_bracket_left_base[k_left] * _dBr_S_L_i
        J[i_left + 1, i_left - 3] += y_br_S_off_l[mask_left]
        y_br_Y_off_l = np.zeros(N)
        y_br_Y_off_l[k_left] = y_bracket_left_base[k_left] * _dBr_Y_L_i
        J[i_left + 1, i_left - 2] += y_br_Y_off_l[mask_left]
        y_br_Z_off_l = np.zeros(N)
        y_br_Z_off_l[k_left] = y_bracket_left_base[k_left] * _dBr_Z_L_i
        J[i_left + 1, i_left - 1] += y_br_Z_off_l[mask_left]

        # Diagonal: combine both boundaries
        J[i_array + 1, i_array] += y_br_S_diag_r + y_br_S_diag_l
        J[i_array + 1, i_array + 1] += y_br_Y_diag_r + y_br_Y_diag_l
        J[i_array + 1, i_array + 2] += y_br_Z_diag_r + y_br_Z_diag_l

        # ------------------------------------------------------------------
        # Stage 2: chain-rule corrections from d(supp_b)/dY and d(supp_b)/dZ
        # in the Y-equation.
        #
        # sigma1p, sigma1m contain a factor supp_i (length N-1) that
        # depends on Y_i, Z_i.  Each of Y_i[k], Z_i[k] is the arithmetic
        # mean of two adjacent cells, so:
        #   d(supp_i[k])/dY[k]   = d(supp_i[k])/dY[k+1] = dsupp_dYi_i[k] / 2
        #   d(supp_i[k])/dZ[k]   = d(supp_i[k])/dZ[k+1] = dsupp_dZi_i[k] / 2
        # supp_b[k+1] = supp_i[k], so the derivative propagates unchanged.
        # When rain_conv_supp_k = 0, dsupp_dXi_i is identically zero and
        # this block is a no-op (Stage 1 behaviour preserved bit-for-bit).
        # ------------------------------------------------------------------
        if rain_conv_supp_k != 0.0:
            _supp_stage2_rows(1, Yflux1_b_pre, diffY)

        # Y residual, B[i_array+1]
        # (the Y-displacement counterflow dY = -Y' dZ is applied inside the
        #  explicit energy-audited deposit in the outer body; no residual
        #  coupling is needed here)
        _comp_residual(1, Y, Yold, sigma1p, sigma1m, sigma2p, sigma2m,
                       Yflux_new_b)

        # ================== Z-Equation (Advection-Diffusion) ================== #
        #
        # Unified path for legacy and bottom_up (same structure as the
        # Y-equation above).  Flux coefficients are identical for both
        # models; the activation product Zflux_new gates the advective flux.
        # ----------------------------------------------------------------

        sigma1p_z = Zflux1_b[1:-1]
        sigma1m_z = Zflux1_b[1:-1]

        sgn_z_flux = smooth_sign(Zflux_new_b[1:-1], rain_sign_eps)
        sigma2p_z = Zflux2_b[1:-1] * sgn_z_flux
        sigma2m_z = Zflux2_b[1:-1] * sgn_z_flux

        # Fill the Jacobian blocks for Z: J[i+2][...]

        z_adv_flux_im3 = np.zeros(N)
        z_adv_flux_im3[k_left] =  dt / dm[k_left] * (
            sigma2m_z * (
                (
                 (dZ1m_dT_i + dZ1m_dP_i * dP_dT_i) * T_i / Cp_i * S0 / 2.0 * dact1_Z
                ) * Zflux2_new_i
                + (
                    (dZ2m_dT_i + dZ2m_dP_i * dP_dT_i) * T_i / Cp_i * S0 / 2.0 * dact2_Z
                ) * Zflux1_new_i
            )
        )

        J[i_array + 2, i_array - 3] = z_adv_flux_im3

        J[i_array + 2, i_array - 2] = 0.0

        z_adv_flux_im1 = np.zeros(N)
        z_adv_flux_im1[k_left] = dt / dm[k_left] * (
                sigma1m_z
                - sigma2m_z
                * dact1_Z
                / 2
                * Zflux2_new_i

                + sigma2m_z
                * dact2_Z
                / 2
                * Zflux1_new_i
        )
        J[i_array + 2, i_array - 1] = z_adv_flux_im1

        z_adv_flux_i_p = np.zeros(N)
        z_adv_flux_i_p[k_right] = (
            dt
            / dm[k_right]
            * ( - sigma2p_z
                * (
                    ((dZ1m_dT_i
                    + dZ1m_dP_i * dP_dT_i)
                    * T_i
                    / Cp_i
                    * S0
                    / 2.0
                    * dact1_Z) * Zflux2_new_i

                    +

                    ((dZ2m_dT_i
                    + dZ2m_dP_i * dP_dT_i)
                    * T_i
                    / Cp_i
                    * S0
                    / 2.0
                    * dact2_Z) * Zflux1_new_i
                )

            )
        )

        z_adv_flux_i_n = np.zeros(N)
        z_adv_flux_i_n[k_left] = (
            dt
            / dm[k_left]
            * (
                sigma2m_z
                * (
                    ((dZ1m_dT_i
                    + dZ1m_dP_i * dP_dT_i)
                    * T_i
                    / Cp_i
                    * S0
                    / 2.0
                    * dact1_Z) * Zflux2_new_i

                    +

                    ((dZ2m_dT_i
                    + dZ2m_dP_i * dP_dT_i)
                    * T_i
                    / Cp_i
                    * S0
                    / 2.0
                    * dact2_Z) * Zflux1_new_i

                )
            )
        )
        J[i_array + 2, i_array] = z_adv_flux_i_p + z_adv_flux_i_n

        J[i_array + 2, i_array + 1] = 0.0

        z_adv_flux_ip2_kp1 = np.zeros(N)
        z_adv_flux_ip2_kp1[k_right] = (
            dt
            / dm[k_right]
            * (
                - sigma1p_z
                + sigma2p_z
                * dact1_Z
                / 2
                * Zflux2_new_i

                - sigma2p_z
                * dact2_Z
                / 2
                * Zflux1_new_i
                )
        )

        z_adv_flux_ip2_km1 = np.zeros(N)
        z_adv_flux_ip2_km1[k_left] = (
            dt
            / dm[k_left]
            * (
                - sigma1m_z
                - sigma2m_z
                * dact1_Z
                / 2
                * Zflux2_new_i
                + sigma2m_z
                * dact2_Z
                / 2
                * Zflux1_new_i
            )
        )

        J[i_array + 2, i_array + 2] = 1 + z_adv_flux_ip2_kp1 + z_adv_flux_ip2_km1

        z_adv_flux_ip3 = np.zeros(N)
        z_adv_flux_ip3[k_right] = (
            dt
            / dm[k_right]
            * (
                -sigma2p_z
                * (
                    ((dZ1m_dT_i
                + dZ1m_dP_i * dP_dT_i)
                * T_i
                / Cp_i
                * S0
                / 2.0
                * dact1_Z) * Zflux2_new_i

                +
                    ((dZ2m_dT_i
                + dZ2m_dP_i * dP_dT_i)
                * T_i
                / Cp_i
                * S0
                / 2.0
                * dact2_Z) * Zflux1_new_i
                )
            )
        )

        J[i_right + 2, i_right + 3] = z_adv_flux_ip3[mask_right]

        z_adv_flux_ip5 = np.zeros(N)
        z_adv_flux_ip5[k_right] = (
            dt
            / dm[k_right]
            * (
                sigma1p_z
                + sigma2p_z
                * dact1_Z
                / 2
                * Zflux2_new_i

                - sigma2p_z
                * dact2_Z
                / 2
                * Zflux1_new_i
            )
        )

        #J[i_right + 2, i_right + 4] = 0.0
        J[i_right + 2, i_right + 5] = z_adv_flux_ip5[mask_right]

        # ------------------------------------------------------------------
        # Bracket-through-DZ_b corrections to the Z-equation Jacobian.
        # Same structure as Y-equation above. Also populates the
        # previously-zero dB_Z/dY entries since DZ_b depends on Y
        # through the convective bracket (dSdY * dY/dr).
        # ------------------------------------------------------------------
        diffZ = np.diff(Z)

        z_bracket_right_base = np.zeros(N)
        z_bracket_right_base[k_right] = (
            dt / dm[k_right] * sigma1p_z * dSigma1z_frac_b[1:-1] * diffZ
        )
        z_bracket_left_base = np.zeros(N)
        z_bracket_left_base[k_left] = (
            -dt / dm[k_left] * sigma1m_z * dSigma1z_frac_b[1:-1] * diffZ
        )

        # Right boundary corrections
        z_br_S_diag_r = np.zeros(N)
        z_br_S_diag_r[k_right] = z_bracket_right_base[k_right] * _dBr_S_L_i
        z_br_Y_diag_r = np.zeros(N)
        z_br_Y_diag_r[k_right] = z_bracket_right_base[k_right] * _dBr_Y_L_i
        z_br_Z_diag_r = np.zeros(N)
        z_br_Z_diag_r[k_right] = z_bracket_right_base[k_right] * _dBr_Z_L_i

        J[i_right + 2, i_right + 3] += z_bracket_right_base[mask_right] * _dBr_S_R_i
        # Y[k+1]: was zero, now populated through bracket!
        J[i_right + 2, i_right + 4] = z_bracket_right_base[mask_right] * _dBr_Y_R_i
        J[i_right + 2, i_right + 5] += z_bracket_right_base[mask_right] * _dBr_Z_R_i

        # Left boundary corrections
        z_br_S_diag_l = np.zeros(N)
        z_br_S_diag_l[k_left] = z_bracket_left_base[k_left] * _dBr_S_R_i
        z_br_Y_diag_l = np.zeros(N)
        z_br_Y_diag_l[k_left] = z_bracket_left_base[k_left] * _dBr_Y_R_i
        z_br_Z_diag_l = np.zeros(N)
        z_br_Z_diag_l[k_left] = z_bracket_left_base[k_left] * _dBr_Z_R_i

        z_br_S_off_l = np.zeros(N)
        z_br_S_off_l[k_left] = z_bracket_left_base[k_left] * _dBr_S_L_i
        J[i_left + 2, i_left - 3] += z_br_S_off_l[mask_left]
        # Y[k-1]: was zero, now populated!
        z_br_Y_off_l = np.zeros(N)
        z_br_Y_off_l[k_left] = z_bracket_left_base[k_left] * _dBr_Y_L_i
        J[i_left + 2, i_left - 2] += z_br_Y_off_l[mask_left]
        z_br_Z_off_l = np.zeros(N)
        z_br_Z_off_l[k_left] = z_bracket_left_base[k_left] * _dBr_Z_L_i
        J[i_left + 2, i_left - 1] += z_br_Z_off_l[mask_left]

        # Diagonal: combine both boundaries
        J[i_array + 2, i_array] += z_br_S_diag_r + z_br_S_diag_l
        # Y[k]: was zero, now populated through bracket!
        J[i_array + 2, i_array + 1] += z_br_Y_diag_r + z_br_Y_diag_l
        J[i_array + 2, i_array + 2] += z_br_Z_diag_r + z_br_Z_diag_l

        # ------------------------------------------------------------------
        # Stage 2: chain-rule corrections from d(supp_b)/dY and d(supp_b)/dZ
        # in the Z-equation (mirrors the Y-equation block above, using
        # diffZ and Zflux1_b_pre).
        # ------------------------------------------------------------------
        if rain_conv_supp_k != 0.0:
            _supp_stage2_rows(2, Zflux1_b_pre, diffZ)

        # Z residual, B[i_array+2]
        _comp_residual(2, Z, Zold, sigma1p_z, sigma1m_z, sigma2p_z, sigma2m_z,
                       Zflux_new_b)

        return (J, B, combined_flux_b, conv_flux_b, rad_flux_b, radiogenic_flux_b, semi_flux_b, latent_flux_b,
                Yflux1_b, Yflux2_b,
                Zflux1_b, Zflux2_b,
                Lambda_b, D_b, DZ_b, D2_b, surface_flux,
                R0_inv_b,
                Lambda_cond_i,
                Lambda_rad_i,
                bracket_dsdr_b,
                Ym_lag_i,
                Z1m_lag_i,
                Z2m_lag_i,
                Cp_i,
                chi_curr_full,
                y_rain_gap_new,
                Nu_T_eff_i,
                Nu_X_eff_i)

    # ---------------------------------------------------------------------- #
    # Global iteration to solve for S, Y, Z together
    # ---------------------------------------------------------------------- #
    error = 1.0
    flux_error = 1.0
    S_trial = Sold.copy() * norm / S0
    Y_trial = Yold.copy()
    Z_trial = Zold.copy()

    # Warm-start: nudge Y toward miscibility in supersaturated zones.
    # Restricted to LIVE envelope cells: composition-pinned cells (the core,
    # and the implicit-source saturated band) must never be written here --
    # an out-of-bounds value in a pinned cell can never be repaired by the
    # (zeroed) Newton corrections and deadlocks Guard 4.  The production
    # (sentinel Z=0) path never triggered at the core interface, so this
    # mask is behaviour-preserving there.
    if advection_local and y_rain_gap_old is not None:
        Y_trial_i = get_arith_mean(Y_trial)
        Z_trial_i = get_arith_mean(Z_trial)
        supersaturated = Y_trial_i > Ym_i * (1 - Z_trial_i)
        if np.any(supersaturated):
            Y_target_i = Ym_i * (1 - Z_trial_i)
            excess_i = np.maximum(Y_trial_i - Y_target_i, 0.0)
            blend = 0.5
            for k in range(len(excess_i)):
                if supersaturated[k] and (k + 1 < kcore):
                    delta = blend * excess_i[k]
                    Y_trial[k] -= delta
                    Y_trial[k + 1] -= delta
            np.clip(Y_trial, 0.0, 1.0, out=Y_trial)

    zold = np.zeros(3 * N)
    zold[::3] = S_trial
    zold[1::3] = Y_trial
    zold[2::3] = Z_trial

    itr = 1
    prev_fluxes = None

    # local copy so we can reduce it on failure without touching the global config
    weight_local = weight

    # Optional: allow full NR restarts if step keeps failing
    max_restarts = 3

    def _crash_report(e):
        """Print a detailed post-mortem crash report and raise.

        Diagnostics only (print statements + the terminal RuntimeError);
        reads the current Newton-loop state (S_trial, dZ, B, J, D_b,
        DZ_b, bracket_dsdr_b, itr, weight_local, ...) via closure at
        call time. Never returns.
        """
        print("\n" + "=" * 72)
        print("  TRANSPORT NR CRASH REPORT")
        print("=" * 72)
        print(f"  Error: {e}")
        print(f"  Age = {t:.6e} s = {t / const.Myr_to_s / 1e3:.6e} Gyr")
        print(f"  dt  = {dt:.6e} s = {dt / const.Myr_to_s:.6e} Myr")
        print(f"  NR iteration = {itr}, restarts = {restart_count}, weight_local = {weight_local:.3e}")
        print(f"  kcore = {kcore}, N = {N}")

        # Identify problem zones from the guard failure message
        # Parse zones from the error string if possible
        _prob_zones = []
        import re as _re
        _zone_match = _re.search(r'zone[s]?[= ]+\[([0-9, ]+)\]', str(e))
        _single_match = _re.search(r'zone=(\d+)', str(e))
        if _zone_match:
            _prob_zones = [int(x.strip()) for x in _zone_match.group(1).split(',')]
        elif _single_match:
            _prob_zones = [int(_single_match.group(1))]

        if not _prob_zones:
            # Fall back to zones with largest |dZ| correction
            try:
                _top5 = np.argsort(np.abs(dZ))[-15:][::-1]
                _prob_zones = sorted(set(idx // 3 for idx in _top5))[:5]
            except Exception:
                _prob_zones = []

        # Expand to include neighbors
        _report_zones = sorted(set(
            z for c in _prob_zones
            for z in range(max(0, c - 2), min(N, c + 3))
        ))

        if _report_zones:
            print(f"\n  Problem zones: {_prob_zones}")
            print(f"  Reporting on zones: {_report_zones}")

            print(f"\n  {'zone':>5s}  {'S_trial':>12s}  {'Y_trial':>12s}  {'Z_trial':>12s}"
                  f"  {'S_old':>12s}  {'Z_old':>12s}")
            for c in _report_zones:
                if c < N:
                    print(f"  {c:5d}  {S_trial[c]:12.6e}  {Y_trial[c]:12.6e}  {Z_trial[c]:12.6e}"
                          f"  {Sold[c]:12.6e}  {Zold[c]:12.6e}")

            print(f"\n  {'zone':>5s}  {'dZ_S':>12s}  {'dZ_Y':>12s}  {'dZ_Z':>12s}"
                  f"  {'B_S':>12s}  {'B_Y':>12s}  {'B_Z':>12s}")
            try:
                for c in _report_zones:
                    if c < N:
                        _i = 3 * c
                        print(f"  {c:5d}  {dZ[_i]:12.4e}  {dZ[_i+1]:12.4e}  {dZ[_i+2]:12.4e}"
                              f"  {B[_i]:12.4e}  {B[_i+1]:12.4e}  {B[_i+2]:12.4e}")
            except Exception:
                print("  (dZ/B not available — may be NaN)")

            print(f"\n  {'zone':>5s}  {'J_SS':>12s}  {'J_YY':>12s}  {'J_ZZ':>12s}"
                  f"  {'brkt_L':>12s}  {'brkt_R':>12s}  {'D_b_R':>12s}  {'DZ_b_R':>12s}")
            try:
                for c in _report_zones:
                    if c < N:
                        _i = 3 * c
                        _bL = bracket_dsdr_b[c] if c > 0 else 0.0
                        _bR = bracket_dsdr_b[c + 1] if c < N else 0.0
                        _Db = D_b[c + 1] if c + 1 <= N else 0.0
                        _DZb = DZ_b[c + 1] if c + 1 <= N else 0.0
                        print(f"  {c:5d}  {J[_i,_i]:12.4e}  {J[_i+1,_i+1]:12.4e}  {J[_i+2,_i+2]:12.4e}"
                              f"  {_bL:12.4e}  {_bR:12.4e}  {_Db:12.4e}  {_DZb:12.4e}")
            except Exception:
                print("  (Jacobian/bracket not available)")

            # Thermodynamic state at problem zones
            print(f"\n  {'zone':>5s}  {'T [K]':>12s}  {'P [GPa]':>12s}  {'rho [g/cc]':>12s}")
            for c in _report_zones:
                if c < N:
                    print(f"  {c:5d}  {T[c]:12.4e}  {P[c]:12.4e}  {rho[c]:12.4e}")

        # --- Automated diagnosis ---
        # Detect D_b cliff: ratio of max to min D_b among report zones
        _has_Db_cliff = False
        try:
            _Db_vals = [DZ_b[c + 1] for c in _report_zones
                        if c < N and c + 1 <= N and DZ_b[c + 1] > 0]
            if len(_Db_vals) >= 2:
                _Db_ratio = max(_Db_vals) / max(min(_Db_vals), 1e-99)
                _has_Db_cliff = _Db_ratio > 1e4
        except Exception:
            pass

        _guard_str = str(e)
        _is_Z_neg = 'Z<0' in _guard_str or 'guard=5' in _guard_str
        _is_S_issue = 'S<floor' in _guard_str or 'rel_dS' in _guard_str
        _is_nan = 'NaN' in _guard_str or 'non-finite' in _guard_str or 'guard=1' in _guard_str

        # Detect bracket sign flip (convective/stable transition)
        _has_bracket_flip = False
        try:
            _brkt_vals = [bracket_dsdr_b[c + 1] for c in _prob_zones if c + 1 <= N]
            if any(b == 0 for b in _brkt_vals) or (any(b > 0 for b in _brkt_vals) and any(b <= 0 for b in _brkt_vals)):
                _has_bracket_flip = True
        except Exception:
            pass

        # --- Plain-English diagnosis ---
        print("\n  DIAGNOSIS:")
        print("  -" * 36)
        if _is_Z_neg and _has_Db_cliff:
            print(f"  The crash was caused by a D_b CLIFF at the Ledoux boundary.")
            print(f"  The composition diffusion coefficient (DZ_b) varies by ~{_Db_ratio:.0e}x")
            print(f"  across the failing zones.  On the convective side, DZ_b is huge")
            print(f"  (~{max(_Db_vals):.1e}), but on the stable side it drops to ~{min(_Db_vals):.1e}.")
            print(f"  This creates a one-sided Z drain: the zone between the two sides")
            print(f"  loses Z through the high-DZ_b boundary but cannot replenish from")
            print(f"  the low-DZ_b side.  The Newton solver overcorrects and drives Z")
            print(f"  below zero.")
        elif _is_Z_neg:
            print(f"  Z (heavy-element fraction) was driven negative at zones {_prob_zones}.")
            print(f"  The Newton correction for Z at these zones is too large and in the")
            print(f"  wrong direction, likely because the Jacobian off-diagonal coupling")
            print(f"  (from neighboring zones with very different DZ_b) overwhelms the")
            print(f"  diagonal term.")
        elif _is_S_issue and _has_Db_cliff:
            print(f"  Entropy (S) diverged at zones near a D_b cliff (DZ_b ratio ~{_Db_ratio:.0e}x).")
            print(f"  The steep D_b transition at the Ledoux boundary creates stiff S-Z")
            print(f"  coupling: changes in Z alter the convective bracket, which changes")
            print(f"  the heat flux, which changes S.  The linearized Newton step")
            print(f"  overshoots because the frozen EOS derivatives (dSdZ, dTdZ) may be")
            print(f"  inaccurate at the trial state.")
        elif _is_S_issue:
            print(f"  Entropy (S) was driven to unphysical values at zones {_prob_zones}.")
            print(f"  This may indicate that the convective flux sensitivity (d_flux_dS)")
            print(f"  is poorly conditioned at these zones, or that EOS derivatives")
            print(f"  (dSdZ, Cv) have glitches.")
        elif _is_nan:
            print(f"  The Newton correction (dZ) contains NaN or Inf values, indicating")
            print(f"  a singular or nearly singular Jacobian matrix.  This often occurs")
            print(f"  when D_b or DZ_b at a boundary is exactly zero while the")
            print(f"  off-diagonal coupling from a neighboring boundary is very large.")
        else:
            print(f"  The safe_newton_step could not find a damped step satisfying all")
            print(f"  physical guards (positive S, non-negative Z, bounded Y, limited")
            print(f"  relative changes) even after 10 halvings of the step size.")

        if _has_bracket_flip:
            print(f"\n  A convective/stable TRANSITION was detected at the failing zones:")
            print(f"  the convective bracket is zero or changes sign, indicating the")
            print(f"  Ledoux boundary passes through these zones.  This is where D_b")
            print(f"  drops most sharply and the Newton solver is most stressed.")

        # --- Context-aware debugging suggestions ---
        print("\n  SUGGESTED FIXES (in order of likelihood):")
        print("  -" * 36)

        _sug_num = 1
        if _is_Z_neg and _has_Db_cliff:
            print(f"\n  {_sug_num}. D_b CLIFF DETECTED (DZ_b ratio ~ {_Db_ratio:.1e} across report zones).")
            print(f"     The diffusion coefficient drops sharply at the Ledoux boundary,")
            print(f"     creating asymmetric Z transport that drives Z negative.")
            print(f"     Try: increase max_Db_change in [diffusion] (current: {max_Db_change:.0e}).")
            print(f"     This allows D_b to vary more freely, reducing the cliff.")
            _sug_num += 1

        if _is_Z_neg:
            print(f"\n  {_sug_num}. Z driven negative — the Newton correction overshoots.")
            print(f"     The off-diagonal Jacobian coupling from zones with large DZ_b")
            print(f"     overwhelms the diagonal, producing a Z correction that's too large.")
            print(f"     Try: reduce the timestep with a smaller early_max_step or max_step,")
            print(f"     so the old and new states are closer and the linearization is better.")
            _sug_num += 1

        if _is_S_issue:
            print(f"\n  {_sug_num}. Entropy (S) guard violated — large or negative S correction.")
            print(f"     This often follows from Z transport instability at the Ledoux")
            print(f"     boundary feeding back through the coupled S-Z system.")
            print(f"     Try: reduce the timestep, or check if dSdZ EOS derivatives are")
            print(f"     noisy at the failing zones (enable smooth_eos_derivs = True).")
            _sug_num += 1

        print(f"\n  {_sug_num}. Enable hinge_smooth = True in [transport] to smooth the")
        print(f"     convective bracket transition (max(0, bracket) -> softplus).")
        print(f"     Use hinge_func = softplus with hinge_k = 1e5.")
        print(f"     This removes the discontinuous derivative at bracket=0,")
        print(f"     but may require more NR iterations (higher niter) at early times.")
        _sug_num += 1

        print(f"\n  {_sug_num}. Run with debug_transport_NR = True in [transport] to get")
        print(f"     per-iteration NR convergence diagnostics. This shows the full")
        print(f"     Newton iteration history at the failing zones, including bracket")
        print(f"     sign flips and D_b oscillations.")
        _sug_num += 1

        print(f"\n  {_sug_num}. Smooth the initial composition profile. Steep Z gradients")
        print(f"     at the Ledoux boundary create stiff transitions that stress the")
        print(f"     Newton solver. Increasing init_gauss_stdev_z or reducing")
        print(f"     maximum_init_z can help.")

        print("\n" + "=" * 72 + "\n")

        raise RuntimeError(
            f"update_SYZ Newton failed after {max_restarts} restarts: {e}."
        )

    restart_count = 0
    # One-shot flag for the last-resort restart that lags the surface-BC
    # Jacobian coupling (see the except-block below).
    bc_jac_lagged = False

    # If high_precision, we also demand flux_error < flux_tol for stability
    if high_precision:
        flux_tol = flux_tolerance
    else:
        flux_tol = np.inf

    z_floor = 1e-10

    # Residual-based convergence: accept when the residual infinity-norm
    # drops below residual_tol, even if the step metric hasn't reached tol.
    # This handles cases where the Jacobian is slightly inaccurate (e.g.
    # missing D_b derivatives in the rain zone), causing the step metric
    # to plateau above tol while the actual residual is tiny.
    residual_tol = 100.0  # absolute tolerance on max(|B|)
    residual_inf = np.inf  # will be set after first iteration

    # For auto-damping: track if NR is oscillating
    prev_error = np.inf
    oscillation_count = 0
    decrease_count = 0

    # Newton iteration
    while (error > tol) or (flux_error > flux_tol):

        try:
            (
                J,
                B,
                total_flux,
                conv_flux_b,
                rad_flux_b,
                radiogenic_flux_b,
                semi_flux_b,
                latent_flux_b,
                Yflux1_b,
                Yflux2_b,
                Zflux1_b,
                Zflux2_b,
                Lambda_b,
                D_b,
                DZ_b,
                D2_b,
                loss_energy,
                R0_inv_b,
                Lambda_cond_i,
                Lambda_rad_i,
                bracket_dsdr_b,
                Ym_lag_i,
                Z1m_lag_i,
                Z2m_lag_i,
                Cp_i,
                chi_curr_full,
                y_rain_gap_new,
                Nu_T_eff_i,
                Nu_X_eff_i,
            ) = Jacobian(S_trial, Y_trial, Z_trial, T_int, kcore_fe, chi_prev_full)

            # --- On-demand FD Jacobian probe (EXPERIMENT; env-gated) ---
            # Fires once, at the first Newton iteration of the first step
            # past ORCHARD_FD_PROBE_GYR, probing S/Y/Z columns at
            # ORCHARD_FD_CELL +- 1.  Same save/restore discipline as the
            # max-iter FD check below.
            if (_FD_PROBE_GYR > 0.0 and not _fd_jac_check_done
                    and t >= _FD_PROBE_GYR * const.Gyr_to_s
                    and _FD_CELL >= 0):
                _fd_jac_check_done = True
                print(f"\n=== ON-DEMAND FD JACOBIAN PROBE (t={t*const.s_to_Gyr:.4f} "
                      f"Gyr, cell {_FD_CELL}) ===", flush=True)
                _sv_phi = phi_b.copy()
                _sv_Db1 = np.copy(D_b_first) if D_b_first is not None else None
                _sv_DZ1 = np.copy(DZ_b_first) if DZ_b_first is not None else None
                _J_ref, _B_ref = J.copy(), B.copy()
                for _c in range(max(0, _FD_CELL - 1), min(N, _FD_CELL + 2)):
                    for _off, _vn in ((0, 'S'), (1, 'Y'), (2, 'Z')):
                        _col = 3 * _c + _off
                        _Sp, _Yp, _Zp = (S_trial.copy(), Y_trial.copy(),
                                         Z_trial.copy())
                        _base = (_Sp, _Yp, _Zp)[_off]
                        _h = 1e-7 * max(abs(_base[_c]), 1e-10)
                        _base[_c] += _h
                        phi_b = _sv_phi.copy()
                        D_b_first = (np.copy(_sv_Db1) if _sv_Db1 is not None
                                     else None)
                        DZ_b_first = (np.copy(_sv_DZ1) if _sv_DZ1 is not None
                                      else None)
                        _pr = Jacobian(_Sp, _Yp, _Zp, T_int, kcore_fe,
                                       chi_prev_full)
                        _fd = (_pr[1] - _B_ref) / _h
                        _an = _J_ref[:, _col]
                        _ad = np.abs(_fd - _an)
                        _sc = np.maximum(np.maximum(np.abs(_fd), np.abs(_an)),
                                         1e-30)
                        _rel = _ad / _sc
                        _wi = int(np.argmax(_ad))
                        print(f"  col {_vn}[{_c}]: max|fd-an|={_ad[_wi]:.4e} "
                              f"at row {('S','Y','Z')[_wi%3]}[{_wi//3}] "
                              f"(an={_an[_wi]:.4e} fd={_fd[_wi]:.4e} "
                              f"rel={_rel[_wi]:.3e}); "
                              f"rows rel>1%: {int(np.sum(_rel > 0.01))}",
                              flush=True)
                phi_b = _sv_phi.copy()
                D_b_first = np.copy(_sv_Db1) if _sv_Db1 is not None else None
                DZ_b_first = np.copy(_sv_DZ1) if _sv_DZ1 is not None else None
                _rr = Jacobian(S_trial, Y_trial, Z_trial, T_int, kcore_fe,
                               chi_prev_full)
                J, B = _rr[0], _rr[1]
                print("=== FD PROBE DONE ===", flush=True)

            y_rain_gap_old_out = y_rain_gap_new

            J_sparse = _csc_from_banded(J)
            # Solve the linear system
            dZ = spsolve(J_sparse, B)

            # If the solve itself produced NaNs, don't even try to step
            if not np.all(np.isfinite(dZ)):
                # Diagnostic: locate non-finite inputs (rows are (cell, eq)
                # with eq 0=S, 1=Y, 2=Z); a finite J/B with non-finite dZ
                # means the matrix is singular.
                _badB = np.where(~np.isfinite(np.asarray(B)))[0]
                if _badB.size:
                    print(f"  [nan-debug] non-finite B rows "
                          f"{[(int(i)//3, int(i)%3) for i in _badB[:8]]}")
                _Jco = J_sparse.tocoo()
                _badJ = ~np.isfinite(_Jco.data)
                if np.any(_badJ):
                    print(f"  [nan-debug] non-finite J entries (row,col): "
                          f"{list(zip(_Jco.row[_badJ][:8].tolist(), _Jco.col[_badJ][:8].tolist()))}")
                if not _badB.size and not np.any(_badJ):
                    print("  [nan-debug] J and B finite -> singular matrix; "
                          f"zero-diagonal rows: "
                          f"{[int(i) for i in np.where(np.abs(J_sparse.diagonal()) < 1e-300)[0][:8]]}")
                raise RuntimeError("dZ has NaNs or infs")

            # Pin Y and Z corrections to zero in core/mantle zones.
            # Composition (Y, Z) is frozen below the envelope -- DZ_b and
            # D_b are already zeroed for k >= kcore, so the Jacobian rows
            # are identity and the residual should be zero.  The sparse LU
            # solve can still introduce O(1e-12) fill-in noise that trips
            # Guard 5 (Z<0) when Z_old == 0.  Any nonzero Y or Z
            # correction in the core is by definition wrong.
            if kcore < N:
                dZ[3*kcore + 1::3] = 0.0   # Y frozen in the whole compact core
                dZ[3*kcore + 2::3] = 0.0   # Z frozen in core

            # # Weighted correction step
            # znew = zold - weight * dZ

            # ------------------------------------------------------------------
            # Use safe step with backtracking
            #   znew : new iterate
            #   eff_step : scalar factor actually used (<= weight_local)
            #   step : vector step actually applied = eff_step * dZ
            # ------------------------------------------------------------------
            znew, eff_step, step = safe_newton_step(zold, dZ, weight_local)

            # ------------------------------------------------------------------
            # Convergence error based on the *applied* step, with a floor
            # error = max_i |step_i| / max(|zold_i|, z_floor)
            # This automatically includes the weight (eff_step) via 'step'.
            # ------------------------------------------------------------------
            denom = np.maximum(np.abs(zold), z_floor)
            error = np.max(np.abs(step) / denom)

            # --- NR convergence diagnostics ---
            residual_inf = np.max(np.abs(B))
            residual_l2 = np.linalg.norm(B)
            rel_step = np.abs(step) / denom
            error_S = np.max(rel_step[::3])
            error_Y = np.max(rel_step[1::3])
            error_Z = np.max(rel_step[2::3])
            worst_idx = np.argmax(rel_step)
            worst_var = worst_idx % 3  # 0=S, 1=Y, 2=Z
            worst_cell = worst_idx // 3
            var_names = {0: 'S', 1: 'Y', 2: 'Z'}
            if debug_transport_NR and (itr <= 10 or itr % 25 == 0 or error <= tol):
                print(
                    f"  NR itr={itr:3d}: step_err={error:.3e}  "
                    f"|B|_inf={residual_inf:.3e}  |B|_L2={residual_l2:.3e}  "
                    f"errS={error_S:.3e} errY={error_Y:.3e} errZ={error_Z:.3e}  "
                    f"worst={var_names[worst_var]}[{worst_cell}]  eff_step={eff_step:.3e}"
                )
            # Flux error comparison as before
            curr_fluxes = (
                conv_flux_b, rad_flux_b, semi_flux_b, latent_flux_b,
                Yflux1_b, Yflux2_b, Zflux1_b, Zflux2_b, D_b, D2_b
            )
            if prev_fluxes is not None:
                flux_error = 0.0
                for fnew, fold in zip(curr_fluxes, prev_fluxes):
                    denom = np.maximum(np.abs(fnew), np.abs(fold))
                    idxs = np.where(denom != 0)[0]
                    if idxs.size == 0:
                        continue
                    local_err = np.max(np.abs(fnew - fold)[idxs] / denom[idxs])
                    flux_error = max(flux_error, local_err)
            prev_fluxes = curr_fluxes

            # Now, and only now, commit the step
            zold = znew
            S_trial = zold[::3]
            Y_trial = zold[1::3]
            Z_trial = zold[2::3]

            assert_all_not_nan(S_trial, Y_trial, Z_trial)

            # Prevent negative metal
            Z_trial[Z_trial < 0] = 0.0

            # ------------------------------------------------------------------
            # Dual convergence criterion: accept when the residual is small
            # enough, even if the step metric hasn't reached tol.  This handles
            # cases where the Jacobian is slightly inaccurate (missing D_b
            # derivatives in the rain zone), causing the step to plateau while
            # the residual is effectively zero.
            # ------------------------------------------------------------------
            if (residual_inf < residual_tol) and (error < 1e4 * tol) and itr >= 3:
                if debug_transport_NR:
                    print(
                        f"  NR converged (residual): itr={itr} "
                        f"|B|_inf={residual_inf:.3e} < {residual_tol:.1e}  "
                        f"step_err={error:.3e}"
                    )
                break

            # ------------------------------------------------------------------
            # Auto-damping: if the step error increases for several iterations
            # in a row, the NR is oscillating.  Reduce the weight to stabilise.
            # ------------------------------------------------------------------
            if error > prev_error:
                oscillation_count += 1
                decrease_count = 0
            else:
                decrease_count = decrease_count + 1 if error < prev_error else 0
                oscillation_count = 0
            prev_error = error

            if oscillation_count >= 3 and weight_local > 0.01:
                weight_local *= 0.5
                oscillation_count = 0
                if debug_transport_NR:
                    print(
                        f"  NR auto-damp: weight_local -> {weight_local:.3f} "
                        f"(oscillation detected, itr={itr})"
                    )

            # Weight recovery: if error decreases steadily, cautiously
            # increase weight to accelerate convergence.
            if decrease_count >= 5 and weight_local < weight:
                weight_local = min(weight_local * 1.5, weight)
                decrease_count = 0
                if debug_transport_NR:
                    print(
                        f"  NR weight-recover: weight_local -> {weight_local:.3f} "
                        f"(steady decrease, itr={itr})"
                    )

            # ----------------------------------------------------------
            # Stalled-NR acceptance: if the step error has been
            # oscillating (not monotonically decreasing) for many
            # iterations and the residual is small, the Jacobian is
            # likely slightly inaccurate (e.g. missing D_b bracket
            # sensitivities in the S equation during rain).  In this
            # case the step metric will plateau while the residual
            # stays bounded.  Accept the solution rather than wasting
            # iterations.
            # ----------------------------------------------------------
            if itr > 30 and residual_inf < 10 * residual_tol and error < 1e3 * tol:
                if debug_transport_NR:
                    print(
                        f"  NR stalled-accept: itr={itr} "
                        f"|B|_inf={residual_inf:.3e}  step_err={error:.3e}"
                    )
                break

            itr += 1
            if itr > max_itr:
                if debug_transport_NR:
                    logger.warning(
                        f"  NR MAX ITER ({max_itr}) reached: "
                        f"step_err={error:.3e} (tol={tol:.3e})  "
                        f"|B|_inf={residual_inf:.3e}  |B|_L2={residual_l2:.3e}  "
                        f"errS={error_S:.3e} errY={error_Y:.3e} errZ={error_Z:.3e}  "
                        f"worst={var_names[worst_var]}[{worst_cell}]  "
                        f"weight={weight_local:.4f}"
                    )

                # --- Finite-difference Jacobian check (once only) ---
                if debug_transport_NR and not _fd_jac_check_done:
                    _fd_jac_check_done = True
                    print("\n=== FINITE-DIFFERENCE JACOBIAN CHECK ===")
                    print(f"  Checking at worst_cell={worst_cell} and neighbors")

                    # Save ALL side-effected state
                    _save_phi_b = phi_b.copy()
                    _save_D_b_first = np.copy(D_b_first) if D_b_first is not None else None
                    _save_DZ_b_first = np.copy(DZ_b_first) if DZ_b_first is not None else None
                    def _restore_state():
                        nonlocal phi_b
                        global D_b_first, DZ_b_first
                        phi_b = _save_phi_b.copy()
                        D_b_first = np.copy(_save_D_b_first) if _save_D_b_first is not None else None
                        DZ_b_first = np.copy(_save_DZ_b_first) if _save_DZ_b_first is not None else None

                    # Get clean reference B at current state
                    _restore_state()
                    ref_result = Jacobian(S_trial, Y_trial, Z_trial, T_int, kcore_fe, chi_prev_full)
                    J_ref = ref_result[0].copy()
                    B_ref = ref_result[1].copy()

                    check_cells = list(range(max(0, worst_cell - 2),
                                             min(N, worst_cell + 3)))
                    fd_eps = 1e-7
                    fd_var_names = {0: 'S', 1: 'Y', 2: 'Z'}

                    for c in check_cells:
                        for var_off, vname in fd_var_names.items():
                            col_idx = 3 * c + var_off
                            S_p, Y_p, Z_p = S_trial.copy(), Y_trial.copy(), Z_trial.copy()
                            if var_off == 0:
                                h = fd_eps * max(abs(S_trial[c]), 1e-10)
                                S_p[c] += h
                            elif var_off == 1:
                                h = fd_eps * max(abs(Y_trial[c]), 1e-10)
                                Y_p[c] += h
                            else:
                                h = fd_eps * max(abs(Z_trial[c]), 1e-10)
                                Z_p[c] += h

                            _restore_state()
                            p_result = Jacobian(S_p, Y_p, Z_p, T_int, kcore_fe, chi_prev_full)
                            B_p = p_result[1]

                            dBdz_fd = (B_p - B_ref) / h
                            dBdz_an = J_ref[:, col_idx]

                            abs_diff = np.abs(dBdz_fd - dBdz_an)
                            scale = np.maximum(np.abs(dBdz_fd), np.abs(dBdz_an))
                            scale = np.where(scale > 0, scale, 1.0)
                            rel_diff = abs_diff / scale

                            bad_mask = rel_diff > 0.01
                            n_bad = np.sum(bad_mask)
                            worst_row_fd = np.argmax(rel_diff)
                            worst_rel_fd = rel_diff[worst_row_fd]
                            rv = worst_row_fd % 3
                            rc = worst_row_fd // 3

                            key = f"{vname}[{c}]"
                            if worst_rel_fd > 0.01:
                                print(f"  FD col={key}: max_rel={worst_rel_fd:.3e} "
                                      f"at {fd_var_names[rv]}[{rc}] "
                                      f"(an={dBdz_an[worst_row_fd]:.4e} fd={dBdz_fd[worst_row_fd]:.4e}) "
                                      f"n_bad={n_bad}")
                                # Top 5 worst
                                for wi in np.argsort(rel_diff)[::-1][:5]:
                                    rv2, rc2 = wi % 3, wi // 3
                                    print(f"    {fd_var_names[rv2]}[{rc2}]: "
                                          f"an={dBdz_an[wi]:.4e} fd={dBdz_fd[wi]:.4e} "
                                          f"rel={rel_diff[wi]:.3e}")
                            else:
                                print(f"  FD col={key}: OK (max_rel={worst_rel_fd:.3e})")

                    _restore_state()
                    # Rain zone info
                    print(f"\n  Rain zone info at check cells:")
                    for c in check_cells:
                        print(f"    cell {c}: Y={Y_trial[c]:.6f} "
                              f"Yflux1={Yflux1_b[c]:.6e} Yflux2={Yflux2_b[c]:.6e} "
                              f"Zflux1={Zflux1_b[c]:.6e} Zflux2={Zflux2_b[c]:.6e}")
                    print("=== END FD CHECK ===\n")

                break

        except RuntimeError as e:
            # We get here if:
            #  - dZ has NaNs (singular J),
            #  - or safe_newton_step could not find a finite update
            restart_count += 1
            if restart_count > max_restarts:
                # Last-resort rescue before declaring failure: lag the
                # surface-BC Jacobian coupling. The surface-loss residual uses
                # the step-start T_int (explicit within this solve), so the
                # consistent Jacobian contribution is zero; dTdS/dTint_dY/
                # dTint_dZ are a normally-helpful semi-implicit acceleration
                # that can destabilize Newton when the surface zone thermally
                # decouples (e.g. radiation = False ice giants at the
                # surface-convection shutoff -> runaway S correction at zone
                # 0). Retry once with the coupling removed (exact residual,
                # zero surface J coupling -- the same lagged/Picard pattern
                # used for the semiconvective flux) at the original
                # relaxation weight.
                if (not bc_jac_lagged) and losses and (
                        dTdS != 0.0 or dTint_dY != 0.0 or dTint_dZ != 0.0):
                    bc_jac_lagged = True
                    logger.warning(
                        "update_SYZ: Newton failed after %d restarts with the "
                        "surface-BC Jacobian coupling active; retrying once "
                        "with dTdS/dTint_dY/dTint_dZ lagged (Picard).",
                        max_restarts,
                    )
                    dTdS = 0.0
                    dTint_dY = 0.0
                    dTint_dZ = 0.0
                    weight_local = weight
                else:
                    _crash_report(e)
            else:
                # soften the relaxation for the conventional restarts
                weight_local *= 0.5

            # *restart from the old state*
            S_trial = Sold.copy() * norm / S0
            Y_trial = Yold.copy()
            Z_trial = Zold.copy()
            zold[::3] = S_trial
            zold[1::3] = Y_trial
            zold[2::3] = Z_trial

            error = 1.0
            flux_error = 1.0
            prev_fluxes = None
            itr = 1
            continue

    # Ratio of He advect/diffuse
    D2_div = np.zeros_like(D2_b)
    mask = (D_b > 0)
    D2_div[mask] = D2_b[mask] / D_b[mask]


    # Return final arrays to evolution.py

    return TransportResult(
        S_trial * S0 / norm,
        Y_trial,
        Z_trial,
        f_rock_old,
        loss_energy,
        D2_div,
        D_b,
        DZ_b,
        total_flux * M0 * T0 * S0,
        conv_flux_b * M0 * T0 * S0,
        rad_flux_b * M0 * T0 * S0,
        radiogenic_flux_b * M0 * T0 * S0,
        semi_flux_b * M0 * T0 * S0,
        latent_flux_b * M0 * T0 * S0,
        Lambda_b,
        phi_b,
        J,
        B,
        itr,
        R0_inv_b,
        Lambda_cond_i,
        Lambda_rad_i,
        bracket_dsdr_b,
        Ym_lag_i,
        Z1m_lag_i,
        Z2m_lag_i,
        innermisc_index,
        Cp_i,
        chi_curr_full,
        y_rain_gap_old_out,
        Nu_T_eff_i,
        Nu_X_eff_i,
    )