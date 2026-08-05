### ======================================================================================
"""
Module: initial.py — Planetary structure initialization at t = 0.

Constructs the initial mass grid, composition profiles (S, Y, Z, f_rock),
and hydrostatic structure (r, P, rho, T) for a planet model. This module is
called once at the start of an evolution run and provides the starting state
for the coupled transport-hydrostatic loop in evolution.py.

Key responsibilities:
    1. Mass mesh construction:
       - Legacy mode: single exponentially-refined mesh from center to surface
       - Partitioned mode: separate meshes for envelope and mantle/core regions
       - Surface and core boundary refinement via configurable exponential parameters

    2. Mass boundary anchoring:
       - Maps normalized [0,1] mesh to physical mass coordinates (grams)
       - Anchors core (mantle+Fe) boundaries at correct mass fractions
       - Uses PCHIP interpolation for smooth, monotonicity-preserving boundaries

    3. Composition profile initialization:
       - Supports uniform, piecewise-linear (struct), Gaussian, exponential,
         and sigmoid profiles for S, Y, Z, and f_rock
       - Optional Gaussian smoothing on struct profiles
       - Enforces total heavy-element mass constraint when mz_struct=True
       - Zeros out envelope composition (Y, Z) in core/mantle cells

    4. Initial structure via two methods:
       - Isentrope interpolation: loads pre-computed structure from utils/
       - RK4 shooting (rk4_initial=True): integrates HSE from scratch for
         arbitrary mass and entropy via bisection on central pressure

    5. Equation of state initialization:
       - Instantiates and caches mixtures_eos, mantle_eos, core_eos as
         module-level singletons (also stored in builtins for cross-module access)
       - Supports multiple EOS backends: CD21, CMS19, SCVH95 for H-He;
         AQUA, ice_mixture, rock mixture for Z; Mg2SiO4/MgSiO3 for mantle;
         Fe_alloy/Fe_pure variants for core

Array conventions:
    - Arrays are ordered outside-in: index 0 = surface, index N-1 = center
    - m_b[0] = M_total (surface), m_b[N] ~ 0 (center), descending
    - Cell-centered arrays (length N): r, p, rho, T, S, Y, Z, f_rock
    - Boundary arrays (length N+1): r_b, m_b
    - kcore = -1 sentinel means coreless (all envelope)
    - kcore_fe = -1 sentinel means no iron core

Developers:
    - Roberto Tejada Arevalo (APPLE+ORCHARD)
    - Ankan Sur (APPLE)
    - Yubo Su (APPLE)

Affiliation:
    Department of Astrophysical Sciences, Princeton University
"""
### ======================================================================================

import numpy as np
from scipy.interpolate import interp1d
from utils import const
import json
import builtins
from eos import eos_class
from utils.common import *
from utils import profiles
from scipy.interpolate import PchipInterpolator

# ------------------ GENERAL PARAMETERS & CONFIG --------------------- #
surf_width = float(config['general']['surf_width'])
surf_width2 = float(config['general']['surf_width2'])

mesh_mode = config['general'].get('mesh_mode', fallback='legacy').strip().lower()
if mesh_mode not in ('legacy', 'partitioned'):
    raise ValueError(
        f"Invalid general.mesh_mode='{mesh_mode}'. Supported values are 'legacy' and 'partitioned'."
    )


def _parse_optional_int_cfg(value, key_name):
    """
    Parse optional integer config values.
    Accepts empty/None/null to mean "unset".
    """
    if value is None:
        return None
    text = str(value).strip()
    if text == '' or text.lower() in ('none', 'null'):
        return None
    out = int(text)
    if out < 0:
        raise ValueError(f'general.{key_name} must be >= 0 (got {out}).')
    return out


N_env_cfg = _parse_optional_int_cfg(config['general'].get('N_env', fallback=None), 'N_env')
N_core_cfg = _parse_optional_int_cfg(config['general'].get('N_core', fallback=None), 'N_core')

if N_env_cfg is not None and N_core_cfg is not None and N_env_cfg != N_core_cfg:
    raise ValueError(
        f'general.N_env ({N_env_cfg}) and general.N_core ({N_core_cfg}) must match when both are set.'
    )

# ------------------ INITIAL PARAMETERS ------------------------------ #
M_factor = float(config['initial'].get('M_Mearth', config['initial'].get('M_factor', '317.907')))
M_factor_MJ = float(config['initial'].get('M_MJup', config['initial'].get('M_factor_MJ', '1.0')))
rk4_initial = config['initial'].getboolean('rk4_initial', fallback=False)
M_z_Me = float(config['initial']['M_z'])
M_z = M_z_Me * const.mearth
mz_struct = config['initial'].getboolean('mz_struct')

# ------------------ BOUNDARY CONDITIONS ----------------------------- #
# Use centralized solar reference constants to avoid drift across modules.
Z_atm = float(config['boundary_condition']['Z_atm']) * const.Z_SOLAR_BAHCALL06

# ------------------ CORE PARAMETERS --------------------------------- #
Mcore_Me = float(config['core']['mass_core'])
Mcore_fe_Me = float(config['core']['mass_core_fe'])
mantle_comp = str(config['core']['mantle_comp']) # either mgsio3, mg2sio4, or h2o
eos_mantle = str(config['core']['eos_mantle']) # either comb or jj for mgsio3 mantle; ignored for mg2sio4 for now
core_comp = str(config['core']['core_comp']) # either Fe_liquid or Fe_alloy for inner core
eos_core = str(config['core']['eos_core'])
mantle_entropy = float(config['core']['mantle_entropy']) # entropy values of the mantle in kb/baryon

# ------------------ EQUATION OF STATE ------------------------------- #
hhe_eos = str(config['equation_of_state']['hhe_eos'])

if Mcore_fe_Me > Mcore_Me:
    raise ValueError(
        f'Mcore_fe ({Mcore_fe_Me:.5f} Mearth) cannot be greater than Mcore ({Mcore_Me:.5f} Mearth).'
    )
if Mcore_Me > M_factor:
    raise ValueError(
        f'Mcore ({Mcore_Me:.5f} Mearth) cannot be less than M_factor ({M_factor:.5f} Mearth).'
    )

if mantle_comp =='mgsio3':
    if eos_mantle == 'comb':
        from eos import mgsio3_comb_eos
        mantle_eos = mgsio3_comb_eos.MGSIO3_COMBINED_EOS()
    elif eos_mantle == 'JJ':
        from eos import mgsio3_JJ_eos
        mantle_eos = mgsio3_JJ_eos.MGSIO3_JJ_EOS()
    elif eos_mantle == 'PPV_2':
        from eos import ppv2_core_eos
        mantle_eos = ppv2_core_eos.PPV2_CORE_EOS()
elif mantle_comp == 'mg2sio4':
    from eos import mg2sio4_aneos_eos
    mantle_eos = mg2sio4_aneos_eos.MG2SIO4_ANEOS_EOS()
elif mantle_comp == 'h2o':
    # eos_mantle selects the AQUA water core EOS variant:
    #   'revised' -> revised AQUA (Cano Amoros et al.), the v2 tables that match
    #               the z_eos='aqua_revised' envelope mixture
    #   anything else (default 'comb') -> original AQUA (Haldemann et al. 2020)
    if eos_mantle == 'revised':
        from eos import aqua_revised_core_eos
        mantle_eos = aqua_revised_core_eos.AQUA_REVISED_CORE_EOS()
    else:
        from eos import aqua_core_eos
        mantle_eos = aqua_core_eos.AQUA_CORE_EOS()
elif mantle_comp == 'aquarock':
    # Water/rock core at rock fraction f_rock_core (default 0.5).
    #   0.5   -> AQUAROCK_CORE_EOS (VAL mixture, cached f_rock=0.50 S-P
    #            inversion table) — the validated legacy path, unchanged.
    #   other -> ROCKWATER_INTERP_CORE_EOS: interpolates between the two
    #            validated endpoint EOSes (revised-AQUA water and mg2sio4
    #            ANEOS rock; the same objects mantle_comp='h2o'/'mg2sio4'
    #            use) with additive-volume/mass-weighted mixing — no
    #            per-f_rock tables needed. Pair with a matching envelope
    #            f_rock profile (f_rock_ini) so eroding-core runs stay
    #            EOS-continuous across the boundary (the counterflow
    #            refill carries the envelope-base f_rock).
    f_rock_core = config['core'].getfloat('f_rock_core', fallback=0.5)
    from eos import aquarock_core_eos
    if abs(f_rock_core - 0.5) < 1e-9:
        mantle_eos = aquarock_core_eos.AQUAROCK_CORE_EOS(f_rock=0.5)
    else:
        mantle_eos = aquarock_core_eos.ROCKWATER_INTERP_CORE_EOS(
            f_rock=f_rock_core)


# Initializing core EOS calls
if core_comp == 'Fe_alloy':
    from eos import fesi16_eos as core_eos # loading Fischer et al. 2012 iron alloy eos
elif core_comp == 'Fe_pure': 
    if eos_core == 'I14':
        # loading Ichikawa et al. 2014 pure liquid iron eos
        from eos import ichikawa_iron_eos as iron_eos
        core_eos = iron_eos.Fe_EOS()
    elif eos_core == 'D17':
        # loading Dorogokupets & Oganov 2007 pure liquid iron eos
        from eos import dorogo_iron_eos as iron_eos
        core_eos = iron_eos.Fe_EOS(phase='liquid')
    elif eos_core == 'D17_comb':
        # loading combined Dorogokupets & Oganov 2007 solid and liquid iron eos
        from eos import iron_comb_eos as iron_eos
        core_eos = iron_eos.Fe_COMBINED_EOS()
    elif eos_core == 'Fe_2':
        from eos import iron2_core_eos as iron_eos
        core_eos = iron_eos.IRON2_CORE_EOS()

    elif eos_core == 'G23_comb':
        # loading combined Gonzalez 2023 solid and liquid iron eos
        from eos import gonzalez_iron_eos as iron_eos
        core_eos = iron_eos.Fe_GONZALEZ_EOS()

    else:
        raise Exception('Invalid core_eos choice in parameter file; choices are I14, D17, comb.')
    
else:
    raise Exception('Invalid core_comp choice in parameter file; choices are Fe_liquid or Fe_alloy.')

initial_profiles_S = str(config['initial']['initial_profiles_S'])
initial_profiles_Y = str(config['initial']['initial_profiles_Y'])
initial_profiles_Z = str(config['initial']['initial_profiles_Z'])
initial_profiles_f_rock = str(config['initial']['initial_profiles_f_rock'])

struct_interp = config['initial'].get('struct_interp', fallback='linear').strip()
struct_smooth = config['initial'].getboolean('struct_smooth')
struct_sigma = float(config['initial']['struct_sigma'])

initial_profile_name = str(config['initial']['initial_profile_name'])

mean = float(config['initial']['init_gauss_center'])
std_dev_z = float(config['initial']['init_gauss_stdev_z'])
std_dev_s = float(config['initial']['init_gauss_stdev_s'])
minimum_s = float(config['initial']['minimum_init_s'])
maximum_z = float(config['initial']['maximum_init_z'])
rate_exp = float(config['initial']['init_exp_rate'])
outer_entropy_gradient = config['initial'].getboolean('outer_entropy_gradient', fallback=False)
outer_entropy_use_mass = config['initial'].getboolean('outer_entropy_use_mass', fallback=True)
outer_entropy_inner_coord = float(config['initial'].get('outer_entropy_inner_coord', fallback='0.9'))
outer_entropy_outer_coord = float(config['initial'].get('outer_entropy_outer_coord', fallback='1.0'))
outer_entropy_delta = float(config['initial'].get('outer_entropy_delta', fallback='0.0'))
outer_entropy_smooth_sigma = float(config['initial'].get('outer_entropy_smooth_sigma', fallback='0.0'))
outer_entropy_min = float(config['initial'].get('outer_entropy_min', fallback=str(minimum_s)))

# Sigmoid parameters

midpoint = float(config['initial']['midpoint'])
alpha = float(config['initial']['alpha'])

mass_struct = config['initial'].getboolean('mass_struct')

# EOS choices
eos_version    = float(config['equation_of_state'].get('eos_version', '1.0'))
hhe_eos        = str(config['equation_of_state']['hhe_eos'])
hg             = config['equation_of_state'].getboolean('hg')
tab            = config['equation_of_state'].getboolean('eos_tab')
z_eos          = str(config['equation_of_state']['z_eos'])
water_eos      = str(config['equation_of_state']['water_eos'])
rock_mixtures  = config['equation_of_state'].getboolean('rock_mixtures')
# Rock (mg2sio4) mass fraction within Z, taken from [initial] f_rock_ini.
# This single parameter drives the v2.0 metal EOS (table selection /
# interpolation); there is no separate [equation_of_state] f_rock.
f_rock_ini     = float(config['initial']['f_rock_ini'])

# --- Invalidate the cached EOS if any key parameter has changed ---
key = (eos_version, hhe_eos, rock_mixtures, z_eos, water_eos, f_rock_ini)
prev_key = getattr(builtins, "_APPLE_prev_eos_key", None)
if prev_key != key:
    if hasattr(builtins, "_APPLE_mixtures_eos"):
        del builtins._APPLE_mixtures_eos
    builtins._APPLE_prev_eos_key = key

print('Loading EOS ...')
# --- Initialize H-He-Z EOS class in builtins if not cached ---
if not hasattr(builtins, "_APPLE_mixtures_eos"):
    from eos.check_eos_data import verify_eos_data
    verify_eos_data()

    if eos_version >= 2.0:
        # --------------------------------------------------------
        # EOS v2.0: hhe_z_mixtures (smoothed, rhomboid tables,
        # thermodynamic-identity derivatives, P-T dependent mu_H)
        # --------------------------------------------------------

        # Map config z_eos name to (species_list, table label).
        #   aqua_revised : pure-water revised AQUA metal.
        #   aquarock     : revised-AQUA water + mg2sio4 rock mixture.
        _z_species_map = {
            'aqua_revised': (['water_revised'],              'aqua_revised'),
            'aquarock':     (['water_revised', 'mg2sio4'],   'aqua_revised'),
        }
        if z_eos in _z_species_map:
            _species_list, _z_label = _z_species_map[z_eos]
        else:
            _species_list, _z_label = [z_eos], z_eos

        # Rock content is driven entirely by f_rock_ini (EOS v2.0 does NOT
        # use rock_mixtures).  hhe_z_mixtures auto-detects interpolation:
        #   f_rock_ini in {0, 0.5, 1.0} -> single precomputed table set
        #   any other value             -> load all three sets and
        #                                  3-point interpolate per cell.
        # Any rock requires the water+rock (aqua_revised + mg2sio4) family.
        if f_rock_ini > 0.0:
            _species_list = ['water_revised', 'mg2sio4']
            _z_label = 'aqua_revised'

        builtins._APPLE_mixtures_eos = eos_class.hhe_z_mixtures(
            hhe_eos_name = hhe_eos,
            hg           = hg,
            smooth_hhe   = True,
            smooth_z     = True,
            mu_h_vary    = True,
            species_list = _species_list,
            z_eos        = _z_label,
            f_rock       = f_rock_ini,
            pt_tab       = tab,
            inv_tab      = tab,
            srho_tab     = tab,
            y_prime      = False,
        )

    elif rock_mixtures:
        # --------------------------------------------------------
        # EOS v1.0: multifraction_mixtures (rock mixtures)
        # --------------------------------------------------------
        if hhe_eos != 'cd':
            raise Exception('Only the CD EOS is currently supported for H-He-Z mixtures!')

        # build the list of Z‐mixture EOS names
        if z_eos != 'ice_mixture':
            if water_eos == 'aqua' or water_eos == 'aqua_mlcp':
                f_ppv_eos_vals = [0.0, 0.25, 0.5, 0.75, 1.0]
                z_eos_list = [f'{z_eos}_ppv2_{f_ppv}' for f_ppv in f_ppv_eos_vals]

            elif water_eos == 'aqua_mgsio3_l':
                z_eos_list = [
                '1.0_0.0_aqua_mgsio3_l',
                '0.75_0.25_aqua_mgsio3_l',
                '0.5_0.5_aqua_mgsio3_l',
                '0.25_0.75_aqua_mgsio3_l',
                '0.0_1.0_aqua_mgsio3_l',
                ]
        else:
            if water_eos == 'aqua':
                z_eos_list = [
                    '1.0_0.0_ice_rock_mixture',
                    '0.75_0.25_ice_rock_mixture',
                    '0.5_0.5_ice_rock_mixture',
                    '0.25_0.75_ice_rock_mixture',
                    '0.0_1.0_ice_rock_mixture',
                ]
            elif water_eos == 'aqua_mlcp':
                z_eos_list = [
                    '1.0_0.0_ice_rock_mixture_aqua_mlcp',
                    '0.75_0.25_ice_rock_mixture_aqua_mlcp',
                    '0.5_0.5_ice_rock_mixture_aqua_mlcp',
                    '0.25_0.75_ice_rock_mixture_aqua_mlcp',
                    '0.0_1.0_ice_rock_mixture_aqua_mlcp',
                ]

            else:
                raise Exception('Only AQUA or AQUA_MLCP (2021 correction) are allowed as water EOSes.')
        builtins._APPLE_mixtures_eos = eos_class.multifraction_mixtures(
            hhe_eos   = hhe_eos,
            z_eos_list= z_eos_list,
            y_prime   = False,
        )

    else:
        # --------------------------------------------------------
        # EOS v1.0: original mixtures class
        # --------------------------------------------------------
        builtins._APPLE_mixtures_eos = eos_class.mixtures(
            hhe_eos = hhe_eos,
            z_eos   = z_eos,
            hg      = hg,
            y_prime = False
        )

    # optional flag if you still want to track rock vs. non‐rock
    builtins._APPLE_mixtures_eos_rock_flag = rock_mixtures

# Alias for convenience
mixtures_eos = builtins._APPLE_mixtures_eos

# Read in initial profile coordinates & deltas for S, Y, Z, f_rock
z_init_coords = json.loads(config.get('initial', 'z_init_coords'))
f_rock_init_coords = json.loads(config.get('initial', 'f_rock_init_coords'))
y_init_coords = json.loads(config.get('initial', 'y_init_coords'))
s_init_coords = json.loads(config.get('initial', 's_init_coords'))

z_init_deltas = json.loads(config.get('initial', 'z_init_deltas'))
f_rock_init_deltas = json.loads(config.get('initial', 'f_rock_init_deltas'))
y_init_deltas = json.loads(config.get('initial', 'y_init_deltas'))
s_init_deltas = json.loads(config.get('initial', 's_init_deltas'))

z_init_values = json.loads(config.get('initial', 'z_init_values'))
y_init_values = json.loads(config.get('initial', 'y_init_values'))
f_rock_init_values = json.loads(config.get('initial', 'f_rock_init_values'))
s_init_values = json.loads(config.get('initial', 's_init_values'))

def _legacy_unit_mesh(n_cells, surf_amp, surf_amp2, surf_width, surf_width2):
    """
    Legacy single-segment mesh constructor on [0, 1].

    Generates a normalized mesh with exponential density enhancement near the
    surface (f -> 1) and center (f -> 0). The algorithm works in "density space":
    start with uniform mesh, compute point density, add exponential refinement
    terms, then integrate reciprocal density to get the refined mesh positions.

    Parameters
    ----------
    n_cells : int
        Number of cells (zones). Returns n_cells + 1 boundary points.
    surf_amp, surf_amp2 : float
        Amplitudes for surface and center refinement (larger = denser near boundary).
    surf_width, surf_width2 : float
        Exponential decay scales (smaller = more concentrated refinement).

    Returns
    -------
    np.ndarray
        Mesh coordinates normalized to [0, 1], shape (n_cells + 1,).
    """
    if n_cells < 0:
        raise ValueError("n_cells must be >= 0")
    if n_cells == 0:
        return np.array([0.0], dtype=float)

    f0 = np.linspace(0.0, 1.0, n_cells + 1)
    density_f0 = 1.0 / np.gradient(f0)
    df = np.mean(density_f0)

    density_f0 += float(surf_amp) * np.exp((f0 - 1.0) / max(float(surf_width), 1e-12)) * df
    density_f0 += float(surf_amp2) * np.exp(-f0 / max(float(surf_width2), 1e-12)) * df

    out = np.cumsum(1.0 / density_f0)
    out -= out[0]
    out /= out[-1]
    return out


def create_mass_mesh(N,
                     surf_amp=1e5,
                     surf_amp2=1e6,
                     surf_width=1e-2,
                     surf_width2=6.5e-3,
                     *,
                     mode='legacy',
                     N_env=None,
                     N_mcore=None,
                     return_meta=False):
    """
    Build the normalized mesh coordinate on [0, 1].

    Parameters
    ----------
    N : int
        Total number of cells.
    surf_amp, surf_amp2 : float
        Surface and center refinement amplitudes.
    surf_width, surf_width2 : float
        Exponential decay scales for refinement.
    mode : str
        'legacy' — single-segment mesh (historical default).
        'partitioned' — two independent meshes stitched at the core-envelope
        boundary, allowing independent resolution control for each region.
    N_env, N_mcore : int, optional
        Cell counts for envelope and mantle+core (partitioned mode).
        At least one must be specified; the other is inferred from N.
    return_meta : bool
        If True, return (mesh, metadata_dict) instead of just mesh.

    Returns
    -------
    mesh : np.ndarray, shape (N+1,)
        Normalized mesh coordinates on [0, 1].
    meta : dict (only if return_meta=True)
        Contains 'mode', 'N_env', 'N_mcore', 'kcore', 'split_coord'.
    """
    if mode == 'legacy':
        mesh = _legacy_unit_mesh(N, surf_amp, surf_amp2, surf_width, surf_width2)
        meta = {'mode': 'legacy', 'kcore': None, 'N_env': None, 'N_mcore': None, 'split_coord': None}
        return (mesh, meta) if return_meta else mesh

    if mode != 'partitioned':
        raise ValueError("mode must be 'legacy' or 'partitioned'")

    if N_env is None and N_mcore is None:
        raise ValueError("partitioned mode requires N_env and/or N_mcore")
    if N_env is None:
        N_env = N - int(N_mcore)
    if N_mcore is None:
        N_mcore = N - int(N_env)
    N_env, N_mcore = int(N_env), int(N_mcore)

    if N_env < 0 or N_mcore < 0 or (N_env + N_mcore != N):
        raise ValueError("Need N_env >= 0, N_mcore >= 0, and N_env + N_mcore = N")

    mc_local = _legacy_unit_mesh(N_mcore, surf_amp, surf_amp2, surf_width, surf_width2)
    env_local = _legacy_unit_mesh(N_env, surf_amp, surf_amp2, surf_width, surf_width2)

    # Envelope always occupies the highest mesh coordinates.
    split = N_mcore / float(max(N, 1))

    if N_mcore == 0:
        mesh = env_local
    elif N_env == 0:
        mesh = mc_local
    else:
        mesh = np.concatenate([
            split * mc_local[:-1],
            split + (1.0 - split) * env_local,
        ])

    mesh[0], mesh[-1] = 0.0, 1.0
    meta = {
        'mode': 'partitioned',
        'N_env': N_env,
        'N_mcore': N_mcore,
        'kcore': N_env,
        'split_coord': split,
    }
    return (mesh, meta) if return_meta else mesh


def make_mass_boundaries_with_anchors(mesh, M_planet, Mcore=None, Mcore_fe=None,
                                      kcore=None, kcore_fe=None, *,
                                      mode='legacy', N_env=None, N_mcore=None,
                                      snap_fe_anchor=True):
    """
    Convert normalized [0,1] mesh to physical mass boundaries (grams).

    Maps the abstract mesh coordinate to enclosed mass, respecting structural
    anchors at the core-mantle boundary (Mcore) and iron core boundary (Mcore_fe).
    Uses PCHIP interpolation for smooth, monotonicity-preserving boundaries.

    Parameters
    ----------
    mesh : array-like
        Normalized mesh [0, 1], shape (N+1,).
    M_planet : float
        Total planet mass in grams.
    Mcore : float, optional
        Total core mass (mantle + iron) in grams. None for coreless planets.
    Mcore_fe : float, optional
        Iron core mass in grams. None if no iron core.
    kcore, kcore_fe : int, optional
        Pre-specified cell indices for core boundaries.
    mode : str
        'legacy' or 'partitioned' mesh strategy.
    snap_fe_anchor : bool
        If True, smoothly enforce mass anchors via PCHIP interpolation.

    Returns
    -------
    m_b : np.ndarray, shape (N+1,)
        Mass boundaries, descending from M_planet (surface) toward 0 (center).
    kcore_out : int
        Core boundary index (-1 if coreless).
    kcore_fe_out : int
        Iron core boundary index (-1 if no iron core).
    """
    mesh = np.asarray(mesh, dtype=float)
    if mesh.ndim != 1 or mesh.size < 2:
        raise ValueError("mesh must be a 1D array with length >= 2")
    if not np.all(np.diff(mesh) >= 0.0):
        raise ValueError("mesh must be nondecreasing")
    if M_planet is None or float(M_planet) <= 0.0:
        raise ValueError("M_planet must be positive")

    if mode == 'legacy':
        # Keep legacy behavior unchanged.
        s = mesh[::-1]
        Np1 = s.size
        N = Np1 - 1
        anchor_eps = 1e-12

        anchors_s = [0.0, 1.0]
        anchors_y = [0.0, 1.0]

        def add_anchor(M_target, k_target):
            if M_target is None or M_target <= 0.0:
                return None
            alpha = float(M_target) / float(M_planet)
            if alpha >= 1.0 - anchor_eps:
                return 0
            if alpha <= anchor_eps:
                return N - 1
            if k_target is None:
                k = int(np.argmin(np.abs(s - alpha)))
            else:
                k = int(k_target)
            k = int(np.clip(k, 1, Np1 - 2))
            anchors_s.append(float(s[k]))
            anchors_y.append(alpha)
            return k

        kcore_fe = add_anchor(Mcore_fe, kcore_fe)
        kcore = add_anchor(Mcore, kcore)

        order = np.argsort(anchors_s)
        x_nodes = np.array(anchors_s, dtype=float)[order]
        y_nodes = np.array(anchors_y, dtype=float)[order]

        if np.any(np.diff(y_nodes) < -1e-14):
            raise ValueError("Anchor mass fractions are not nondecreasing with radius. "
                             "Expected 0 ≤ Mcore_fe ≤ Mcore ≤ M_planet.")

        g = PchipInterpolator(x_nodes, y_nodes, extrapolate=False)
        y = g(s)
        y[0], y[-1] = 1.0, 0.0

        m_b = M_planet * y
        if kcore is None:
            kcore = N - 1
        if kcore_fe is None:
            kcore_fe = N - 1
        return m_b, int(kcore), int(kcore_fe)

    if mode != 'partitioned':
        raise ValueError("mode must be 'legacy' or 'partitioned'")

    N = mesh.size - 1
    if Mcore is None:
        raise ValueError("partitioned mode requires Mcore")

    if N_env is None and N_mcore is None:
        if kcore is None:
            raise ValueError("Need N_env/N_mcore or kcore in partitioned mode")
        N_env = int(kcore)
    if N_env is None:
        N_env = N - int(N_mcore)
    if N_mcore is None:
        N_mcore = N - int(N_env)

    N_env = int(N_env)
    N_mcore = int(N_mcore)

    if N_env < 0 or N_mcore < 0 or (N_env + N_mcore != N):
        raise ValueError("Need N_env >= 0, N_mcore >= 0, and N_env + N_mcore = N")

    alpha = float(np.clip(float(Mcore) / float(M_planet), 0.0, 1.0))
    eps = 1e-14
    if alpha >= 1.0 - eps and N_env != 0:
        raise ValueError("Mcore ~= M_planet requires N_env=0 and N_mcore=N in partitioned mode")
    if alpha <= eps and N_mcore != 0:
        raise ValueError("Mcore ~= 0 requires N_mcore=0 and N_env=N in partitioned mode")

    # Split in mesh ordering (center->surface): [0..N_mcore] interior, [N_mcore..N] envelope.
    mc = mesh[:N_mcore + 1]
    env = mesh[N_mcore:]

    def _normalize(x):
        if len(x) <= 1:
            return np.array([0.0], dtype=float)
        dx = x[-1] - x[0]
        if dx <= 0.0:
            raise ValueError("Non-monotonic mesh segment")
        return (x - x[0]) / dx

    u_mc = _normalize(mc)
    u_env = _normalize(env)

    # Build enclosed-mass fraction in center->surface order.
    y_cs_mc = alpha * u_mc
    y_cs_env = alpha + (1.0 - alpha) * u_env

    if N_mcore == 0:
        y_center_to_surface = y_cs_env
    elif N_env == 0:
        y_center_to_surface = y_cs_mc
    else:
        y_center_to_surface = np.concatenate([y_cs_mc[:-1], y_cs_env])

    # Convert to surface->center ordering used by m_b.
    y = y_center_to_surface[::-1]
    y[0], y[-1] = 1.0, 0.0

    # Surface->center index convention used throughout the codebase.
    kcore = N_env

    if Mcore_fe is None or Mcore_fe <= 0.0:
        if kcore_fe is None:
            kcore_fe = N - 1
        else:
            kcore_fe = int(np.clip(int(kcore_fe), kcore, N))
    else:
        alpha_fe = float(np.clip(float(Mcore_fe) / float(M_planet), 0.0, alpha))
        if kcore_fe is None:
            kcore_fe = int(np.argmin(np.abs(y - alpha_fe)))
        else:
            kcore_fe = int(kcore_fe)
        kcore_fe = int(np.clip(kcore_fe, kcore, N))

        if snap_fe_anchor:
            s = mesh[::-1]

            # Smoothly enforce anchors instead of single-point snapping.
            anchor_pairs = [(float(s[-1]), 0.0), (float(s[0]), 1.0)]
            if 0 < kcore < N:
                anchor_pairs.append((float(s[kcore]), alpha))
            if 0 < kcore_fe < N:
                anchor_pairs.append((float(s[kcore_fe]), alpha_fe))

            anchor_pairs.sort(key=lambda p: p[0])
            xs, ys = [], []
            for x_i, y_i in anchor_pairs:
                if xs and abs(x_i - xs[-1]) < 1e-15:
                    ys[-1] = y_i
                else:
                    xs.append(x_i)
                    ys.append(y_i)

            xs = np.asarray(xs, dtype=float)
            ys = np.asarray(ys, dtype=float)
            if np.any(np.diff(ys) < -1e-14):
                raise ValueError("Non-monotonic anchors after Fe-core smoothing.")

            g = PchipInterpolator(xs, ys, extrapolate=False)
            y = g(s)
            y[0], y[-1] = 1.0, 0.0

    m_b = M_planet * y
    dm = -np.diff(m_b)
    if np.any(dm <= 0.0):
        raise RuntimeError("Non-positive zone masses after stitching. "
                           "Try adjusting N_env/N_mcore, kcore_fe, or set snap_fe_anchor=False.")

    return m_b, int(kcore), int(kcore_fe)


def initialize_grid(
    N,
    M_planet,
    S_ini,
    Y_ini,
    Z_ini,
    f_rock_ini,
):
    """
    Set up the planetary structure at t=0.

    This is the main initialization routine called once by evolution.py.
    It builds the mass grid, loads or computes the initial hydrostatic
    structure, and constructs composition profiles from config parameters.

    Steps:
        1. Load background structure from isentrope file (or RK4 shooting)
        2. Create mass mesh and physical mass boundaries with core anchors
        3. Initialize entropy profile S (uniform, struct, gaussian, exponential)
        4. Initialize helium profile Y (as Y' = Y/(X+Y))
        5. Initialize metallicity profile Z
        6. Initialize rock fraction profile f_rock
        7. Zero out composition in core/mantle cells (Y=0, Z=0)
        8. Enforce total heavy-element mass constraint (if mz_struct=True)
        9. Convert Y' to Y: Y = Y' * (1 - Z)

    Parameters
    ----------
    N : int
        Number of mass zones.
    M_planet : float
        Total planet mass in grams (CGS).
    S_ini : float
        Initial entropy in k_B/baryon for the envelope.
    Y_ini : float
        Initial He fraction Y/(X+Y) (protosolar ~ 0.277).
    Z_ini : float
        Initial heavy-element mass fraction in the envelope.
    f_rock_ini : float
        Initial rock fraction within Z (0 = pure water, 1 = pure rock).

    Returns
    -------
    r_b_old : np.ndarray, shape (N+1,)
        Radial boundaries in cm.
    p_old : np.ndarray, shape (N,)
        Pressures at cell centers in dyn/cm^2.
    m_b : np.ndarray, shape (N+1,)
        Mass boundaries in grams, descending from surface to center.
    dm : np.ndarray, shape (N,)
        Cell masses in grams (positive).
    S : np.ndarray, shape (N,)
        Entropy profile in k_B/baryon.
    Y : np.ndarray, shape (N,)
        Helium mass fraction (Y, not Y').
    Z : np.ndarray, shape (N,)
        Heavy-element mass fraction.
    f_rock : np.ndarray, shape (N,)
        Rock fraction within Z.
    kcore : int
        Core boundary cell index (-1 if coreless).
    kcore_fe : int
        Iron core boundary cell index (-1 if no iron core).
    """

    rad_p, m_p, rho_p, p_p, t_p = np.genfromtxt('utils/isentrope_ankan_s9_adam.txt', unpack=True)

    if M_factor_MJ != 1.0:
        m_p = m_p * M_factor_MJ

    f_p = interp1d(m_p, p_p, fill_value='extrapolate')
    f_r = interp1d(m_p, rad_p, fill_value='extrapolate')
    f_t = interp1d(m_p, t_p, fill_value='extrapolate')

    # Adjust for a solid core if specified (Mcore)
    Mcore = Mcore_Me * const.mearth
    Mcore_fe = Mcore_fe_Me * const.mearth

    mesh, mesh_meta = create_mass_mesh(
        N,
        surf_width=surf_width,
        surf_width2=surf_width2,
        mode=mesh_mode,
        N_env=N_env_cfg,
        N_mcore=N_core_cfg,
        return_meta=True,
    )

    m_b, kcore, kcore_fe = make_mass_boundaries_with_anchors(
        mesh, M_planet,
        Mcore=Mcore, Mcore_fe=Mcore_fe,
        kcore=mesh_meta.get('kcore'),
        kcore_fe=None,   # or pass explicit index if you want
        mode=mesh_mode,
        N_env=mesh_meta.get('N_env'),
        N_mcore=mesh_meta.get('N_mcore'),
    )

    # Sentinel conventions:
    #   - coreless model (mass_core=mass_core_fe=0): kcore = -1 externally
    #   - ironless rocky core (mass_core>0, mass_core_fe=0): kcore_fe = -1 externally
    # Internally, convert to slice-safe indices so negative Python slicing does
    # not accidentally act on the last cell.
    is_coreless = (Mcore <= 0.0 and Mcore_fe <= 0.0)
    is_ironless_core = (Mcore > 0.0 and Mcore_fe <= 0.0)
    if is_coreless:
        kcore_idx = N
        kcore_fe_idx = N
        kcore_out = -1
        kcore_fe_out = N
    else:
        kcore_idx = int(kcore)
        kcore_fe_idx = N if is_ironless_core else int(kcore_fe)
        kcore_out = kcore_idx
        kcore_fe_out = -1 if is_ironless_core else kcore_fe_idx

    # center boundary must be zero; do NOT set m_b[-1]=1.0
    dm = -np.diff(m_b)                      # shell masses, all positive
    assert np.all(dm > 0)
    assert np.isclose(dm.sum(), M_planet, rtol=1e-12, atol=0.0)

    if rk4_initial:
        # Build initial structure from scratch using RK4 shooting method.
        # This replaces the isentrope file interpolation and works for
        # any planet mass and initial entropy.
        from rk4_initial import build_initial_structure
        p_atm_dyn = float(config['hydrostatic_equilibrium'].get('p_atm', 1.0)) * 1e6
        mantle_s = float(config['core'].get('mantle_entropy', S_ini))
        core_s = float(config['core'].get('core_entropy', S_ini))

        # If RK4 shooting fails (exception or an unphysical structure), fall
        # back to the legacy S9 Jupiter isentrope interpolation below.  The
        # isentrope start is rough for non-Jupiter configurations, but the
        # initial Henyey HSE relaxation usually converges from it anyway.
        _rk4_failed = False
        try:
            r_b_old, p_old, _rho_rk4, _T_rk4 = build_initial_structure(
                N, M_planet, S_ini, Y_ini, Z_ini, f_rock_ini,
                m_b, kcore_idx, kcore_fe_idx, p_atm_dyn,
                mantle_entropy=mantle_s if not is_coreless else None,
                core_entropy=core_s if not is_coreless else None,
            )
            if (not np.all(np.isfinite(r_b_old[:-1]))
                    or not np.all(np.isfinite(p_old))
                    or np.any(r_b_old[:-1] <= 0.0)
                    or np.any(p_old <= 0.0)):
                _rk4_failed = True
                logger.warning(
                    "RK4 initial structure contains non-finite or non-positive "
                    "values; falling back to the S9 Jupiter isentrope."
                )
        except Exception as _rk4_err:
            _rk4_failed = True
            logger.warning(
                "RK4 initial structure failed (%s); falling back to the "
                "S9 Jupiter isentrope." % _rk4_err
            )
        if _rk4_failed:
            r_b_old = f_r(m_b)
            r_b_old[-1] = 1.0
            p_old = f_p(m_b[:-1])
    else:
        r_b_old = f_r(m_b)
        r_b_old[-1] = 1.0
        p_old = f_p(m_b[:-1])

    r = get_geom_mean(r_b_old)  # cell-centered radius
    m = get_geom_mean(m_b)      # cell-centered mass

    # Validate struct-profile inputs explicitly.
    def _validate_struct_inputs(name, mode, coords, deltas, values):
        if mode != 'struct':
            return
        has_deltas = len(deltas) > 0
        has_values = len(values) > 0
        has_coords = len(coords) > 0
        # Cannot specify both deltas and values for the same property.
        if has_deltas and has_values:
            raise ValueError(
                f'For {name}: specify either {name.lower()}_init_deltas or '
                f'{name.lower()}_init_values, not both.'
            )
        # If coords are given, must have either deltas or values.
        if has_coords and not has_deltas and not has_values:
            raise ValueError(
                f'If you use "struct" for {name}, provide either '
                f'{name.lower()}_init_deltas or {name.lower()}_init_values '
                f'alongside {name.lower()}_init_coords.'
            )
        # If deltas/values are given, must have coords.
        if (has_deltas or has_values) and not has_coords:
            raise ValueError(
                f'If you use "struct" for {name}, {name.lower()}_init_coords '
                f'must be provided.'
            )
        # Validate lengths.
        if has_deltas and len(coords) != len(deltas) + 1:
            raise ValueError(
                f'For {name} struct profile: len(coords) must equal len(deltas)+1 '
                f'(got {len(coords)} coords, {len(deltas)} deltas).'
            )
        if has_values and len(coords) != len(values):
            raise ValueError(
                f'For {name} struct profile: len(coords) must equal len(values) '
                f'(got {len(coords)} coords, {len(values)} values).'
            )

    _validate_struct_inputs('S', initial_profiles_S, s_init_coords, s_init_deltas, s_init_values)
    _validate_struct_inputs('Y', initial_profiles_Y, y_init_coords, y_init_deltas, y_init_values)
    _validate_struct_inputs('Z', initial_profiles_Z, z_init_coords, z_init_deltas, z_init_values)
    _validate_struct_inputs('f_rock', initial_profiles_f_rock, f_rock_init_coords, f_rock_init_deltas, f_rock_init_values)

    # Start with uniform S = S_ini
    S = np.zeros(N) + S_ini
    S[kcore_idx:] = mantle_entropy
    # Initialize Y, Z, f_rock arrays
    Y_prime = np.zeros(N) + Y_ini
    Z = np.zeros(N) + Z_ini
    f_rock = np.zeros(N) + f_rock_ini
    # Profile builder (mass or radius-based)
    builder = profiles.profile_builder(r, m, Mcore, kcore_idx)

    # Extra kwargs shared by all n_point_profile_mass/radius calls
    _values_kw = dict(
        s_values=s_init_values or None,
        y_values=y_init_values or None,
        z_values=z_init_values or None,
        f_rock_values=f_rock_init_values or None,
        interp_method=struct_interp,
    )

    def _struct_profiles():
        """One full 4-quantity n_point_profile call with the CURRENT arrays.

        Each 'struct' branch below runs the full 4-quantity builder once and
        keeps a single output index. The calls are intentionally sequential:
        the closure reads the current S/Y_prime/Z/f_rock bindings at call
        time, so each quantity's call consumes the arrays already updated by
        the previous branch (S -> Y' -> Z -> f_rock threading). Do NOT hoist
        this into a single shared call — that would change the threading and
        the results.
        """
        fn = builder.n_point_profile_mass if mass_struct else builder.n_point_profile_radius
        if struct_smooth:
            return fn(
                s_init_coords, y_init_coords, z_init_coords, f_rock_init_coords,
                s_init_deltas, y_init_deltas, z_init_deltas, f_rock_init_deltas,
                S, Y_prime, Z, f_rock, **_values_kw,
                smooth_method='global_gaussian', smooth_sigma=struct_sigma
            )
        return fn(
            s_init_coords, y_init_coords, z_init_coords, f_rock_init_coords,
            s_init_deltas, y_init_deltas, z_init_deltas, f_rock_init_deltas,
            S, Y_prime, Z, f_rock, **_values_kw
        )

    # ----------------------- S Profile ----------------------- #
    if initial_profiles_S == 'struct':
        S = _struct_profiles()[0]

    elif initial_profiles_S == 'gaussian':
        S = builder.inverted_gaussian_profile(
            mean, std_dev_s, offset=minimum_s, amplitude=(S_ini - minimum_s)
        )

    elif initial_profiles_S == 'exponential':
        S = builder.inverted_exponential_profile(
            mean, rate_exp, offset=minimum_s, amplitude=(S_ini - minimum_s)
        )

    if outer_entropy_gradient:
        S = builder.apply_outer_linear_ramp(
            S,
            inner_coord=outer_entropy_inner_coord,
            outer_coord=outer_entropy_outer_coord,
            delta_outer=outer_entropy_delta,
            use_mass_coords=outer_entropy_use_mass,
            q_min=outer_entropy_min,
            smooth_sigma=outer_entropy_smooth_sigma,
        )

    # ----------------------- Y Profile ----------------------- #
    if initial_profiles_Y == 'struct':
        Y_prime = _struct_profiles()[1]

    elif initial_profiles_Y == 'gaussian':
        Y_prime = builder.gaussian_profile(mean, std_dev_z, amplitude=maximum_z)

    elif initial_profiles_Y == 'exponential':
        Y_prime = builder.exponential_profile(mean, rate_exp, amplitude=maximum_z)

    # ----------------------- Z Profile ----------------------- #
    if initial_profiles_Z == 'struct':
        Z = _struct_profiles()[2]

    elif initial_profiles_Z == 'gaussian':
        if mz_struct:
            Z = builder.gaussian_profile(mean, std_dev_z, amplitude=maximum_z)
        else:
            Z = builder.gaussian_profile(mean, std_dev_z, offset=Z_ini, amplitude=(maximum_z - Z_ini))

    elif initial_profiles_Z == 'exponential':
        Z = builder.exponential_profile(mean, rate_exp, amplitude=maximum_z)
    elif initial_profiles_Z == 'sigmoid':
        Z = builder.sigmoid_profile(midpoint, alpha, Z_ini=Z_ini, mass_profile=mass_struct)

    # ------------------ f_rock Profile ------------------ #
    if initial_profiles_f_rock == 'struct':
        f_rock = _struct_profiles()[3]



    # Zero out composition in the core
    Y_prime[kcore_idx:] = 0.0
    Z[kcore_idx:] = 0.0
    f_rock[kcore_idx:] = 0.0

    # set up Z profiles including normalization

    if mz_struct:
        if M_z_Me > 0:
            sl_env = np.s_[:kcore_idx]

            # Mass conservation-- specified Z mass needs to be above the minimum value.
            Mz_env_desired_Me = M_z_Me - Mcore_Me
            Mz_env_curr_Me = np.sum((Z * dm)[sl_env]) / const.mearth
            M_env_tot_Me = (M_planet / const.mearth) - Mcore_Me
            deltaZ = (Mz_env_desired_Me - Mz_env_curr_Me) / M_env_tot_Me
            Z[sl_env] += deltaZ

            if Z.min() < 0:
                raise ValueError((
                    'Requested Z profile "{}" could not be made to match total M_z ' +
                    '({} M_earth) with M_core = {} Mearth. ' +
                    'Profile Mz_env={}, M_env={}, Desired Mz_env={}, deltaZ_env={}'
                ).format(initial_profiles_Z, M_z_Me, Mcore_Me, Mz_env_curr_Me, M_env_tot_Me, Mz_env_desired_Me, deltaZ))

    print('Ready!')

    return (r_b_old, p_old, m_b, dm, S, Y_prime*(1-Z), Z, f_rock, kcore_out, kcore_fe_out)
