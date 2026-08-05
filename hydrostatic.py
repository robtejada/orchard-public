"""
Module: hydrostatic.py — Henyey relaxation solver for hydrostatic equilibrium.

Solves for the pressure and radius structure of a multi-layer planet (envelope +
rocky mantle + iron core) given entropy and composition profiles. Called every
timestep by evolution.py after the transport solve updates S, Y, Z.

Numerical method:
    Henyey relaxation (Newton-Raphson on the discretized structure equations).
    The state vector contains 2N unknowns interleaved as:
        [ln P_0, ln r_0, ln P_1, ln r_1, ..., ln P_{N-1}, ln r_{N-1}]

    Residual equations (2N total):
        - N pressure equations (hydrostatic balance with centrifugal term):
          dP/dm = -Gm/(4*pi*r^4) + omega^2*r/(6*pi)
        - N radius equations (mass continuity):
          dr/dm = 1/(4*pi*r^2*rho)

    Boundary conditions:
        - Outer: P[0] matched to P_atm (+ gravity correction)
        - Inner: uniform-sphere regularity condition at center

    The 2N x 2N Jacobian is sparse block-tridiagonal (CSC format), solved via
    scipy.sparse.linalg.spsolve. Under-damped Newton with configurable initial
    alpha (default 0.4) and optional adaptive ramping toward 1.0 as convergence
    progresses. Convergence tolerance ~1e-10, max 2000 iterations. Returns NaN
    sentinel on failure (triggers step rejection in evolution.py).

EOS calls per zone:
    - Envelope (k < kcore): mixtures_eos — H-He-Z mixture
    - Mantle (kcore <= k < kcore_fe): mantle_eos — silicate (MgSiO3, Mg2SiO4, H2O)
    - Core (k >= kcore_fe): core_eos — iron alloy or pure iron

Rotation:
    get_omega() computes angular velocity from angular momentum conservation.
    Optional Theory of Figures (TOF) corrections for J2, J4, J6, J8 via
    utils/TOF.py, activated by tof_calc=True.

Developers:
    - Roberto Tejada Arevalo (APPLE+ORCHARD)
    - Ankan Sur (APPLE)
    - Yubo Su (APPLE)

Affiliation:
    Department of Astrophysical Sciences, Princeton University
"""

import numpy as np
from utils import const
from utils.const import G, kbbar_to_erg
from scipy.sparse.linalg import spsolve
from scipy.sparse import csc_array

# from eos import metals_eos, core_eos
from initial import mixtures_eos, mantle_eos, core_eos
from utils.common import *
from utils.TOF import get_moments

# ------------------ GENERAL PARAMETERS & CONFIG --------------------- #
N = int(config['general']['N'])

# ------------------ INITIAL PARAMETERS ------------------------------ #
M_factor = float(config['initial'].get('M_Mearth', config['initial'].get('M_factor', '317.907')))
M_factor_MJ = float(config['initial'].get('M_MJup', config['initial'].get('M_factor_MJ', '1.0')))
M_planet = float(config['initial'].get('M_planet_unit_cgs', config['initial'].get('M_planet', '5.972167867791379e27'))) * M_factor
if M_factor_MJ != 1.0:
    M_planet *= M_factor_MJ
outer_entropy_gradient = config['initial'].getboolean('outer_entropy_gradient', fallback=False)
outer_entropy_use_mass = config['initial'].getboolean('outer_entropy_use_mass', fallback=True)

# ------------------ BOUNDARY CONDITIONS ----------------------------- #
planet = config['boundary_condition']['planet']
bc_atm = str(config['boundary_condition'].get('bc_atm', '')).lower()

# ------------------ HYDROSTATIC EQUILIBRIUM ------------------------- #
tol = float(config['hydrostatic_equilibrium']['tol_hydro'])
p_atm_dyn = config['hydrostatic_equilibrium'].getfloat('p_atm') * 1e6 # convert from bar to dyn/cm^2
hse_alpha_init = config['hydrostatic_equilibrium'].getfloat('hse_alpha_init', fallback=0.4)
hse_adaptive_alpha = config['hydrostatic_equilibrium'].getboolean('hse_adaptive_alpha', fallback=False)
rotation = config['hydrostatic_equilibrium'].getboolean('rotation')
# Pressure-factor centering of the gravity term in the Henyey stencil.
# 'legacy'   : historical one-sided e^{-y_k} factor (first-order; carries an
#              O(1/N) work-vs-gravity energy defect, C ~ -4.7/N * E_rad).
# 'midpoint' : e^{-(y_{k-1}+y_k)/2} factor, matching the rotation term's
#              centering (second-order; removes the O(1/N) defect and
#              converges observables at much lower N, but CHANGES RESULTS —
#              ~1% radius at N=300 — so it is opt-in until fully validated).
# See writeups/energy_conservation/energy_conservation_methods.tex.
hse_gravity_centering = config['hydrostatic_equilibrium'].get(
    'hse_gravity_centering', fallback='legacy').strip().lower()
if hse_gravity_centering not in ('legacy', 'midpoint'):
    raise ValueError(
        f"hse_gravity_centering must be 'legacy' or 'midpoint', got "
        f"'{hse_gravity_centering}'")
# Backward compatibility: prefer `tof_calc`, fall back to legacy `I_correction`.
_hse_cfg = config['hydrostatic_equilibrium']
if 'tof_calc' in _hse_cfg:
    tof_calc = _hse_cfg.getboolean('tof_calc')
else:
    tof_calc = _hse_cfg.getboolean('I_correction', fallback=False)
C_MoI = float(config['hydrostatic_equilibrium']['C_MoI'])
period = float(config['hydrostatic_equilibrium']['period'])
env_s_start = config['hydrostatic_equilibrium'].getboolean('env_s_start')
isothermal_compact_core = config['hydrostatic_equilibrium'].getboolean(
    'isothermal_compact_core', fallback=False
)
# Theory-of-Figures order for gravity harmonics and figure computation.
#   4 (default): Nettelmann 2017 ToF4 via utils/TOF.py   (J_2..J_8)
#   7          : Nettelmann 2021 ToF7 via utils/TOF7.py  (J_2..J_14)
tof_order = int(config['hydrostatic_equilibrium'].get('tof_order', fallback=4))
if tof_order not in (4, 7):
    raise ValueError(
        f"[hydrostatic_equilibrium] tof_order must be 4 or 7, got {tof_order}"
    )
if tof_order == 7:
    from utils.TOF7 import get_moments7
else:
    get_moments7 = None

# Gravity backend: 'tof' (default; ToF4/ToF7 selected by tof_order above) or
# 'cms' (Concentric Maclaurin Spheroids, utils/cms_hubbard.py). CMS is
# nonperturbative (matches MH24) but MUCH slower than ToF7 per call — intended
# as a validation/post-processing backend, not the Gyr-evolution inner loop.
gravity_method = str(
    config['hydrostatic_equilibrium'].get('gravity_method', fallback='tof')
).lower()
if gravity_method == 'cms':
    from utils.cms_hubbard import get_moments_cms_from_mean
    _cms_kmax = int(config['hydrostatic_equilibrium'].get('cms_kmax', fallback=10))
    _cms_nmu = int(config['hydrostatic_equilibrium'].get('cms_nmu', fallback=48))

# Module-level warm-start cache for the CMS shape field (zeta, shape (N, L)),
# reused as the Jacobi initial guess across calls. Reset by init=True.
_cms_shape_cache = None
_cms_shape_cache_nz = None

# Module-level warm-start cache for the full 8-tuple shape functions returned
# by TOF7. Reset by hydrostatic_equilibrium(init=True) at the start of a fresh
# evolution run. Unused when tof_order == 4.
_tof7_shape_cache = None
_tof7_shape_cache_nz = None

# Module-level warm-start cache for the 5-tuple (s0, s2, s4, s6, s8) returned
# by TOF4. Mirrors the TOF7 pattern above. Reset by hydrostatic_equilibrium(
# init=True) at the start of a fresh evolution run. Using the previous call's
# converged shapes as the Jacobi initial guess cuts the iteration count from
# ~cold-start to typically 3-5 iterations once HSE has stabilized.
_tof4_shape_cache = None
_tof4_shape_cache_nz = None

def _tof7_reset_cache():
    """Clear the TOF7 warm-start cache (called at init=True)."""
    global _tof7_shape_cache, _tof7_shape_cache_nz
    _tof7_shape_cache = None
    _tof7_shape_cache_nz = None


def _tof4_reset_cache():
    """Clear the TOF4 warm-start cache (called at init=True)."""
    global _tof4_shape_cache, _tof4_shape_cache_nz
    _tof4_shape_cache = None
    _tof4_shape_cache_nz = None


def _cms_reset_cache():
    """Clear the CMS warm-start (zeta) cache (called at init=True)."""
    global _cms_shape_cache, _cms_shape_cache_nz
    _cms_shape_cache = None
    _cms_shape_cache_nz = None


# One-shot flag for the rotation breakup diagnostic. Set True after the first
# call within an evolution run so the check doesn't repeat for every Henyey
# Newton iteration. Reset to False by _reset_breakup_check() when
# hydrostatic_equilibrium(init=True) is entered at the start of a fresh run.
_breakup_check_done = False


def _reset_breakup_check():
    """Reset the breakup-check one-shot flag (called at init=True)."""
    global _breakup_check_done
    _breakup_check_done = False


def _call_tof(r_in, rho_in, m_in, omega_in):
    """
    Unified Theory-of-Figures dispatcher.

    Always returns the TOF.py-compatible 8-element j2n array
        [J2*1e6, J4*1e6, J6*1e6, J8*1e6, oblateness, R_eq, R_pol, I_MoI]
    and 5-tuple shape functions (s0, s2, s4, s6, s8), so downstream consumers
    (evolution.py's unpack, shapes_list flatten) need not special-case
    tof_order. Dispatches on the module-level `tof_order` config:

      * tof_order == 4 -> utils.TOF.get_moments, with warm-started Jacobi
        initial guess drawn from the module-level cache of the previous call's
        converged s-functions (s2, s4, s6, s8). Mirrors the TOF7 pattern.
      * tof_order == 7 -> utils.TOF7.get_moments7, with warm-started Jacobi
        initial guess drawn from the module-level cache of the previous call's
        converged s-functions. The 11-element TOF7 j2n is repacked down to
        the 8-element layout and the 8-tuple shapes is sliced to 5 (s0..s8).

    The warm-start cache is reset whenever hydrostatic_equilibrium(init=True)
    is entered, which corresponds to a fresh evolution run.
    """
    global _tof7_shape_cache, _tof7_shape_cache_nz
    global _tof4_shape_cache, _tof4_shape_cache_nz
    global _cms_shape_cache, _cms_shape_cache_nz
    if gravity_method == 'cms':
        # CMS backend (Concentric Maclaurin Spheroids). r_in are mean/volumetric
        # radii (center->surface) as for ToF; the helper converts to CMS's
        # equatorial-radius grid internally. Warm-start zeta from the cache.
        ig = (_cms_shape_cache if (_cms_shape_cache is not None
              and _cms_shape_cache_nz == len(r_in)) else None)
        j2n, zeta, _ = get_moments_cms_from_mean(
            r_in[::-1], rho_in[::-1], m_in[-1], omega_in,
            G=G, kmax=_cms_kmax, L=_cms_nmu, initial_guess=ig,
        )
        _cms_shape_cache = zeta
        _cms_shape_cache_nz = len(r_in)
        # CMS does not produce ToF s_n figure functions; return the standard
        # 5-array zero placeholder so downstream consumers are unaffected.
        return j2n, tuple(np.zeros(len(r_in)) for _ in range(5))
    if tof_order == 7:
        ig = None
        if (_tof7_shape_cache is not None
                and _tof7_shape_cache_nz == len(r_in)):
            # TOF7.get_moments7 takes (s2, s4, s6, s8, s10, s12, s14) only
            # (no s0, which is closed-form from the equal-volume condition).
            ig = tuple(_tof7_shape_cache[1:])
        j2n, shapes_7 = get_moments7(
            r_in, rho_in, m_in, omega_in, initial_guess=ig
        )
        # TOF7 now returns the same 8-element j2n as TOF4 (J_10..J_14 are no
        # longer computed by default). We still cache the full 8-tuple of
        # shape functions for Jacobi warm-start — the higher s_{10..14} are
        # still iterated even though their J_n are not output, so that the
        # low-order J's retain TOF7's accuracy advantage over TOF4.
        _tof7_shape_cache = shapes_7
        _tof7_shape_cache_nz = len(r_in)
        shapes = tuple(shapes_7[:5])  # (s0, s2, s4, s6, s8) for consumers
        return j2n, shapes
    else:
        ig = None
        if (_tof4_shape_cache is not None
                and _tof4_shape_cache_nz == len(r_in)):
            # TOF.get_moments takes (s2, s4, s6, s8) only (no s0 — it's
            # closed-form from Nettelmann 2017 eq. (25) after Jacobi converges).
            ig = tuple(_tof4_shape_cache[1:5])
        j2n, shapes = get_moments(
            r_in, rho_in, m_in, omega_in, initial_guess=ig
        )
        _tof4_shape_cache = shapes
        _tof4_shape_cache_nz = len(r_in)
        return j2n, shapes


# ------------------ CORE PARAMETERS --------------------------------- #
mc = float(config['core']['mass_core'])
mantle_entropy = float(config['core']['mantle_entropy'])
core_entropy = float(config['core']['core_entropy'])
fe_core_offset = float(config['core']['fe_core_offset'])

# ------------------ EQUATION OF STATE ------------------------------- #
tab = config['equation_of_state'].getboolean('eos_tab')

log10e = np.log10(np.e)


# Reference (equatorial) radii used to seed rotation / Theory-of-Figures.
# The omega expression is identical for every planet type, so the old
# 8-branch elif ladder reduces to this lookup table.
_PLANET_REFERENCE_RADII = {
    "Jupiter": const.rjup_eq,
    "Saturn": const.rsat_eq,
    "Uranus": 4.007 * const.rearth,   # Helled 2010
    "Neptune": 3.883 * const.rearth,  # Helled 2010
    "Sub_Neptune": 3.00 * const.rearth,
    "Super_Earth": 1.4 * const.rearth,
    "Exoplanet": const.rjup_eq,       # placeholder; rotation unused for exoplanets
    "Super_Jupiter": const.rjup_eq,   # placeholder; rotation unused
}

def _fill_nonfinite_from_neighbors(arrays, outward_fallback):
    """Replace non-finite cells with the nearest finite neighbor value (in place).

    EOS extrapolation at very low surface pressures is the usual source of
    these NaNs; a single NaN cell would otherwise poison the entire sparse
    Henyey solve. Searches inward (toward the center) first. When
    `outward_fallback` is True (the in-Newton-loop guard), arrays with no
    finite values at all are skipped, and cells with no finite inward
    neighbor take the nearest finite outward value instead — matching the
    two originally distinct inline guards exactly.
    """
    for _arr in arrays:
        _bad = ~np.isfinite(_arr)
        if not np.any(_bad):
            continue
        if outward_fallback and not np.any(np.isfinite(_arr)):
            continue
        for _idx in np.where(_bad)[0]:
            _found = False
            for _jj in range(_idx + 1, len(_arr)):
                if np.isfinite(_arr[_jj]):
                    _arr[_idx] = _arr[_jj]
                    _found = True
                    break
            if not _found and outward_fallback:
                for _jj in range(_idx - 1, -1, -1):
                    if np.isfinite(_arr[_jj]):
                        _arr[_idx] = _arr[_jj]
                        break

def get_omega(r, rho, m, omega_planet, R_p, omega_old, t, init):
    """
    Compute the angular velocity from angular momentum conservation.

    The moment of inertia is approximated via numerical integration:
        I_structure ~ |∫ (8 pi / 3) r^4 rho(r) dr|

    The rotation rate is then (angular momentum conservation, L anchored to
    the present-day observed value):
        omega = omega_planet * (I_planet / I_structure)
    where I_planet = C_MoI * M * R_p^2 (observed constraint).

    On the init call the I_ratio is clamped to <= 1.0 (unconverged seed
    guard); on evolution calls I_ratio > 1 is allowed, so models that
    contract past the present-day MoI spin faster than omega_planet.
    TOF (Theory of Figures) corrections are deferred until t > 1e-3 Gyr
    to allow the structure to stabilize first.

    Parameters
    ----------
    r : np.ndarray
        Cell-centered radii (cm), length N.
    rho : np.ndarray
        Cell-centered densities (g/cm^3), length N.
    m : np.ndarray
        Cell-centered enclosed masses (g), length N.
    omega_planet : float
        Nominal rotation rate (rad/s) from config (2*pi/period).
    R_p : float
        Planetary reference radius (cm).
    omega_old : float
        Previous rotation rate for iterative TOF correction.
    t : float
        Current simulation time (seconds).
    init : bool
        If True, use approximate integration (no TOF correction).

    Returns
    -------
    omega : float
        Updated rotation rate (rad/s). 0 if rotation=False.
    j2n : list
        Gravitational moments [J2, J4, J6, J8, oblat, Req, Rpol, I_structure].
    shapes : list of np.ndarray
        5 shape coefficient arrays from TOF, each length N.
    """

    if not rotation:
        I_planet = C_MoI * m[0] * R_p**2
        j2n = [0, 0, 0, 0, 0, R_p, R_p, I_planet]
        return 0, j2n, [np.zeros(N)] * 5

    # Physical breakup-period check — runs once per evolution run.
    # P_breakup = 2*pi*sqrt(R^3 / (G*M)) is the Keplerian period at which
    # centrifugal support at the equator equals gravity. Below it the planet
    # cannot be hydrostatic. The dimensionless rotation parameter
    # q = (omega/omega_breakup)^2 = (P_breakup/period)^2 is the same `small`
    # used inside utils/TOF.py (line ~260), so this log gives users a direct
    # view of the exact quantity that drives J2 ~ q and J4 ~ q^2. The
    # one-shot flag prevents re-logging during every Henyey Newton iteration.
    global _breakup_check_done
    if init and not _breakup_check_done:
        # Evaluate breakup at the OBSERVED reference radius R_p (the same one
        # used for I_planet), NOT the current iterate r[0]: on init the guess
        # may be a deliberately over-expanded RK4-fallback structure
        # (rk4_initial hands HSE the more expanded candidate to relax), whose
        # transient Keplerian period would false-positive this check.  R_p
        # asks the physical question — can the actual planet spin at the
        # configured period? — independent of the iterate; during init the
        # applied omega is I_ratio-clamped and error-gated anyway.
        omega_breakup = np.sqrt(G * m[0] / R_p ** 3)
        period_breakup = 2.0 * np.pi / omega_breakup
        q = (omega_planet / omega_breakup) ** 2
        ratio = period / period_breakup

        if period <= period_breakup:
            raise ValueError(
                f"Configured rotation period {period:.1f} s is at/below the "
                f"Keplerian breakup period {period_breakup:.1f} s at the "
                f"reference radius R_p = {R_p:.3e} cm "
                f"(q = {q:.3f} >= 1). Planet cannot physically rotate that fast; "
                f"centrifugal support exceeds gravity at the equator."
            )
        if ratio < 1.5:
            logger.warning(
                f"Configured period {period:.1f} s is only {ratio:.2f}x breakup "
                f"({period_breakup:.1f} s); q = {q:.3f}. Marginal HSE stability."
            )
        logger.info(
            f"Rotation check: period={period:.1f} s, "
            f"P_breakup={period_breakup:.1f} s, ratio={ratio:.2f}, q={q:.4f}."
        )
        _breakup_check_done = True

    I_planet = C_MoI * m[0] * R_p**2
    if init:
        I_structure = abs(8 * np.pi / 3 * np.trapezoid(r**4 * rho, x=r))
        j2n = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, I_structure]
        shapes = [np.zeros(N)] * 5
    else:
        # If tof_calc, compute from TOF moments; else approximate.
        if tof_calc:
            j2n, shapes = _call_tof(r[::-1], rho[::-1], m[::-1], omega_old)
            I_structure = j2n[-1]
        else:
            I_structure = abs(8 * np.pi / 3 * np.trapezoid(r**4 * rho, x=r))
            # Keep outputs defined even when skipping TOF correction.
            j2n = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, I_structure]
            shapes = [np.zeros(N)] * 5

    I_ratio = I_planet / I_structure
    if init:
        # Clamp I_ratio <= 1 on the init HSE call only: I_structure is computed
        # from the UNCONVERGED initial radius guess (surface r ~ a few x 1e9 cm
        # before the Henyey solve expands it), so I_structure is far too small
        # and I_ratio can reach ~1e3 -- giving omega ~1e3 x nominal, a huge
        # centrifugal term, and a diverging (NaN) solve.  As the structure
        # converges I_structure -> I_planet and the clamp becomes inactive.
        I_ratio = min(I_ratio, 1.0)
    else:
        # Evolution calls: angular momentum conservation is followed honestly,
        # so a model that contracts past the present-day reference MoI spins
        # FASTER than omega_planet (I_ratio > 1) and the J2n keep evolving
        # with the structure (previously the clamp froze omega at the
        # present-day rate the moment I_structure < I_planet -- the "J-kink
        # pin").  The x2 ceiling is only a backstop against transient
        # mid-Newton I_structure excursions, far above any physical value
        # for these models (I_ratio ~ 1.00-1.05 at 4.57 Gyr).
        I_ratio = min(I_ratio, 2.0)

    return omega_planet * I_ratio, j2n, shapes


def hydrostatic_equilibrium(
    s_values, log_r_b_old, log_p_old, m_b, y_values, z_values, rock_fraction,
    kcore, kcore_fe, omega_prev, time_sc, logt_old, init=False
):
    """
    Solve for hydrostatic equilibrium using the Henyey method of relaxation.

    Iteratively refines pressure and radius profiles via damped Newton-Raphson
    on a 2N state vector [ln P_0, ln r_0, ln P_1, ln r_1, ...]. The Jacobian
    is sparse block-tridiagonal (2N x 2N) and solved via scipy.sparse.linalg.spsolve.

    Algorithm:
        1. Initialize state vector from previous (log P, log r)
        2. Compute T, rho from EOS at each cell (envelope, mantle, core)
        3. Build 2N residual vector (pressure balance + mass continuity)
        4. Build 2N x 2N sparse Jacobian
        5. Solve: dY = J^{-1} A, update: Y -= alpha * dY
        6. Check convergence: max|dY/Y| < tol
        7. Optional: adaptive alpha ramping (0.4 -> 0.95 as error decreases)
        8. Repeat up to 2000 iterations

    Core seeding (init=True):
        - Isothermal mode: seeds core at envelope-base temperature
        - Standard mode: seeds mantle with mantle_entropy, Fe core from
          mantle-base temperature continuity + fe_core_offset

    Parameters
    ----------
    s_values : np.ndarray, shape (N,)
        Entropy in k_B/baryon.
    log_r_b_old : np.ndarray, shape (N+1,)
        ln(r) at cell boundaries from previous step.
    log_p_old : np.ndarray, shape (N,)
        ln(P) at cell centers from previous step.
    m_b : np.ndarray, shape (N+1,)
        Mass boundaries in grams.
    y_values : np.ndarray, shape (N,)
        Helium mass fraction.
    z_values : np.ndarray, shape (N,)
        Heavy-element mass fraction.
    rock_fraction : np.ndarray, shape (N,)
        Rock fraction within Z.
    kcore : int
        Envelope/mantle boundary index (-1 if coreless).
    kcore_fe : int
        Mantle/iron-core boundary index (-1 if no iron core).
    omega_prev : float
        Previous rotation rate for iterative TOF correction.
    time_sc : float
        Current simulation time in seconds.
    logt_old : np.ndarray, shape (N,)
        log10(T) from previous step (used for NaN recovery).
    init : bool
        If True, use initialization-only core seeding rules.

    Returns
    -------
    r_b : np.ndarray, shape (N+1,)
        Converged radial boundaries (cm).
    p : np.ndarray, shape (N,)
        Converged pressures (dyn/cm^2).
    rho : np.ndarray, shape (N,)
        Converged densities (g/cm^3).
    temp : np.ndarray, shape (N,)
        Converged temperatures (K).
    s_values : np.ndarray, shape (N,)
        Entropy (possibly adjusted in core during init).
    g : float
        Surface gravity (cm/s^2).
    omega : float
        Final rotation rate (rad/s).
    j2n : list
        Gravitational moments [J2, J4, J6, J8, oblat, Req, Rpol, I].
    shapes : list of np.ndarray
        5 shape coefficient arrays from TOF.
    itr : int
        Number of iterations to convergence.
    """
    # Fresh evolution runs (init=True) clear any stale TOF warm-start caches
    # (both TOF4 and TOF7) and re-arm the rotation breakup one-shot diagnostic
    # in get_omega.
    if init:
        _tof4_reset_cache()
        _tof7_reset_cache()
        _cms_reset_cache()
        _reset_breakup_check()

    # Sentinel normalization:
    #   - kcore=-1    => coreless (all cells are envelope)
    #   - kcore_fe=-1 => no Fe core (mantle extends to the center)
    # Convert to slice-safe indices locally.
    n_cells = len(log_p_old)
    kcore_raw = int(kcore)
    kcore_fe_raw = int(kcore_fe)
    if kcore_raw < 0:
        kcore = n_cells
        kcore_fe = n_cells
    else:
        kcore = int(np.clip(kcore_raw, 0, n_cells))
        if kcore_fe_raw < 0:
            kcore_fe = n_cells
        else:
            kcore_fe = int(np.clip(kcore_fe_raw, kcore, n_cells))

    # Identify the planetary reference radius and rotation.
    if planet not in _PLANET_REFERENCE_RADII:
        raise ValueError('Rotation only available for Jupiter, Saturn, Uranus, Neptune, Sub_Neptune, and Super_Earth at present.')
    R_p = _PLANET_REFERENCE_RADII[planet]
    omega_planet = 2 * np.pi / period if rotation else 0

    p_atm = np.log(p_atm_dyn) # 1e6 -> 1 bar in dyn/cm^2
    mass_mid = 0.5 * (m_b[1:] + m_b[:-1])

    # Unpack old radial and pressure data (Henye(y) vector).
    r_old = log_r_b_old[:-1]       # ln(r) for the main N cells
    Y_old = np.zeros(2 * N)        # Combined vector of (ln P, ln r)
    Y_old[::2] = log_p_old
    Y_old[1::2] = r_old

    # Pre-allocate arrays that will be updated iteratively.
    rho = np.zeros_like(log_p_old)
    temp = np.zeros_like(log_p_old)
    omega_array = np.zeros(N)
    # Rotation handling (omega is centralised here, NOT recomputed inside
    # Jacobian/matrix_A, so the residual and its Jacobian share one consistent omega).
    #
    # init=True solves are the hard case: the initial guess can be badly collapsed for
    # small cores (r ~ 0.1 R_p), forcing a stiff, oscillatory ~10x expansion.  Applying
    # full omega during that expansion drives an omega<->centrifugal feedback (q =
    # (omega/omega_breakup)^2 grows as R^3 as the structure inflates) that amplifies the
    # overshoot and diverges the solve to NaN.  So on init we suppress the centrifugal
    # term with a latched error gate (0 -> 1 as the Newton error drops) that engages
    # only once the structure is genuinely near equilibrium; the gate reaches 1 well
    # before convergence, so the converged rotating figure is unchanged.
    #
    # Evolution steps (init=False) start from the previous converged rotating structure
    # -- a good guess with no large expansion -- so omega is tracked directly each
    # iteration (cheap, stable, and gives the self-consistent figure immediately).
    rot_gate = 0.0
    ROT_GATE_HI = 1e-4   # error above which rotation is suppressed on init (gate -> 0)
    ROT_GATE_LO = 1e-6   # error below which rotation is fully on on init  (gate -> 1)

    def Jacobian(x, y, q, dqdy=None):
        """
        Build the 2N x 2N Jacobian matrix for the Henyey relaxation.
        x, y, q are ln(r), ln(P), ln(rho) respectively.
        dqdy (optional): d(ln rho)/d(ln P)|_S at each cell (length N).
            When provided, adds pressure-density coupling terms to the
            radius equations, improving Newton convergence.
        Assumes global variables: N, G, m_b, omega_array, mass_mid, omega_planet,
        R_p, omega_prev, time_sc, init.
        """
        n_rows = 2 * N
        # Local rotation for each shell.  omega_array is set once per Newton iteration
        # by the error-gated homotopy in the main loop (see note at definition), so the
        # Jacobian and residual share a single, consistent omega; do not recompute it
        # from the live iterate here (that feedback diverges small-core solves).

        # Preallocate arrays for the contributions from each zone.
        # For pressure equations:
        pressure_km1 = np.zeros(N)   # contribution from zone k-1 (left)
        pressure_k   = np.zeros(N)   # self contribution at zone k
        pressure_kp1 = np.zeros(N)   # contribution from zone k+1 (right)
        # For radius equations:
        radius_k     = np.zeros(N)   # self contribution at zone k
        radius_kp1   = np.zeros(N)   # contribution from zone k+1 (right)

        # Use a single index array for all zones.
        k_array = np.arange(N)

        # --- Interior zones: k = 1 ... N-2 ---
        mask_interior = (k_array >= 1) & (k_array <= N - 2)
        k_interior = k_array[mask_interior]

        # Compute common factors that use neighboring values.
        dm_i = 0.5 * (m_b[k_interior - 1] - m_b[k_interior + 1])
        common_exp_int = np.exp(-x[k_interior] - 0.5 * (y[k_interior - 1] + y[k_interior]))

        if hse_gravity_centering == 'midpoint':
            # Second-order gravity term: midpoint pressure factor (matches the
            # rotation term's centering). Derivatives: d/dy_{k-1} = d/dy_k =
            # -Tg/2 (chain through -0.5(y_{k-1}+y_k)), d/dx_k = -4 Tg.
            Tg_mid = (G / (4 * np.pi)) * dm_i * m_b[k_interior] * np.exp(
                -0.5 * (y[k_interior - 1] + y[k_interior]) - 4 * x[k_interior])
            pressure_km1[k_interior] = 1 - 0.5 * Tg_mid \
                                + (0.5 / (6 * np.pi)) * (omega_array[k_interior] ** 2) * dm_i * common_exp_int
            pressure_k[k_interior] = (-1
                                - 0.5 * Tg_mid
                                + (0.5 / (6 * np.pi)) * (omega_array[k_interior] ** 2) * dm_i * common_exp_int)
            pressure_kp1[k_interior] = -4 * Tg_mid \
                                + (1 / (6 * np.pi)) * (omega_array[k_interior] ** 2) * dm_i * common_exp_int
        else:
            # Legacy one-sided factor e^{-y_k}; expressions kept verbatim for
            # bit-identity with all pre-flag results.
            pressure_km1[k_interior] = 1 + (0.5 / (6 * np.pi)) * (omega_array[k_interior] ** 2) * dm_i * common_exp_int
            pressure_k[k_interior] = (-1
                                - (G / (4 * np.pi)) * dm_i * m_b[k_interior] * np.exp(-y[k_interior] - 4 * x[k_interior])
                                + (0.5 / (6 * np.pi)) * (omega_array[k_interior] ** 2) * dm_i * common_exp_int)
            pressure_kp1[k_interior] = (-4 * G / (4 * np.pi)) * dm_i * m_b[k_interior] * np.exp(-y[k_interior] - 4 * x[k_interior]) \
                                + (1 / (6 * np.pi)) * (omega_array[k_interior] ** 2) * dm_i * common_exp_int

        radius_k[k_interior] = 1 + 1.5 * (m_b[k_interior] - m_b[k_interior + 1]) / (4 * np.pi) \
                        * np.exp(-1.5 * (x[k_interior] + x[k_interior + 1]) - q[k_interior])
        radius_kp1[k_interior] = -1 + 1.5 * (m_b[k_interior] - m_b[k_interior + 1]) / (4 * np.pi) \
                            * np.exp(-1.5 * (x[k_interior] + x[k_interior + 1]) - q[k_interior])

        # --- Top zone: k = 0 ---
        pressure_k[0] = np.exp(y[0])
        pressure_kp1[0] = ((4 * G / (4 * np.pi)) * m_b[0] * (m_b[0] - m_b[1]) * 0.5 * np.exp(-4 * x[0])
                        - 0.5 * (m_b[0] - m_b[1]) * (omega_array[0] ** 2) * np.exp(-x[0]) / (6 * np.pi))
        radius_k[0] = 1 + 1.5 * (m_b[0] - m_b[1]) / (4 * np.pi) \
                    * np.exp(-1.5 * (x[0] + x[1]) - q[0])
        radius_kp1[0] = -1 + 1.5 * (m_b[0] - m_b[1]) / (4 * np.pi) \
                        * np.exp(-1.5 * (x[0] + x[1]) - q[0])

        # --- Pressure-density coupling for radius equations ---
        # d(radius_eqn)/d(y[k]) through d(q[k])/d(y[k]) = d(ln rho)/d(ln P)|_S
        # The radius residual term C[k] = (m[k]-m[k+1])/(4π) exp(-1.5(x[k]+x[k+1]) - q[k])
        # has d(C)/d(y[k]) = C[k] · dqdy[k]  (chain rule through -q[k])
        radius_pressure_k = np.zeros(N)
        if dqdy is not None:
            # Interior + top zones (k = 0..N-2): radius eqn has -C[k] term
            # d(-C[k])/d(y[k]) = C[k] * dqdy[k]  (from -exp(-q) → +exp(-q)*dq/dy)
            C_interior = (m_b[k_interior] - m_b[k_interior + 1]) / (4 * np.pi) \
                * np.exp(-1.5 * (x[k_interior] + x[k_interior + 1]) - q[k_interior])
            radius_pressure_k[k_interior] = C_interior * dqdy[k_interior]

            C_top = (m_b[0] - m_b[1]) / (4 * np.pi) \
                * np.exp(-1.5 * (x[0] + x[1]) - q[0])
            radius_pressure_k[0] = C_top * dqdy[0]

        # --- Bottom zone: k = N-1 ---
        pressure_km1[N - 1] = np.exp(y[N - 2])
        pressure_k[N - 1] = -np.exp(y[N - 1])
        # No pressure_kp1 for the bottom zone.
        radius_k[N - 1] = 1
        # No radius_kp1 for the bottom zone.

        # Add pressure-density coupling to bottom zone if dqdy available
        if dqdy is not None:
            rho_bottom = np.exp(q[N - 1])
            # Inner boundary (row 2N-2): d(rho^(4/3))/d(y[N-1])
            # = (4/3) * (G/2) * (4π/3)^(1/3) * m^(2/3) * rho^(4/3) * dqdy[N-1]
            pressure_k[N - 1] += (
                (4.0 / 3.0) * G / 2.0 * (4 * np.pi / 3.0) ** (1.0 / 3.0)
                * m_b[N - 1] ** (2.0 / 3.0)
                * rho_bottom ** (4.0 / 3.0)
                * dqdy[N - 1]
            )
            # Central condition (row 2N-1): d(+(1/3)*q[N-1])/d(y[N-1])
            radius_k[N - 1] += (1.0 / 3.0) * dqdy[N - 1]

        # --- Now build the sparse matrix.
        # For a single index array k_array, the row indices for pressure equations are 2*k and for radius equations 2*k+1.
        pressure_rows = 2 * k_array       # pressure equations
        radius_rows = 2 * k_array + 1      # radius equations

        # Build lists for row, column, and data entries.
        row_list = []
        col_list = []
        data_list = []

        # Pressure equations:
        # Self contribution: J[2*k, 2*k] = pressure_k[k]
        row_list.append(pressure_rows)
        col_list.append(2 * k_array)
        data_list.append(pressure_k)

        # Left contribution (for k > 0): J[2*k, 2*k - 2] = pressure_km1[k]
        mask_left = (k_array > 0)
        row_list.append(pressure_rows[mask_left])
        col_list.append(2 * k_array[mask_left] - 2)
        data_list.append(pressure_km1[mask_left])

        # Right contribution (for k < N-1): J[2*k, 2*k + 1] = pressure_kp1[k]
        mask_right = (k_array < N - 1)
        row_list.append(pressure_rows[mask_right])
        col_list.append(2 * k_array[mask_right] + 1)
        data_list.append(pressure_kp1[mask_right])

        # Radius equations:
        # Self contribution: J[2*k+1, 2*k+1] = radius_k[k]
        row_list.append(radius_rows)
        col_list.append(2 * k_array + 1)
        data_list.append(radius_k)

        # Right contribution (for k < N-1): J[2*k+1, 2*k+3] = radius_kp1[k]
        row_list.append(radius_rows[mask_right])
        col_list.append(2 * k_array[mask_right] + 3)
        data_list.append(radius_kp1[mask_right])

        # Pressure-density coupling: J[2*k+1, 2*k] = radius_pressure_k[k]
        # Only add nonzero entries to avoid filling the sparse matrix unnecessarily.
        if dqdy is not None:
            nz_mask = radius_pressure_k != 0
            if np.any(nz_mask):
                row_list.append(radius_rows[nz_mask])
                col_list.append(2 * k_array[nz_mask])
                data_list.append(radius_pressure_k[nz_mask])

        # Concatenate the lists into arrays.
        row_arr = np.concatenate(row_list)
        col_arr = np.concatenate(col_list)
        jacobian_arr = np.concatenate(data_list)

        # Build the sparse matrix in CSC format directly using csc_array.
        J_sparse = csc_array((jacobian_arr, (row_arr, col_arr)), shape=(n_rows, n_rows))

        return J_sparse


    def matrix_A(x, y, q):
        """
        Builds the 2N residual vector for the Henyey method.
        x, y, q are ln(r), ln(P), ln(rho).
        """
        A = np.zeros(2 * N)

        rho_vals = np.exp(q)
        # omega_array is set once per Newton iteration by the error-gated homotopy in
        # the main loop (see note at definition); use it directly.  J2n / shape
        # figures are computed from the CONVERGED structure after the loop
        # (get_omega and the `if rotation:` block), not here.

        # Top boundary: P - Patm - GM/r^2 + rotation
        A[0] = (
            -np.exp(p_atm)
            + np.exp(y[0])
            - (G / (4 * np.pi)) * m_b[0] * (m_b[0] - m_b[1]) * 0.5 * np.exp(-4 * x[0])
            + 0.5 * (m_b[0] - m_b[1]) * omega_array[0] ** 2 * np.exp(-x[0]) / (6.0 * np.pi)
        )

        # Middle region: differences in lnP, plus gravitational + rotational terms.
        # The gravity term's pressure factor is centering-dependent (see the
        # hse_gravity_centering parse note at module top): 'midpoint' uses the
        # second-order e^{-(y_{k-1}+y_k)/2}; 'legacy' keeps the historical
        # one-sided e^{-y_k}, verbatim for bit-identity.
        if hse_gravity_centering == 'midpoint':
            _grav_pressure_factor = np.exp(
                -0.5 * (y[: N - 2] + y[1 : N - 1]) - 4 * x[1 : N - 1])
        else:
            _grav_pressure_factor = np.exp(-y[1 : N - 1] - 4 * x[1 : N - 1])
        A[2 : 2 * N - 2 : 2] = (
            y[: N - 2]
            - y[1 : N - 1]
            + (G / (4.0 * np.pi))
            * (m_b[: N - 2] - m_b[2 : N])
            * 0.5
            * m_b[1 : N - 1]
            * _grav_pressure_factor
            - (m_b[: N - 2] - m_b[2 : N])
            * 0.5
            / (6.0 * np.pi)
            * omega_array[1 : N - 1] ** 2
            * np.exp(-x[1 : N - 1] - 0.5 * (y[: N - 2] + y[1 : N - 1]))
        )

        # Inner boundary: P continuity + approximate central boundary
        A[2 * N - 2] = (
            np.exp(y[N - 2])
            - np.exp(y[N - 1])
            + G
            / 2.0
            * (4 * np.pi / 3.0) ** (1.0 / 3.0)
            * m_b[N - 1] ** (2.0 / 3.0)
            * (rho_vals[N - 1]) ** (4.0 / 3.0)
        )

        # Next boundary condition: ln(r[k]) - ln(r[k+1]) ...
        A[1 : 2 * N - 1][::2] = (
            x[: N - 1]
            - x[1 : N]
            - (m_b[: N - 1] - m_b[1 : N])
            / (4.0 * np.pi)
            * np.exp(-1.5 * (x[: N - 1] + x[1 : N]) - q[: N - 1])
        )

        # Central condition: x[N-1] - 1/3 ( ln(...) - ln(rho[N-1]) )
        A[2 * N - 1] = x[N - 1] - (1.0 / 3.0) * (np.log(3 * m_b[N - 1] / (4 * np.pi)) - q[N - 1])

        return A

    # ---------------------------------------------------------------------- #
    # Iteration loop (Henyey relaxation)
    # ---------------------------------------------------------------------- #
    maxiter = 2000
    error = 1.0
    itr = 0

    mantle_init_s = mantle_entropy
    core_init_s = core_entropy
    # Bare-rock runs set kcore=0, so guard any kcore-1 indexing and skip empty
    # envelope slices; kcore_fe==kcore means no rocky mantle (pure Fe core).
    has_envelope = kcore > 0
    has_mantle = kcore_fe > kcore
    preserve_core_entropy_profile = (
        outer_entropy_gradient
        and outer_entropy_use_mass
        and planet == 'Super_Earth'
        and bc_atm == 'bare'
        and not has_envelope
    )


    # ---------------------------------------------------------------------- #
    # Fix 3: Better initial guess for core pressures.
    # When initializing a model with a compact core (init=True, mc>0), the
    # isentrope-derived pressure profile has no knowledge of the dense core
    # material.  Replace core pressure guesses with a simple inward
    # hydrostatic integration using approximate densities.
    # ---------------------------------------------------------------------- #
    _rk4_initial = config['initial'].getboolean('rk4_initial', fallback=False)
    if init and mc > 0 and kcore < N and not _rk4_initial:
        rho_mantle_approx = 7.0    # g/cm^3, silicate at giant planet core conditions
        rho_fe_approx     = 13.0   # g/cm^3, iron at giant planet core conditions

        if has_envelope:
            P_base = np.exp(log_p_old[kcore - 1])
            r_base = np.exp(r_old[kcore - 1])
        else:
            P_base = np.exp(log_p_old[0])
            r_base = np.exp(r_old[0])

        P_running = P_base
        r_running = r_base

        for k in range(kcore, N):
            rho_approx = rho_fe_approx if k >= kcore_fe else rho_mantle_approx
            dm_k = m_b[k] - m_b[k + 1]
            m_mid = 0.5 * (m_b[k] + m_b[k + 1])

            dr = dm_k / (4.0 * np.pi * r_running**2 * rho_approx)
            r_new = max(r_running - dr, 1e5)
            r_mid = 0.5 * (r_running + r_new)

            dP = G * m_mid * dm_k / (4.0 * np.pi * r_mid**4)
            P_running = P_running + dP
            r_running = r_new

            log_p_old[k] = np.log(P_running)
            r_old[k] = np.log(r_running)

        Y_old[::2] = log_p_old
        Y_old[1::2] = r_old

    def _seed_and_evaluate_state(final):
        """Evaluate T/rho (and, on init, seed core S) from the current pressures.

        Shared by the two call sites that previously duplicated this cascade
        verbatim: once per Henyey iteration (final=False, reading the live
        `log_p_old` iterate via closure) and once after convergence
        (final=True, reading the converged binding). The EOS call sequence is
        identical at both sites; `final` selects only the historically
        different NaN-guard behavior:
        - final=False: guard (with outward fallback) runs mid-cascade, between
          the mantle and Fe-core density evaluations and only when a core
          exists, so the Fe-core rho sees guard-cleaned temperatures.
        - final=True: guard (inward-only) runs once at the very end,
          unconditionally (also for coreless envelopes).
        Mutates temp / rho / s_values in place.
        """
        # Envelope: mixtures_eos. Skip if there is no envelope (kcore=0).
        if has_envelope:
            temp[:kcore] = 10 ** mixtures_eos.get_logt_sp(
                s_values[:kcore],
                log_p_old[:kcore] * log10e,
                y_values[:kcore],
                z_values[:kcore],
                rock_fraction[:kcore],
                ideal_guess=init,
                arr_guess=logt_old[:kcore],
                tab=tab,
            )
            rho[:kcore] = 10 ** mixtures_eos.get_logrho_pt(
                log_p_old[:kcore] * log10e,
                np.log10(temp[:kcore]),
                y_values[:kcore],
                z_values[:kcore],
                rock_fraction[:kcore],
            )

        if mc > 0:
            if init:
                if isothermal_compact_core and has_mantle and has_envelope:
                    # Seed the full compact core (rock + Fe) with the envelope-base T.
                    # Entropies are derived from that constant T (strict isothermal seed).
                    core_seed_temp = temp[kcore - 1]
                    temp[kcore:] = core_seed_temp

                    s_values[kcore:kcore_fe] = mantle_eos.get_s_pt(
                        np.exp(log_p_old[kcore:kcore_fe]) * mantle_eos.dyn_to_GPa,
                        core_seed_temp,
                    ) * mantle_eos.erg_to_kbbar

                    s_values[kcore_fe:] = core_eos.get_s_pt(
                        np.exp(log_p_old[kcore_fe:]) * core_eos.dyn_to_GPa,
                        core_seed_temp,
                    ) * const.erg_to_kbbar

                else:
                    # Mantle initialization only makes sense if a mantle exists.
                    if has_mantle:
                        if preserve_core_entropy_profile:
                            # Keep the initialized rocky-interior entropy profile.
                            pass
                        elif env_s_start and has_envelope:
                            # Envelope-mantle continuity is only valid if kcore>0.
                            temp[kcore] = temp[kcore - 1]
                            s_values[kcore:kcore_fe] = mantle_eos.get_s_pt(
                                np.exp(log_p_old[kcore]) * mantle_eos.dyn_to_GPa,
                                temp[kcore]
                            ) * mantle_eos.erg_to_kbbar
                        else:
                            # For bare rock or no envelope, seed mantle with a uniform entropy.
                            s_values[kcore:kcore_fe] = mantle_init_s

                        temp[kcore:kcore_fe] = mantle_eos.get_t_sp(
                            s_values[kcore:kcore_fe],
                            np.exp(log_p_old[kcore:kcore_fe]) * mantle_eos.dyn_to_GPa,
                        )

                    # If there is a mantle boundary, use its temperature to seed the Fe core.
                    if has_mantle and (0 < kcore_fe < N):
                        temp[kcore_fe] = temp[kcore_fe - 1] # temperature continuity into Fe core
                        s_values[kcore_fe:] = core_eos.get_s_pt(
                            np.exp(log_p_old[kcore_fe]) * core_eos.dyn_to_GPa,
                            temp[kcore_fe]
                        ) * const.erg_to_kbbar
                        s_values[kcore_fe:] += fe_core_offset
                    else:
                        # No mantle boundary to anchor; use a uniform core entropy.
                        s_values[kcore_fe:] = core_init_s

                    temp[kcore_fe:] = core_eos.get_T_sp(
                        s_values[kcore_fe:],
                        np.exp(log_p_old[kcore_fe:]) * core_eos.dyn_to_GPa
                    )

            else:
                # For later iterations, only update mantle temperatures if it exists.
                if has_mantle:
                    temp[kcore:kcore_fe] = mantle_eos.get_t_sp(
                        s_values[kcore:kcore_fe],
                        np.exp(log_p_old[kcore:kcore_fe]) * mantle_eos.dyn_to_GPa,
                    )

                temp[kcore_fe:] = core_eos.get_T_sp(s_values[kcore_fe:], np.exp(log_p_old[kcore_fe:]) * core_eos.dyn_to_GPa)

            # Mantle density is only defined when the mantle slice is non-empty.
            if has_mantle:
                rho[kcore:kcore_fe] = mantle_eos.get_rho_sp(
                    s_values[kcore:kcore_fe],
                    np.exp(log_p_old[kcore:kcore_fe]) * mantle_eos.dyn_to_GPa,
                )

            if not final:
                # Guard: if EOS extrapolation produced NaN in T or rho (e.g.,
                # low-pressure mantle cells outside the EOS table), replace
                # with nearest finite neighbor values so the Henyey iteration
                # can proceed. Without this, a single NaN cell poisons the
                # entire sparse solve. Placed BEFORE the Fe-core rho so that
                # evaluation sees guard-cleaned temperatures.
                _fill_nonfinite_from_neighbors((temp, rho), outward_fallback=True)

            rho[kcore_fe:] = core_eos.get_rho_pt(
                np.exp(log_p_old[kcore_fe:]) * core_eos.dyn_to_GPa,
                temp[kcore_fe:]
            )


        if final:
            # Guard: fill any remaining NaN in final T/rho from nearest valid
            # neighbor (inward-only). Runs unconditionally — also for coreless
            # envelopes — matching the historical post-convergence behavior.
            _fill_nonfinite_from_neighbors((temp, rho), outward_fallback=False)

    while error > tol:
        itr += 1
        if itr > maxiter:
            raise ValueError('Max iterations exceeded for hydrostatic equilibrium.')

        # Evaluate T/rho from the current pressure iterate (and seed the core
        # entropies on the first pass); see _seed_and_evaluate_state.
        _seed_and_evaluate_state(final=False)

        # ----------------------------------------------------------
        # Compute d(ln rho)/d(ln P)|_S for the Jacobian coupling.
        # The radius equation depends on rho which depends on P
        # through the EOS; without this term the Jacobian treats
        # rho as constant w.r.t. P, slowing convergence.
        # Only enable after a few initial iterations to let the
        # state stabilize (avoids NaN on init=True first call).
        # ----------------------------------------------------------
        dlnrho_dlnP = None  # None = no coupling this iteration
        dp_frac = 1e-4  # fractional pressure perturbation

        # Gate: skip the very first iterations (avoids NaN on unrelaxed
        # init seeds) and engage once the state is merely SANE (1e-1),
        # not nearly-converged (the historical 1e-3 gate made the first
        # ~10 iterations run with the rho-P coupling missing -- the
        # Jacobian error that dominates the iteration count).
        if hse_adaptive_alpha and itr >= 3 and error < 1e-1:
            dlnrho_dlnP = np.zeros(N)

            if has_envelope:
                # Use analytic Gamma1 from the mixture EOS
                gamma1_env = mixtures_eos.get_gamma1(
                    s_values[:kcore],
                    log_p_old[:kcore] * log10e,  # ln(P) -> log10(P)
                    y_values[:kcore],
                    z_values[:kcore],
                    rock_fraction[:kcore],
                    tab=tab,
                )
                # Clamp Gamma1 away from zero to prevent singular derivatives
                gamma1_env = np.maximum(gamma1_env, 0.1)
                dlnrho_dlnP[:kcore] = 1.0 / gamma1_env

            if mc > 0:
                if has_mantle:
                    P_GPa_m = np.exp(log_p_old[kcore:kcore_fe]) * mantle_eos.dyn_to_GPa
                    rho_p = mantle_eos.get_rho_sp(
                        s_values[kcore:kcore_fe], P_GPa_m * (1 + dp_frac))
                    rho_m = mantle_eos.get_rho_sp(
                        s_values[kcore:kcore_fe], P_GPa_m * (1 - dp_frac))
                    dlnrho_dlnP[kcore:kcore_fe] = (
                        np.log(rho_p / np.maximum(rho_m, 1e-99)) / (2 * dp_frac)
                    )

                if kcore_fe < N:
                    P_GPa_c = np.exp(log_p_old[kcore_fe:]) * core_eos.dyn_to_GPa
                    T_p = core_eos.get_T_sp(
                        s_values[kcore_fe:], P_GPa_c * (1 + dp_frac))
                    T_m = core_eos.get_T_sp(
                        s_values[kcore_fe:], P_GPa_c * (1 - dp_frac))
                    rho_p = core_eos.get_rho_pt(P_GPa_c * (1 + dp_frac), T_p)
                    rho_m = core_eos.get_rho_pt(P_GPa_c * (1 - dp_frac), T_m)
                    dlnrho_dlnP[kcore_fe:] = (
                        np.log(rho_p / np.maximum(rho_m, 1e-99)) / (2 * dp_frac)
                    )

        # Recompute omega from the current structure (consumed by Jacobian/matrix_A this
        # iteration).  On init, ramp it in via a latched error gate; on evolution steps,
        # track it directly (see note at definition).
        if rotation:
            omega_target, _, _ = get_omega(
                np.exp(r_old), rho, mass_mid, omega_planet, R_p, omega_prev, time_sc, init
            )
            if init:
                _gate = np.clip(
                    (np.log10(ROT_GATE_HI) - np.log10(max(error, 1e-30)))
                    / (np.log10(ROT_GATE_HI) - np.log10(ROT_GATE_LO)),
                    0.0, 1.0,
                )
                rot_gate = max(rot_gate, float(_gate))   # latch: never decrease
                omega_array[:] = omega_target * rot_gate
            else:
                omega_array[:] = omega_target

        # Build and solve the linear system
        _log_rho = np.log(rho)
        A_vec = matrix_A(r_old, log_p_old, _log_rho)
        J_sparse = Jacobian(r_old, log_p_old, _log_rho, dlnrho_dlnP)
        dY = spsolve(J_sparse, A_vec)
        # Adaptive damping: start at hse_alpha_init (safe, linear convergence)
        # and ramp toward ~0.9 as error decreases (quadratic convergence).
        if hse_adaptive_alpha:
            if error > 1e-2:
                alpha = hse_alpha_init
            elif error < 1e-5:
                alpha = 0.9
            else:
                t = (np.log10(error) - np.log10(1e-2)) / (np.log10(1e-5) - np.log10(1e-2))
                alpha = hse_alpha_init + t * (0.95 - hse_alpha_init)
        else:
            alpha = hse_alpha_init

        # Guard: if the linear solve produced non-finite values (singular J),
        # bail out so _get_hse_state_issues can reject the timestep.
        if not np.all(np.isfinite(dY)):
            log_p_old[:] = np.nan
            r_old[:] = np.nan
            break

        # -------------------------------------------------------------- #
        # Newton update with optional core damping + clamping.
        # Fix 1: Reduced alpha for core zones during init to prevent
        #        overshooting in stiff EOS regions.
        # Fix 2: Clamp max change in ln(P) and ln(r) per iteration
        #        to prevent catastrophic EOS extrapolation.
        # -------------------------------------------------------------- #
        Y_new = Y_old - alpha * dY

        # Fix 2: Clamp pressure and radius changes during init
        if init:
            max_dlnP = 2.0
            max_dlnr = 2.0
            pressure_indices = slice(0, 2 * N, 2)
            radius_indices   = slice(1, 2 * N, 2)
            dlnP = Y_new[pressure_indices] - Y_old[pressure_indices]
            Y_new[pressure_indices] = Y_old[pressure_indices] + np.clip(dlnP, -max_dlnP, max_dlnP)
            dlnr = Y_new[radius_indices] - Y_old[radius_indices]
            Y_new[radius_indices] = Y_old[radius_indices] + np.clip(dlnr, -max_dlnr, max_dlnr)

        # Guard: if Y_new itself is non-finite, bail out.
        if not np.all(np.isfinite(Y_new)):
            log_p_old[:] = np.nan
            r_old[:] = np.nan
            break

        error = np.max(np.abs(Y_new - Y_old) / np.abs(Y_old + 1e-30))

        # Update for next iteration
        Y_old = Y_new
        log_p_old = Y_old[::2]
        r_old = Y_old[1::2]

    # ---------------------------------------------------------------------- #
    # Final update of T, rho once converged (log_p_old is now the converged
    # binding); the inward-only NaN guard runs at the end of the helper.
    # ---------------------------------------------------------------------- #
    _seed_and_evaluate_state(final=True)

    # ---------------------------------------------------------------------- #
    # Final outputs
    # ---------------------------------------------------------------------- #
    r = np.exp(r_old)
    p = np.exp(log_p_old)
    r_b = np.insert(r, N, 1)  # Insert boundary at the end
    g = G * m_b[0] / r[0] ** 2

    # Recompute omega and the gravitational moments from the CONVERGED structure.
    # During the Newton iteration omega was error-gated for stability; here we close
    # one operator-split step by re-evaluating omega on the relaxed structure so the
    # returned omega / J2n are self-consistent with it (across timesteps this converges
    # to the fully self-consistent rotating figure).  get_omega also supplies the
    # correct non-rotating sentinels (omega=0, J2n=0, radii=R_p, MoI=I_planet) since the
    # in-loop matrix_A no longer fills j2n.
    omega_final, j2n, shapes = get_omega(
        r, rho, mass_mid, omega_planet, R_p, omega_prev, time_sc, init
    )
    omega_array[:] = omega_final
    if rotation:
        j2n, shapes = _call_tof(r[::-1], rho[::-1], mass_mid[::-1], omega_array[0])

    return r_b, p, rho, temp, s_values, g, omega_array[0], j2n, shapes, itr
