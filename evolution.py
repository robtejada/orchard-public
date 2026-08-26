#!/usr/bin/python
"""
ORCHARD: A Generalized Planetary Evolution Code (derived from APPLE)
Version: February 2026

Main evolution driver that orchestrates the coupled structure-transport loop
for modeling the thermal and compositional evolution of planets from 0.5 Earth
masses to 10+ Jupiter masses over gigayear timescales.

The main loop in run() follows this operator-split structure:

    Initialization (initial.py):
        Build mass mesh, load EOS, set S/Y/Z/f_rock profiles.
        Solve initial HSE -> r, P, rho, T.
        Evaluate atmosphere BC -> T_int, T_eff.

    Main loop (while t < final_age):
        1. update_SYZ()              [transport.py]   -> S_new, Y_new, Z_new + fluxes
        2. hydrostatic_equilibrium() [hydrostatic.py] -> r_b, P, rho, T, g, omega
        3. Validate state (NaN, non-monotonic P, excessive changes)
           - If invalid: dt /= 2, retry
        4. Compute error metric (max fractional change in S, T, rho, Y, Z above P_floor)
           - If error > threshold: dt /= 2, retry
        5. Evaluate atmosphere BC    [atm_bc.py]      -> T_int, T_eff
        6. [Optional] regrid_profiles() [regrid.py]   - refine rain zone
        7. Adaptive timestep: dt_new = min(f_err * f_itr * dt, dt_max)
        8. Save output (text or HDF5)

Adaptive Timestepping:
    - Error metric: max fractional change in (S, Y, Z, T, rho) above pressure floor
    - Early-time smoothing: stricter at t=0, relaxes toward user tolerance by ~0.1 Gyr
    - Step rejection halves dt; accepted steps grow dt by min(f_err, 2x cap)
    - Growth cap reduced to 1.25x after a retry to prevent oscillations
    - Rain-aware: He rain onset suppresses dt growth to avoid composition jumps

Energy Conservation:
    - Tracks internal energy (U), gravitational energy (E_g), and radiative losses (L)
    - Uses midpoint or predictor-corrector luminosity estimates (configurable)
    - Reports conservation residual: (U_f - E_g_f + L_f) / (U_0 - E_g_0 + L_0) - 1

Output:
    - Text files: one per saved timestep in models/<config_name>/
    - HDF5: single file with time-indexed datasets (SWMR for live reading)

Reference:
    Henyey, Kippenhahn & Weigert; Tejada Arevalo et al. 2026b; Sur et al. 2024

Developers:
    - Roberto Tejada Arevalo (APPLE+ORCHARD)
    - Ankan Sur (APPLE)
    - Yubo Su (APPLE)

Affiliation:
    Department of Astrophysical Sciences, Princeton University
"""
### ======================================================================================

import subprocess
import numpy as np
from utils import const
from utils.const import G, kbbar_to_erg
from utils.hdf5_evol_saver import H5EvolutionWriter
from scipy.interpolate import interp1d
import time
import atm_bc
from hydrostatic import hydrostatic_equilibrium
#from hydrostatic_sp_test import hydrostatic_equilibrium
from initial import initialize_grid, mixtures_eos, mantle_eos, core_eos
import os
from transport import update_SYZ, max_itr
import shutil
import glob as _glob
from utils.common import *
from utils.regrid import regrid_profiles

def _get_hse_state_issues(r_b, p, rho, temp, S):
    """
    Validate a trial hydrostatic equilibrium state for physical consistency.

    Checks for NaN/inf in all state variables, non-positive pressures, and
    non-monotonic pressure profiles. Used by the main loop to decide whether
    to accept a timestep or reject and retry with halved dt.

    Parameters
    ----------
    r_b : array-like
        Radial boundaries (N+1).
    p : array-like
        Cell-centered pressures (N). Must be positive and monotonically
        increasing with depth (i.e., P[k+1] > P[k]).
    rho : array-like
        Cell-centered densities (N).
    temp : array-like
        Cell-centered temperatures (N).
    S : array-like
        Cell-centered entropies (N).

    Returns
    -------
    list of str
        Human-readable issue descriptions. Empty list means the state is valid.
    """
    issues = []
    for name, arr in (("r_b", r_b), ("p", p), ("rho", rho), ("T", temp), ("S", S)):
        arr_np = np.asarray(arr)
        bad_count = int(np.count_nonzero(~np.isfinite(arr_np)))
        if bad_count:
            issues.append(f"{name}: {bad_count} non-finite")

    p_np = np.asarray(p)
    if np.all(np.isfinite(p_np)):
        non_pos = int(np.count_nonzero(p_np <= 0.0))
        if non_pos:
            issues.append(f"p: {non_pos} non-positive")
        bad_mon_idx = np.where(np.diff(p_np) <= 0.0)[0]
        if bad_mon_idx.size > 0:
            i0 = int(bad_mon_idx[0])
            issues.append(f"p: non-monotonic at idx {i0}->{i0 + 1}")

    return issues

def _residual_triplet(uf, egf, e0, lost):
    """One energy-conservation residual channel.

    Total-energy residual (U_f - E_g_f + E_lost) - e0 when `lost` is used as
    the radiated-energy accumulator, normalized by the initial total energy
    and by the loss itself. Shared by the per-step diagnostics and the final
    summary; the expressions match the previously duplicated inline code
    exactly (op order preserved).

    Returns (residual/e0, residual/lost-or-NaN, residual).
    """
    ef_x = uf - egf + lost
    residual = ef_x - e0
    return residual / e0, (residual / lost if lost != 0 else np.nan), residual

def run():
    """
    Main evolution routine:

    1) Reads configuration parameters to set up a planet model.
    2) Initializes planetary structure (mass grid, S, Y, Z, f_rock, core indices, etc.).
    3) Selects and initializes the atmosphere boundary condition object.
    4) Loops over time until the final age is reached:
        a) Updates composition and entropy with `update_SYZ` (transport solve).
        b) Solves for hydrostatic equilibrium (`hydrostatic_equilibrium`).
        c) Validates the trial state and rejects/retries with halved timestep if needed.
        d) Computes diagnostics (energy conservation, Brunt-Vaisala, rotation/TOF harmonics).
        e) Computes error metric and adapts the timestep based on relative error in S, Y, Z, rho, T.
        f) Evaluates the atmosphere boundary condition to obtain T_int and T_eff.
        g) Saves intermediate results at configured intervals (text or HDF5).
    5) Restarts are NOT supported (restart_flag=True raises immediately): the
       energy-conservation ledger cannot be reconstructed from saved output,
       and ORCHARD policy is to always run models fresh.
    6) Upon exit or exception, final data is saved and summary statistics are logged.
    """

    # ------------------ GENERAL PARAMETERS & CONFIG --------------------- #
    N = int(config['general']['N'])
    final_age = float(config['general']['final_age']) * const.Gyr_to_s
    dt_ = float(config['general']['dt_'])
    err = float(config['general']['error'])
    # Stall guard (0 = disabled): end the run cleanly if the adaptive timestep
    # collapses below this many Myr after early evolution (dt-independent
    # per-step errors, e.g. rain-zone flicker at sharp Z fronts, reject every
    # halving and would otherwise spin at the dt floor until walltime).
    dt_abort_myr = config['general'].getfloat('dt_abort_myr', fallback=0.0)
    # Rain-drain kill switch (0 = disabled): end the run cleanly if the
    # committed outer-cell metal mass fraction falls below this many x solar
    # (Chen+2023, Z_sun = 0.017) after early evolution. Rain-active corners
    # that drain the outer envelope toward Z_out ~ 0 can never recover the
    # Z_atm target; stop them instead of burning walltime to final_age.
    z_atm_abort_solar = config['general'].getfloat('z_atm_abort_solar',
                                                   fallback=0.0)
    # Post-step repair of unphysical envelope temperature inversions
    # (Schwarzschild-unstable layers left behind by under-converged transport
    # NR updates during rain eras). Default False = exact no-op.
    repair_T_inv = config['transport'].getboolean('repair_T_inversions',
                                                  fallback=False)
    max_step = float(config['general']['max_step']) * const.Myr_to_s
    early_max_step_myr = config['general'].getfloat('early_max_step_myr', fallback=0.0)
    early_max_step_until_gyr = config['general'].getfloat('early_max_step_until_gyr', fallback=0.0)
    if early_max_step_myr > 0.0 and early_max_step_until_gyr > 0.0:
        early_max_step = early_max_step_myr * const.Myr_to_s
        early_max_step_until = early_max_step_until_gyr * const.Gyr_to_s
    else:
        early_max_step = None
        early_max_step_until = 0.0
    late_max_step_myr = config['general'].getfloat('late_max_step_myr', fallback=0.0)
    late_max_step_from_gyr = config['general'].getfloat('late_max_step_from_gyr', fallback=0.0)
    if late_max_step_myr > 0.0 and late_max_step_from_gyr > 0.0:
        late_max_step = late_max_step_myr * const.Myr_to_s
        late_max_step_from = late_max_step_from_gyr * const.Gyr_to_s
    else:
        late_max_step = None
        late_max_step_from = float('inf')
    save_interval = float(config['general']['save_interval']) * const.Myr_to_s
    early_save_interval_myr = config['general'].getfloat('early_save_interval_myr', fallback=0.0)
    early_save_until_gyr = config['general'].getfloat('early_save_until_gyr', fallback=0.0)
    if early_save_interval_myr > 0.0 and early_save_until_gyr > 0.0:
        early_save_interval = early_save_interval_myr * const.Myr_to_s
        early_save_until = early_save_until_gyr * const.Gyr_to_s
    else:
        early_save_interval = None
        early_save_until = 0.0

    def _effective_save_interval(t_now):
        """Return the active save interval at model age *t_now*, or 0 if saving every step."""
        if early_save_interval is not None and t_now < early_save_until:
            return early_save_interval
        return save_interval

    def _effective_max_step(t_now):
        """Return the active timestep cap at model age *t_now* (seconds).

        Combines the baseline `max_step` with optional time-windowed tighteners:
        `early_max_step` (active for t < early_max_step_until) and
        `late_max_step` (active for t >= late_max_step_from). Returns the
        smallest of whichever caps are currently in effect.
        """
        cap = max_step
        if early_max_step is not None and t_now < early_max_step_until:
            cap = min(cap, early_max_step)
        if late_max_step is not None and t_now >= late_max_step_from:
            cap = min(cap, late_max_step)
        return cap

    save_data = config['general'].getboolean('save_data')
    verbose = config['general'].getboolean('verbose', fallback=False)
    energy_conservation_debug = config['general'].getboolean('energy_conservation_debug', fallback=False)
    save_verbose = config['general'].getboolean('save_verbose', fallback=False)
    restart_flag = config['general'].getboolean('restart_flag')
    if restart_flag:
        # Accounting fix (2026-07): fail fast instead of crashing later. The
        # restart loader never reconstructed the energy/luminosity histories
        # (energy[0], lum_arr seeds), so the first accepted step would die
        # with an IndexError inside the energy bookkeeping — and the budget
        # would be wrong even if it didn't. Project policy is to always run
        # models fresh; make that explicit here.
        raise RuntimeError(
            "restart_flag=True is not supported: the energy-conservation "
            "accounting cannot be reconstructed from a restart, and ORCHARD "
            "policy is to always run models fresh."
        )
    high_precision = config['general'].getboolean('high_precision')
    save_hdf5 = config['general'].getboolean('save_hdf5')
    hdf5_filename = config["general"].get("hdf5_filename", "evolution.h5").strip()
    h5w = None

    # ------------------ BOUNDARY CONDITIONS ----------------------------- #
    BC_atm = config['boundary_condition']['bc_atm']
    Y_atm_bc = float(config['boundary_condition']['Y_atm'])
    # Use shared solar reference for C23-style atmosphere metallicity conversions.
    Z_atm_bc = float(config['boundary_condition']['Z_atm']) * const.Z_SOLAR_CHEN23
    cloud = config['boundary_condition'].getboolean('cloud')
    irradiation = config['boundary_condition'].getboolean('irradiation')
    y_atm_fixed = config['boundary_condition'].getboolean('y_atm_fixed')
    z_atm_fixed = config['boundary_condition'].getboolean('z_atm_fixed')
    planet = config['boundary_condition']['planet']
    # Radius display unit for the verbose per-step readouts (constant for the
    # whole run): gas giants in R_Jup, everything else in R_Earth.
    if planet in ('Jupiter', 'Saturn', 'Super_Jupiter'):
        _r_scale, _r_unit_label = const.rjup, 'R_Jup'
    else:
        _r_scale, _r_unit_label = const.rearth, 'R_Earth'
    T_eq = float(config['boundary_condition']['T_eq'])
    bond_albedo = float(config['boundary_condition']['bond_albedo'])
    rock_gray_kappa_ir = float(config['boundary_condition'].get('rock_gray_kappa_ir', 1e-3))
    rock_gray_p_match_bar = float(config['boundary_condition'].get('rock_gray_p_match_bar', 1.0))
    rock_gray_p_match = rock_gray_p_match_bar * 1e6

    # ------------------ INITIAL PARAMETERS ------------------------------ #
    M_factor = float(config['initial'].get('M_Mearth', config['initial'].get('M_factor', '317.907')))
    M_factor_MJ = float(config['initial'].get('M_MJup', config['initial'].get('M_factor_MJ', '1.0')))
    M_planet = float(config['initial'].get('M_planet_unit_cgs', config['initial'].get('M_planet', '5.972167867791379e27'))) * M_factor
    if M_factor_MJ != 1.0:
        M_planet *= M_factor_MJ
    S_ini = float(config['initial']['S_ini'])
    Y_ini = float(config['initial']['Y_ini'])
    Z_ini = float(config['initial']['Z_ini'])
    f_rock_ini = float(config['initial']['f_rock_ini'])
    folder_name = '%s/%s' % (args.data_folder, model_fldr)
    h5_path = None
    if save_hdf5:
        h5_name = hdf5_filename
        if h5_name.endswith(os.sep) or os.path.splitext(h5_name)[1] == "":
            h5_name = os.path.join(h5_name, "evolution.h5")
        h5_path = h5_name if os.path.isabs(h5_name) else os.path.join(folder_name, h5_name)

    # ------------------ TRANSPORT & DIFFUSION --------------------------- #
    convection = config['transport'].getboolean('convection')
    DY_micro_val = get_diffusion_param('Dmicro_y')
    DZ_micro_val = get_diffusion_param('Dmicro_z')
    DZ_micro_val_in = get_diffusion_param('Dmicro_z_in') # interior diffusion coefficient
    # Dimensionless enclosed-mass threshold for stronger interior Z diffusion.
    # Using m/M avoids unit bugs from comparing grams to dimensionless numbers.
    z_diff_inner_mfrac = get_diffusion_param('z_diff_inner_mfrac', default=0.7)
    # Maximum allowed melt-fraction change per timestep before the chi limiter
    # halves dt.  Lower values → tighter control, smaller timesteps.
    max_delta_chi = float(config['transport'].get('max_delta_chi', 0.05))
    # Pressure floor [GPa] for the adaptive error metric.  Zones below
    # this threshold are excluded from S/T/rho error so the rapidly-cooling
    # surface doesn't force impractically small timesteps.
    p_err_floor_GPa = float(config['transport'].get('p_err_floor', 0.01))

    # rain_error_factor (>= 1): multiplicative relaxation of the adaptive
    # timestep error threshold when helium rain is active.  With the implicit
    # rain solver, a value of 1.0 (no relaxation) is appropriate.
    rain_error_factor = float(config['diffusion'].get('rain_error_factor', 1.0))
    alpha_herain = float(config['diffusion'].get('alpha_herain', 1.0))

    # ------------------ MASS LOSS PARAMETERS ----------------------------- #
    mass_loss_enabled = config.getboolean('mass_loss', 'mass_loss', fallback=False)
    if mass_loss_enabled:
        raise RuntimeError(
            "mass_loss is an experimental feature and is not yet available in this version of ORCHARD. "
            "It will be included in a future release."
        )
    # ------------- SPATIAL D SMOOTHING PARAMETER ----------------------------- #
    smooth_D_spatial = config.getint('transport', 'smooth_D_spatial', fallback=0)

    # ------------------ REGRID PARAMETERS --------------------------------- #
    regrid_enabled = config.getboolean('regrid', 'regrid_enabled', fallback=False)
    regrid_every = config.getint('regrid', 'regrid_every', fallback=50)
    t_regrid_start = config.getfloat('regrid', 't_regrid_start_gyr', fallback=0.15) * const.Gyr_to_s
    regrid_step_counter = 0  # counts accepted steps since last regrid

    # Whether rotation / TOF harmonics are enabled (controls Js_tof.txt output).
    rotation = config['hydrostatic_equilibrium'].getboolean('rotation', fallback=False)

    # ------------------ Active Plotting Setup ----------------------------- #
    _active_plot = args.plot_active
    _active_frame_num = 0
    _active_movie_outdir = os.path.join('plots', model_fldr + '_movie')
    _active_temp_ylim_kK = None  # set from initial frame, reused for all
    _active_next_plot_time = max_step  # first non-init frame at t = max_step
    _plot_rock_phases = planet in ('Sub_Neptune', 'Super_Earth')
    if _active_plot:
        import matplotlib
        matplotlib.use('Agg')
        matplotlib.rcParams['text.usetex'] = False  # faster text rendering for movie frames
        from plot_class import init_plot as _init_plot
        if os.path.isdir(_active_movie_outdir):
            shutil.rmtree(_active_movie_outdir)
        os.makedirs(_active_movie_outdir, exist_ok=True)

    # ------------------ Initialize Variables & Grids -------------------- #
    save_time = 0
    t = 0
    i = 0
    retry = 0
    # Chi-limiter plateau trackers (see the chi-based timestep limiter below).
    _chi_last_absdchi = None
    _chi_plateau_n = 0
    # Official energy-loss integral used for conservation diagnostics.
    # We use the midpoint of the transport and post-HSE luminosity estimates.
    energy_lost = 0
    energy_lost_midpoint = 0.0
    energy_lost_hse = 0

    (
        r_b_old,
        p_old,
        m_b,
        dm,
        S,
        Y,
        Z,
        f_rock,
        kcore,
        kcore_fe,
    ) = initialize_grid(N, M_planet, S_ini, Y_ini, Z_ini, f_rock_ini)
    # Sentinel normalization for runtime calculations (preserve raw values for metadata):
    #   - kcore=-1    => coreless (all cells are envelope)
    #   - kcore_fe=-1 => no Fe core (mantle extends to the center)
    kcore_raw = int(kcore)
    kcore_fe_raw = int(kcore_fe)
    is_coreless = (kcore_raw < 0)
    if is_coreless:
        kcore = N
        kcore_fe = N
    else:
        kcore = int(np.clip(kcore_raw, 0, N))
        if kcore_fe_raw < 0:
            kcore_fe = N
        else:
            kcore_fe = int(np.clip(kcore_fe_raw, kcore, N))

    has_core_boundary = not is_coreless
    has_fe_boundary = (kcore_fe < N)

    def _cell_value_or_last(arr, idx):
        idx = int(idx)
        if idx < 0 or idx >= len(arr):
            return arr[-1]
        return arr[idx]

    def _cell_value_or_nan(arr, idx, exists=True):
        idx = int(idx)
        if (not exists) or idx < 0 or idx >= len(arr):
            return float('nan')
        return arr[idx]

    def _boundary_value_or_nan(arr, idx, exists=True):
        idx = int(idx)
        if (not exists) or idx < 0 or idx >= len(arr):
            return float('nan')
        return arr[idx]

    # Cache envelope/mantle existence once; these flags keep bare-rock runs
    # from calling H/He EOS or using kcore-1 indices.
    has_envelope = kcore > 0
    has_mantle = kcore_fe > kcore

    # Store initial mass in helium and metals
    mass_Y_init = np.sum(Y * dm)
    mass_Z_init = np.sum(Z * dm)

    # Arithmetic mean of m_b for references
    m = get_arith_mean(m_b)
    innermisc_index = kcore

    eos_tab_flag = config['equation_of_state'].getboolean('eos_tab', fallback=True)

    # ------------------ ARRAYS FOR SAVING MODEL OUTPUT ------------------ #
    data_r = []
    data_r_b = []
    data_p = []
    data_rho = []
    data_temp = []
    data_dt = []
    data_save_time = []
    data_tint = []
    data_T_eff = []
    data_age = []
    data_grav = []
    data_S = []
    data_t10 = []
    data_totrad = []
    data_mantle_rad = []
    data_core_rad = []
    data_omega = []
    data_Y = []
    data_Z = []
    data_Y_atm = []
    data_Z_atm = []
    data_f_rock = []
    energy = []
    D2_arr = []
    D_b_old = np.zeros(N+1) + DY_micro_val
    DZ_b_old = np.zeros(N+1) + DZ_micro_val
    Nu_T_eff_old = np.ones(N - 1)
    # Previous accepted step's cell-centered dT/dt [K/s], used by the opt-in
    # time_centered_heat_factor mode in transport.py (eps_S fix). None means
    # "no history yet" (first step, or right after a regrid) -> transport
    # falls back to the legacy old-state T weighting for that step.
    dTdt_prev = None
    Nu_X_eff_old = np.ones(N - 1)
    mfrac_b = m_b / m_b[0]
    inner_z_diff_mask_b = mfrac_b < z_diff_inner_mfrac
    DZ_b_old[inner_z_diff_mask_b] = DZ_micro_val_in
    D2 = np.zeros(N+1)
    lum_arr = []
    lum_hse_bsurf_arr = []
    loss_lum_arr = []
    loss_lum_pc_arr = []
    lum_compare_rows = []
    lum_hse_radius_compare_rows = []
    lum_transport_pc_compare_rows = []
    tint_compare_rows = []
    energy_diag_rows = []
    energy_lost_transport = 0.0
    energy_lost_transport_pc = 0.0
    energy_lost_hse_bsurf = 0.0
    #pgap_arr = []
    fluxes = []
    flux = np.zeros(N + 1)
    brunt_arr_SY = []
    brunt_arr_Prho = []
    brunt_arr_PT = []
    Ym_arr = []
    Z1m_arr = []
    Z2m_arr = []
    conser_energy = []
    grad_l, grad_ldx_l, cp_l, chit_chirho_l = [], [], [], []
    grad_a_l, B_Y_l, B_Z_l = [], [], []
    dSdZ_prho_l, dSdZ_pt_l, dSdY_prho_l, dSdY_pt_l = [], [], [], []
    dsdr_l, dydr_l, dzdr_l = [], [], []
    data_R0_inv = []
    conv_flux_arr = []
    semi_flux_arr = []
    rad_flux_arr = []
    radio_flux_arr = []
    lambda_cond_arr = [] if save_verbose else None
    lambda_rad_arr = [] if save_verbose else None
    bracket_dsdr_arr = [] if save_verbose else None
    D_arr = []
    Ym_old_i = np.ones(N - 1)
    Z1m_old_i = np.ones(N - 1)
    Z2m_old_i = np.ones(N - 1)
    # Helium rain gap bounds (pressure) from previous timestep.
    # None means no detected rain region in the previous converged step.
    y_rain_gap_old = None

    data_int_lum = []
    data_phases = []

    melt_fraction = []

    J2_list, J4_list, J6_list, J8_list, f_list, I_list, I2_list, Req_list, Rpol_list, Rvol_list = [], [], [], [], [], [], [], [], [], []
    shapes_list = []

    # ------------------------------------------------------------------ #
    # Save-path helpers.  The initial-step and main-loop save paths are
    # structurally identical and differ only in WHICH values they record
    # (initial state vs newly accepted state); those differences enter
    # through the parameters below.  Quantities whose variable names are
    # identical at both call sites (t, dt_, save_time, T_int, T_eff, g,
    # t10_bar, the J2n/shape figures, N_i_SY, R0_inv_b, flux, D2, the
    # verbose *_i diagnostics, ...) are read from the enclosing scope at
    # call time.  Both helpers only append references / build dicts --
    # no arithmetic -- so they cannot perturb the saved values.
    # ------------------------------------------------------------------ #
    def _record_saved_step(*, r_now, r_b_now, p_now, rho_now, temp_now, S_now,
                           Y_now, Z_now, f_rock_now, omega_now, mantle_rad,
                           core_rad, Ym_i, Z1m_i, Z2m_i, int_lum, melt_chi,
                           conserv_row, cp_row, dsdy_prho_row, dsdy_pt_row,
                           dsdz_prho_row, dsdz_pt_row, D_row, phases_row):
        """Append one saved timestep to every in-memory output series."""
        data_r.append(r_now)
        data_r_b.append(r_b_now)
        data_p.append(p_now)
        data_rho.append(rho_now)
        data_temp.append(temp_now)
        data_tint.append(T_int)
        data_T_eff.append(T_eff)
        data_Y_atm.append(Y_now[0])
        data_Z_atm.append(Z_now[0])
        data_age.append(t)
        data_dt.append(dt_)
        data_save_time.append(save_time)
        data_S.append(S_now)
        data_grav.append(g)
        data_t10.append(t10_bar)
        data_totrad.append(r_now[0])
        data_mantle_rad.append(mantle_rad)
        data_core_rad.append(core_rad)
        data_Y.append(Y_now)
        data_Z.append(Z_now)
        data_f_rock.append(f_rock_now)
        data_omega.append(omega_now)
        if conserv_row is not None:
            conser_energy.append(conserv_row)
        brunt_arr_SY.append(N_i_SY)
        data_R0_inv.append(R0_inv_b)
        Ym_arr.append(Ym_i)
        Z1m_arr.append(Z1m_i)
        Z2m_arr.append(Z2m_i)
        data_int_lum.append(int_lum)
        melt_fraction.append(melt_chi)
        if save_verbose:
            conv_flux_arr.append(conv_flux)
            semi_flux_arr.append(semi_flux)
            rad_flux_arr.append(rad_flux)
            radio_flux_arr.append(radiogenic_flux)
            lambda_cond_arr.append(Lambda_cond_i)
            lambda_rad_arr.append(Lambda_rad_i)
            bracket_dsdr_arr.append(bracket_dsdr_b)
            grad_a_l.append(grad_a_i)
            grad_l.append(grad_i)
            grad_ldx_l.append(grad_ldx_i)
            cp_l.append(cp_row)
            chit_chirho_l.append(chit_chirho_i)
            brunt_arr_Prho.append(N_i_Prho)
            brunt_arr_PT.append(N_i_PT)
            B_Y_l.append(B_Y_i)
            B_Z_l.append(B_Z_i)
            dSdY_prho_l.append(dsdy_prho_row)
            dSdY_pt_l.append(dsdy_pt_row)
            dSdZ_prho_l.append(dsdz_prho_row)
            dSdZ_pt_l.append(dsdz_pt_row)
            dsdr_l.append(dsdr_i)
            dydr_l.append(dydr_i)
            dzdr_l.append(dzdr_i)
            D_arr.append(D_row)
            D2_arr.append(D2)
            fluxes.append(flux)
            data_phases.append(phases_row)
        J2_list.append(j2)
        J4_list.append(j4)
        J6_list.append(j6)
        J8_list.append(j8)
        shapes_list.append(np.array(shapes).flatten())
        f_list.append(obl)
        I_list.append(MoI)
        I2_list.append(MoI)
        Req_list.append(eq_rad)
        Rpol_list.append(pol_rad)
        Rvol_list.append(eq_rad**(2/3) * pol_rad**(1/3))

    def _build_snapshot(*, r_now, r_b_now, p_now, rho_now, temp_now, S_now,
                        Y_now, Z_now, f_rock_now, omega_now, mantle_rad,
                        core_rad, Ym_i, Z1m_i, Z2m_i, int_lum, melt_chi,
                        conserv_row, Dcoeff_row, cp_row, dsdy_prho_row,
                        dsdy_pt_row, dsdz_prho_row, dsdz_pt_row):
        """Build the HDF5 snapshot dict for one saved timestep."""
        snap = {
            "age": t,
            "dt": dt_,
            "save_time": save_time,
            "T_int": T_int,
            "T_eff": T_eff,
            "Y_atm": Y_now[0],
            "Z_atm": Z_now[0],
            "grav": g,
            "totrad": r_now[0],
            "mantle_rad": mantle_rad,
            "core_rad": core_rad,
            "kcore": float(kcore),
            "kcore_fe": float(kcore_fe),
            "t10": t10_bar,
            "omega": omega_now,
            "int_lum": int_lum,
            "j2": j2,
            "j4": j4,
            "j6": j6,
            "j8": j8,
            "obl": obl,
            "eq_rad": eq_rad,
            "pol_rad": pol_rad,
            "MoI": MoI,

            "m_b": m_b,
            "dm": dm,
            "r": r_now,
            "r_b": r_b_now,
            "p": p_now,
            "rho": rho_now,
            "temp": temp_now,
            "S": S_now,
            "Y": Y_now,
            "Z": Z_now,
            "f_rock": f_rock_now,

            "Ym": Ym_i,
            "Z1m": Z1m_i,
            "Z2m": Z2m_i,
            "melt_fraction": melt_chi,
            "R0_inv": R0_inv_b,

            "brunt": N_i_SY,

            "energies_row": [t, internal_energy, grav_potential, energy_lost,
                             rot_energy],
            "conserv_row": conserv_row,
        }
        if save_verbose:
            snap["conv_flux"] = conv_flux
            snap["rad_flux"] = rad_flux
            snap["semi_flux"] = semi_flux
            snap["radio_flux"] = radiogenic_flux
            snap["Dcoeff"] = Dcoeff_row
            snap["D2"] = D2
            snap["cp"] = cp_row
            snap["dsdy_prho"] = dsdy_prho_row
            snap["dsdy_pt"] = dsdy_pt_row
            snap["dsdz_prho"] = dsdz_prho_row
            snap["dsdz_pt"] = dsdz_pt_row
            snap["Lambda_cond_i"] = Lambda_cond_i
            snap["Lambda_rad_i"] = Lambda_rad_i
            snap["bracket_dsdr"] = bracket_dsdr_b
        return snap

    def _write_text_profiles():
        """Write the text-mode output file set (one file per data series).

        Called both periodically inside the evolution loop and once from the
        final cleanup path. Every file is fully rewritten from the in-memory
        series on each call, so the write order of the files is irrelevant to
        the on-disk result.
        """
        np.savetxt(os.path.join(folder_name, 'rad_profiles.txt'), data_r, fmt='%.8e')
        np.savetxt(os.path.join(folder_name, 'rad_boundary_profiles.txt'), data_r_b, fmt='%.8e')
        np.savetxt(os.path.join(folder_name, 'press_profiles.txt'), data_p, fmt='%.8e')
        np.savetxt(os.path.join(folder_name, 'rho_profiles.txt'), data_rho, fmt='%.8e')
        np.savetxt(os.path.join(folder_name, 'temp_profiles.txt'), data_temp, fmt='%.8e')
        np.savetxt(os.path.join(folder_name, 'Sprofiles.txt'), data_S, fmt='%.8e')
        np.savetxt(os.path.join(folder_name, 'Yprofiles.txt'), data_Y, fmt='%.8e')
        np.savetxt(os.path.join(folder_name, 'Zprofiles.txt'), data_Z, fmt='%.8e')
        np.savetxt(os.path.join(folder_name, 'f_rock_profiles.txt'), data_f_rock, fmt='%.8e')
        if energy_conservation_debug:
            np.savetxt(os.path.join(folder_name, 'energies'), energy, fmt='%.8e')
            np.savetxt(os.path.join(folder_name, 'conserv_energies'), conser_energy, fmt='%.8e')
        np.savetxt(os.path.join(folder_name, 'R0_inv.txt'), data_R0_inv, fmt='%.8e')
        if save_verbose:
            np.savetxt(os.path.join(folder_name, 'conv_flux.txt'), conv_flux_arr, fmt='%.8e')
            np.savetxt(os.path.join(folder_name, 'semi_flux.txt'), semi_flux_arr, fmt='%.8e')
            np.savetxt(os.path.join(folder_name, 'rad_flux.txt'), rad_flux_arr, fmt='%.8e')
            np.savetxt(os.path.join(folder_name, 'radio_flux.txt'), radio_flux_arr, fmt='%.8e')
            np.savetxt(os.path.join(folder_name, 'cp.txt'), cp_l, fmt='%.8e')
            np.savetxt(os.path.join(folder_name, 'dsdy_prho.txt'), dSdY_prho_l, fmt='%.8e')
            np.savetxt(os.path.join(folder_name, 'dsdy_pt.txt'), dSdY_pt_l, fmt='%.8e')
            np.savetxt(os.path.join(folder_name, 'dsdz_prho.txt'), dSdZ_prho_l, fmt='%.8e')
            np.savetxt(os.path.join(folder_name, 'dsdz_pt.txt'), dSdZ_pt_l, fmt='%.8e')
            np.savetxt(os.path.join(folder_name, 'Lambda_cond_i.txt'), lambda_cond_arr, fmt='%.8e')
            np.savetxt(os.path.join(folder_name, 'Lambda_rad_i.txt'), lambda_rad_arr, fmt='%.8e')
            np.savetxt(os.path.join(folder_name, 'bracket_dsdr.txt'), bracket_dsdr_arr, fmt='%.8e')
            np.savetxt(os.path.join(folder_name, 'mantle_phases.txt'), data_phases, fmt='%.8e')
            np.savetxt(os.path.join(folder_name, 'Dcoeff.txt'), D_arr, fmt='%.8e')
        np.savetxt(os.path.join(folder_name, 'melt_fraction.txt'), melt_fraction, fmt='%.8e')
        np.savetxt(
            os.path.join(folder_name, 'evolution_data.txt'),
            np.c_[data_age, data_tint, data_T_eff, data_grav, data_totrad, data_t10, data_omega, data_int_lum],
            fmt='%.8e')
        np.savetxt(
            os.path.join(folder_name, 'evolution_yz.txt'),
            np.c_[data_Y_atm, data_Z_atm],
            fmt='%.8e')
        np.savetxt(
            os.path.join(folder_name, 'timestep_data.txt'),
            np.c_[data_dt, data_age, data_save_time],
            fmt='%.8e')
        if rotation:
            np.savetxt(os.path.join(folder_name, 'Js_tof.txt'), np.c_[ J2_list, J4_list, J6_list, J8_list, f_list, I_list, I2_list, Req_list, Rpol_list, Rvol_list], fmt='%.8e')
        np.savetxt(os.path.join(folder_name, 'brunt.txt'), brunt_arr_SY, fmt='%.8e')
        np.savetxt(os.path.join(folder_name, 'Ym.txt'), Ym_arr, fmt='%.8e')
        np.savetxt(os.path.join(folder_name, 'Z1m.txt'), Z1m_arr, fmt='%.8e')
        np.savetxt(os.path.join(folder_name, 'Z2m.txt'), Z2m_arr, fmt='%.8e')

    def _atm_boundary_tint_teff(p_now, temp_now, Z_now, T_emb_now,
                                enforce_g75_planet=False):
        """Evaluate the atmosphere boundary condition -> (T_int, T_eff).

        One 11-branch dispatch shared by the initial-step and main-loop BC
        evaluations (previously two near-identical elif ladders that could
        drift apart). S_atm, Y_atm, Z_atm, g, t1_bar, t10_bar, t1000_bar and
        t_p_interp are read from the enclosing scope at call time; the state
        arrays that differ between the call sites (p/temp/Z vs
        p_new/temp_new/Z_new) and the envelope-mantle boundary temperature
        enter as parameters. enforce_g75_planet reproduces the initial-step
        check rejecting g75 for Jupiter/Saturn.
        """
        if BC_atm == 'c23' and y_atm_fixed:
            return atm.get_tint_teff(S_atm, np.log10(g), Y_atm, Z_atm)
        elif BC_atm == 'c23' and not y_atm_fixed:
            Yprime = Y_atm/(1 - Z_atm)
            return atm.get_tint_teff(S_atm, np.log10(g), Yprime)
        elif BC_atm == 'c26':
            return atm.get_tint_teff(S_atm, np.log10(g), Y_atm, Z_atm)
        elif BC_atm == 'm21':
            return atm.get_tint_teff(np.log10(g), t10_bar, Z_atm=Z_atm/const.Z_SOLAR_CHEN23)
        elif BC_atm == 'p20':
            return atm.get_tint_teff(np.log10(g), t10_bar)
        elif BC_atm == 'f11':
            return atm.get_tint_teff(np.log10(g), t10_bar)
        elif BC_atm == 'b97':
            return atm.get_tint_teff(np.log10(g), t10_bar)
        elif BC_atm == 'g75':
            result = atm.get_tint_teff(g, t1_bar)
            if enforce_g75_planet and (planet == 'Jupiter' or planet == 'Saturn'):
                raise Exception('Graboske et al. 1975 only valid for Uranus and Neptune.')
            return result
        elif BC_atm == 'f07':
            # Fortney+07 tables use their own solar reference.
            # NOTE: a local name is required here -- assigning to t1000_bar
            # would make it function-local for the whole helper and break the
            # closure read on the normal path (UnboundLocalError).
            t1000 = t1000_bar
            p_core = _cell_value_or_last(p_now, kcore)
            if p_core < 990000000:
                # The envelope is too thin to reach the tables' 1 kbar
                # reference level: the 1 kbar surface would fall inside the
                # core/mantle, where no envelope T(P) exists.  Fall back to
                # the deepest envelope cell.  (Normal, thicker envelopes are
                # sampled at 1 kbar via t1000_bar above, which is the axis
                # the tables are built on and matches the Jacobian path in
                # transport.py.)
                p_last_env = p_now[kcore - 1]
                t1000 = t_p_interp(p_last_env)
            # Cap Z at 0.94 before computing the Fortney+2007 metallicity
            # factor: the f07 atmosphere tables become unreliable beyond this
            # point, so clipping here keeps the BC lookup inside its trusted
            # range rather than extrapolating.
            f_Z = z_to_metallicity_factor(np.clip(Z_now[0], 0.0, 0.94), const.Z_SOLAR_FORTNEY07)
            return atm.get_tint_teff(np.log10(g), np.log10(t1000), mh_ = f_Z)
        elif BC_atm == 'bare':
            return atm.get_tint_teff(T_surf=T_emb_now)
        elif BC_atm == 'gray':
            return atm.get_tint_teff(T_surf=T_emb_now, g=g)
        elif BC_atm == 'ideal_gray':
            return atm.get_tint_teff(T_surf=T_emb_now, g=g)
        else:
            raise Exception('Atmospheric boundary condition name error.')

    # Time the entire program
    program_starts = time.time()

    # ------------------ Initialize BC (Atmosphere) Object --------------- #
    # One object for the selected BC (see atm_bc.create_atm); every BC
    # evaluation below goes through this single `atm`.
    atm = atm_bc.create_atm(
        BC_atm, planet,
        cloud=cloud, irradiation=irradiation,
        T_eq=T_eq, bond_albedo=bond_albedo,
        rock_gray_kappa_ir=rock_gray_kappa_ir,
        rock_gray_p_match=rock_gray_p_match,
    )

    # ------------------ START EVOLUTION LOOP ---------------------------- #
    try:
        if not restart_flag:
            if args.clean:
                logger.info(f"Cleaning '{folder_name}' for models...")
                try:
                    shutil.rmtree(folder_name, ignore_errors=True)
                    logger.info(f"Folder '{folder_name}' deleted successfully.")
                except OSError as e:
                    logger.info(f"Error: {e}")
                    raise(e)
            os.makedirs(folder_name, exist_ok=True)

            # Remove stale output files from the opposite format so that
            # load_models never mixes old text data with new HDF5 data
            # (or vice-versa).
            if save_hdf5:
                stale = _glob.glob(os.path.join(folder_name, "*.txt"))
                # Also catch legacy files without a .txt extension
                for name in ("energies", "conserv_energies"):
                    fp = os.path.join(folder_name, name)
                    if os.path.isfile(fp):
                        stale.append(fp)
            else:
                stale = (
                    _glob.glob(os.path.join(folder_name, "*.h5"))
                    + _glob.glob(os.path.join(folder_name, "*.hdf5"))
                )

            for fp in stale:
                try:
                    os.remove(fp)
                except OSError:
                    pass
            if stale:
                logger.info(
                    f"Cleared {len(stale)} stale "
                    f"{'text' if save_hdf5 else 'HDF5'} file(s) from '{folder_name}'."
                )

            if save_hdf5:
                # Ensure parent directory exists
                os.makedirs(os.path.dirname(h5_path) or ".", exist_ok=True)

                # Delete any previous file to avoid shape/schema mismatches
                if os.path.exists(h5_path):
                    try:
                        os.remove(h5_path)
                        logger.info(f"Deleted previous HDF5 file: {h5_path}")
                    except OSError as e:
                        raise OSError(f"Could not delete existing HDF5 file '{h5_path}': {e}") from e
                    
                h5w = H5EvolutionWriter(
                    path=h5_path,
                    mode="w",
                    flush_every=1,
                    swmr=True,
                )
                metadata = {
                    "N": int(N),
                    "M_planet": float(M_planet),
                    "kcore": int(kcore_raw),
                    "kcore_fe": int(kcore_fe_raw),
                    "BC_atm": str(BC_atm),
                    "planet": str(planet),
                    "save_verbose": bool(save_verbose),
                    "config_basename": str(config_basename),
                }
                h5w.write_constants(metadata=metadata)

        j = 0
        # -------------- Handle a Restart Run --------------- #
        if restart_flag:
            dt_, t, save_time = np.genfromtxt(
                os.path.join(folder_name, "timestep_data.txt"),
                unpack=True
            )[:, -1]

            # `save_time` in timestep_data.txt is the last save checkpoint that was hit.
            # On restart we want the *next* checkpoint strictly after the current age `t`,
            # otherwise later dt adjustment (save_time - t) can go negative.
            _esi = _effective_save_interval(t)
            if _esi > 0:
                while save_time <= t:
                    save_time += _effective_save_interval(save_time)

            temp_old = np.genfromtxt(
                os.path.join(folder_name, "temp_profiles.txt"),
                unpack=True
            )[:, -1]

            r_b_old = np.genfromtxt(
                os.path.join(folder_name, "rad_boundary_profiles.txt"),
                unpack=True
            )[:, -1]

            p_old = np.genfromtxt(
                os.path.join(folder_name, "press_profiles.txt"),
                unpack=True
            )[:, -1]

            rho_old = np.genfromtxt(
                os.path.join(folder_name, "rho_profiles.txt"),
                unpack=True
            )[:, -1]

            T_int = np.genfromtxt(
                os.path.join(folder_name, 'evolution_data.txt'),
                unpack=True,
                usecols=1
            )[-1]

            Y_old = np.genfromtxt(
                os.path.join(folder_name, "Yprofiles.txt"),
                unpack=True
            )[:, -1]

            Z_old = np.genfromtxt(
                os.path.join(folder_name, "Zprofiles.txt"),
                unpack=True
            )[:, -1]

            S_old = np.genfromtxt(
                os.path.join(folder_name, "Sprofiles.txt"),
                unpack=True
            )[:, -1]

            dcoeff_fp = os.path.join(folder_name, "Dcoeff.txt")
            if os.path.isfile(dcoeff_fp):
                D_b_old = np.genfromtxt(dcoeff_fp, unpack=True)[:, -1]
            else:
                # save_verbose=False omits Dcoeff.txt; fall back to the configured baseline diffusion.
                D_b_old = np.zeros(N + 1) + DY_micro_val

            Ym_old_i = np.genfromtxt(
                os.path.join(folder_name, "Ym.txt"),
                unpack=True
            )[:, -1]

            i = 1
            TdS_energy = 0
            path = folder_name

            def _restart_read_last_profile(name, nout):
                fp = os.path.join(path, name)
                if not os.path.isfile(fp):
                    return np.zeros(nout)
                arr = np.genfromtxt(fp, unpack=True)
                arr = np.asarray(arr)
                if arr.ndim == 1:
                    return arr.astype(float)
                return np.asarray(arr[:, -1], dtype=float)

            conv_flux = _restart_read_last_profile("conv_flux.txt", N + 1)
            rad_flux = _restart_read_last_profile("rad_flux.txt", N + 1)
            dsdy_prho_i = _restart_read_last_profile("dsdy_prho.txt", N - 1)
            dsdz_prho_i = _restart_read_last_profile("dsdz_prho.txt", N - 1)
            dsdy_pt_i = _restart_read_last_profile("dsdy_pt.txt", N - 1)
            dsdz_pt_i = _restart_read_last_profile("dsdz_pt.txt", N - 1)

            cp_i = np.zeros(N - 1)
            grad_i = np.zeros(N - 1)
            grad_ldx_i = np.zeros(N - 1)
            chit_chirho_i = np.zeros(N - 1)
            N_i_SY = np.zeros(N - 1)
            N_i_Prho = np.zeros(N - 1)
            N_i_PT = np.zeros(N - 1)
            grad_a_i = np.zeros(N - 1)
            B_Y_i = np.zeros(N - 1)
            B_Z_i = np.zeros(N - 1)
            R0_inv_b = np.zeros(N + 1)

            print("Restarting at time (Gyr)=", t * const.s_to_Gyr)
            if save_hdf5:
                os.makedirs(os.path.dirname(h5_path) or ".", exist_ok=True)
                h5w = H5EvolutionWriter(
                    path=h5_path,
                    mode="a",
                    flush_every=1,
                    swmr=True,
                )

        # ------------------ TIME EVOLUTION LOOP ------------------------ #
        while t <= final_age:

            if i > 0:
                logger.info(
                    f"Timestep: {dt_ * const.s_to_Myr} Myr, Age: \033[38;5;208m{t * const.s_to_Gyr} Gyr\033[0m"
                )

            # dt-collapse stall guard: every retry path halves dt_ and
            # re-enters this loop, so a single check here catches all stall
            # modes. Only after early evolution (t > 0.05 Gyr) — the run
            # legitimately starts at dt_ ~ 1e2 s.
            if (dt_abort_myr > 0.0 and t > 0.05 * const.Gyr_to_s
                    and dt_ < dt_abort_myr * const.Myr_to_s):
                logger.error(
                    "\033[91mTimestep %.3e Myr fell below dt_abort_myr=%.1e "
                    "at age %.4f Gyr — dt-collapse stall detected; ending the "
                    "run early (state saved through the last accepted step).\033[0m"
                    % (dt_ * const.s_to_Myr, dt_abort_myr, t * const.s_to_Gyr)
                )
                break

            # Rain-drain kill switch: Z_old[0] is the committed outer-cell
            # metal fraction (the same quantity saved as data_Z_atm). Z_old
            # is first bound at the end of step 0 / the restart load; the
            # t > 0.05 Gyr test short-circuits before it is referenced on
            # the first fresh-run pass (t = 0 there). has_envelope guards
            # bare-rock runs, where cell 0 is a mantle cell carrying the
            # Z = 0 sentinel and would trip the threshold spuriously.
            if (z_atm_abort_solar > 0.0 and has_envelope
                    and t > 0.05 * const.Gyr_to_s
                    and Z_old[0] < z_atm_abort_solar * const.Z_SOLAR_CHEN23):
                logger.error(
                    "\033[91mOuter-cell Z=%.3e (%.3f x solar) fell below "
                    "z_atm_abort_solar=%.2f at age %.4f Gyr — rain-drained "
                    "envelope detected; ending the run early (state saved "
                    "through the last accepted step).\033[0m"
                    % (Z_old[0], Z_old[0] / const.Z_SOLAR_CHEN23,
                       z_atm_abort_solar, t * const.s_to_Gyr)
                )
                break

            # ========== INITIALIZE (FIRST TIMESTEP) ========== #
            if i == 0 and not restart_flag:
                # Solve hydrostatic equilibrium for the first time step
                (
                    r_b, p, rho, temp, S, g, omega, j2n, shapes, niter_HSE
                ) = hydrostatic_equilibrium(
                    S,
                    np.log(r_b_old),
                    np.log(p_old),
                    m_b,
                    Y,
                    Z,
                    f_rock,
                    kcore,
                    kcore_fe,
                    0,
                    t,
                    logt_old=np.zeros_like(S),
                    init=True
                )

                assert_all_not_nan(r_b, p, rho, temp, S, g, omega)

                r = get_arith_mean(r_b)

                j2, j4, j6, j8, obl, eq_rad, pol_rad, MoI = j2n

                # Temperature at P=1e7 barye
                t10 = interp1d(p, temp, kind='linear', fill_value='extrapolate')(1e7)

                # Compute internal energy (avoid H/He EOS if no envelope exists).
                u = np.zeros(N)
                if has_envelope:
                    u[:kcore] = 10 ** mixtures_eos.get_logu_pt(
                        np.log10(p[:kcore]),
                        np.log10(temp[:kcore]),
                        Y[:kcore],
                        Z[:kcore],
                        f_rock[:kcore]
                    )
                if has_mantle:
                    u[kcore:kcore_fe] = mantle_eos.get_u_pt(
                        p[kcore:kcore_fe] * mantle_eos.dyn_to_GPa,
                        temp[kcore:kcore_fe]) #* mantle_eos.U_conv_cgs
                # Core (Fe) energy is always computed for the inner region.
                if kcore_fe < N:
                    u[kcore_fe:] = core_eos.get_u_pt(
                        p[kcore_fe:] * core_eos.dyn_to_GPa,
                        temp[kcore_fe:]
                    )

                internal_energy = np.sum(u * dm)
                grav_potential = np.sum(G * m * dm / r)
                # Rotational kinetic energy: E_rot = I omega^2 / 2 with the
                # SAME I_structure (j2n[-1]) that get_omega's angular-momentum
                # conservation uses -- the spin-up energy drawn from
                # contraction must appear in the ledger or it reads as a
                # spurious loss (~2% of E_rad over 4.5 Gyr for Jupiter).
                # Exactly 0.0 when rotation = False (omega = 0).
                rot_energy = (0.5 * float(j2n[-1]) * float(omega) ** 2
                              if rotation else 0.0)
                TdS_energy = 0
                chemical_potential = 0

                energy.append([t, internal_energy, grav_potential, energy_lost,
                               rot_energy])
                conser_energy.append([0, 0, 0])

                t_p_interp = interp1d(p, temp, kind='linear', fill_value='extrapolate')
                t1_bar = t_p_interp(1e6)
                t10_bar = t_p_interp(1e7)
                t1000_bar = t_p_interp(990000000)
                T_emb = _cell_value_or_last(temp, kcore)

                pbase1 = 10  # bars
                pbase2 = 30  # bars
                cond = ((p >= pbase1 * 1e6) & (p <= pbase2 * 1e6))

                # atmosphere boundary condition expect single values for S, Y, Z at the atmosphere
                # take mean over the range of pressures corresponding to the atmosphere
                S_atm = np.mean(S[cond]) if len(S[cond]) > 0 else S[0] 
                Y_atm = np.mean(Y[cond]) if not y_atm_fixed else Y_atm_bc
                Z_atm = np.mean(Z[cond]) if not z_atm_fixed else Z_atm_bc

                T_int, T_eff = _atm_boundary_tint_teff(
                    p, temp, Z, T_emb, enforce_g75_planet=True)

                assert_all_not_nan(T_int, T_eff, temp, p, r_b, rho, S, Y, Z, f_rock)

                # ---- Clean initialization report ---- #
                _hhe_eos = config['equation_of_state'].get('hhe_eos', 'cd')
                _z_eos = config['equation_of_state'].get('z_eos', 'aqua')
                _mantle_comp = config['core'].get('mantle_comp', 'mg2sio4')
                _eos_core = config['core'].get('eos_core', 'D17_comb')
                _core_comp = config['core'].get('core_comp', 'Fe_pure')
                _mass_core_Me = float(config['core'].get('mass_core', 0.0))
                _mass_core_fe_Me = float(config['core'].get('mass_core_fe', 0.0))

                # Total heavy-element mass: envelope Z + mantle + iron core
                _Z_env_mass = np.sum(Z * dm) / const.mearth
                _total_Z_mass = _Z_env_mass + _mass_core_Me

                # Proto-solar metallicity at surface
                _zsurf = np.clip(Z[0], 0.0, 1.0 - 1e-12)
                _Z_solar = z_to_metallicity_factor(_zsurf, const.Z_SOLAR_CHEN23)

                # Radius in appropriate units
                _use_earth_radii = planet in ('Sub_Neptune', 'Super_Earth', 'Uranus', 'Neptune')
                if _use_earth_radii:
                    _R_display = r_b[0] / const.rearth
                    _R_unit = 'R_Earth'
                else:
                    _R_display = r_b[0] / const.rjup
                    _R_unit = 'R_Jup'

                # Total energy budget
                _E_total = internal_energy - grav_potential

                # BC description
                _bc_names = {
                    'c23': 'Chen+2023', 'c26': 'Chen+2026', 'f11': 'Fortney+2011',
                    'f07': 'Fortney+2007', 'b97': 'Burrows+1997', 'g75': 'Graboske+1975',
                    'm21': 'Marley+2021', 'p20': 'Phillips+2020',
                    'bare': 'Bare (no atmosphere)', 'gray': 'Semi-gray (rock)',
                    'ideal_gray': 'Ideal gray',
                }
                _bc_desc = _bc_names.get(BC_atm, BC_atm)

                logger.info("=" * 70)
                logger.info("ORCHARD — Initial Model Summary")
                logger.info("=" * 70)
                logger.info(f"  Planet type           : {planet}")
                logger.info(f"  Total mass            : {M_planet / const.mearth:.3f} M_Earth "
                            f"({M_planet / const.M_jup:.5f} M_Jup)")
                logger.info(f"  Mantle + core mass    : {_mass_core_Me:.3f} M_Earth "
                            f"(iron core: {_mass_core_fe_Me:.3f} M_Earth)")
                logger.info(f"  Total Z mass          : {_total_Z_mass:.3f} M_Earth "
                            f"(envelope: {_Z_env_mass:.3f}, mantle+core: {_mass_core_Me:.3f})")
                logger.info(f"  Surface metallicity   : {_Z_solar:.3f} x solar (Chen+2023)")
                logger.info(f"  Initial radius        : {_R_display:.3f} {_R_unit}")
                logger.info(f"  T_eff / T_int         : {T_eff:.2f} / {T_int:.2f} K")
                logger.info(f"  Energy budget (U-E_g) : {_E_total:.4e} erg")
                logger.info("-" * 70)
                logger.info(f"  H-He EOS              : {_hhe_eos}")
                logger.info(f"  Z EOS                 : {_z_eos}")
                if _mass_core_Me > 0:
                    logger.info(f"  Mantle EOS            : {_mantle_comp}")
                    logger.info(f"  Iron core EOS         : {_core_comp} ({_eos_core})")
                logger.info(f"  Boundary condition    : {_bc_desc}")
                logger.info("=" * 70)

                L_int0 = 4 * np.pi * r[0]**2 * const.sigma_sb * T_int**4
                L_int0_bsurf = 4 * np.pi * r_b[0]**2 * const.sigma_sb * T_int**4
                lum_arr.append(L_int0)
                lum_hse_bsurf_arr.append(L_int0_bsurf)
                # `loss_energy` from transport is actually a luminosity [erg/s]. There is no
                # transport solve on the initial state, so use the same initial luminosity.
                loss_lum_arr.append(L_int0)
                loss_lum_pc_arr.append(L_int0)
                if energy_conservation_debug:
                    lum_compare_rows.append([t, L_int0, L_int0, 0.0, 0.0])
                    lum_hse_radius_compare_rows.append([t, L_int0, L_int0_bsurf, L_int0_bsurf - L_int0, 0.0])
                    lum_transport_pc_compare_rows.append([t, L_int0_bsurf, L_int0, L_int0, 0.0, 0.0])
                    tint_compare_rows.append([t, T_int, T_int, 0.0, 0.0])

                phases_old = np.ones(len(p[kcore:kcore_fe]), dtype=np.uint8)  # 1=True


                chi_prev_full = np.zeros(N)
                chi_prev_full[:] = -1.0

                # Initialize zero arrays for various gradient and flux placeholders
                grad_i = np.zeros(N - 1)
                grad_ldx_i = np.zeros(N - 1)
                cp_i = np.zeros(N - 1)
                chit_chirho_i = np.zeros(N - 1)
                N_i_SY = np.zeros(N - 1)
                N_i_Prho = np.zeros(N - 1)
                N_i_PT = np.zeros(N - 1)
                grad_a_i = np.zeros(N - 1)
                B_Y_i = np.zeros(N - 1)
                B_Z_i = np.zeros(N - 1)
                dsdy_prho_i = np.zeros(N - 1)
                dsdz_prho_i = np.zeros(N - 1)
                dsdy_pt_i = np.zeros(N - 1)
                dsdz_pt_i = np.zeros(N - 1)
                dsdr_i = np.zeros(N - 1)
                dydr_i = np.zeros(N - 1)
                dzdr_i = np.zeros(N - 1)
                R0_inv_b = np.zeros(N + 1)
                conv_flux = np.zeros(N + 1)
                semi_flux = np.zeros(N + 1)
                rad_flux = np.zeros(N + 1)
                radiogenic_flux = np.zeros(N + 1)
                Lambda_cond_i = np.zeros(N - 1)
                Lambda_rad_i = np.zeros(N - 1)
                bracket_dsdr_b = np.zeros(N + 1)

                # Save initial data (initial state; no conservation row yet)
                _record_saved_step(
                    r_now=r, r_b_now=r_b, p_now=p, rho_now=rho, temp_now=temp,
                    S_now=S, Y_now=Y, Z_now=Z, f_rock_now=f_rock,
                    omega_now=omega,
                    mantle_rad=_boundary_value_or_nan(r_b, kcore, exists=has_core_boundary),
                    core_rad=_boundary_value_or_nan(r_b, kcore_fe, exists=has_fe_boundary),
                    Ym_i=Ym_old_i, Z1m_i=Z1m_old_i, Z2m_i=Z2m_old_i,
                    int_lum=L_int0, melt_chi=chi_prev_full,
                    conserv_row=None,
                    cp_row=cp_i if save_verbose else None,
                    dsdy_prho_row=dsdy_prho_i if save_verbose else None,
                    dsdy_pt_row=dsdy_pt_i if save_verbose else None,
                    dsdz_prho_row=dsdz_prho_i if save_verbose else None,
                    dsdz_pt_row=dsdz_pt_i if save_verbose else None,
                    D_row=np.zeros(N + 1) if save_verbose else None,
                    phases_row=phases_old if save_verbose else None,
                )

                if save_hdf5 and (h5w is not None):
                    snap = _build_snapshot(
                        r_now=r, r_b_now=r_b, p_now=p, rho_now=rho,
                        temp_now=temp, S_now=S, Y_now=Y, Z_now=Z,
                        f_rock_now=f_rock, omega_now=omega,
                        mantle_rad=_boundary_value_or_nan(r_b, kcore, exists=has_core_boundary),
                        core_rad=_boundary_value_or_nan(r_b, kcore_fe, exists=has_fe_boundary),
                        Ym_i=Ym_old_i, Z1m_i=Z1m_old_i, Z2m_i=Z2m_old_i,
                        int_lum=L_int0, melt_chi=chi_prev_full,
                        conserv_row=[t, 0.0, 0.0],
                        Dcoeff_row=np.zeros(N + 1) if save_verbose else None,
                        cp_row=cp_i if save_verbose else None,
                        dsdy_prho_row=dsdy_prho_i if save_verbose else None,
                        dsdy_pt_row=dsdy_pt_i if save_verbose else None,
                        dsdz_prho_row=dsdz_prho_i if save_verbose else None,
                        dsdz_pt_row=dsdz_pt_i if save_verbose else None,
                    )
                    h5w.append(snap)

                # Schedule the next save checkpoint.
                _esi = _effective_save_interval(0.0)
                if _esi > 0:
                    save_time = _esi

                # Active plotting: generate frame for initial state
                if _active_plot:
                    _active_frame_num += 1
                    _age_gyr = t * const.s_to_Gyr
                    _frame_name = f"{_active_frame_num:04d}_t{_age_gyr:0.3f}Gyr"
                    _save_path = os.path.join(_active_movie_outdir, _frame_name + '.png')
                    try:
                        _plotter = _init_plot(paths=[args.config], plot_rock_phases=_plot_rock_phases)
                        # First frame has the hottest state; fix y-axis for all frames
                        if _active_temp_ylim_kK is None:
                            _active_temp_ylim_kK = float(np.nanmax(np.asarray(_plotter.temp[0]))) * 1e-3
                        _plotter.still_plot(
                            age=_age_gyr,
                            output_file=_frame_name,
                            labels=[model_fldr],
                            save=True,
                            save_path=_save_path,
                            movie_mode=True,
                            temp_ylim_kK=_active_temp_ylim_kK,
                        )
                        del _plotter
                    except Exception as _e:
                        logger.warning('Active plot frame %d failed: %s' % (_active_frame_num, _e))

                i += 1

                # Prepare for next iteration
                log_p_old = np.log(data_p[-1])
                p_old = data_p[-1]
                S_old = data_S[-1]
                temp_old = data_temp[-1]
                Y_old = data_Y[-1]
                Z_old = data_Z[-1]
                f_rock_old = f_rock
                rho_old = data_rho[-1]
                p1_old = 1.e12
                p2_old = 1.05e12
                r_b_old = r_b
                t += dt_
                omega_old = omega

            # ========== SUBSEQUENT TIMESTEPS ========== #
            else:
                T_int_transport_in = T_int

                # 2) Update composition and entropy
                SYargs = (
                    temp_old,
                    r_b_old,
                    p_old,
                    rho_old,
                    m_b,
                    dm,
                    T_int,
                    dt_,
                    t,
                    Y_old,
                    Z_old,
                    f_rock_old,
                    S_old,
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
                    omega_old,
                    dTdt_prev,
                )
                try:
                    # TransportResult fields -> the loop's working names.
                    # (latent_flux, phi, J, B are diagnostics the evolution
                    #  loop does not consume; see TransportResult docstring.)
                    _tr = update_SYZ(*SYargs)
                    S_new = _tr.S
                    Y_new = _tr.Y
                    Z_new = _tr.Z
                    f_rock_new = _tr.f_rock
                    loss_energy = _tr.loss_energy
                    D2 = _tr.D2
                    D_b_new = _tr.D_b
                    DZ_b_new = _tr.DZ_b
                    flux = _tr.total_flux
                    conv_flux = _tr.conv_flux
                    rad_flux = _tr.rad_flux
                    radiogenic_flux = _tr.radiogenic_flux
                    semi_flux = _tr.semi_flux
                    Lambda = _tr.Lambda
                    niter = _tr.niter
                    R0_inv_b = _tr.R0_inv_b
                    Lambda_cond_i = _tr.Lambda_cond_i
                    Lambda_rad_i = _tr.Lambda_rad_i
                    bracket_dsdr_b = _tr.bracket_dsdr_b
                    Ym_new_i = _tr.Ym_i
                    Z1m_new_i = _tr.Z1m_i
                    Z2m_new_i = _tr.Z2m_i
                    innermisc_index = _tr.innermisc_index
                    cp_new_i = _tr.Cp_i
                    chi_curr_full = _tr.chi_full
                    y_rain_gap_new = _tr.y_rain_gap
                    Nu_T_eff_new = _tr.Nu_T_eff_i
                    Nu_X_eff_new = _tr.Nu_X_eff_i
                except RuntimeError as _nr_err:
                    # Newton solver failed — halve the timestep and retry.
                    dt_ = dt_ / 2.0
                    t -= dt_
                    logger.info(
                        "\033[91mTransport NR failed: %s. "
                        "Halving timestep to %.3e Myr\033[0m",
                        str(_nr_err)[:120],
                        dt_ * const.s_to_Myr,
                    )
                    retry += 1
                    if dt_ < 0.001:
                        raise
                    continue

                # Carry rain-gap bounds forward to the next timestep.
                y_rain_gap_prev = y_rain_gap_old
                y_rain_gap_old = y_rain_gap_new

                # 3) Solve hydrostatic equilibrium
                (
                    r_b_new,
                    p_new,
                    rho_new,
                    temp_new,
                    S_new,
                    g,
                    omega,
                    j2n,
                    shapes,
                    niter_HSE

                ) = hydrostatic_equilibrium(
                    S_new,
                    np.log(r_b_old),
                    np.log(p_old),
                    m_b,
                    Y_new,
                    Z_new,
                    f_rock_new,
                    kcore,
                    kcore_fe,
                    data_omega[-1],
                    t,
                    logt_old=np.log10(temp_old),
                    init=False
                )

                state_issues = _get_hse_state_issues(r_b_new, p_new, rho_new, temp_new, S_new)
                if state_issues:
                    if verbose:
                        logger.info("\033[95mRejected trial HSE state: %s\033[0m" % "; ".join(state_issues))
                    dt_ = dt_ / 2.0
                    # `t` stores the target age for the current trial step (end-of-step age).
                    # After halving a rejected step, retarget the end age accordingly.
                    t -= dt_
                    logger.info(
                            "\033[91mRejecting [HSE] (%d) timestep, new timestep is = %.3e Myr\033[0m"
                            % (retry, dt_ * const.s_to_Myr)
                                )
                    retry += 1
                    if dt_ < 0.0001:
                        raise Exception("time step killed")
                    continue

                # ----------------------------------------------------------
                # Chi-based timestep limiter:
                # If the melt fraction changed by more than max_delta_chi in
                # any mantle cell, reject the timestep and retry with dt/2.
                # This prevents violent viscosity/kappa swings near the melt
                # front in bare-rock models that cool quickly.
                # With the MAX_LOG_KAPPA_RATIO cap on the inviscid/viscous
                # dynamic range (transport.py), the kappa sensitivity per
                # unit chi is bounded, so max_delta_chi can be relaxed
                # (set via config [transport] max_delta_chi).
                # ----------------------------------------------------------
                chi_prev_arr = np.asarray(chi_prev_full, dtype=float)
                chi_curr_arr = np.asarray(chi_curr_full, dtype=float)
                if (
                    chi_prev_arr.shape == chi_curr_arr.shape
                    and np.all(chi_prev_arr >= 0.0)
                    and np.all(np.isfinite(chi_prev_arr))
                    and np.all(np.isfinite(chi_curr_arr))
                ):
                    abs_dchi = np.max(np.abs(chi_curr_arr - chi_prev_arr))
                    if abs_dchi > max_delta_chi:
                        # Plateau detection: if |dchi| is unchanged (within 1%)
                        # across consecutive halvings, the jump is dt-independent
                        # and further halving is provably useless -- accept at the
                        # CURRENT dt rather than grinding down 40 more halvings to
                        # the collapse threshold.  Recovering models show |dchi|
                        # decreasing with dt and never plateau (verified across
                        # the example suite); the retry >= 6 gate adds margin.
                        if (_chi_last_absdchi is not None
                                and abs(abs_dchi - _chi_last_absdchi) < 0.01 * abs_dchi):
                            _chi_plateau_n += 1
                        else:
                            _chi_plateau_n = 0
                        _chi_last_absdchi = abs_dchi
                        if dt_ / 2.0 < 0.0001 or (_chi_plateau_n >= 2 and retry >= 6):
                            # dt has collapsed: |dchi| is dt-independent, i.e. a
                            # genuinely bistable melt transition no timestep can
                            # resolve.  Accept it once with a warning instead of
                            # killing the run -- the latent heat of the jump is
                            # already accounted for in the transport residual
                            # via the dchi/dt term.
                            logger.warning(
                                "Chi limiter: max|dchi|=%.4f > %.4f is "
                                "dt-independent (plateau %d, retry %d, dt=%.3e "
                                "Myr); accepting discontinuous melt transition "
                                "at zone %d."
                                % (abs_dchi, max_delta_chi, _chi_plateau_n,
                                   retry, dt_ * const.s_to_Myr,
                                   int(np.argmax(np.abs(chi_curr_arr - chi_prev_arr))))
                            )
                            _chi_last_absdchi = None
                            _chi_plateau_n = 0
                        else:
                            dt_ = dt_ / 2.0
                            t -= dt_
                            logger.info(
                                "\033[93mChi limiter: max|Δχ|=%.4f > %.4f, "
                                "halving timestep to %.3e Myr\033[0m"
                                % (abs_dchi, max_delta_chi, dt_ * const.s_to_Myr)
                            )
                            retry += 1
                            continue

                # Step accepted past the chi limiter: clear plateau trackers so
                # halvings from different steps are never compared.
                _chi_last_absdchi = None
                _chi_plateau_n = 0

                j2, j4, j6, j8, obl, eq_rad, pol_rad, MoI = j2n # unpack j2n tuple

                # 4) Compute Brunt-Väisälä and other diagnostics
                r_new_i = r_b_new[1:-1]
                r_new = get_arith_mean(r_b_new)
                S_new_i = get_arith_mean(S_new)
                rho_new_i = get_geom_mean(rho_new)
                p_new_i = get_geom_mean(p_new)
                Y_new_i = get_arith_mean(Y_new)
                Z_new_i = get_arith_mean(Z_new)
                f_rock_new_i = get_arith_mean(f_rock_new)
                temp_new_i = get_geom_mean(temp_new)
                g_i = G * m_b[1:-1] / r_new_i**2

                phases_new = np.ones(len(p[kcore:kcore_fe]), dtype=np.uint8)  # 1=True

                # Prepare arrays for derivative evaluations
                drhodS_PY_i = np.zeros_like(Y_new_i)
                dSdY_prho_i = np.zeros_like(Y_new_i)
                dSdZ_prho_i = np.zeros_like(Y_new_i)
                dSdY_pt_i = np.zeros_like(Y_new_i)
                dSdZ_pt_i = np.zeros_like(Y_new_i)

                ChiY_ChiT_i = np.zeros_like(Y_new_i)
                ChiZ_ChiT_i = np.zeros_like(Y_new_i)
                ChiT_Chirho_i = np.zeros_like(Y_new_i)

                # Envelope-only thermodynamic derivatives live on interior boundaries.
                # For bare-rock runs (kcore=0) or tiny envelopes (kcore<=1), skip.
                env_int = kcore - 1
                has_env_int = env_int > 0
                if has_env_int:
                    drhodS_PY_i[:env_int] = mixtures_eos.get_dlogrho_ds_py(
                        S_new_i[:env_int],
                        np.log10(p_new_i[:env_int]),
                        Y_new_i[:env_int],
                        Z_new_i[:env_int],
                        f_rock_new_i[:env_int],
                        tab=True
                    )
                    ChiT_Chirho_i[:env_int] = mixtures_eos.get_dlogrho_dlogt_py(
                        np.log10(p_new_i[:env_int]),
                        np.log10(temp_new_i[:env_int]),
                        Y_new_i[:env_int],
                        Z_new_i[:env_int],
                        f_rock_new_i[:env_int]
                    )
                    ChiY_ChiT_i[:env_int] = mixtures_eos.get_dlogt_dy_rhop_rhot(
                        np.log10(rho_new_i[:env_int]),
                        np.log10(temp_new_i[:env_int]),
                        Y_new_i[:env_int],
                        Z_new_i[:env_int],
                        f_rock_new_i[:env_int]
                    )
                    ChiZ_ChiT_i[:env_int] = mixtures_eos.get_dlogt_dz_rhop_rhot(
                        np.log10(rho_new_i[:env_int]),
                        np.log10(temp_new_i[:env_int]),
                        Y_new_i[:env_int],
                        Z_new_i[:env_int],
                        f_rock_new_i[:env_int]
                    )

                    dz_i = mixtures_eos.adaptive_dx(Z_new_i[:env_int])
                    dy_i = mixtures_eos.adaptive_dx(Y_new_i[:env_int])

                    dSdY_pt_i[:env_int] = mixtures_eos.get_dsdy_pt(
                        np.log10(p_new_i[:env_int]),
                        np.log10(temp_new_i[:env_int]),
                        Y_new_i[:env_int],
                        Z_new_i[:env_int],
                        f_rock_new_i[:env_int],
                        dy=dy_i
                    )
                    dSdZ_pt_i[:env_int] = mixtures_eos.get_dsdz_pt(
                        np.log10(p_new_i[:env_int]),
                        np.log10(temp_new_i[:env_int]),
                        Y_new_i[:env_int],
                        Z_new_i[:env_int],
                        f_rock_new_i[:env_int],
                        dz=dz_i
                    )
                    dSdY_prho_i[:env_int] = mixtures_eos.get_dsdy_rhop_srho(
                        S_new_i[:env_int],
                        np.log10(rho_new_i[:env_int]),
                        Y_new_i[:env_int],
                        Z_new_i[:env_int],
                        f_rock_new_i[:env_int],
                        ideal_guess=False,
                        dy=dy_i,
                        ds=0.1,
                        arr_guess=np.log10(temp_new_i[:env_int]),
                        tab=True
                    )
                    dSdZ_prho_i[:env_int] = mixtures_eos.get_dsdz_rhop_srho(
                        S_new_i[:env_int],
                        np.log10(rho_new_i[:env_int]),
                        Y_new_i[:env_int],
                        Z_new_i[:env_int],
                        f_rock_new_i[:env_int],
                        ideal_guess=False,
                        dz=dz_i,
                        ds=0.1,
                        arr_guess=np.log10(temp_new_i[:env_int]),
                        tab=True
                    )

                dsdr_i = np.diff(S_new) / np.diff(r_new) * kbbar_to_erg
                dydr_i = np.diff(Y_new) / np.diff(r_new)
                dzdr_i = np.diff(Z_new) / np.diff(r_new)

                grad_ldx_i = (
                    -dsdr_i
                    + dSdY_prho_i * dydr_i
                    + dSdZ_prho_i * dzdr_i
                )

                grad_a_i = np.zeros_like(Y_new_i)
                grad_i = np.zeros_like(Y_new_i)
                grad_Y_i = np.zeros_like(Y_new_i)
                grad_Z_i = np.zeros_like(Y_new_i)
                B_Y_i = np.zeros_like(Y_new_i)
                B_Z_i = np.zeros_like(Y_new_i)

                # Adiabatic and composition gradients only exist in the envelope.
                if has_env_int:
                    grad_a_i[:env_int] = mixtures_eos.get_nabla_ad(
                        S_new_i[:env_int],
                        np.log10(p_new_i[:env_int]),
                        Y_new_i[:env_int],
                        Z_new_i[:env_int],
                        f_rock_new_i[:env_int]
                    )

                    grad_i[:env_int] = np.diff(np.log(temp_new[:kcore])) / np.diff(np.log(p_new[:kcore]))
                    grad_Y_i[:env_int] = np.diff(Y_new[:kcore]) / np.diff(np.log(p_new[:kcore]))
                    grad_Z_i[:env_int] = np.diff(Z_new[:kcore]) / np.diff(np.log(p_new[:kcore]))
                    B_Y_i[:env_int] = -ChiY_ChiT_i[:env_int] * grad_Y_i[:env_int]
                    B_Z_i[:env_int] = -ChiZ_ChiT_i[:env_int] * grad_Z_i[:env_int]

                # Brunt-Väisälä frequencies
                #
                # N_sq_i_SY: uses only drhodS_PY_i, which we set only in the H/He envelope,
                # so it is naturally ~0 in the mantle/core for bare-rock or core regions.
                N_sq_i_SY = g_i * drhodS_PY_i * grad_ldx_i
                #
                # N_sq_i_Prho and N_sq_i_PT are written in terms of the *mixtures* EOS
                # (gamma1, Chi derivatives, etc). Those quantities are undefined in the
                # mantle/core (no Y/Z profiles there; mantle is handled by mantle_eos).
                # Therefore we explicitly compute them ONLY on the envelope interior
                # boundaries (indices 0..kcore-2 == [:kcore-1]).
                N_sq_i_Prho = np.zeros_like(Y_new_i)
                N_sq_i_PT = np.zeros_like(Y_new_i)
                if has_env_int:
                    diff_log_rho = np.diff(np.log(rho_new))
                    diff_log_p = np.diff(np.log(p_new))
                    diff_log_p = np.where(np.abs(diff_log_p) < 1e-30, 1e-30, diff_log_p)

                    gamma1_env = mixtures_eos.get_gamma1(
                        S_new_i[:env_int],
                        np.log10(p_new_i[:env_int]),
                        Y_new_i[:env_int],
                        Z_new_i[:env_int],
                        f_rock_new_i[:env_int],
                        dp=0.1,
                        tab=True,
                        ideal_guess=False,
                        arr_guess=np.log10(p_new_i[:env_int]),
                    )

                    N_sq_i_Prho[:env_int] = (
                        g_i[:env_int] ** 2
                        * (
                            diff_log_rho[:env_int] / diff_log_p[:env_int]
                            - (1.0 / gamma1_env) * rho_new_i[:env_int] / p_new_i[:env_int]
                        )
                    )
                    N_sq_i_PT[:env_int] = (
                        g_i[:env_int] ** 2
                        * rho_new_i[:env_int] / p_new_i[:env_int]
                        * ChiT_Chirho_i[:env_int]
                        * (grad_a_i[:env_int] - grad_i[:env_int] - B_Y_i[:env_int] - B_Z_i[:env_int])
                    )

                N_i_SY = np.sqrt(np.maximum(0, N_sq_i_SY))
                N_i_Prho = np.sqrt(np.maximum(0, N_sq_i_Prho))
                N_i_PT = np.sqrt(np.maximum(0, N_sq_i_PT))

                # 5) Error-based Timestep

                # For bare-rock / low-kappa gray models the outermost
                # mantle zones cool radiatively so fast that they dominate
                # the adaptive error and force impractically small timesteps.
                # Exclude zones below p_err_floor from the error metric.
                P_ERR_FLOOR_CGS = p_err_floor_GPa * 1e10  # GPa → dyn/cm^2
                err_mask = p_new >= P_ERR_FLOOR_CGS
                rel_S_err = np.abs((S_new - S_old) / (np.abs(S_old) + ZERO))
                if np.any(err_mask):
                    error_S = np.max(rel_S_err[err_mask])
                else:
                    error_S = np.max(rel_S_err)
                # Y/Z are only meaningful in the envelope. In the mantle/core we set
                # them to zero by construction, so dividing by Y_old or Z_old across
                # the full array will generate inf/NaN for bare-rock and core zones.
                if has_envelope:
                    env_slice = slice(0, kcore)
                    # Floor the relative-error denominators at a dynamically
                    # meaningful composition (1e-4): in fully rained-out cells
                    # (Z ~ ZERO = 1e-20) solver-level jitter would otherwise
                    # register as percent-level relative changes independent of
                    # dt, collapsing the timestep. Absolute composition changes
                    # below ~1e-4 are structurally irrelevant.
                    COMP_ERR_FLOOR = 1e-4
                    rel_Y_env = (np.abs(Y_new[env_slice] - Y_old[env_slice])
                                 / np.maximum(np.abs(Y_old[env_slice]), COMP_ERR_FLOOR))
                    rel_Z_env = (np.abs(Z_new[env_slice] - Z_old[env_slice])
                                 / np.maximum(np.abs(Z_old[env_slice]), COMP_ERR_FLOOR))
                    error_Y = float(np.max(rel_Y_env))
                    error_Z = float(np.max(rel_Z_env))
                else:
                    error_Y = 0.0
                    error_Z = 0.0
                error = max(error_S, error_Y, error_Z)
                error_temp = 0.0
                error_rho = 0.0
                rel_temp_err = None
                rel_rho_err = None

                if i > 0:
                    rel_temp_err = np.abs((temp_new - temp_old) / (temp_old + ZERO))
                    rel_rho_err = np.abs((rho_new - rho_old) / (rho_old + ZERO))
                    _trho_mask = err_mask.copy() if np.any(err_mask) \
                        else np.ones(rel_temp_err.shape[0], dtype=bool)
                    if np.any(_trho_mask):
                        error_temp = np.max(rel_temp_err[_trho_mask])
                        error_rho = np.max(rel_rho_err[_trho_mask])
                    else:
                        error_temp = 0.0
                        error_rho = 0.0
                    error = max(error, error_temp, error_rho)

                bad_error_names = [
                    name for (name, val) in (
                        ("S", error_S),
                        ("Y", error_Y),
                        ("Z", error_Z),
                        ("rho", error_rho),
                        ("T", error_temp),
                    )
                    if not np.isfinite(val)
                ]
                if bad_error_names:
                    logger.warning(
                        "Non-finite timestep error component(s): %s",
                        ", ".join(bad_error_names),
                    )
                    error = np.inf

                conv_cells = np.where(abs(conv_flux) >= 1)[0]
                innerstabidx = -1 if len(conv_cells) == 0 else conv_cells[-1] + 1
                # Guard sparse/non-convective edge cases to avoid IndexError.
                nonconv_cells = np.where(abs(conv_flux) < 1)[0]
                outerconvidx = nonconv_cells[1] if nonconv_cells.size > 1 else -1

                if verbose:
                    logger.info(
                        "\tErrs (S, Y, Z, rho, T) kcore, innerstab, outerconv: (%.4e, %.4e, %.4e, %.4e, %.4e) %d, %d, %d",
                        error_S,
                        error_Y,
                        error_Z,
                        error_rho,
                        error_temp,
                        kcore,
                        innerstabidx,
                        outerconvidx,
                    )
                    if y_rain_gap_new is not None:
                        logger.info(
                            "\tHe rain gap (P): [%.6e, %.6e] Mbar",
                            y_rain_gap_new[0] * 1e-12,
                            y_rain_gap_new[1] * 1e-12,
                        )

                # Early-time error smoothing: at t=0 the tolerance is ~5x stricter
                # than the user-specified 'error', then relaxes quadratically toward
                # the user value by t ~ 0.1 Gyr. This prevents the model from taking
                # excessively large early steps when the initial structure is far from
                # thermal equilibrium and small changes are expected.
                err_smooth = (5 - err) * max(1 - (t / (1e-1 * const.Gyr_to_s))**2, 0) + err

                # ---------------------------------------------------------
                # Rain-aware error relaxation (bottom_up model)
                #
                # When helium/Z rain is active, composition changes at the
                # rain boundaries propagate into T and rho through the EOS.
                # These changes are expected and physical, but cells that
                # sit marginally at the miscibility activation can flicker
                # between attempts, producing a dt-independent error that
                # forces the adaptive controller to take impractically
                # small steps (or stall outright).  The rain_error_factor
                # parameter (>= 1, default 1 = no relaxation) multiplies
                # err_smooth whenever a rain gap is open, allowing larger
                # relative changes per step.  This applies to both rain
                # models; with strongly displaced miscibility curves (e.g.
                # the hhe Z-rain stand-in) the legacy model needs it too,
                # because the curve sits far below attainable composition
                # and supersaturation does not self-regulate.
                # ---------------------------------------------------------
                if y_rain_gap_new is not None:
                    err_smooth *= rain_error_factor

                # If error is too big, reduce step size.
                if error > err_smooth / 100:
                    if verbose and i > 0 and rel_rho_err is not None and rel_temp_err is not None:
                        n_top = min(5, rel_rho_err.size)
                        if n_top > 0:
                            top_rho_idx = np.argpartition(rel_rho_err, -n_top)[-n_top:]
                            top_rho_idx = top_rho_idx[np.argsort(rel_rho_err[top_rho_idx])[::-1]]
                            for idx in top_rho_idx:
                                logger.info(
                                    "\tTop rho_err zone idx=%d: rel=%.4e, p=%.4e GPa, rho_old=%.4e, rho_new=%.4e, T_old=%.4e, T_new=%.4e",
                                    int(idx),
                                    rel_rho_err[idx],
                                    p_new[idx] * 1e-10,
                                    rho_old[idx],
                                    rho_new[idx],
                                    temp_old[idx],
                                    temp_new[idx],
                                )

                        n_top_t = min(3, rel_temp_err.size)
                        if n_top_t > 0:
                            top_temp_idx = np.argpartition(rel_temp_err, -n_top_t)[-n_top_t:]
                            top_temp_idx = top_temp_idx[np.argsort(rel_temp_err[top_temp_idx])[::-1]]
                            for idx in top_temp_idx:
                                logger.info(
                                    "\tTop T_err zone idx=%d: rel=%.4e, p=%.4e GPa, T_old=%.4e, T_new=%.4e, rho_old=%.4e, rho_new=%.4e",
                                    int(idx),
                                    rel_temp_err[idx],
                                    p_new[idx] * 1e-10,
                                    temp_old[idx],
                                    temp_new[idx],
                                    rho_old[idx],
                                    rho_new[idx],
                                )

                        # S/Y/Z argmax cells: these components often dominate the
                        # rejection (error = max over S, Y, Z, T, rho) yet were
                        # invisible in the T/rho listings above.
                        if np.any(err_mask):
                            s_idx = int(np.flatnonzero(err_mask)[np.argmax(rel_S_err[err_mask])])
                        else:
                            s_idx = int(np.argmax(rel_S_err))
                        logger.info(
                            "\tTop S_err zone idx=%d: rel=%.4e, S_old=%.4e, S_new=%.4e",
                            s_idx, rel_S_err[s_idx], S_old[s_idx], S_new[s_idx],
                        )
                        if has_envelope:
                            y_idx = int(np.argmax(rel_Y_env))
                            z_idx = int(np.argmax(rel_Z_env))
                            logger.info(
                                "\tTop Y_err zone idx=%d: rel=%.4e, Y_old=%.6f, Y_new=%.6f | "
                                "Top Z_err zone idx=%d: rel=%.4e, Z_old=%.6f, Z_new=%.6f",
                                y_idx, rel_Y_env[y_idx], Y_old[y_idx], Y_new[y_idx],
                                z_idx, rel_Z_env[z_idx], Z_old[z_idx], Z_new[z_idx],
                            )

                    dt_ = dt_ / 2.0
                    # `t` is the current trial end age. Retarget after halving the step.
                    t -= dt_
                    if verbose:
                        logger.info(
                                "\033[91mRejecting [ERR] (%d) timestep (err=%.3e), new timestep is = %.3e Myr\033[0m"
                                % (retry, error, dt_ * const.s_to_Myr)
                            )
                    retry += 1

                    if dt_ < 0.001:
                        raise Exception("time step killed")

                else:
                    had_retry = retry > 0
                    retry = 0
                    old_dt = dt_

                    # Timestep growth control:
                    # - growth_cap: max allowed growth factor per accepted step.
                    #   Reduced to 1.25x after a retry to prevent ping-pong oscillations
                    #   where dt doubles then immediately gets rejected again.
                    # - err_mult: inversely proportional to error (smaller error -> bigger step)
                    # - itr_mult: penalizes high Newton iteration counts (high_precision mode)
                    growth_cap = 1.25 if had_retry else 2.0
                    err_mult = min(err_smooth / 100.0 / max(error, ZERO), growth_cap)
                    if high_precision:
                        itr_mult = (max_itr - niter - 1) / max_itr + 0.6
                    else:
                        itr_mult = 1
                    eff_max_step = _effective_max_step(t)
                    next_dt = min(old_dt * err_mult * itr_mult, eff_max_step)

                    # ---- Adaptive dt suppression when rain zone changes rapidly ----
                    # If the He rain gap size changed by >20%, cap dt to prevent
                    # composition jumps. If rain just appeared, halve the step.
                    # If the NR solver used many iterations, don't grow dt.
                    if y_rain_gap_new is not None and y_rain_gap_prev is not None:
                        gap_old = abs(y_rain_gap_prev[1] - y_rain_gap_prev[0])
                        gap_new = abs(y_rain_gap_new[1] - y_rain_gap_new[0])
                        if gap_old > 0 and abs(gap_new - gap_old) / gap_old > 0.2:
                            next_dt = min(next_dt, old_dt)
                    elif y_rain_gap_new is not None and y_rain_gap_prev is None:
                        next_dt = min(next_dt, old_dt * 0.5)
                    if niter > max_itr * 0.5:
                        next_dt = min(next_dt, old_dt)

                    # ---- Rain-aware CFL cap for aggressive alpha_herain ----
                    #
                    # The helium rain advection velocity is v = D2 / (alpha_herain * Hp).
                    # When alpha_herain < 1, this velocity is 1/alpha_herain times
                    # larger than the baseline, making the advection timescale
                    # (how quickly a cell drains its excess He) proportionally shorter.
                    #
                    # If dt exceeds this advection timescale, the implicit Newton
                    # solver receives a residual dominated by the advection term.
                    # The Newton correction then tries to drain Y by more than the
                    # cell holds, triggering safe_newton_step guard failures (Y < 0),
                    # which cascade into repeated backtracking, NR restarts, and
                    # ultimately severe timestep collapse (dt -> 1e-7 Myr).
                    #
                    # The fix is simple: cap dt so the advection Courant number
                    # stays ~O(1).  Since the baseline runs stably at the effective
                    # max_step (which respects early_max_step / late_max_step
                    # tighteners at the current age), scaling by alpha_herain keeps
                    # the same effective Courant number.  For alpha_herain = 0.1
                    # and an effective cap of 50 Myr, this means dt_max = 5 Myr —
                    # 10x more steps than the no-rain phase, but each converges
                    # cleanly in ~5-10 NR iterations instead of hitting the
                    # 150-iteration ceiling or crashing.  If a late-time tightener
                    # has already pulled the effective cap to 10 Myr, the rain cap
                    # tracks down to 1 Myr automatically.
                    #
                    # The cap only activates when rain is present (y_rain_gap_new
                    # is not None) and alpha_herain < 1.  At alpha_herain >= 1
                    # (the default), this block is skipped entirely, preserving
                    # bit-for-bit backward compatibility.
                    if y_rain_gap_new is not None and alpha_herain < 1.0:
                        next_dt = min(next_dt, alpha_herain * eff_max_step)

                    state_issues = _get_hse_state_issues(r_b_new, p_new, rho_new, temp_new, S_new)
                    if state_issues:
                        logger.info("\033[95mRejected pre-BC trial state: %s\033[0m" % "; ".join(state_issues))
                        dt_ = old_dt / 2.0
                        # `t` still points at the rejected trial's end age (= start + old_dt).
                        # Shift it to the new halved-step target end age (= start + dt_).
                        t -= dt_
                        logger.info("Rejecting [pre-BC] (%d) timestep, new timestep is = %.3e Myr" % (retry, dt_ * const.s_to_Myr))
                        retry += 1
                        if dt_ < 1:
                            raise Exception("time step killed")
                        continue

                    # 6) Set boundary condition from atmosphere
                    pbase1 = 10  # bars
                    pbase2 = 30  # bars
                    cond = ((p_new >= pbase1 * 1e6) & (p_new <= pbase2 * 1e6))

                    # YS: for very hot ICs, pressure at base of envelope never
                    # reaches 10 bars...
                    if convection and len(S_new[cond]) > 0:
                        S_atm = np.mean(S_new[cond])
                        Y_atm = np.mean(Y_new[cond]) if not y_atm_fixed else Y_atm_bc # fixed Y or varied Y for atm bc
                        Z_atm = np.mean(Z_new[cond]) if not z_atm_fixed else Z_atm_bc
                    else:
                        S_atm = S_new[0]
                        Y_atm = Y_new[0] if not y_atm_fixed else Y_atm_bc
                        Z_atm = Z_new[0] if not z_atm_fixed else Z_atm_bc

                    t_p_interp = interp1d(p_new, temp_new, kind='linear', fill_value='extrapolate')

                    t1_bar = t_p_interp(1e6)
                    t10_bar = t_p_interp(1e7)
                    t1000_bar = t_p_interp(990000000)

                    # (bare/gray/ideal_gray use the just-updated surface
                    #  temperature as the envelope-mantle boundary value.)
                    # A failed S->Tint inversion means the accepted trial
                    # state's surface entropy left the atmosphere table's
                    # invertible range (seen in violent late-time surface
                    # oscillations of heavily eroded small-core models) --
                    # an invalid trial state, not a fatal error: reject the
                    # step and retry at dt/2, exactly like the pre-BC state
                    # check above.  A truly diverged state collapses dt into
                    # the dt_abort guard, which ends the run with state saved.
                    try:
                        T_int, T_eff = _atm_boundary_tint_teff(
                            p_new, temp_new, Z_new, temp_new[0])
                    except atm_bc.TintInversionError as tint_err:
                        logger.info("\033[95mRejected BC trial state: %s\033[0m" % tint_err)
                        dt_ = old_dt / 2.0
                        # `t` still points at the rejected trial's end age
                        # (= start + old_dt); shift to the halved-step target.
                        t -= dt_
                        logger.info("Rejecting [BC] (%d) timestep, new timestep is = %.3e Myr" % (retry, dt_ * const.s_to_Myr))
                        retry += 1
                        if dt_ < 1:
                            raise Exception("time step killed")
                        continue

                    assert_all_not_nan(T_int, T_eff, p_new, temp_new, S_new, Y_new, Z_new)

                    # 7) Update energies for energy conservation step
                    #lum_arr.append(loss_energy)
                    L_int = 4.0 * np.pi * r_new[0]**2 * const.sigma_sb * T_int**4
                    L_int_hse_bsurf = 4.0 * np.pi * r_b_new[0]**2 * const.sigma_sb * T_int**4
                    # `loss_energy` from update_SYZ is a luminosity [erg/s] computed in the
                    # transport solve before the subsequent HSE/BC update.
                    loss_lum = loss_energy
                    # Cheap split predictor-corrector for the transport-side luminosity:
                    # predictor = transport-returned surface luminosity (pre-HSE),
                    # corrector = post-HSE luminosity using the boundary radius.
                    # Diagnostic-only; the official ledger below uses the raw
                    # transport luminosity as a backward-Euler rectangle.
                    loss_lum_pc = 0.5 * (loss_lum + L_int_hse_bsurf)

                    lum_arr.append(L_int)
                    lum_hse_bsurf_arr.append(L_int_hse_bsurf)
                    loss_lum_arr.append(loss_lum)
                    loss_lum_pc_arr.append(loss_lum_pc)
                    if energy_conservation_debug:
                        lum_diff = L_int - loss_lum
                        lum_rel = lum_diff / max(abs(L_int), abs(loss_lum), ZERO)
                        lum_compare_rows.append([t, L_int, loss_lum, lum_diff, lum_rel])
                        lum_hse_radius_diff = L_int_hse_bsurf - L_int
                        lum_hse_radius_rel = lum_hse_radius_diff / max(abs(L_int), abs(L_int_hse_bsurf), ZERO)
                        lum_hse_radius_compare_rows.append(
                            [t, L_int, L_int_hse_bsurf, lum_hse_radius_diff, lum_hse_radius_rel]
                        )
                        lum_transport_raw_rel_b = (loss_lum - L_int_hse_bsurf) / max(
                            abs(loss_lum), abs(L_int_hse_bsurf), ZERO
                        )
                        lum_transport_pc_rel = (loss_lum_pc - L_int_hse_bsurf) / max(
                            abs(loss_lum_pc), abs(L_int_hse_bsurf), ZERO
                        )
                        lum_transport_pc_compare_rows.append(
                            [t, L_int_hse_bsurf, loss_lum, loss_lum_pc, lum_transport_raw_rel_b, lum_transport_pc_rel]
                        )
                        tint_diff = T_int - T_int_transport_in
                        tint_rel = tint_diff / max(abs(T_int), abs(T_int_transport_in), ZERO)
                        tint_compare_rows.append([t, T_int_transport_in, T_int, tint_diff, tint_rel])

                    # Compute internal energy and gravitational potential energy.
                    # Important: mixtures_eos is only valid for the H/He envelope region.
                    u_new = np.zeros_like(p_new)
                    if has_envelope:
                        u_new[:kcore] = 10**mixtures_eos.get_logu_pt(
                            np.log10(p_new[:kcore]),
                            np.log10(temp_new[:kcore]),
                            Y_new[:kcore],
                            Z_new[:kcore],
                            f_rock_new[:kcore]
                        )
                    if has_mantle:
                        u_new[kcore:kcore_fe] = mantle_eos.get_u_pt(
                        p_new[kcore:kcore_fe] * mantle_eos.dyn_to_GPa,
                        temp_new[kcore:kcore_fe]) #* mantle_eos.U_conv_cgs
                    if kcore_fe < N:
                        u_new[kcore_fe:] = core_eos.get_u_pt(
                            p_new[kcore_fe:] * core_eos.dyn_to_GPa,
                            temp_new[kcore_fe:]
                        )

                    internal_energy = np.sum(u_new * dm) # internal energy of EOS
                    grav_potential = np.sum(G * m * dm / r_new) # gravitational potential energy
                    # Rotational kinetic energy (see the t=0 ledger seed):
                    # same I_structure as get_omega; exactly 0.0 when
                    # rotation = False.
                    rot_energy = (0.5 * float(j2n[-1]) * float(omega) ** 2
                                  if rotation else 0.0)

                    temp_mid = 0.5 * (temp_new + temp_old)

                    # If S is boundary-centered (len N+1), convert to cell-centered (len N)
                    if S_new.shape[0] == dm.shape[0] + 1:
                        S_old_cell = get_arith_mean(S_old)
                        S_new_cell = get_arith_mean(S_new)
                    else:
                        S_old_cell = S_old
                        S_new_cell = S_new

                    TdS_step = np.sum(temp_mid * (S_new_cell - S_old_cell) * dm / (const.erg_to_kbbar))

                    #TdS_energy += TdS_step

                    E_rad_step_hse = 0.5 * (lum_arr[-2] + lum_arr[-1]) * old_dt
                    if energy_conservation_debug and len(lum_arr) >= 2:
                        energy_lost_hse += E_rad_step_hse
                    E_rad_step_hse_bsurf = 0.5 * (lum_hse_bsurf_arr[-2] + lum_hse_bsurf_arr[-1]) * old_dt
                    if energy_conservation_debug and len(lum_hse_bsurf_arr) >= 2:
                        energy_lost_hse_bsurf += E_rad_step_hse_bsurf
                    E_rad_step_transport = 0.5 * (loss_lum_arr[-2] + loss_lum_arr[-1]) * old_dt
                    if energy_conservation_debug and len(loss_lum_arr) >= 2:
                        energy_lost_transport += E_rad_step_transport
                    E_rad_step_transport_pc = 0.5 * (loss_lum_pc_arr[-2] + loss_lum_pc_arr[-1]) * old_dt
                    if energy_conservation_debug and len(loss_lum_pc_arr) >= 2:
                        energy_lost_transport_pc += E_rad_step_transport_pc
                    E_rad_step_mid = 0.5 * (E_rad_step_hse + E_rad_step_transport)
                    if len(lum_arr) >= 2:
                        energy_lost_midpoint += E_rad_step_mid
                    # OFFICIAL bookkeeping (2026-07 accounting fix): meter exactly the
                    # energy the implicit transport step removed through the surface —
                    # the solver applies the surface flux `loss_lum` over the whole
                    # accepted step (backward-Euler), so the consistent quadrature is
                    # the rectangle loss_lum * dt, NOT a trapezoid of averaged
                    # channels. The old midpoint-of-(HSE, transport) ledger mixed a
                    # cell-center-radius, post-BC luminosity into the budget, i.e.
                    # it metered something other than the heat drained from the
                    # entropy field (measured bias O(1e-4) * E_rad on the N=300
                    # Jupiter baselines, model-dependent in sign). The historical
                    # channels above remain as debug diagnostics under
                    # energy_conservation_debug; the former
                    # energy_conservation_use_transport_pc switch is removed.
                    E_rad_step_official = loss_lum_arr[-1] * old_dt
                    energy_lost += E_rad_step_official

                    S_mid = 0.5 * (S_new_cell + S_old_cell)
                    rho_mid = 0.5 * (rho_new + rho_old)
                    Y_mid = 0.5 * (Y_new + Y_old)
                    Z_mid = 0.5 * (Z_new + Z_old)

                    dUdY = np.zeros_like(S_old_cell)
                    dUdZ = np.zeros_like(S_old_cell)
                    # Chemical potential terms are only defined in the H/He envelope.
                    # The mantle/core have no Y/Z profiles (mantle is effectively all Z),
                    # so mixtures_eos should never be called there.
                    if has_envelope:
                        dz_ = mixtures_eos.adaptive_dx(Z_mid[:kcore])
                        dy_ = mixtures_eos.adaptive_dx(Y_mid[:kcore])
                        dUdY[:kcore] = mixtures_eos.get_dudy_srho(
                            S_mid[:kcore],
                            np.log10(rho_mid[:kcore]),
                            Y_mid[:kcore],
                            Z_mid[:kcore],
                            f_rock_new[:kcore],
                            dy=dy_,
                            tab=True
                        )
                        dUdZ[:kcore] = mixtures_eos.get_dudz_srho(
                            S_mid[:kcore],
                            np.log10(rho_mid[:kcore]),
                            Y_mid[:kcore],
                            Z_mid[:kcore],
                            f_rock_new[:kcore],
                            dz=dz_,
                            tab=True
                        )

                    # Energy conservation calculation
                    chemical_potential = np.sum(
                        (dUdY * (Y_new - Y_old) + dUdZ * (Z_new - Z_old)) * dm
                    )
                    TdS_energy += TdS_step + chemical_potential #+ 0.5 * (lum_old + lum_new) * (t - data_age[-1])
                    dE_S = (TdS_energy + energy_lost) / energy_lost # new internal heat added to lost heat should be conserved
                    energy.append([t, internal_energy, grav_potential, energy_lost,
                                   rot_energy])

                    _, u0, eg0, lum0, erot0 = energy[0] # initial internal, potential, loss, rotational
                    _, uf, egf, lumf, erotf = energy[-1] # current internal, potential, loss, rotational
                    e0 = u0 - eg0 + lum0 + erot0 # initial total energy (potential energy is negative)
                    # Official bookkeeping: residual of (U + E_rot - E_g +
                    # E_radiated) vs t=0, normalized by e0 and by the radiated
                    # energy (see _residual_triplet).  E_rot is folded into the
                    # internal-energy argument; it is identically 0 for
                    # rotation = False, reproducing the legacy residual.
                    uf_rot = uf + erotf
                    deltaE, deltaE_loss, energy_residual = _residual_triplet(uf_rot, egf, e0, lumf)
                    if energy_conservation_debug:
                        # Same residual, evaluated per alternative loss accumulator.
                        (deltaE_midpoint, deltaE_midpoint_loss,
                         energy_residual_midpoint) = _residual_triplet(uf_rot, egf, e0, energy_lost_midpoint)
                        deltaE_hse, deltaE_hse_loss, _ = _residual_triplet(uf_rot, egf, e0, energy_lost_hse)
                        (deltaE_transport, deltaE_transport_loss,
                         _) = _residual_triplet(uf_rot, egf, e0, energy_lost_transport)
                        (deltaE_transport_pc, deltaE_transport_pc_loss,
                         _) = _residual_triplet(uf_rot, egf, e0, energy_lost_transport_pc)
                        (deltaE_hse_bsurf, deltaE_hse_bsurf_loss,
                         _) = _residual_triplet(uf_rot, egf, e0, energy_lost_hse_bsurf)
                        energy_diag_rows.append([
                            t,
                            deltaE,
                            deltaE_loss,
                            deltaE_midpoint,
                            deltaE_midpoint_loss,
                            deltaE_hse,
                            deltaE_hse_loss,
                            deltaE_hse_bsurf,
                            deltaE_hse_bsurf_loss,
                            deltaE_transport,
                            deltaE_transport_loss,
                            deltaE_transport_pc,
                            deltaE_transport_pc_loss,
                        ])

                    T_emb = _cell_value_or_last(temp_new, kcore)
                    T_cmb = _cell_value_or_nan(temp_new, kcore_fe, exists=has_fe_boundary)

                    mu_atm = mean_molecular_weight_HHe_plus_water(Z_new[0])

                    R_transit = get_rtransit_isothermal_guillot2010(
                        r_new[0], g, T_eff, mu_atm)

                    # Inner-envelope metallicity is undefined when kcore=0; log NaN instead.
                    if has_envelope:
                        Z_inner = Z_new[kcore - 1]
                        # Keep metallicity diagnostics on the same Chen+23 reference.
                        z_inner_for_log = np.clip(Z_inner, 0.0, 1.0 - 1e-12)
                        Z_in_solar = z_to_metallicity_factor(z_inner_for_log, const.Z_SOLAR_CHEN23)
                    else:
                        Z_in_solar = float('nan')
                    z_out_for_log = np.clip(Z_new[0], 0.0, 1.0 - 1e-12)
                    Z_out_solar = z_to_metallicity_factor(z_out_for_log, const.Z_SOLAR_CHEN23)

                    if verbose:
                        logger.info(
                            ('\tTransit Radius [{unit}]: {:.3f}, Mean Radius [{unit}]: {:.3f}, Core Radius [{unit}]: {:.3f}, '
                            'T_eff: {:.3f}, Tint: {:.3f}, T_EMB: {:.3f}, T_CMB: {:.3f}, '
                            'Z_in (solar): {:.3f}, Z_out (solar): {:.3f}, Y_out: {:.3f}, S_out: {:.3f}, rho_avg: {:.3f}, '
                            'niter: \033[94m{}\033[0m, niter_HSE: \033[94m{}\033[0m, dE_S: {:.3e}, dE_G: \033[92m{:.3e}\033[0m'
                            .format(
                                R_transit / _r_scale,
                                r_new[0] / _r_scale,
                                _cell_value_or_nan(r_new, kcore, exists=has_core_boundary) / _r_scale,
                                T_eff, T_int, T_emb, T_cmb,
                                Z_in_solar,
                                Z_out_solar,
                                Y_new[0],
                                S_new[0],
                                m_b[0] / ((4/3) * np.pi * r_new[0] ** 3),
                                int(niter),
                                int(niter_HSE),
                                dE_S,
                                deltaE,
                                unit=_r_unit_label,
                            ))
                        )
                        if energy_conservation_debug:
                            logger.info(
                                '\tEnergy residual metrics (official dE_G bookkeeping = transport backward-Euler rectangle): dE/e0=%+.5e, dE/E_lost=%+.5e, dE_abs=%+.5e erg, E_lost=%+.5e erg',
                                deltaE,
                                deltaE_loss,
                                energy_residual,
                                lumf,
                            )
                            logger.info(
                                '\tEnergy residual metrics (midpoint luminosity): dE/e0=%+.5e, dE/E_lost=%+.5e, dE_abs=%+.5e erg, E_lost=%+.5e erg',
                                deltaE_midpoint,
                                deltaE_midpoint_loss,
                                energy_residual_midpoint,
                                energy_lost_midpoint,
                            )
                            logger.info(
                                '\tTransport/HSE luminosity compare: L_transport=%+.5e, L_HSE=%+.5e, frac_diff=%+.5e; dE_HSE/e0=%+.5e, dE_transport/e0=%+.5e',
                                loss_lum,
                                L_int,
                                lum_rel,
                                deltaE_hse,
                                deltaE_transport,
                            )
                            logger.info(
                                '\tTransport-PC/HSE(boundary) compare: L_transport_pc=%+.5e, L_HSE(boundary)=%+.5e, frac_diff=%+.5e; dE_transport_pc/e0=%+.5e',
                                loss_lum_pc,
                                L_int_hse_bsurf,
                                lum_transport_pc_rel,
                                deltaE_transport_pc,
                            )
                            logger.info(
                                '\tHSE radius-choice compare: L_HSE(center)=%+.5e, L_HSE(boundary)=%+.5e, frac_diff=%+.5e; dE_HSE_boundary/e0=%+.5e',
                                L_int,
                                L_int_hse_bsurf,
                                lum_hse_radius_rel,
                                deltaE_hse_bsurf,
                            )
                            logger.info(
                                '\tTint compare: T_int_transport=%+.5e K, T_int_postHSE=%+.5e K, frac_diff=%+.5e',
                                T_int_transport_in,
                                T_int,
                                tint_rel,
                            )
                    else:
                        logger.info(
                            '\tRadius [{unit}]: {:.3f}, T_eff [K]: {:.3f}, T_EMB [K]: {:.3f}, '
                            'Luminosity [Lsun]: {:.3e}, Y_out: {:.3f}, Z_out [solar]: {:.3f}, P_center [GPa]: {:.3f}'
                            .format(
                                r_new[0] / _r_scale,
                                T_eff,
                                T_emb,
                                L_int / const.lsun,
                                Y_new[0],
                                Z_out_solar,
                                p_new[-1] * 1e-10,
                                unit=_r_unit_label,
                            )
                        )

                    dt_ = next_dt

                    # Repair unphysical envelope temperature inversions on the
                    # accepted state (before it is rolled and saved). These are
                    # Schwarzschild-unstable layers injected by under-converged
                    # transport NR updates during rain eras; convection would
                    # erase them physically, but the composition gating keeps
                    # them "stable" numerically. Minimal monotone (isotonic)
                    # adjustment; S and rho recomputed per repaired cell.
                    if repair_T_inv and kcore > 2:
                        from utils.thermal_repair import (
                            repair_temperature_inversions)
                        repair_temperature_inversions(
                            S_new, temp_new, rho_new, p_new, Y_new, Z_new,
                            f_rock_new, dm, kcore, mixtures_eos, eos_tab_flag,
                        )

                    # 8) Prepare for next iteration
                    # Accepted-step temperature rate for the opt-in
                    # time_centered_heat_factor mode (must be computed BEFORE
                    # temp_old is overwritten below).
                    dTdt_prev = (temp_new - temp_old) / old_dt
                    log_p_old = np.log(p_new)
                    p_old = p_new
                    S_old = S_new
                    temp_old = temp_new
                    Y_old = Y_new
                    Z_old = Z_new
                    f_rock_old = f_rock_new
                    rho_old = rho_new
                    D_b_old = D_b_new
                    DZ_b_old = DZ_b_new

                    # --- Spatial smoothing of D_b_old / DZ_b_old ---
                    # Applied BETWEEN timesteps (not inside the NR loop)
                    # so the Jacobian stays consistent during iteration.
                    # Log-space running geometric mean suppresses grid-
                    # scale D oscillations from bracket flickering without
                    # affecting NR convergence.
                    if smooth_D_spatial > 0 and has_envelope and kcore > 2 * smooth_D_spatial + 1:
                        hw = smooth_D_spatial
                        _lo, _hi = 1, min(kcore, N)
                        for _D_arr in (D_b_old, DZ_b_old):
                            _log = np.log(np.maximum(_D_arr[_lo:_hi], 1e-99))
                            _sm = np.convolve(
                                _log,
                                np.ones(2 * hw + 1) / (2 * hw + 1),
                                mode='same',
                            )
                            _D_arr[_lo:_hi] = np.exp(_sm)

                    Nu_T_eff_old = Nu_T_eff_new
                    Nu_X_eff_old = Nu_X_eff_new
                    r_b_old = r_b_new
                    Ym_old_i = Ym_new_i
                    Z1m_old_i = Z1m_new_i
                    Z2m_old_i = Z2m_new_i
                    omega_old = omega

                    chi_prev_full = chi_curr_full

                    # --- Adaptive mesh regridding ---
                    if regrid_enabled:
                        regrid_step_counter += 1
                        if (regrid_step_counter >= regrid_every
                                and t > t_regrid_start
                                and has_envelope
                                and y_rain_gap_old is not None):
                            regrid_step_counter = 0
                            M_core_regrid = m_b[kcore] if kcore < N else 0.0
                            M_core_fe_regrid = m_b[kcore_fe] if kcore_fe < N else 0.0
                            # Save backup of all state arrays before regridding
                            _bak = (m_b.copy(), dm.copy(), S_old.copy(), Y_old.copy(),
                                    Z_old.copy(), f_rock_old.copy(),
                                    r_b_old.copy(), p_old.copy(), temp_old.copy(), rho_old.copy(),
                                    D_b_old.copy(), DZ_b_old.copy(),
                                    Ym_old_i.copy(), Z1m_old_i.copy(), Z2m_old_i.copy(),
                                    chi_prev_full.copy(), Nu_T_eff_old.copy(), Nu_X_eff_old.copy(),
                                    int(kcore), int(kcore_fe), log_p_old.copy())
                            regrid_ok = False
                            try:
                                (m_b, dm, S_old, Y_old, Z_old, f_rock_old,
                                 r_b_old, p_old, temp_old, rho_old,
                                 D_b_old, DZ_b_old,
                                 Ym_old_i, Z1m_old_i, Z2m_old_i,
                                 chi_prev_full, Nu_T_eff_old, Nu_X_eff_old,
                                 kcore, kcore_fe,
                                ) = regrid_profiles(
                                    _bak[0], _bak[1], _bak[2], _bak[3], _bak[4], _bak[5],
                                    _bak[6], _bak[7], _bak[8], _bak[9],
                                    _bak[10], _bak[11],
                                    _bak[12], _bak[13], _bak[14],
                                    _bak[15],
                                    _bak[16], _bak[17],
                                    _bak[18], _bak[19],
                                    M_planet, M_core_regrid, M_core_fe_regrid,
                                    bracket_dsdr_b, y_rain_gap_old,
                                    config,
                                )
                                # Restore HSE consistency on new grid
                                (
                                    r_b_old, p_old,
                                    rho_old, temp_old,
                                    S_old,
                                    _g_regrid, _omega_regrid,
                                    _j2n_regrid, _shapes_regrid,
                                    _niter_hse_regrid,
                                ) = hydrostatic_equilibrium(
                                    S_old,
                                    np.log(r_b_old),
                                    np.log(p_old),
                                    m_b,
                                    Y_old,
                                    Z_old,
                                    f_rock_old,
                                    kcore,
                                    kcore_fe,
                                    data_omega[-1],
                                    t,
                                    logt_old=np.log10(temp_old),
                                    init=True,
                                )
                                # Verify no NaN
                                if (np.any(~np.isfinite(temp_old)) or
                                        np.any(~np.isfinite(p_old)) or
                                        np.any(~np.isfinite(r_b_old))):
                                    raise RuntimeError("HSE produced NaN after regrid")
                                regrid_ok = True
                                log_p_old = np.log(p_old)
                                # Accounting fix (2026-07): the cell-centered
                                # enclosed-mass array feeds grav_potential; it must
                                # track the regridded m_b or the energy budget mixes
                                # stale masses with the new grid.
                                m = get_arith_mean(m_b)
                                # The lagged dT/dt history lives on the old grid;
                                # drop it so the next transport step uses the
                                # legacy T weighting once, then rebuilds.
                                dTdt_prev = None
                                has_envelope = kcore > 0
                                has_mantle = kcore_fe > kcore
                                logger.info("Regrid HSE cleanup: %d iterations",
                                            _niter_hse_regrid)
                            except Exception as e:
                                logger.warning("Regrid failed (%s), reverting.", e)
                            if not regrid_ok:
                                # Restore backup
                                (m_b, dm, S_old, Y_old, Z_old, f_rock_old,
                                 r_b_old, p_old, temp_old, rho_old,
                                 D_b_old, DZ_b_old,
                                 Ym_old_i, Z1m_old_i, Z2m_old_i,
                                 chi_prev_full, Nu_T_eff_old, Nu_X_eff_old,
                                 kcore, kcore_fe, log_p_old) = _bak

                    adiabatic_exponent = np.zeros_like(S_old)
                    grad_a = np.zeros_like(S_old)
                    # Adiabatic indices are only defined for the H/He envelope;
                    # the mantle/core do not use mixtures_eos or Y/Z profiles.
                    if has_envelope:
                        adiabatic_exponent[:kcore] = mixtures_eos.get_gamma1(
                            S_old[:kcore],
                            np.log10(p_old[:kcore]),
                            Y_old[:kcore],
                            Z_old[:kcore],
                            f_rock_old[:kcore],
                        )
                        grad_a[:kcore] = mixtures_eos.get_nabla_ad(
                            S_new[:kcore],
                            np.log10(p_new[:kcore]),
                            Y_new[:kcore],
                            Z_new[:kcore],
                            f_rock_new[:kcore]
                        )

                    # 9) Save data if needed
                    if (save_interval == 0 and early_save_interval is None) or t >= save_time:
                        # Record the newly accepted state (see _record_saved_step)
                        _record_saved_step(
                            r_now=r_new, r_b_now=r_b_new, p_now=p_new,
                            rho_now=rho_new, temp_now=temp_new, S_now=S_new,
                            Y_now=Y_new, Z_now=Z_new, f_rock_now=f_rock_new,
                            omega_now=omega_old,
                            mantle_rad=_cell_value_or_nan(r_new, kcore, exists=has_core_boundary),
                            core_rad=_cell_value_or_nan(r_new, kcore_fe, exists=has_fe_boundary),
                            Ym_i=Ym_new_i, Z1m_i=Z1m_new_i, Z2m_i=Z2m_new_i,
                            int_lum=L_int, melt_chi=chi_curr_full,
                            conserv_row=[t, deltaE, dE_S],
                            cp_row=cp_new_i if save_verbose else None,
                            dsdy_prho_row=dSdY_prho_i if save_verbose else None,
                            dsdy_pt_row=dSdY_pt_i if save_verbose else None,
                            dsdz_prho_row=dSdZ_prho_i if save_verbose else None,
                            dsdz_pt_row=dSdZ_pt_i if save_verbose else None,
                            D_row=D_b_new if save_verbose else None,
                            phases_row=phases_new if save_verbose else None,
                        )

                        if save_hdf5 and (h5w is not None):
                            snap = _build_snapshot(
                                r_now=r_new, r_b_now=r_b_new, p_now=p_new,
                                rho_now=rho_new, temp_now=temp_new, S_now=S_new,
                                Y_now=Y_new, Z_now=Z_new, f_rock_now=f_rock_new,
                                omega_now=omega_old,
                                mantle_rad=_cell_value_or_nan(r_new, kcore, exists=has_core_boundary),
                                core_rad=_cell_value_or_nan(r_new, kcore_fe, exists=has_fe_boundary),
                                Ym_i=Ym_new_i, Z1m_i=Z1m_new_i, Z2m_i=Z2m_new_i,
                                int_lum=L_int, melt_chi=chi_curr_full,
                                conserv_row=[t, deltaE, dE_S],
                                Dcoeff_row=D_b_new if save_verbose else None,
                                cp_row=cp_new_i if save_verbose else None,
                                dsdy_prho_row=dSdY_prho_i if save_verbose else None,
                                dsdy_pt_row=dSdY_pt_i if save_verbose else None,
                                dsdz_prho_row=dSdZ_prho_i if save_verbose else None,
                                dsdz_pt_row=dSdZ_pt_i if save_verbose else None,
                            )
                            h5w.append(snap)

                        if not save_hdf5:
                            # Save out to files occasionally
                            if j == 0:
                                np.savetxt(os.path.join(folder_name, 'mgrid.txt'), m_b, fmt='%.8e')

                            if j % 1 == 0:
                                _write_text_profiles()
                        j += 1

                        # Advance the save checkpoint so it always refers to the next
                        # save time strictly after the current model age.
                        _esi = _effective_save_interval(t)
                        if _esi > 0:
                            while save_time <= t:
                                save_time += _effective_save_interval(save_time)

                        # Active plotting: generate frame every max_step interval
                        if _active_plot and t >= _active_next_plot_time:
                            _active_frame_num += 1
                            _age_gyr = t * const.s_to_Gyr
                            _frame_name = f"{_active_frame_num:04d}_t{_age_gyr:0.3f}Gyr"
                            _save_path = os.path.join(_active_movie_outdir, _frame_name + '.png')
                            try:
                                _plotter = _init_plot(paths=[args.config], plot_rock_phases=_plot_rock_phases)
                                _plotter.still_plot(
                                    age=_age_gyr,
                                    output_file=_frame_name,
                                    labels=[model_fldr],
                                    save=True,
                                    save_path=_save_path,
                                    movie_mode=True,
                                    temp_ylim_kK=_active_temp_ylim_kK,
                                )
                                del _plotter
                            except Exception as _e:
                                logger.warning('Active plot frame %d failed: %s' % (_active_frame_num, _e))
                            # Advance checkpoint so we don't double-plot
                            while _active_next_plot_time <= t:
                                _active_next_plot_time += max_step

                    # Adjust next dt to avoid overshooting the next save checkpoint.
                    _esi = _effective_save_interval(t)
                    if _esi > 0:
                        while save_time <= t:
                            save_time += _effective_save_interval(save_time)
                        if t + dt_ >= save_time:
                            dt_ = save_time - t
                    t += dt_
                    i += 1

            now = time.time()

    # =============== END OF EVOLUTION LOOP & EXCEPTION HANDLING =============== #
    except:
        # Re-raise any exceptions so they are caught externally/logged
        raise

    finally:
        # -------------- Final Data Save & Clean Up --------------- #
        if h5w is not None:
            try:
                h5w.close()
            except Exception:
                pass

        shutil.copy(args.config, os.path.join(args.data_folder, config_basename))
        try:
            label = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode()
            logger.info(f"Code SHA: {label}")
        except:
            pass

        if save_data and (not save_hdf5):
            _write_text_profiles()

        # Keep a compact TOF summary table available for downstream plotting
        # regardless of whether the run was saved as text or HDF5.
        if save_data and rotation and len(J2_list) > 0:
            np.savetxt(
                os.path.join(folder_name, 'Js_tof.txt'),
                np.c_[J2_list, J4_list, J6_list, J8_list, f_list, I_list, I2_list, Req_list, Rpol_list, Rvol_list],
                fmt='%.8e')

        # Log final messages (E_rot folded as in the per-step residual;
        # identically 0 for rotation = False)
        _, u0, eg0, lum0, erot0 = energy[0]
        _, uf, egf, lumf, erotf = energy[-1]
        e0 = u0 - eg0 + lum0 + erot0
        uf_rot = uf + erotf
        deltaE, deltaE_loss, energy_residual = _residual_triplet(uf_rot, egf, e0, lumf)
        if energy_conservation_debug:
            # Same residual, evaluated per alternative loss accumulator.
            (deltaE_midpoint, deltaE_midpoint_loss,
             energy_residual_midpoint) = _residual_triplet(uf_rot, egf, e0, energy_lost_midpoint)
            (deltaE_hse, deltaE_hse_loss,
             energy_residual_hse) = _residual_triplet(uf_rot, egf, e0, energy_lost_hse)
            (deltaE_transport, deltaE_transport_loss,
             energy_residual_transport) = _residual_triplet(uf_rot, egf, e0, energy_lost_transport)
            (deltaE_transport_pc, deltaE_transport_pc_loss,
             energy_residual_transport_pc) = _residual_triplet(uf_rot, egf, e0, energy_lost_transport_pc)
            (deltaE_hse_bsurf, deltaE_hse_bsurf_loss,
             energy_residual_hse_bsurf) = _residual_triplet(uf_rot, egf, e0, energy_lost_hse_bsurf)

        _elapsed = now - program_starts
        _h, _rem = divmod(_elapsed, 3600)
        _m, _s = divmod(_rem, 60)
        _age_final = (t - dt_) * const.s_to_Myr / 1e3
        _use_earth_radii_final = planet in ('Sub_Neptune', 'Super_Earth', 'Uranus', 'Neptune')
        if _use_earth_radii_final:
            _R_final = r_b_old[0] / const.rearth
            _R_unit_final = 'R_Earth'
        else:
            _R_final = r_b_old[0] / const.rjup
            _R_unit_final = 'R_Jup'
        _Z_atm_solar_final = z_to_metallicity_factor(np.clip(Z_old[0], 0.0, 1.0 - 1e-12), const.Z_SOLAR_CHEN23)
        logger.info("ORCHARD — Final Model Summary")
        logger.info("=" * 70)
        logger.info(f"  Planet type           : {planet}")
        logger.info(f"  Total mass            : {M_planet / const.mearth:.3f} M_Earth "
                    f"({M_planet / const.M_jup:.5f} M_Jup)")
        logger.info(f"  Final age             : {_age_final:.3f} Gyr")
        logger.info(f"  Final radius          : {_R_final:.3f} {_R_unit_final}")
        logger.info(f"  T_eff / T_int         : {T_eff:.2f} / {T_int:.2f} K")
        logger.info(f"  Luminosity            : {L_int / const.lsun:.3e} L_sun")
        logger.info(f"  Y atm / Z atm         : {Y_old[0]:.3f} / {_Z_atm_solar_final:.3f} x solar (Chen+2023)")
        logger.info(f"  Central pressure      : {p_old[-1] * 1e-10:.3f} GPa")
        logger.info(f"  Central density       : {rho_old[-1]:.3f} g/cm^3")
        if verbose:
            # Two normalizations of the same absolute residual (see
            # _residual_triplet): dE/E_rad divides by the energy actually
            # radiated over the run and is comparable across planets;
            # dE/e0 divides by the EOS-zero-point-dependent initial budget
            # e0 = u0 - eg0 + lum0, so its scale (and even its usefulness)
            # varies with composition. Prefer dE/E_rad when quoting a number.
            logger.info(f"  Energy nonconservation: {deltaE_loss:.3e} (dE/E_rad)")
            logger.info(f"                          {deltaE:.3e} (dE/e0, zero-point dependent)")
        logger.info("-" * 70)
        logger.info(f"  Code time             : {int(_h):02d}:{int(_m):02d}:{_s:04.1f}")
        logger.info("=" * 70)
        if energy_conservation_debug:
            logger.info('Energy residual diagnostics (official dE_G bookkeeping = transport backward-Euler rectangle): dE/e0=' + str(np.round(deltaE, 6))
                        + ', dE/E_lost=' + str(np.round(deltaE_loss, 6))
                        + ', dE_abs=' + f'{energy_residual:.6e}' + ' erg')
            logger.info('Energy residual diagnostics (midpoint luminosity): dE/e0=' + str(np.round(deltaE_midpoint, 6))
                        + ', dE/E_lost=' + str(np.round(deltaE_midpoint_loss, 6))
                        + ', dE_abs=' + f'{energy_residual_midpoint:.6e}' + ' erg')
            logger.info('Energy residual diagnostics (HSE luminosity): dE/e0=' + str(np.round(deltaE_hse, 6))
                        + ', dE/E_lost=' + str(np.round(deltaE_hse_loss, 6))
                        + ', dE_abs=' + f'{energy_residual_hse:.6e}' + ' erg')
            logger.info('Energy residual diagnostics (HSE luminosity, boundary radius): dE/e0=' + str(np.round(deltaE_hse_bsurf, 6))
                        + ', dE/E_lost=' + str(np.round(deltaE_hse_bsurf_loss, 6))
                        + ', dE_abs=' + f'{energy_residual_hse_bsurf:.6e}' + ' erg')
            logger.info('Energy residual diagnostics (transport luminosity): dE/e0=' + str(np.round(deltaE_transport, 6))
                        + ', dE/E_lost=' + str(np.round(deltaE_transport_loss, 6))
                        + ', dE_abs=' + f'{energy_residual_transport:.6e}' + ' erg')
            logger.info('Energy residual diagnostics (transport luminosity, predictor-corrected): dE/e0=' + str(np.round(deltaE_transport_pc, 6))
                        + ', dE/E_lost=' + str(np.round(deltaE_transport_pc_loss, 6))
                        + ', dE_abs=' + f'{energy_residual_transport_pc:.6e}' + ' erg')
            logger.info('Integrated E_lost comparison: official=' + f'{lumf:.6e}'
                        + ' erg, HSE=' + f'{energy_lost_hse:.6e}'
                        + ' erg, HSE_boundary=' + f'{energy_lost_hse_bsurf:.6e}'
                        + ' erg, transport=' + f'{energy_lost_transport:.6e}'
                        + ' erg, transport_pc=' + f'{energy_lost_transport_pc:.6e}'
                        + ' erg, midpoint_acc=' + f'{energy_lost_midpoint:.6e}'
                        + ' erg, frac(HSE-transport)/max='
                        + f'{((energy_lost_hse - energy_lost_transport) / max(abs(energy_lost_hse), abs(energy_lost_transport), ZERO)):.6e}')
            if len(lum_compare_rows) > 1:
                lum_compare_arr = np.asarray(lum_compare_rows[1:], dtype=float)
                t_cmp = lum_compare_arr[:, 0]
                rel_cmp = lum_compare_arr[:, 4]
                abs_rel_cmp = np.abs(rel_cmp)
                logger.info('L_transport vs L_HSE mismatch: mean_rel=' + f'{np.mean(rel_cmp):.6e}'
                            + ', median_abs_rel=' + f'{np.median(abs_rel_cmp):.6e}'
                            + ', p95_abs_rel=' + f'{np.percentile(abs_rel_cmp, 95):.6e}'
                            + ', max_abs_rel=' + f'{np.max(abs_rel_cmp):.6e}')
                early_mask_01 = t_cmp <= 0.1 * const.Gyr_to_s
                early_mask_1 = t_cmp <= 1.0 * const.Gyr_to_s
                if np.any(early_mask_01):
                    logger.info('L mismatch <=0.1 Gyr: mean_abs_rel=' + f'{np.mean(abs_rel_cmp[early_mask_01]):.6e}'
                                + ', max_abs_rel=' + f'{np.max(abs_rel_cmp[early_mask_01]):.6e}')
                if np.any(early_mask_1):
                    logger.info('L mismatch <=1.0 Gyr: mean_abs_rel=' + f'{np.mean(abs_rel_cmp[early_mask_1]):.6e}'
                                + ', max_abs_rel=' + f'{np.max(abs_rel_cmp[early_mask_1]):.6e}')
            if len(lum_hse_radius_compare_rows) > 1:
                hse_radius_cmp = np.asarray(lum_hse_radius_compare_rows[1:], dtype=float)
                rel_hse_radius = hse_radius_cmp[:, 4]
                abs_rel_hse_radius = np.abs(rel_hse_radius)
                logger.info('L_HSE(center) vs L_HSE(boundary): mean_rel=' + f'{np.mean(rel_hse_radius):.6e}'
                            + ', median_abs_rel=' + f'{np.median(abs_rel_hse_radius):.6e}'
                            + ', p95_abs_rel=' + f'{np.percentile(abs_rel_hse_radius, 95):.6e}'
                            + ', max_abs_rel=' + f'{np.max(abs_rel_hse_radius):.6e}')
            if len(tint_compare_rows) > 1:
                tint_cmp = np.asarray(tint_compare_rows[1:], dtype=float)
                rel_tint = tint_cmp[:, 4]
                abs_rel_tint = np.abs(rel_tint)
                abs_dtint = np.abs(tint_cmp[:, 3])
                logger.info('T_int transport vs post-HSE: mean_rel=' + f'{np.mean(rel_tint):.6e}'
                            + ', median_abs_rel=' + f'{np.median(abs_rel_tint):.6e}'
                            + ', p95_abs_rel=' + f'{np.percentile(abs_rel_tint, 95):.6e}'
                            + ', max_abs_rel=' + f'{np.max(abs_rel_tint):.6e}'
                            + ', max_abs_dT=' + f'{np.max(abs_dtint):.6e}' + ' K')
            if len(lum_transport_pc_compare_rows) > 1:
                pc_cmp = np.asarray(lum_transport_pc_compare_rows[1:], dtype=float)
                raw_rel_b = pc_cmp[:, 4]
                pc_rel_b = pc_cmp[:, 5]
                logger.info('L_transport(raw) vs L_HSE(boundary): mean_abs_rel=' + f'{np.mean(np.abs(raw_rel_b)):.6e}'
                            + ', p95_abs_rel=' + f'{np.percentile(np.abs(raw_rel_b), 95):.6e}'
                            + ', max_abs_rel=' + f'{np.max(np.abs(raw_rel_b)):.6e}')
                logger.info('L_transport(PC) vs L_HSE(boundary): mean_abs_rel=' + f'{np.mean(np.abs(pc_rel_b)):.6e}'
                            + ', p95_abs_rel=' + f'{np.percentile(np.abs(pc_rel_b), 95):.6e}'
                            + ', max_abs_rel=' + f'{np.max(np.abs(pc_rel_b)):.6e}')

        if energy_conservation_debug and len(lum_compare_rows) > 0:
            try:
                np.savetxt(
                    os.path.join(folder_name, 'luminosity_compare.txt'),
                    np.asarray(lum_compare_rows, dtype=float),
                    header='age_s L_int_HSE_erg_s L_transport_erg_s L_diff_erg_s rel_diff',
                    fmt='%.8e')
            except Exception as e:
                logger.warning('Failed to save luminosity_compare.txt: %s', e)
        if energy_conservation_debug and len(lum_hse_radius_compare_rows) > 0:
            try:
                np.savetxt(
                    os.path.join(folder_name, 'luminosity_hse_radius_compare.txt'),
                    np.asarray(lum_hse_radius_compare_rows, dtype=float),
                    header='age_s L_hse_center_erg_s L_hse_boundary_erg_s L_diff_erg_s rel_diff',
                    fmt='%.8e')
            except Exception as e:
                logger.warning('Failed to save luminosity_hse_radius_compare.txt: %s', e)
        if energy_conservation_debug and len(tint_compare_rows) > 0:
            try:
                np.savetxt(
                    os.path.join(folder_name, 'tint_compare.txt'),
                    np.asarray(tint_compare_rows, dtype=float),
                    header='age_s T_int_transport_in_K T_int_postHSE_K T_diff_K rel_diff',
                    fmt='%.8e')
            except Exception as e:
                logger.warning('Failed to save tint_compare.txt: %s', e)
        if energy_conservation_debug and len(lum_transport_pc_compare_rows) > 0:
            try:
                np.savetxt(
                    os.path.join(folder_name, 'luminosity_transport_pc_compare.txt'),
                    np.asarray(lum_transport_pc_compare_rows, dtype=float),
                    header='age_s L_hse_boundary_erg_s L_transport_raw_erg_s L_transport_pc_erg_s rel_raw_vs_hse_boundary rel_pc_vs_hse_boundary',
                    fmt='%.8e')
            except Exception as e:
                logger.warning('Failed to save luminosity_transport_pc_compare.txt: %s', e)
        if energy_conservation_debug and len(energy_diag_rows) > 0:
            try:
                np.savetxt(
                    os.path.join(folder_name, 'energy_diagnostics.txt'),
                    np.asarray(energy_diag_rows, dtype=float),
                    header=(
                        'age_s dE_G_e0 dE_G_E_lost '
                        'dE_midpoint_e0 dE_midpoint_E_lost '
                        'dE_hse_e0 dE_hse_E_lost '
                        'dE_hse_boundary_e0 dE_hse_boundary_E_lost '
                        'dE_transport_e0 dE_transport_E_lost '
                        'dE_transport_pc_e0 dE_transport_pc_E_lost'
                    ),
                    fmt='%.8e')
            except Exception as e:
                logger.warning('Failed to save energy_diagnostics.txt: %s', e)

        logger.info('Evolution complete. Final data saved to: ' + folder_name)

    # ---------- Optional post-evolution movie ----------
    # If both flags are set, active plotting already ran, so skip end plotting.
    do_plot_end = args.plot_end and not args.plot_active

    movie_outdir = os.path.join('plots', model_fldr + '_movie')

    if do_plot_end:
        import matplotlib
        matplotlib.use('Agg')
        from plot_class import init_plot

        max_step_myr = float(config['general']['max_step'])
        dt_gyr = max_step_myr * 1e-3  # convert Myr to Gyr

        logger.info('Generating evolution movie with dt_gyr=%.4f ...' % dt_gyr)

        plotter = init_plot(paths=[args.config], plot_rock_phases=_plot_rock_phases)
        plotter.movie_evol_plots_by_age(
            output_file=model_fldr + '_movie',
            labels=[model_fldr],
            dt_gyr=dt_gyr,
        )
        logger.info('Movie frames saved to %s/' % movie_outdir)

    # Generate mp4 from frames if any plotting mode was active
    if args.plot_end or args.plot_active:
        mp4_path = os.path.join(movie_outdir, 'output.mp4')
        ffmpeg_cmd = (
            f"ffmpeg -y -framerate 30 -pattern_type glob -i '{movie_outdir}/*.png' "
            f"-vf \"scale=trunc(iw/2)*2:trunc(ih/2)*2\" "
            f"-c:v libx264 -pix_fmt yuv420p {mp4_path}"
        )
        logger.info('Generating movie: %s' % mp4_path)
        ret = subprocess.run(ffmpeg_cmd, shell=True, capture_output=True, text=True)
        if ret.returncode == 0:
            logger.info('Movie saved to %s' % mp4_path)
        else:
            logger.warning('ffmpeg failed (returncode %d): %s' % (ret.returncode, ret.stderr.strip()))

if __name__ == '__main__':
    run()
