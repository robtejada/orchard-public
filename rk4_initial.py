"""
RK4 shooting method for planetary initial conditions.

Builds self-consistent (P, r, rho, T) profiles for any planet mass and
entropy by integrating the hydrostatic structure equations from center
to surface using 4th-order Runge-Kutta, then iterating on the central
pressure P_c until the surface pressure matches p_atm.

This replaces the pre-computed isentrope file interpolation, providing
accurate initial conditions for any mass (Saturn to super-Jupiters)
and any initial entropy.

Developer:
    - Roberto Tejada Arevalo (APPLE+ORCHARD)
"""

import numpy as np
from utils.const import G
from initial import mixtures_eos, mantle_eos, core_eos
from utils.common import logger

# Conversion factors
log10e = np.log10(np.e)
dyn_to_GPa = 1e-10


def _get_rho_scalar(ln_P, S, Y, Z, f_rock, is_envelope, is_mantle, is_core,
                     mantle_entropy=None, core_entropy=None):
    """
    Get density at a single point given ln(P) and composition.
    Returns density in g/cm³.
    """
    P_dyn = np.exp(ln_P)
    log10_P = ln_P * log10e

    if is_envelope:
        s_arr = np.array([S])
        lgp_arr = np.array([log10_P])
        y_arr = np.array([Y])
        z_arr = np.array([Z])
        fr_arr = np.array([f_rock])
        log_rho = mixtures_eos.get_logrho_sp(s_arr, lgp_arr, y_arr, z_arr, _frock=fr_arr, tab=True)
        return float(10.0 ** log_rho[0])

    elif is_mantle:
        P_GPa = P_dyn * dyn_to_GPa
        P_min = getattr(mantle_eos, "P_min", 1e-4)
        P_GPa_safe = max(P_GPa, P_min)
        s_val = mantle_entropy if mantle_entropy is not None else S
        rho = mantle_eos.get_rho_sp(np.array([s_val]), np.array([P_GPa_safe]))
        return float(np.asarray(rho).ravel()[0])

    elif is_core:
        P_GPa = P_dyn * dyn_to_GPa
        P_min = getattr(core_eos, "P_min", 1.0)
        P_GPa_safe = max(P_GPa, P_min)
        s_val = core_entropy if core_entropy is not None else S
        T = core_eos.get_T_sp(np.array([s_val]), np.array([P_GPa_safe]))
        T_val = float(np.asarray(T).ravel()[0])
        rho = core_eos.get_rho_pt(np.array([P_GPa_safe]), np.array([T_val]))
        return float(np.asarray(rho).ravel()[0])

    else:
        return 1.0  # fallback


def rk4_integrate(P_c, S_arr, Y_arr, Z_arr, f_rock_arr, m_b,
                   kcore, kcore_fe,
                   mantle_entropy=None, core_entropy=None):
    """
    Integrate hydrostatic structure from center to surface using RK4
    on a fine internal mass grid, then interpolate onto orchard's grid.

    Uses a log-spaced internal grid (N_fine points) for the RK4 to handle
    the steep gradients near the center, then maps the result back to
    the orchard mass boundaries m_b.

    Parameters
    ----------
    P_c : float
        Central pressure in dyn/cm².
    S_arr, Y_arr, Z_arr, f_rock_arr : arrays (N,)
        Composition profiles (cell 0=surface, N-1=center).
    m_b : array (N+1,)
        Mass boundaries (m_b[0]=M_total, m_b[N]≈0, surface to center).
    kcore, kcore_fe : int
        Core boundary indices.

    Returns
    -------
    r_b : array (N+1,)
        Radius at boundaries in cm.
    P_cells : array (N,)
        Pressure at cell centers in dyn/cm².
    rho_cells : array (N,)
        Density at cell centers in g/cm³.
    T_cells : array (N,)
        Temperature at cell centers in K.
    """
    N = len(S_arr)
    M_total = m_b[0]

    # --- Build a fine hybrid mass grid for RK4 ---
    # Must resolve all regions: center, mantle/core, and envelope.
    # For rocky planets the envelope can be < 0.01% of total mass,
    # so a fixed 50/50 split misses it entirely.
    N_total_fine = 2000
    m_min = M_total * 1e-8

    # Core boundary mass (ascending convention: m_b[kcore] is mass at
    # envelope-mantle boundary, with m_b descending from surface to center).
    if kcore == 0:
        # Bare rock: all cells are mantle/core, no envelope.
        # Full log-spaced grid from center to surface.
        m_fine = np.logspace(np.log10(m_min), np.log10(M_total),
                             N_total_fine + 1)
    elif 0 < kcore < N:
        # Planet with core and envelope: split grid at core boundary.
        m_core_boundary = m_b[kcore]
        env_fraction = (M_total - m_core_boundary) / M_total
        # Guarantee minimum 50 cells in the envelope for interpolation.
        N_env = max(50, int(N_total_fine * env_fraction))
        N_env = min(N_env, N_total_fine // 2)
        N_interior = N_total_fine - N_env

        # Interior: log-spaced from center to core boundary
        m_interior = np.logspace(
            np.log10(m_min), np.log10(m_core_boundary), N_interior + 1
        )
        # Envelope: use hybrid log+linear grid (like the coreless path)
        # so that the outer atmosphere is well-resolved. A pure log-spaced
        # envelope puts the coarsest cells at the surface, missing the
        # steep pressure drop from ~100 bar to ~1 bar. The linear outer
        # half provides uniform resolution where it matters most.
        m_env_split = 0.5 * (m_core_boundary + M_total)
        N_env_log = N_env // 2
        N_env_lin = N_env - N_env_log
        m_env_log = np.logspace(
            np.log10(m_core_boundary), np.log10(m_env_split), N_env_log + 1
        )[1:]  # skip duplicate at m_core_boundary
        m_env_lin = np.linspace(m_env_split, M_total, N_env_lin + 1)[1:]
        m_fine = np.concatenate([m_interior, m_env_log, m_env_lin])
    else:
        # Coreless planet (gas giant): original grid
        N_log = 1000
        N_lin = 1000
        m_split = 0.5 * M_total
        m_log = np.logspace(np.log10(m_min), np.log10(m_split), N_log + 1)
        m_lin = np.linspace(m_split, M_total, N_lin + 1)[1:]
        m_fine = np.concatenate([m_log, m_lin])

    N_fine = len(m_fine) - 1

    # For composition, we need to know which region each fine cell belongs to.
    # Map fine grid masses to orchard cell indices to look up S, Y, Z, f_rock.
    # orchard: m_b[0]=M_total (surface), m_b[N]=0 (center).
    # Fine grid: m_fine[0]≈0 (center), m_fine[N_fine]=M_total (surface).
    # Orchard cell k contains mass between m_b[k] and m_b[k+1].
    # For a mass m in the fine grid, find which orchard cell it maps to.
    # Since m_b is decreasing, use searchsorted on reversed m_b.
    m_b_reversed = m_b[::-1]  # now ascending: [0, ..., M_total]
    # Cell j in reversed grid = original cell (N-1-j).
    fine_cell_idx = np.searchsorted(m_b_reversed, m_fine[:-1], side='right') - 1
    fine_cell_idx = np.clip(fine_cell_idx, 0, N - 1)
    # Convert to original orchard index
    fine_orig_idx = N - 1 - fine_cell_idx

    # Look up composition for each fine cell
    S_fine = S_arr[fine_orig_idx]
    Y_fine = Y_arr[fine_orig_idx]
    Z_fine = Z_arr[fine_orig_idx]
    fr_fine = f_rock_arr[fine_orig_idx]

    # Region flags
    fine_is_env = fine_orig_idx < kcore
    fine_is_mnt = (fine_orig_idx >= kcore) & (fine_orig_idx < kcore_fe)
    fine_is_cor = fine_orig_idx >= kcore_fe

    def get_rho_fine(ln_P, j):
        """Get density for fine cell j. Returns rho_floor on EOS failure."""
        try:
            rho = _get_rho_scalar(
                ln_P, S_fine[j], Y_fine[j], Z_fine[j], fr_fine[j],
                fine_is_env[j], fine_is_mnt[j], fine_is_cor[j],
                mantle_entropy, core_entropy
            )
            if not np.isfinite(rho) or rho <= 0:
                return rho_floor
            return rho
        except Exception:
            return rho_floor

    # Derivative functions with safety guards.
    # Cap the radius growth per unit mass to prevent runaway expansion
    # in the low-density outer envelope where rho → 0.
    rho_floor = 1e-6  # g/cm³, prevents 1/(r³ρ) from blowing up

    # Maximum ln(r) to prevent exp overflow (r ~ 1e12 cm ~ 100 R_J)
    max_lnr = np.log(1e12)

    # Minimum ln(P) to prevent exp() underflow to 0 during the reverse mapping
    # onto the orchard grid. 1 dyn/cm^2 is ~1 uPa — well below any physical
    # planetary pressure, so this never perturbs a converged trajectory, but
    # keeps a catastrophically bad iteration from producing P_cells = 0 that
    # strict-edge EOSes (e.g. G23 iron) reject.
    min_lnP = np.log(1.0)

    def dlnr_dm(ln_r, rho):
        rho_safe = max(rho, rho_floor)
        lr_safe = min(ln_r, max_lnr)
        return 1.0 / (4.0 * np.pi * np.exp(3.0 * lr_safe) * rho_safe)

    def dlnP_dm(ln_r, ln_P, m):
        lr_safe = min(ln_r, max_lnr)
        return -G * m / (4.0 * np.pi * np.exp(4.0 * lr_safe) * np.exp(ln_P))

    # --- Central boundary condition ---
    rho_c = get_rho_fine(np.log(P_c), 0)
    r_center = (3.0 * m_fine[0] / (4.0 * np.pi * rho_c)) ** (1.0 / 3.0)
    r_center = max(r_center, 1.0)
    P_c_corr = P_c - (3.0 * G / (8.0 * np.pi)) * (4.0 * np.pi * rho_c / 3.0) ** (4.0 / 3.0) * m_fine[0] ** (2.0 / 3.0)
    P_c_corr = max(P_c_corr, P_c * 0.5)

    # State arrays on fine grid boundaries
    ln_r_fine = np.zeros(N_fine + 1)
    ln_P_fine = np.zeros(N_fine + 1)
    ln_r_fine[0] = np.log(r_center)
    ln_P_fine[0] = np.log(P_c_corr)

    # --- RK4 integration ---
    for j in range(N_fine):
        h = m_fine[j + 1] - m_fine[j]
        lr = ln_r_fine[j]
        lp = ln_P_fine[j]
        m_j = m_fine[j]
        m_mid = 0.5 * (m_fine[j] + m_fine[j + 1])

        rho1 = get_rho_fine(lp, j)
        k1r = dlnr_dm(lr, rho1)
        k1p = dlnP_dm(lr, lp, m_j)

        lr2 = lr + 0.5 * h * k1r
        lp2 = lp + 0.5 * h * k1p
        rho2 = get_rho_fine(lp2, j)
        k2r = dlnr_dm(lr2, rho2)
        k2p = dlnP_dm(lr2, lp2, m_mid)

        lr3 = lr + 0.5 * h * k2r
        lp3 = lp + 0.5 * h * k2p
        rho3 = get_rho_fine(lp3, j)
        k3r = dlnr_dm(lr3, rho3)
        k3p = dlnP_dm(lr3, lp3, m_mid)

        lr4 = lr + h * k3r
        lp4 = lp + h * k3p
        j_next = min(j + 1, N_fine - 1)
        rho4 = get_rho_fine(lp4, j_next)
        k4r = dlnr_dm(lr4, rho4)
        k4p = dlnP_dm(lr4, lp4, m_fine[j + 1])

        lr_new = lr + (h / 6.0) * (k1r + 2 * k2r + 2 * k3r + k4r)
        lp_new = lp + (h / 6.0) * (k1p + 2 * k2p + 2 * k3p + k4p)

        # Guard against NaN/inf from EOS failures
        if not np.isfinite(lr_new) or not np.isfinite(lp_new):
            lr_new = lr  # freeze at last good state
            lp_new = lp

        # Cap radius growth
        lr_new = min(lr_new, max_lnr)
        # Floor ln(P) so exp() cannot underflow downstream
        lp_new = max(lp_new, min_lnP)

        ln_r_fine[j + 1] = lr_new
        ln_P_fine[j + 1] = lp_new

    # --- Interpolate from fine grid onto orchard m_b ---
    # Fine grid: m_fine[0]≈0 (center) → m_fine[N_fine]=M_total (surface), ascending.
    # orchard: m_b[0]=M_total (surface) → m_b[N]=0 (center), descending.
    from scipy.interpolate import interp1d
    f_lnr = interp1d(m_fine, ln_r_fine, kind='linear', fill_value='extrapolate')
    f_lnP = interp1d(m_fine, ln_P_fine, kind='linear', fill_value='extrapolate')

    # Evaluate at orchard boundaries (clamp to [m_min, M_total])
    m_b_clamped = np.clip(m_b, m_min, M_total)
    r_b = np.exp(f_lnr(m_b_clamped))
    P_b = np.exp(f_lnP(m_b_clamped))

    # Cell-centered pressures (geometric mean of boundary pressures)
    P_cells = np.sqrt(P_b[:-1] * P_b[1:])
    # Fix center cell where P_b[-1] might be at m_min ≈ 0
    P_cells[-1] = max(P_cells[-1], P_c * 0.5)

    # Compute density and temperature from EOS at cell-centered pressures
    rho_cells = np.zeros(N)
    T_cells = np.zeros(N)
    log10_P_cells = np.log10(np.maximum(P_cells, 1e-30))

    if kcore > 0:
        env = slice(0, kcore)
        log_rho_env = mixtures_eos.get_logrho_sp(
            S_arr[env], log10_P_cells[env], Y_arr[env], Z_arr[env],
            _frock=f_rock_arr[env], tab=True
        )
        rho_cells[env] = 10.0 ** log_rho_env
        T_cells[env] = 10.0 ** mixtures_eos.get_logt_sp(
            S_arr[env], log10_P_cells[env], Y_arr[env], Z_arr[env],
            _frock=f_rock_arr[env], tab=True
        )

    if kcore_fe > kcore:
        mnt = slice(kcore, kcore_fe)
        P_min_mnt_GPa = getattr(mantle_eos, "P_min", 1e-4)
        P_GPa = np.maximum(P_cells[mnt] * dyn_to_GPa, P_min_mnt_GPa)
        s_mnt = np.full(kcore_fe - kcore,
                        mantle_entropy if mantle_entropy is not None else S_arr[kcore])
        rho_cells[mnt] = mantle_eos.get_rho_sp(s_mnt, P_GPa)
        T_cells[mnt] = mantle_eos.get_t_sp(s_mnt, P_GPa)

    if kcore_fe < N:
        fec = slice(kcore_fe, N)
        P_min_core_GPa = getattr(core_eos, "P_min", 1.0)
        P_GPa = np.maximum(P_cells[fec] * dyn_to_GPa, P_min_core_GPa)
        s_fec = np.full(N - kcore_fe,
                        core_entropy if core_entropy is not None else S_arr[kcore_fe])
        T_fec = core_eos.get_T_sp(s_fec, P_GPa)
        rho_cells[fec] = core_eos.get_rho_pt(P_GPa, T_fec)
        T_cells[fec] = T_fec

    return r_b, P_cells, rho_cells, T_cells


def _is_rocky_planet(kcore, N, m_b):
    """Check if the planet is core-dominated (thin or no envelope)."""
    if kcore == 0:
        return True  # bare rock: all cells are mantle/core
    if kcore < 0 or kcore >= N:
        return False  # coreless gas giant
    env_mass_fraction = (m_b[0] - m_b[kcore]) / m_b[0]
    return env_mass_fraction < 0.1


def _shoot_rocky(S_arr, Y_arr, Z_arr, f_rock_arr, m_b,
                 kcore, kcore_fe, p_atm, M_planet,
                 mantle_entropy, core_entropy, max_iter=30):
    """
    Find P_c for a rocky planet by iterating on mean density.

    For rocky planets with thin H/He envelopes, the standard bisection
    on surface pressure fails because P_surface(P_c) is non-monotonic:
    the envelope is too thin to produce the needed pressure drop from
    the core-boundary pressure to p_atm. Instead, iterate P_c until the
    RK4 integration yields a physically reasonable interior with monotonic
    profiles. The main HSE solver in evolution.py will then refine the
    surface pressure and envelope structure.
    """
    N = len(S_arr)

    # Estimate P_c from the uniform-sphere formula:
    #   P_c ~ (3G / 8pi) * (4pi rho_bar / 3)^(4/3) * M^(2/3)
    # Use mantle density at ~100 GPa as a characteristic mean density.
    rho_bar = mantle_eos.get_rho_sp(
        np.array([mantle_entropy if mantle_entropy is not None else 0.54]),
        np.array([100.0])  # GPa
    )
    rho_bar = float(np.asarray(rho_bar).ravel()[0])
    P_c_est = (3.0 * G / (8.0 * np.pi)) * \
              (4.0 * np.pi * rho_bar / 3.0) ** (4.0 / 3.0) * \
              M_planet ** (2.0 / 3.0)
    P_c_est = np.clip(P_c_est, 1e10, 1e16)

    logger.debug(f"  RK4 rocky mode: rho_bar = {rho_bar:.2f} g/cm³, "
                 f"P_c_est = {P_c_est:.3e} dyn/cm²")

    # Iterate: integrate at P_c, check if profiles are monotonic and
    # the planet radius is reasonable, adjust P_c accordingly.
    best_result = None
    best_score = np.inf
    P_c = P_c_est

    for iteration in range(max_iter):
        r_b, P_cells, rho, T = rk4_integrate(
            P_c, S_arr, Y_arr, Z_arr, f_rock_arr, m_b,
            kcore, kcore_fe, mantle_entropy, core_entropy
        )

        r_mono = np.all(np.diff(r_b) < 0)
        p_mono = np.all(np.diff(P_cells) > 0)
        R_planet = r_b[0]

        # Target radius from mean density: R = (3M / 4pi rho_bar)^(1/3)
        R_target = (3.0 * M_planet / (4.0 * np.pi * rho_bar)) ** (1.0 / 3.0)

        score = abs(np.log(R_planet / R_target))
        if not r_mono or not p_mono:
            score += 10.0  # penalize non-monotonic profiles

        if score < best_score:
            best_score = score
            best_result = (r_b, P_cells, rho, T, P_c)

        if r_mono and p_mono and abs(R_planet / R_target - 1.0) < 0.3:
            logger.debug(f"  RK4 rocky converged: P_c = {P_c:.3e}, "
                         f"R = {R_planet / 6.9911e9:.3f} R_J "
                         f"({R_planet / 6.371e8:.2f} R_E), "
                         f"itr = {iteration + 1}")
            return r_b, P_cells, rho, T, P_c

        # Adjust P_c: empirically, the damped correction with negative
        # exponent matches what works in the rest of the codebase for
        # rocky planets.  Don't flip the sign without retesting all
        # rocky-planet configurations.
        ratio = R_planet / R_target
        # Scaling: P_c ~ R^(-4) for HSE, so use a damped correction
        correction = ratio ** (-2.0)  # damped relative to R^-4
        correction = np.clip(correction, 0.3, 3.0)
        P_c *= correction

    logger.warning(f"  RK4 rocky: {max_iter} iterations, using best result "
                   f"(score={best_score:.3f})")
    return best_result


def _has_compact_core(kcore, N, m_b):
    """Check if the planet has a compact core (gas giant with core)."""
    if kcore <= 0 or kcore >= N:
        return False
    env_mass_fraction = (m_b[0] - m_b[kcore]) / m_b[0]
    return env_mass_fraction >= 0.1


def _shoot_gas_giant_with_core(S_arr, Y_arr, Z_arr, f_rock_arr, m_b,
                                kcore, kcore_fe, p_atm, M_planet,
                                mantle_entropy, core_entropy, max_iter=30):
    """
    Find P_c for a gas giant with a compact core by radius matching.

    The thin atmosphere of a gas giant (~1e-7 of total mass) cannot be
    resolved by the RK4 mass grid, so surface-pressure bisection fails
    for models with compact cores (the bisection converges to a degenerate
    tiny-planet solution). Instead, iterate on P_c to match the expected
    planet radius from the coreless uniform-sphere estimate, producing a
    reasonable initial guess that the Henyey HSE solver can refine.
    """
    N = len(S_arr)
    M_J = 1.899e30
    R_J = 6.9911e9

    M_ratio = M_planet / M_J if M_planet is not None else 1.0

    # Estimate target radius from the coreless model.
    # For a homogeneous H-He sphere at entropy S, the radius scales as
    # R ~ M^(-1/3) * f(S). Use the coreless P_c ~ 4e13*M_ratio^2 as
    # reference: at this P_c, the mean density gives a target radius.
    S_env = S_arr[0]  # envelope entropy
    P_c_ref = 4e13 * max(M_ratio, 0.1) ** 2
    P_c_ref = np.clip(P_c_ref, 1e11, 1e16)

    # Estimate mean envelope density from EOS at moderate pressure.
    # Use pressure at ~50% of the envelope mass as representative.
    P_mid_est = P_c_ref * 0.1  # roughly mid-envelope pressure
    log10_P_mid = np.log10(P_mid_est)
    try:
        log_rho_est = mixtures_eos.get_logrho_sp(
            np.array([S_env]), np.array([log10_P_mid]),
            np.array([Y_arr[0]]), np.array([Z_arr[0]]),
            _frock=np.array([f_rock_arr[0]]), tab=True
        )
        rho_env_est = float(10.0 ** log_rho_est[0])
    except Exception:
        rho_env_est = 1.0  # fallback

    # Target radius from mean density
    R_target = (3.0 * M_planet / (4.0 * np.pi * rho_env_est)) ** (1.0 / 3.0)
    R_target = np.clip(R_target, 0.5 * R_J, 10.0 * R_J)

    logger.debug(f"  RK4 gas-giant-with-core mode: "
                 f"rho_env_est = {rho_env_est:.3f} g/cm³, "
                 f"R_target = {R_target / R_J:.2f} R_J")

    # Iterate on P_c: integrate at P_c, compare R to R_target, adjust.
    best_result = None
    best_score = np.inf
    P_c = P_c_ref

    for iteration in range(max_iter):
        r_b, P_cells, rho, T = rk4_integrate(
            P_c, S_arr, Y_arr, Z_arr, f_rock_arr, m_b,
            kcore, kcore_fe, mantle_entropy, core_entropy
        )

        R_planet = r_b[0]
        r_mono = np.all(np.diff(r_b) < 0)
        p_mono = np.all(np.diff(P_cells) > 0)
        # Floor collapse: integration hit min_lnP=ln(1) for some cells.
        # Such a profile carries no usable structural information, so
        # heavily penalise it (+100) and bail out at the end if every
        # iteration looked like this.
        p_collapsed = bool(np.any(P_cells <= 1.0001))

        score = abs(np.log(R_planet / R_target))
        if not r_mono or not p_mono:
            score += 10.0
        if p_collapsed:
            score += 100.0

        if score < best_score:
            best_score = score
            best_result = (r_b, P_cells, rho, T, P_c)

        if r_mono and p_mono and abs(R_planet / R_target - 1.0) < 0.3:
            logger.debug(f"  RK4 gas-giant-with-core converged: P_c = {P_c:.3e}, "
                         f"R = {R_planet / R_J:.3f} R_J, "
                         f"itr = {iteration + 1}")
            return (r_b, P_cells, rho, T, P_c), True

        # Adjust P_c.  The relationship between P_c and R for a real
        # planet (mixture of compressible envelope + dense core) is
        # not the simple polytropic R ~ P_c^{-1/4} sign.  Empirically,
        # for the configurations that work (homog Jupiter, GJ1214b
        # s=2.0), this damped correction with negative exponent
        # converges; flipping the sign breaks them.  Keep the legacy
        # rule and rely on floor-collapse detection above to catch
        # the cases where this iteration cannot make progress.
        ratio = R_planet / R_target
        correction = ratio ** (-2.0)
        correction = np.clip(correction, 0.3, 3.0)
        P_c *= correction

    # Did not converge.  Return the best (non-converged) radius-match structure
    # together with converged=False, so shoot_for_Pc compares it against the
    # surface-pressure-bisection candidate and hands HSE the more expanded
    # (recoverable) of the two.  None means every iteration hit the ln(P) floor
    # (no usable structure at all).
    if best_score >= 100.0:
        logger.warning(
            f"  RK4 gas-giant-with-core: every iteration hit the ln(P) "
            f"floor (best_score={best_score:.3f}); no usable radius-match "
            f"structure.")
        return None, False
    logger.warning(f"  RK4 gas-giant-with-core did not converge "
                   f"(best score={best_score:.3f}); will compare against the "
                   f"surface-pressure-bisection structure.")
    return best_result, False


def _shoot_psurf_bisection(S_arr, Y_arr, Z_arr, f_rock_arr, m_b,
                           kcore, kcore_fe, p_atm, P_c_guess=None,
                           M_planet=None, mantle_entropy=None,
                           core_entropy=None, tol=0.01, max_iter=60):
    """Bisection on central pressure so the surface pressure matches p_atm.

    This is the structure builder for coreless gas giants.  For a *cored* gas
    giant it is one of two candidate structures compared in ``shoot_for_Pc``:
    for a dense core it can converge to a degenerate "tiny-planet" root, so its
    result is compared (by surface radius) against the radius-match structure
    rather than used unconditionally.

    Returns (r_b, P_cells, rho, T, P_c).
    """
    M_J = 1.899e30
    if P_c_guess is None:
        M_ratio = M_planet / M_J if M_planet is not None else 1.0
        P_c_guess = 4e13 * max(M_ratio, 0.1) ** 2
        P_c_guess = np.clip(P_c_guess, 1e11, 1e16)

    logger.debug(f"  RK4 P_surf bisection: P_c_guess = {P_c_guess:.3e} dyn/cm²")

    def get_surface_P(P_c):
        try:
            r_b, P_cells, rho, T = rk4_integrate(
                P_c, S_arr, Y_arr, Z_arr, f_rock_arr, m_b,
                kcore, kcore_fe, mantle_entropy, core_entropy
            )
            return P_cells[0]
        except Exception:
            return np.nan

    # Bracket the root
    P_lo = P_c_guess * 0.01
    P_hi = P_c_guess * 100.0

    P_surf_guess = get_surface_P(P_c_guess)
    if np.isnan(P_surf_guess):
        P_surf_guess = 0.0

    # Higher P_c -> higher P_surf (more compressed planet)
    P_lo_surf = get_surface_P(P_lo)
    P_hi_surf = get_surface_P(P_hi)

    # Expand brackets if needed
    for _ in range(10):
        if not np.isnan(P_lo_surf) and not np.isnan(P_hi_surf):
            if (P_lo_surf - p_atm) * (P_hi_surf - p_atm) <= 0:
                break
        P_lo *= 0.1
        P_hi *= 10.0
        P_lo_surf = get_surface_P(P_lo)
        P_hi_surf = get_surface_P(P_hi)

    # Bisection
    r_b = P_cells = rho = T = None
    P_mid = P_c_guess
    P_surf = np.nan
    for iteration in range(max_iter):
        P_mid = np.sqrt(P_lo * P_hi)  # geometric midpoint
        r_b, P_cells, rho, T = rk4_integrate(
            P_mid, S_arr, Y_arr, Z_arr, f_rock_arr, m_b,
            kcore, kcore_fe, mantle_entropy, core_entropy
        )
        P_surf = P_cells[0]

        if P_surf <= 0 or not np.isfinite(P_surf):
            P_hi = P_mid  # P_c too high
            continue

        rel_err = abs(P_surf - p_atm) / p_atm
        if rel_err < tol:
            logger.debug(f"  RK4 P_surf converged: P_c = {P_mid:.3e}, "
                         f"P_surf = {P_surf:.3e} (target {p_atm:.3e}), "
                         f"itr = {iteration + 1}, R = {r_b[0] / 6.9911e9:.2f} R_J")
            return r_b, P_cells, rho, T, P_mid

        if P_surf > p_atm:
            P_hi = P_mid
        else:
            P_lo = P_mid

    logger.warning(f"  RK4 P_surf bisection: {max_iter} iterations, "
                   f"P_surf = {P_surf:.3e}, target = {p_atm:.3e}")
    return r_b, P_cells, rho, T, P_mid


def shoot_for_Pc(S_arr, Y_arr, Z_arr, f_rock_arr, m_b,
                  kcore, kcore_fe, p_atm,
                  P_c_guess=None, M_planet=None,
                  mantle_entropy=None, core_entropy=None,
                  tol=0.01, max_iter=60):
    """
    Find central pressure P_c for initial structure.

    For gas giants (envelope-dominated), uses bisection on P_surface = p_atm.
    For gas giants with compact cores, uses radius matching (the RK4 mass
    grid cannot resolve the thin atmosphere for surface-pressure bisection).
    For rocky planets (core-dominated with thin envelope), uses iterative
    density matching since surface-pressure bisection is ill-conditioned.
    """
    N = len(S_arr)
    M_J = 1.899e30

    # Rocky planets: use density-matching iteration instead of bisection
    if _is_rocky_planet(kcore, N, m_b):
        return _shoot_rocky(
            S_arr, Y_arr, Z_arr, f_rock_arr, m_b,
            kcore, kcore_fe, p_atm, M_planet,
            mantle_entropy, core_entropy
        )

    # Gas giants with a compact core.  Two shooting strategies can each produce
    # a degenerate (collapsed) initial structure, depending on the core:
    #   * radius matching collapses to a near-vacuum envelope for a *light* core
    #     (pure water or water/rock) -- HSE then diverges to NaN;
    #   * surface-pressure bisection can converge to a degenerate "tiny-planet"
    #     root for a *denser* core (e.g. a sub-Neptune silicate core) -- also
    #     unrecoverable by HSE.
    # In every observed case the HSE-recoverable structure is the more EXPANDED
    # one (larger surface radius): collapse is the shared failure mode.  So we
    # compute BOTH candidates and hand HSE whichever has the larger surface
    # radius -- unless radius matching genuinely converged, which we trust
    # directly.  HSE refines the small core on the first timestep.
    if _has_compact_core(kcore, N, m_b):
        rad_result, rad_converged = _shoot_gas_giant_with_core(
            S_arr, Y_arr, Z_arr, f_rock_arr, m_b,
            kcore, kcore_fe, p_atm, M_planet,
            mantle_entropy, core_entropy
        )
        if rad_converged:
            return rad_result

        psurf_result = _shoot_psurf_bisection(
            S_arr, Y_arr, Z_arr, f_rock_arr, m_b,
            kcore, kcore_fe, p_atm, P_c_guess, M_planet,
            mantle_entropy, core_entropy, tol, max_iter
        )

        candidates = [c for c in (rad_result, psurf_result)
                      if c is not None and np.isfinite(c[0][0]) and c[0][0] > 0.0]
        if candidates:
            best = max(candidates, key=lambda c: c[0][0])  # largest surface radius
            logger.debug(
                f"  RK4 cored gas giant: chose the more expanded of "
                f"{len(candidates)} candidate(s), "
                f"R = {best[0][0] / 6.9911e9:.3f} R_J.")
            return best
        logger.warning(
            "  RK4 cored gas giant: no usable candidate structure; "
            "falling through to surface-pressure bisection.")

    # --- Coreless gas giant path (also the final fallback) ---
    return _shoot_psurf_bisection(
        S_arr, Y_arr, Z_arr, f_rock_arr, m_b,
        kcore, kcore_fe, p_atm, P_c_guess, M_planet,
        mantle_entropy, core_entropy, tol, max_iter
    )


def build_initial_structure(N, M_planet, S_ini, Y_ini, Z_ini, f_rock_ini,
                             m_b, kcore, kcore_fe, p_atm,
                             mantle_entropy=None, core_entropy=None):
    """
    Build self-consistent initial structure using RK4 shooting.
    """
    # Build uniform composition profiles
    S_arr = np.full(N, S_ini)
    Y_arr = np.full(N, Y_ini * (1 - Z_ini))
    Z_arr = np.full(N, Z_ini)
    f_rock_arr = np.full(N, f_rock_ini)

    if kcore < N and mantle_entropy is not None:
        S_arr[kcore:] = mantle_entropy
    if kcore_fe < N and core_entropy is not None:
        S_arr[kcore_fe:] = core_entropy

    if kcore < N:
        Y_arr[kcore:] = 0.0
        Z_arr[kcore:] = 0.0
        f_rock_arr[kcore:] = 0.0

    r_b, P, rho, T, P_c = shoot_for_Pc(
        S_arr, Y_arr, Z_arr, f_rock_arr, m_b,
        kcore, kcore_fe, p_atm,
        M_planet=M_planet,
        mantle_entropy=mantle_entropy,
        core_entropy=core_entropy,
    )

    logger.debug(f"  RK4 initial structure: R = {r_b[0] / 6.9911e9:.3f} R_J, "
                  f"P_c = {P_c:.3e} dyn/cm², rho_c = {rho[-1]:.3f} g/cm³")

    return r_b, P, rho, T
