"""
Adaptive mesh regridding for ORCHARD.

Targeted cell-splitting approach: splits cells in the helium rain zone
and merges cells in the deep adiabatic interior to keep N constant.
Only a few boundaries change per step, minimizing HSE disruption.

References
----------
- Paxton et al. 2011 (MESA I), §6.4–6.5 — gradient-based mesh refinement
  (MESA changes N; we keep N fixed and do split/merge instead).
"""

import numpy as np
from scipy.interpolate import PchipInterpolator
import logging

logger = logging.getLogger(__name__)


def regrid_profiles(
    m_b, dm, S, Y, Z, f_rock,
    r_b, P, T, rho,
    D_b, DZ_b,
    Ym_i, Z1m_i, Z2m_i,
    chi_prev_full,
    Nu_T_eff, Nu_X_eff,
    kcore, kcore_fe,
    M_planet, M_core, M_core_fe,
    bracket_dsdr_b, y_rain_gap,
    config,
):
    """
    Perform one regridding step using targeted cell splitting/merging.

    Strategy:
    - Find the largest cells inside (or near) the rain pressure gap.
    - Split each by inserting a boundary at the cell midpoint.
    - Merge pairs of cells in the deep adiabatic interior to compensate
      (keeping N constant).
    - For split cells: child cells inherit parent's S, Y, Z values.
    - For merged cells: mass-weighted average of the two parents.
    - P, T, rho, r_b are linearly interpolated (HSE cleanup will fix them).

    Only changes a few boundaries per step (controlled by n_splits_per_step).
    """
    N = len(S)

    sec = 'regrid'
    n_splits = int(config[sec].get('n_splits_per_step', 3))

    def _unchanged():
        """Return the input state exactly as received (every skip path)."""
        return (m_b, dm, S, Y, Z, f_rock,
                r_b, P, T, rho,
                D_b, DZ_b,
                Ym_i, Z1m_i, Z2m_i,
                chi_prev_full, Nu_T_eff, Nu_X_eff,
                kcore, kcore_fe)

    if y_rain_gap is None or len(y_rain_gap) != 2:
        logger.info("Regrid: no rain gap, skipping.")
        return _unchanged()

    p_lo = min(y_rain_gap)  # outer boundary (lower pressure)
    p_hi = max(y_rain_gap)  # inner boundary (higher pressure)

    # Extend the target region slightly beyond the rain gap
    # to refine the transition zones on either side
    p_margin = 0.5 * (p_hi - p_lo)  # 50% margin on each side
    p_target_lo = p_lo - p_margin
    p_target_hi = p_hi + p_margin

    # --- Check if rain zone already has enough cells ---
    max_rain_cells = int(config[sec].get('max_rain_cells', 40))
    n_rain_current = np.sum((P >= p_lo) & (P <= p_hi))
    if n_rain_current >= max_rain_cells:
        logger.info("Regrid: rain zone has %d cells (max=%d), skipping.",
                     n_rain_current, max_rain_cells)
        return _unchanged()

    # --- Find cells to SPLIT (in/near rain zone, largest first) ---
    # Cells are indexed 0 (surface) to N-1 (deepest).
    # P[k] is the cell-centre pressure.
    rain_candidates = []
    for k in range(min(kcore, N)):
        if p_target_lo <= P[k] <= p_target_hi:
            rain_candidates.append((dm[k], k))

    if not rain_candidates:
        logger.info("Regrid: no cells in rain zone to split.")
        return _unchanged()

    # Sort by cell mass (largest first) — split the biggest cells
    rain_candidates.sort(reverse=True)
    split_cells = [k for _, k in rain_candidates[:n_splits]]
    split_cells.sort()  # keep in order

    # --- Find cells to MERGE (deep adiabatic interior, smallest gradient) ---
    # Look for pairs of adjacent cells far from the rain zone
    # in the deep interior (high index = deep)
    # Criterion: cells with nearly identical S, Y, Z
    merge_region_start = max(k for k in split_cells) + 20  # at least 20 cells past rain
    merge_region_start = min(merge_region_start, kcore - 2)

    merge_candidates = []
    for k in range(merge_region_start, min(kcore - 1, N - 1)):
        dS = abs(S[k] - S[k + 1])
        dY = abs(Y[k] - Y[k + 1])
        dZ = abs(Z[k] - Z[k + 1])
        gradient = dS + 100 * dY + 100 * dZ  # penalize composition changes
        merge_candidates.append((gradient, k))

    if len(merge_candidates) < len(split_cells):
        logger.info("Regrid: not enough merge candidates (%d < %d splits).",
                     len(merge_candidates), len(split_cells))
        return _unchanged()

    # Sort by gradient (smallest first) — merge the most uniform pairs
    merge_candidates.sort()
    merge_cells = []
    used = set()
    for _, k in merge_candidates:
        if k not in used and (k + 1) not in used:
            merge_cells.append(k)
            used.add(k)
            used.add(k + 1)
        if len(merge_cells) >= len(split_cells):
            break
    merge_cells.sort()

    if len(merge_cells) < len(split_cells):
        # Can't find enough non-overlapping pairs
        n_actual = len(merge_cells)
        split_cells = split_cells[:n_actual]

    if not split_cells:
        logger.info("Regrid: no valid split/merge pairs found.")
        return _unchanged()

    # --- Build new arrays ---
    # Work with lists for easy insert/delete, then convert back
    m_b_list = list(m_b)
    S_list = list(S)
    Y_list = list(Y)
    Z_list = list(Z)
    f_rock_list = list(f_rock)
    T_list = list(T)
    rho_list = list(rho)
    P_list = list(P)
    chi_list = list(chi_prev_full)

    # Track index shifts from operations
    # Process splits first (inserting boundaries), then merges (removing boundaries)
    # Since splits add indices and merges remove them, process in reverse order
    # to avoid index invalidation.

    # --- Gradient-preserving helpers for cell splitting ---
    # When splitting a cell, we use minmod-limited reconstruction so that
    # child cells capture the local slope rather than being flat copies.
    # This reduces the artificial staircase that flat splitting creates at
    # composition gradients (especially important in the He-rain zone).

    def _minmod(a, b):
        """Scalar minmod limiter: returns the value with smaller magnitude
        if both have the same sign, otherwise zero (no slope)."""
        if a * b <= 0.0:
            return 0.0
        return a if abs(a) < abs(b) else b

    def _gradient_split(val_list, k):
        """Return (val_outer, val_inner) for splitting cell k using
        minmod-limited linear reconstruction.

        Estimates the slope from the left and right neighbours, applies
        the minmod limiter to avoid creating new extrema, then offsets
        each child by +/- slope/4 so that the average of the two children
        equals the parent value exactly.

        Falls back to flat copy at domain boundaries where neighbours
        are unavailable.
        """
        if k <= 0 or k + 1 >= len(val_list):
            # At boundaries, fall back to flat copy
            return val_list[k], val_list[k]
        slope_L = val_list[k] - val_list[k - 1]
        slope_R = val_list[k + 1] - val_list[k]
        slope = _minmod(slope_L, slope_R)
        # Outer child (index k, lower pressure) gets +slope/4,
        # inner child (index k+1, higher pressure) gets -slope/4.
        val_outer = val_list[k] + 0.25 * slope
        val_inner = val_list[k] - 0.25 * slope
        return val_outer, val_inner

    # First: do all splits (insert new boundaries, duplicate cell values)
    # Process from highest index to lowest to avoid invalidation
    for k in reversed(split_cells):
        # Split cell k into two cells of equal mass
        # Insert a new boundary at the midpoint: m_b[k] > m_mid > m_b[k+1]
        m_mid = 0.5 * (m_b_list[k] + m_b_list[k + 1])
        m_b_list.insert(k + 1, m_mid)

        # Gradient-preserving split for thermodynamic/composition fields.
        # Uses minmod-limited reconstruction so children carry the local
        # slope; their average equals the parent value (conservative).
        S_outer, S_inner = _gradient_split(S_list, k)
        S_list[k] = S_outer
        S_list.insert(k + 1, S_inner)

        Y_outer, Y_inner = _gradient_split(Y_list, k)
        Y_list[k] = Y_outer
        Y_list.insert(k + 1, Y_inner)

        Z_outer, Z_inner = _gradient_split(Z_list, k)
        Z_list[k] = Z_outer
        Z_list.insert(k + 1, Z_inner)

        fr_outer, fr_inner = _gradient_split(f_rock_list, k)
        f_rock_list[k] = fr_outer
        f_rock_list.insert(k + 1, fr_inner)

        # chi is a binary-like convective flag — keep as flat copy
        chi_list.insert(k + 1, chi_list[k])

        # For P, T, rho: interpolate between parent and neighbor
        if k + 2 < len(T_list):
            T_list.insert(k + 1, 0.5 * (T_list[k] + T_list[k + 1]))
            rho_list.insert(k + 1, 0.5 * (rho_list[k] + rho_list[k + 1]))
            P_list.insert(k + 1, 0.5 * (P_list[k] + P_list[k + 1]))
        else:
            T_list.insert(k + 1, T_list[k])
            rho_list.insert(k + 1, rho_list[k])
            P_list.insert(k + 1, P_list[k])

    # After splits, the array is now N + n_splits long
    # Now merge cells in the deep interior to bring it back to N
    # merge_cells indices need to be adjusted for the splits that happened above
    # All splits are at lower indices than merge region, so each split shifts
    # merge indices up by 1
    n_inserted = len(split_cells)
    adjusted_merge = [k + n_inserted for k in merge_cells]

    # Process merges from highest index to lowest
    for k in reversed(adjusted_merge):
        if k + 1 >= len(S_list):
            continue
        # Merge cells k and k+1: mass-weighted average
        dm_k = m_b_list[k] - m_b_list[k + 1]
        dm_k1 = m_b_list[k + 1] - m_b_list[k + 2]
        dm_tot = dm_k + dm_k1
        if dm_tot > 0:
            w0 = dm_k / dm_tot
            w1 = dm_k1 / dm_tot
        else:
            w0 = w1 = 0.5

        S_list[k] = w0 * S_list[k] + w1 * S_list[k + 1]
        Y_list[k] = w0 * Y_list[k] + w1 * Y_list[k + 1]
        Z_list[k] = w0 * Z_list[k] + w1 * Z_list[k + 1]
        f_rock_list[k] = w0 * f_rock_list[k] + w1 * f_rock_list[k + 1]
        chi_list[k] = w0 * chi_list[k] + w1 * chi_list[k + 1]
        T_list[k] = w0 * T_list[k] + w1 * T_list[k + 1]
        rho_list[k] = w0 * rho_list[k] + w1 * rho_list[k + 1]
        P_list[k] = w0 * P_list[k] + w1 * P_list[k + 1]

        # Remove cell k+1 and boundary k+1
        del S_list[k + 1]
        del Y_list[k + 1]
        del Z_list[k + 1]
        del f_rock_list[k + 1]
        del chi_list[k + 1]
        del T_list[k + 1]
        del rho_list[k + 1]
        del P_list[k + 1]
        del m_b_list[k + 1]  # remove the internal boundary

    # Convert back to arrays
    m_b_new = np.array(m_b_list)
    S_new = np.array(S_list)
    Y_new = np.array(Y_list)
    Z_new = np.array(Z_list)
    f_rock_new = np.array(f_rock_list)
    chi_new = np.array(chi_list)
    T_new = np.array(T_list)
    rho_new = np.array(rho_list)
    P_new = np.array(P_list)

    dm_new = -np.diff(m_b_new)

    # Verify we still have N cells
    assert len(S_new) == N, f"Cell count changed: {len(S_new)} != {N}"
    assert len(m_b_new) == N + 1, f"Boundary count changed: {len(m_b_new)} != {N + 1}"

    # Clamp
    Y_new = np.clip(Y_new, 0.0, 1.0)
    Z_new = np.clip(Z_new, 0.0, 1.0)
    f_rock_new = np.clip(f_rock_new, 0.0, 1.0)
    chi_new = np.clip(chi_new, 0.0, 1.0)

    # Interpolate boundary-centred quantities to new boundary positions
    r_b_new = _interp_boundary(r_b, m_b, m_b_new)
    D_b_new = _interp_boundary(D_b, m_b, m_b_new)
    DZ_b_new = _interp_boundary(DZ_b, m_b, m_b_new)

    # Ensure r_b positive and monotonically decreasing
    r_b_new = np.maximum(r_b_new, 1e-10)
    for i in range(1, len(r_b_new)):
        if r_b_new[i] >= r_b_new[i - 1]:
            r_b_new[i] = r_b_new[i - 1] * 0.9999

    # Interior quantities (N-1)
    Ym_new = _interp_interior(Ym_i, m_b, m_b_new)
    Z1m_new = _interp_interior(Z1m_i, m_b, m_b_new)
    Z2m_new = _interp_interior(Z2m_i, m_b, m_b_new)
    Nu_T_new = _interp_interior(Nu_T_eff, m_b, m_b_new)
    Nu_X_new = _interp_interior(Nu_X_eff, m_b, m_b_new)

    # kcore / kcore_fe: unchanged if no core
    # If there's a core, we need to adjust for the index shifts
    kcore_new = kcore
    kcore_fe_new = kcore_fe
    if kcore < N:
        # The core boundary mass hasn't changed, find its new index
        core_mass = m_b[kcore]
        kcore_new = np.argmin(np.abs(m_b_new - core_mass))
    if kcore_fe < N and kcore_fe > kcore:
        fe_mass = m_b[kcore_fe]
        kcore_fe_new = np.argmin(np.abs(m_b_new - fe_mass))

    # Count rain cells
    n_rain_old = np.sum((P >= p_lo) & (P <= p_hi))
    n_rain_new = np.sum((P_new >= p_lo) & (P_new <= p_hi))
    max_shift = np.max(np.abs(m_b_new - m_b) / M_planet)

    logger.info(
        "Regrid: %d splits, %d merges, max shift=%.2e M_p, "
        "rain cells %d → %d",
        len(split_cells), len(merge_cells), max_shift,
        n_rain_old, n_rain_new,
    )

    return (m_b_new, dm_new, S_new, Y_new, Z_new, f_rock_new,
            r_b_new, P_new, T_new, rho_new,
            D_b_new, DZ_b_new,
            Ym_new, Z1m_new, Z2m_new,
            chi_new, Nu_T_new, Nu_X_new,
            kcore_new, kcore_fe_new)


def _interp_boundary(f_old, m_b_old, m_b_new):
    """PCHIP interpolation for boundary-centred quantities (N+1)."""
    x_old = m_b_old[::-1]
    y_old = f_old[::-1]

    mask = np.concatenate(([True], np.diff(x_old) > 0))
    x_old = x_old[mask]
    y_old = y_old[mask]

    if len(x_old) < 2:
        return f_old.copy()

    interp = PchipInterpolator(x_old, y_old, extrapolate=True)
    x_new = m_b_new[::-1]
    f_new = interp(x_new)[::-1]

    f_new[0] = f_old[0]
    f_new[-1] = f_old[-1]
    return f_new


def _interp_interior(f_old, m_b_old, m_b_new):
    """PCHIP interpolation for interior-boundary quantities (N-1)."""
    N = len(f_old) + 1
    m_int_old = m_b_old[1:N]
    m_int_new = m_b_new[1:N]

    if len(m_int_old) < 2:
        return f_old.copy()

    x_old = m_int_old[::-1]
    y_old = f_old[::-1]

    mask = np.concatenate(([True], np.diff(x_old) > 0))
    x_old = x_old[mask]
    y_old = y_old[mask]

    if len(x_old) < 2:
        return f_old.copy()

    interp = PchipInterpolator(x_old, y_old, extrapolate=True)
    x_new = m_int_new[::-1]
    f_new = interp(x_new)[::-1]

    return f_new
