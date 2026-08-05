"""
Module: atm_bc.py — Atmosphere boundary conditions for planetary evolution.

Provides classes that map interior state variables (surface gravity, entropy,
helium fraction, metallicity) to boundary temperatures (T_int, T_eff) and
their derivatives (dTint/dS, dTint/dY, dTint/dZ). These derivatives couple
the atmosphere to the interior Newton-Raphson solver in transport.py.

Each class loads pre-computed atmosphere tables and builds interpolators for
forward evaluation (state -> T) and inverse root-finding (T -> state).

Available boundary condition models:
    | Class          | Config key   | Best for              | Method                   |
    |----------------|-------------|------------------------|--------------------------|
    | chen_atm       | c23, c26    | Jupiter, Saturn,       | 2D-4D RGI on tables     |
    |                |             | Super-Jupiters         |                          |
    | m21_atm        | m21         | T > 200 K gas giants   | 3D RGI (SONORA-BOBCAT)  |
    | p20_atm        | p20         | Gas giants             | 2D RGI (Phillips 2020)  |
    | f11_atm        | f11         | All solar system       | 3D RGI (Fortney 2011)   |
    | b97_atm        | b97         | Gas giants             | 2D RGI (Burrows 1997)   |
    | g75_atm        | g75         | Uranus, Neptune        | Analytical (Guillot 95) |
    | f07_atm        | f07         | Sub-Neptunes           | 4D RGI (Fortney 2007)   |
    | bare_atm       | bare        | Bare super-Earths      | Stefan-Boltzmann         |
    | rock_gray_atm  | gray        | Rocky w/ atmosphere    | Eddington semi-gray      |
    | ideal_gray_atm | ideal_gray  | Sub-Neptunes           | Dry adiabat + gray       |

Key interface methods (all classes provide):
    get_tint_teff(g, T_ref, Y, Z) -> (T_int, T_eff)
    get_dtintds(g, tint, Y, Z)    -> dTint/dS  (finite-difference)
    get_dtintdy(g, tint, S, Z)    -> dTint/dY  (finite-difference)

Developers:
    - Rob Tejada Arevalo (APPLE+ORCHARD)
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator as RGI
from scipy.optimize import root, root_scalar
from astropy import units as u
from astropy.constants import k_B
from astropy.constants import u as amu

import pandas as pd
from scipy.interpolate import interp1d, griddata
from utils import const
import re
import os
import glob
import logging

mp = amu.to('g') # grams
kb = k_B.to('erg/K') # ergs/K

erg_to_kbbar = (u.erg/u.Kelvin/u.gram).to(k_B/mp)

def _fill_grid_nan_gaps(grid):
    """Fill NaN gaps in a 2D grid: linear griddata first, then
    nearest-neighbor for the edge gaps linear cannot reach. No-op when the
    grid is already complete. Shared by _fill2d_from_scatter and the m21
    loader (previously verbatim copies). p20's loader deliberately uses an
    ungated, linear-only fill and is NOT routed through this helper — doing
    so would change its edge values.
    """
    if np.isnan(grid).any():
        mask = ~np.isnan(grid)
        pts = np.argwhere(mask)
        vals = grid[mask]
        all_pts = np.indices(grid.shape).reshape(2, -1).T

        lin = griddata(pts, vals, all_pts, method='linear').reshape(grid.shape)
        if np.isnan(lin).any():
            nn = griddata(pts, vals, all_pts, method='nearest').reshape(grid.shape)
            lin[np.isnan(lin)] = nn[np.isnan(lin)]
        grid = lin
    return grid


def _fill2d_from_scatter(tint_col, g_col, val_col, tint_axis, g_axis):
    """
    Convert scattered (Tint, g, value) data into a regular 2D grid with gap-filling.

    First maps known data points onto the grid using rounded-value dictionaries
    for fast index lookup, then fills NaN gaps using scipy.griddata (linear
    interpolation first, nearest-neighbor for remaining edge gaps).

    Parameters
    ----------
    tint_col, g_col, val_col : array-like
        Columns of (Tint, gravity, value) data from atmosphere tables.
    tint_axis, g_axis : array-like
        Regular grid axes for the output 2D array.

    Returns
    -------
    np.ndarray, shape (len(tint_axis), len(g_axis))
        Filled 2D grid with no NaN values (unless entire rows/columns are missing).
    """
    Nt, Ng = len(tint_axis), len(g_axis)
    grid = np.full((Nt, Ng), np.nan)

    tint_key = np.round(tint_axis, 12)
    g_key    = np.round(g_axis, 12)
    ti_map = {v: i for i, v in enumerate(tint_key)}
    gi_map = {v: j for j, v in enumerate(g_key)}

    t = np.round(tint_col.astype(float), 12)
    g = np.round(g_col.astype(float), 12)
    v = val_col.astype(float)

    for tt, gg, vv in zip(t, g, v):
        if tt in ti_map and gg in gi_map:
            grid[ti_map[tt], gi_map[gg]] = vv

    # Fill any holes (linear then nearest at edges)
    return _fill_grid_nan_gaps(grid)


class TintInversionError(Exception):
    """The S -> Tint inversion found no root in the bracket: the requested
    atmospheric entropy lies outside the table's invertible range at this
    (g, Y, Z).  During evolution this marks an invalid TRIAL state -- the
    loop rejects the step and retries at dt/2 (same treatment as an HSE
    NaN sentinel) -- while at initialization it still aborts loudly."""


class chen_atm:
    """
    Chen et al. 2023/2026 atmosphere boundary condition tables.

    Supports two table versions:
        c23: Jupiter/Saturn-specific tables with 4 configurations
             (cloud/irradiation combinations). 2D or 3D RGI on (g, Tint, Y).
        c26: Super-Jupiter general tables. Cloudless version supports 4D
             interpolation over (g, Tint, Y, Z) for 6 metallicity-helium
             combinations. Cloud version is 2D at fixed Y=0.275, Z=1 solar.

    Provides forward evaluation (state -> S_atm, T_eff) and inverse
    root-finding (S -> T_int) plus finite-difference derivatives for
    Newton-Raphson coupling to the transport solver.
    """
    def __init__(self, planet, bc_table, cloud=True, irradiation=True, table_shape=(9,7)):
        self.planet = planet
        self.table_shape = table_shape
        self.has_z = False
        self.bc_table = bc_table

        if self.bc_table == 'c23':

            if planet == 'Jupiter':

                if cloud and irradiation: # irradiated, cloud
                    y_25_dir = 'atmospheres/Jupiter_boundary/Jupiter_irradiated_3.16solar_table_ammocloud_size1_FACFLX05_Y025CMS.dat'
                    y_15_dir = 'atmospheres/Jupiter_boundary/Jupiter_irradiated_3.16solar_ammocloud_1micron_FACFLX05_Y015CMS.dat'

                    self.data_y25 = np.genfromtxt(y_25_dir, unpack=True)
                    self.data_y15 = np.genfromtxt(y_15_dir, unpack=True)

                    self.ys = [0.15, 0.25]
                    self.data = [self.data_y15, self.data_y25]

                elif cloud and not irradiation: # isolated, cloud
                    y_25_dir = 'atmospheres/Jupiter_boundary/Jupiter_isolated_3.16solar_ammocloud_1micron_Y25CMS.dat'
                    #y_25_dir = 'atmospheres/Jupiter_boundary/Jupiter_isolated_3.16solar_ammocloud_1micron.dat'
                    self.data_y25 = np.genfromtxt(y_25_dir, unpack=True)

                    self.ys=[0.25]
                    self.data = [self.data_y25]

                elif not cloud and irradiation: # irradiated, no cloud
                    y_25_dir = 'atmospheres/Jupiter_boundary/Jupiter_irradiated_3.16solar_FACFLX05_Y25CMS.dat'
                    #y_25_dir = 'atmospheres/Jupiter_boundary/Jupiter_irradiated_3.16solar_FACFLX05_Y25CMS.dat'
                    self.data_y25 = np.genfromtxt(y_25_dir, unpack=True)

                    self.ys=[0.25]
                    self.data = [self.data_y25]

                elif not (cloud or irradiation): # isolated, no cloud
                    if self.table_shape == (9, 7):
                        y_25_dir = 'atmospheres/Jupiter_boundary/Jupiter_isolated_3.16solar_table_Y025CMS.dat'
                    elif self.table_shape == (8, 10):
                        y_25_dir = 'atmospheres/Jupiter_boundary/Jupiter_isolated_3.16solar_table.dat'
                    else:
                        raise Exception('Only (9,7) or (8, 10) shapes available for Chen+23 BC tables.')
                    self.data_y25 = np.genfromtxt(y_25_dir, unpack=True)

                    self.ys=[0.25]
                    self.data = [self.data_y25]


            elif planet == 'Saturn':
                if cloud and irradiation:
                    y_25_dir = \
                        'atmospheres/Saturn_boundary/Saturn_irradiated_5solar_ammocloud_10micron_FACFLX05_Y25CMS.dat'
                    self.data_y25 = np.genfromtxt(y_25_dir, unpack=True)
                    y_15_dir = \
                        'atmospheres/Saturn_boundary/Saturn_irradiated_5solar_ammocloud_10micron_Y15CMS_new.dat'
                    self.data_y15 = np.genfromtxt(y_15_dir, unpack=True)
                    y_05_dir = \
                        'atmospheres/Saturn_boundary/Saturn_irradiated_5solar_ammocloud_10micron_FACFLX05_Y05CMS.dat'
                    self.data_y05 = np.genfromtxt(y_05_dir, unpack=True)

                    #else:
                    self.ys = [0.05, 0.15, 0.25]
                    self.data = [self.data_y05, self.data_y15, self.data_y25]

                elif cloud and not irradiation:
                    y_15_dir = 'atmospheres/Saturn_boundary/Saturn_isolated_5solar_ammocloud_10micron_Y15CMS.dat'
                    self.data_y15 = np.genfromtxt(y_15_dir, unpack=True)

                    self.ys = [0.15]
                    self.data = [self.data_y15]


                elif not cloud and irradiation:
                    y_15_dir = 'atmospheres/Saturn_boundary/Saturn_irradiated_5solar_Y15CMS_extend.dat'
                    self.data_y15 = np.genfromtxt(y_15_dir, unpack=True)

                    self.ys = [0.15]
                    self.data = [self.data_y15]

                elif not (cloud or irradiation):
                    y_15_dir = 'atmospheres/Saturn_boundary/Saturn_isolated_5solar_Y15CMS.dat'
                    self.data_y15 = np.genfromtxt(y_15_dir, unpack=True)

                    self.ys = [0.15]
                    self.data = [self.data_y15]

        elif self.bc_table == 'c26':
            self.cloud = cloud

            if cloud:
                # ----------------------------------------------------------
                # Ammonia-cloud table: single table at Y=0.275, Z=1 solar.
                # No Y-Z interpolation — build a 2D RGI on (g, tint) only.
                # ----------------------------------------------------------
                self.has_z = False
                cloud_file = 'atmospheres/Super_Jupiter_YZ/Super_Jupiter_Z1_Y275_ammocloud.dat'
                data = np.genfromtxt(cloud_file, unpack=True)

                tint_vals = np.unique(np.round(data[0].astype(float), 12))
                g_vals    = np.unique(np.round(data[1].astype(float), 12))
                self.tint_arr = np.sort(tint_vals)
                self.g_arr    = np.sort(g_vals)
                self.table_shape = (len(self.tint_arr), len(self.g_arr))

                S_2d    = _fill2d_from_scatter(data[0], data[1], data[2],
                                               self.tint_arr, self.g_arr)
                Teff_2d = _fill2d_from_scatter(data[0], data[1], data[3],
                                               self.tint_arr, self.g_arr)

                # RGI axes: (g, tint); values transposed from (Nt, Ng) -> (Ng, Nt)
                self.s_atm_rgi = RGI((self.g_arr, self.tint_arr),
                                     S_2d.T, method='linear',
                                     bounds_error=False, fill_value=None)
                self.teff_rgi  = RGI((self.g_arr, self.tint_arr),
                                     Teff_2d.T, method='linear',
                                     bounds_error=False, fill_value=None)
                return

            # ----------------------------------------------------------
            # Standard cloudless c26: 6-table Y-Z grid (unchanged)
            # ----------------------------------------------------------
            self.has_z = True

            # axes
            self.ys = np.array([0.15, 0.275], dtype=float)
            self.fzs = np.array([1.0, 3.16, 10.0], dtype=float)
            fz_conv_z = 0.017 / (1 - 0.017)  # solar Z=0.017; fz_conv_z is technically Z / (X + Y), or the relative fraction of metals to H/He
            self.zs = self.fzs * fz_conv_z / (1 + self.fzs * fz_conv_z) # converting to mass fraction from solar enhancement factor

            # filenames
            y_15_z_1_dir  = 'atmospheres/Super_Jupiter_YZ/Super_Jupiter_Z1.0_Y0.15.dat'
            y_15_z_3_dir  = 'atmospheres/Super_Jupiter_YZ/Super_Jupiter_Z3.16_Y0.15.dat'
            y_15_z_10_dir = 'atmospheres/Super_Jupiter_YZ/Super_Jupiter_Z10.0_Y0.15.dat'

            y_275_z_1_dir  = 'atmospheres/Super_Jupiter_YZ/Super_Jupiter_Z1.0_Y0.275.dat'
            y_275_z_3_dir  = 'atmospheres/Super_Jupiter_YZ/Super_Jupiter_Z3.16_Y0.275.dat'
            y_275_z_10_dir = 'atmospheres/Super_Jupiter_YZ/Super_Jupiter_Z10.0_Y0.275.dat'

            # load (each is 4 x N arrays: Tint, g(or log_g), S, Teff)
            self.data_y15_z1  = np.genfromtxt(y_15_z_1_dir,  unpack=True)
            self.data_y15_z3  = np.genfromtxt(y_15_z_3_dir,  unpack=True)
            self.data_y15_z10 = np.genfromtxt(y_15_z_10_dir, unpack=True)

            self.data_y275_z1  = np.genfromtxt(y_275_z_1_dir,  unpack=True)
            self.data_y275_z3  = np.genfromtxt(y_275_z_3_dir,  unpack=True)
            self.data_y275_z10 = np.genfromtxt(y_275_z_10_dir, unpack=True)

            # arrange as [iz][iy] to match (Z, Y)
            tables = [
                [self.data_y15_z1,  self.data_y275_z1],   # Z=1.0,   Y=0.15/0.275
                [self.data_y15_z3,  self.data_y275_z3],   # Z=3.16
                [self.data_y15_z10, self.data_y275_z10],  # Z=10.0
            ]

            # infer axes from the first table (robust to any table_shape mismatch)
            t0 = tables[0][0]
            # columns: 0=Tint, 1=g(or log_g), 2=S, 3=Teff
            tint_vals = np.unique(np.round(t0[0].astype(float), 12))
            g_vals    = np.unique(np.round(t0[1].astype(float), 12))

            self.tint_arr = np.sort(tint_vals)
            self.g_arr    = np.sort(g_vals)
            self.table_shape = (len(self.tint_arr), len(self.g_arr))

            Nz, Ny = len(self.zs), len(self.ys)
            Nt, Ng = self.table_shape

            # Build 4D grids: (Nz, Ny, Nt, Ng)
            self.S_grid    = np.empty((Nz, Ny, Nt, Ng), dtype=float)
            self.Teff_grid = np.empty((Nz, Ny, Nt, Ng), dtype=float)

            for iz in range(Nz):
                for iy in range(Ny):
                    tab = tables[iz][iy]
                    self.S_grid[iz, iy] = _fill2d_from_scatter(
                        tab[0], tab[1], tab[2], self.tint_arr, self.g_arr
                    )
                    self.Teff_grid[iz, iy] = _fill2d_from_scatter(
                        tab[0], tab[1], tab[3], self.tint_arr, self.g_arr
                    )

            # RGI expects values shaped as (Ng, Nt, Ny, Nz) for axes (g, tint, y, z)
            self.S_vals    = self.S_grid.transpose(3, 2, 1, 0)
            self.Teff_vals = self.Teff_grid.transpose(3, 2, 1, 0)

            self.s_atm_rgi = RGI((self.g_arr, self.tint_arr, self.ys, self.zs),
                                 self.S_vals, method='linear', bounds_error=False, fill_value=None)
            self.teff_rgi  = RGI((self.g_arr, self.tint_arr, self.ys, self.zs),
                                 self.Teff_vals, method='linear', bounds_error=False, fill_value=None)

            # IMPORTANT: we're done; skip the legacy 3D table-building below.
            return

        # ===== legacy 3D path (unchanged) for Jupiter/Saturn/Exoplanet etc =====
        self.tint_grid = []
        self.teff_grid = []
        self.g_grid = []
        self.s_grid = []

        for tables in self.data:
            if irradiation:
                self.tint_grid.append(tables[0].reshape(self.table_shape))
                self.teff_grid.append(tables[3].reshape(self.table_shape))
                self.g_grid.append(tables[1].reshape(self.table_shape))
                self.s_grid.append(tables[2].reshape(self.table_shape))
            else:
                self.tint_grid.append(tables[0].reshape(self.table_shape))
                self.teff_grid.append(tables[0].reshape(self.table_shape))
                self.g_grid.append(tables[1].reshape(self.table_shape))
                self.s_grid.append(tables[2].reshape(self.table_shape))

        self.g_arr = np.array(self.g_grid)[:, 0, :][0]
        self.tint_arr = np.array(self.tint_grid)[:, :, 0][0]

        self.s_atm_rgi = RGI((self.g_arr, self.tint_arr, self.ys), np.array(self.s_grid).T,
                             method='linear', bounds_error=False, fill_value=None)
        self.teff_rgi = RGI((self.g_arr, self.tint_arr, self.ys), np.array(self.teff_grid).T,
                            method='linear', bounds_error=False, fill_value=None)

    # -------------------
    # small helper
    # -------------------
    def _stack_points(self, *arrs):
        arrs = np.broadcast_arrays(*arrs)
        pts = np.column_stack([a.ravel() for a in arrs])
        return pts, arrs[0].shape

    def get_s_atm(self, g, tint, y_atm, z_atm=1.0):
        if getattr(self, 'cloud', False):
            if np.isscalar(g):
                return float(self.s_atm_rgi((g, tint)))
            return self.s_atm_rgi(np.array([g, tint]).T)
        elif self.has_z:
            if np.isscalar(g) and np.isscalar(tint) and np.isscalar(y_atm) and np.isscalar(z_atm):
                return float(self.s_atm_rgi((g, tint, y_atm, z_atm)))
            pts, shp = self._stack_points(g, tint, y_atm, z_atm)
            return self.s_atm_rgi(pts).reshape(shp)
        else:
            if np.isscalar(g):
                return float(self.s_atm_rgi((g, tint, y_atm)))
            return self.s_atm_rgi(np.array([g, tint, y_atm]).T)

    def get_teff(self, g, tint, y_atm, z_atm=1.0):
        if getattr(self, 'cloud', False):
            if np.isscalar(g):
                return float(self.teff_rgi((g, tint)))
            return self.teff_rgi(np.array([g, tint]).T)
        elif self.has_z:
            if np.isscalar(g) and np.isscalar(tint) and np.isscalar(y_atm) and np.isscalar(z_atm):
                return float(self.teff_rgi((g, tint, y_atm, z_atm)))
            pts, shp = self._stack_points(g, tint, y_atm, z_atm)
            return self.teff_rgi(pts).reshape(shp)
        else:
            if np.isscalar(g):
                return float(self.teff_rgi((g, tint, y_atm)))
            return self.teff_rgi(np.array([g, tint, y_atm]).T)

    # -------------------
    # S -> Tint inversion (now Z-aware)
    # -------------------
    def _tint_root(self, tint, sval, gval, yval, zval=1.0):
        return self.get_s_atm(gval, tint, yval, zval) - sval

    def get_tint(self, s, g, y, z=1.0, bracket=(0, 10000.0)):
        s, g, y, z = np.broadcast_arrays(s, g, y, z)
        out = np.empty_like(s, dtype=float)

        it = np.nditer([s, g, y, z, out],
                       flags=['refs_ok', 'multi_index'],
                       op_flags=[['readonly']]*4 + [['writeonly']])

        for s_, g_, y_, z_, tint_slot in it:
            try:
                sol = root_scalar(self._tint_root,
                                  args=(float(s_), float(g_), float(y_), float(z_)),
                                  bracket=bracket, method='brenth')
                tint_slot[...] = sol.root
            except Exception:
                raise TintInversionError(
                    "S->Tint inversion failed: no root in bracket %s for "
                    "S_atm=%.4f, logg=%.4f, Y=%.4f, Z=%.4f. State is outside "
                    "the atmosphere table's invertible range (at init: try a "
                    "lower initial entropy or adjust the bracket)."
                    % (bracket, float(s_), float(g_), float(y_), float(z_)))

        return out if out.size > 1 else out.item()

    def get_tint_teff(self, s, g, y, z=1.0):
        tint = self.get_tint(s, g, y, z=z)
        teff = self.get_teff(g, tint, y, z_atm=z if self.has_z else 1.0)
        return tint, teff

    # -------------------
    # finite-diff derivatives (now pass Z through where relevant)
    # -------------------
    def get_dtintds(self, g, tint, y_atm, z_atm=1.0, dt=0.1):
        T1 = tint*(1+dt)
        T2 = tint*(1-dt)
        S1 = self.get_s_atm(g, T1, y_atm, z_atm)
        S2 = self.get_s_atm(g, T2, y_atm, z_atm)
        return (T1 - T2)/(S1 - S2)

    def get_dtintdy(self, s, g, y, z=1.0, dy=0.05):
        if getattr(self, 'cloud', False):
            return 0.0
        T1 = self.get_tint(s, g, y*(1+dy), z=z)
        T2 = self.get_tint(s, g, y*(1-dy), z=z)
        return (T1 - T2)/(2*y*dy)

    def get_dtintdz(self, s, g, y, z=1.0, dz=0.05):
        """
        dtint/dz at fixed (Teff, g, y), using Teff inversion (Super_Jupiter only).
        """
        if getattr(self, 'cloud', False):
            return 0.0
        if not self.has_z:
            raise ValueError("dtintdz is only defined for Z-dependent (Super_Jupiter) tables.")
        z1 = z*(1+dz)
        z2 = z*(1-dz)
        T1 = self.get_tint(s, g, y, z=z1)
        T2 = self.get_tint(s, g, y, z=z2)
        return (T1 - T2)/(2*z*dz)
    
class m21_atm:
    """
    SONORA-BOBCAT atmosphere tables (Marley et al. 2021).

    Loads T_10 tables at three metallicities (M/H = -0.5, 0.0, +0.5),
    builds a 3D RGI on (logg, Teff, Z_atm), and provides T_10 -> Teff
    inversion via root-finding. Best for T > 200 K gas giants.
    """
    def __init__(self):
        """
        SONORA-BOBCAT atmosphere tables (Marley et al. 2021). Good for above 200 K.
        Loads all three metallicity tables (M/H = -0.5, 0.0, +0.5) and builds a
        3D interpolator T10(logg, Teff, Z_atm) to allow interpolation across metallicities.
        """
        files = {
            0.316: "atmospheres/sonora-0.5.dat",   # M/H = -0.5
            1.0:   "atmospheres/sonora+0.0.dat",    # M/H =  0.0
            3.16:  "atmospheres/sonora+0.5.dat",    # M/H = +0.5
        }
        self.z_atm_vals = np.array(sorted(files.keys()))

        # Collect the union of logg and Teff axes across all files
        all_logg = set()
        all_teff = set()
        raw_data = {}
        for z_key, fname in files.items():
            data = pd.read_csv(fname, sep=r"\s+", header=None, names=["Teff", "logg", "T10"])
            data = data.dropna()
            data["logg"] = data["logg"].round(5)
            data["Teff"] = data["Teff"].round(5)
            all_logg.update(data["logg"].unique())
            all_teff.update(data["Teff"].unique())
            raw_data[z_key] = data

        self.logg_vals = np.sort(np.array(list(all_logg)))
        self.teff_vals = np.sort(np.array(list(all_teff)))
        Ng, Nt, Nz = len(self.logg_vals), len(self.teff_vals), len(self.z_atm_vals)

        logg_to_i = {val: i for i, val in enumerate(self.logg_vals)}
        teff_to_j = {val: j for j, val in enumerate(self.teff_vals)}

        # Build 3D grid: (logg, Teff, Z_atm)
        T10_3d = np.full((Ng, Nt, Nz), np.nan)

        for iz, z_key in enumerate(self.z_atm_vals):
            data = raw_data[z_key]
            grid_2d = np.full((Ng, Nt), np.nan)
            for _, row in data.iterrows():
                if row["logg"] in logg_to_i and row["Teff"] in teff_to_j:
                    grid_2d[logg_to_i[row["logg"]], teff_to_j[row["Teff"]]] = row["T10"]

            # Fill NaN gaps: linear first, then nearest for edges
            grid_2d = _fill_grid_nan_gaps(grid_2d)

            T10_3d[:, :, iz] = grid_2d

        # Construct 3D interpolator: T10 = f(logg, Teff, Z_atm)
        self.interpolator = RGI(
            (self.logg_vals, self.teff_vals, self.z_atm_vals),
            T10_3d,
            bounds_error=False,
            fill_value=None
        )

    def get_teff(self, logg, t10_target, Z_atm=1.0, bracket=(0, 10000)):
        """
        Invert T10(logg, Teff, Z_atm) to return Teff such that T10(logg, Teff, Z_atm) == t10_target.
        Works for scalars or arrays.
        """
        logg, t10_target = np.broadcast_arrays(logg, t10_target)
        out = np.empty_like(t10_target, dtype=float)

        it = np.nditer([logg, t10_target, out],
                       flags=['refs_ok', 'multi_index'],
                       op_flags=[['readonly'], ['readonly'], ['writeonly']])

        z = float(Z_atm)
        for g, t10_val, teff_out in it:
            def func(teff):
                return self.interpolator((float(g), float(teff), z)) - float(t10_val)

            sol = root_scalar(func, bracket=bracket, method='brentq')
            teff_out[...] = sol.root

        return out if out.size > 1 else out.item()

    def get_tint_teff(self, logg, t10_target, Z_atm=1.0, bracket=(0, 10000)):
        teff = self.get_teff(logg, t10_target, Z_atm=Z_atm, bracket=bracket)
        tint = teff
        return tint, teff

    def get_dtintdt10(self, g_input, t10_input, Z_atm=1.0, dt=0.25):
        teff_plus = self.get_teff(g_input, t10_input + dt, Z_atm=Z_atm)
        teff_minus = self.get_teff(g_input, t10_input - dt, Z_atm=Z_atm)
        return (teff_plus - teff_minus) / (2 * dt)

    def get_dtintdz(self, g_input, t10_input, Z_atm=1.0, dz=0.05):
        """
        dTint/dZ at fixed (logg, T10), using finite differences in Z_atm.
        """
        z1 = Z_atm * (1 + dz)
        z2 = Z_atm * (1 - dz)
        T1 = self.get_teff(g_input, t10_input, Z_atm=z1)
        T2 = self.get_teff(g_input, t10_input, Z_atm=z2)
        return (T1 - T2) / (2 * Z_atm * dz)

class p20_atm:
    """
    Phillips et al. 2020 atmosphere tables.

    2D RGI on (logg, Teff) -> T_10. Similar to m21_atm but without
    metallicity dependence.
    """
    def __init__(self, filename="atmospheres/phillips2020_atmospheres/phillips2020_ATMO.txt"):
        """
        Initialize the TeffInverter by loading data and constructing T10(logg, Teff) interpolator.
        """
        # Load and clean data
        data = pd.read_csv(filename, sep=r"\s+", header=None, names=["Teff", "logg", "T10"])
        data = data.dropna()
        data["logg"] = data["logg"].round(5)
        data["Teff"] = data["Teff"].round(5)

        # Create axes
        self.logg_vals = np.sort(data["logg"].unique())
        self.teff_vals = np.sort(data["Teff"].unique())

        # Initialize grid
        T10_grid = np.full((len(self.logg_vals), len(self.teff_vals)), np.nan)
        logg_to_i = {val: i for i, val in enumerate(self.logg_vals)}
        teff_to_j = {val: j for j, val in enumerate(self.teff_vals)}

        # Populate grid
        for _, row in data.iterrows():
            i = logg_to_i[row["logg"]]
            j = teff_to_j[row["Teff"]]
            T10_grid[i, j] = row["T10"]

        # Fill missing values using linear interpolation
        valid_mask = ~np.isnan(T10_grid)
        known_points = np.argwhere(valid_mask)
        known_values = T10_grid[valid_mask]
        all_points = np.indices(T10_grid.shape).reshape(2, -1).T
        filled_values = griddata(known_points, known_values, all_points, method='linear')
        self.T10_grid_filled = filled_values.reshape(T10_grid.shape)

        # Construct interpolator: T10 = f(logg, Teff)
        self.interpolator = RGI(
            (self.logg_vals, self.teff_vals),
            self.T10_grid_filled,
            bounds_error=False,
            fill_value=None
        )

    def get_teff(self, logg, t10_target, bracket=(0, 10000)):
        """
        Invert T10(logg, Teff) to return Teff such that T10(logg, Teff) == t10_target.
        Works for scalars or arrays.
        """
        logg, t10_target = np.broadcast_arrays(logg, t10_target)
        out = np.empty_like(t10_target, dtype=float)

        it = np.nditer([logg, t10_target, out],
                       flags=['refs_ok', 'multi_index'],
                       op_flags=[['readonly'], ['readonly'], ['writeonly']])

        for g, t10_val, teff_out in it:
            def func(teff):
                return self.interpolator((float(g), float(teff))) - float(t10_val)

            sol = root_scalar(func, bracket=bracket, method='brentq')
            teff_out[...] = sol.root

        return out if out.size > 1 else out.item()
    
    def get_tint_teff(self, logg, t10_target, bracket=(0, 10000)):
        teff = self.get_teff(logg, t10_target, bracket=bracket)
        tint = teff
        return tint, teff

    def get_dtintdt10(self, g_input, t10_input, dt=0.25):
        teff_plus = self.get_teff(g_input, t10_input + dt)
        teff_minus = self.get_teff(g_input, t10_input - dt)
        return (teff_plus - teff_minus) / (2 * dt)

class f11_atm:
    """
    Fortney et al. 2011 atmosphere tables for Jupiter, Saturn, Uranus, Neptune.

    Loads T_eff and T_10 tables at multiple irradiation factors. Builds 3D RGI
    on (irradiation, logg, Tint) for gas giants, or 3D on (flux, g, Tint) for
    ice giants. Supports T_10 -> Tint inversion via root-finding.
    """
    def __init__(self, planet):
        self.planet = planet
        if self.planet == 'Jupiter':
            self._load_gas_giant_tables('atmospheres/fortney2011_jup.txt')
        elif self.planet == 'Saturn':
            self._load_gas_giant_tables('atmospheres/fortney2011_sat.txt')
        elif planet == 'Uranus':
            self._load_ice_giant_tables(flux_axis=np.array([1.0, 1.8]),
                                        col_labels=('1.0U', '1.8U'))
        elif planet == 'Neptune':
            self._load_ice_giant_tables(flux_axis=np.array([0.12, 1.0]),
                                        col_labels=('0.12N', '1.0N'))

    def _load_gas_giant_tables(self, filename):
        """Load a Fortney+2011 gas-giant table (Jupiter or Saturn).

        The two planets shared this ~55-line loader verbatim, differing only
        in the data file. Builds four 2-D (logg, Tint) RGIs at the 1.0x and
        0.7x irradiation factors plus two 3-D RGIs on the (age-like)
        irradiation axis [0, 4.56].
        """
        # reading data and cleaning it
        self.data = pd.read_csv(filename, sep='\t')
        self.data['logg'] = np.log10(self.data['Gravity'].ffill() * 100) # now in log10 of cm/s^2
        self.data = self.data.replace(' ... ', np.nan, regex=False)
        self.data = self.data.astype(float)


        self.teff_1 = []
        self.teff_07 = []
        self.t10_1 = []
        self.t10_07 = []

        self.t_int = []
        self.grav = []

        for name, group in self.data.groupby('logg'):
            group_ = group[group['T_eff, 1.0'].notnull()]
            T_int_ = np.array(group_['T_int'])
            T_eff_1 = np.array(group_['T_eff, 1.0'])
            T_10_1 = np.array(group_['T_10, 1.0'])

            T_eff_07 = np.array(group_['T_eff, 0.7'])
            T_10_07 = np.array(group_['T_10, 0.7'])

            # filling '...' with linear extrapolations to have complete data
            Teff_1_interp = interp1d(T_int_, T_eff_1, kind='linear', fill_value='extrapolate')
            T10_1_interp = interp1d(T_int_, T_10_1, kind='linear', fill_value='extrapolate')
            Teff_07_interp = interp1d(T_int_, T_eff_07, kind='linear', fill_value='extrapolate')
            T10_07_interp = interp1d(T_int_, T_10_07, kind='linear', fill_value='extrapolate')

            self.teff_1.append(Teff_1_interp(np.array(group['T_int'])))
            self.teff_07.append(Teff_07_interp(np.array(group['T_int'])))
            self.t10_1.append(T10_1_interp(np.array(group['T_int'])))
            self.t10_07.append(T10_07_interp(np.array(group['T_int'])))

            self.t_int.append(np.array(group['T_int']))
            self.grav.append(np.array(group['logg']))

        self.teff_1_rgi = RGI((np.array(self.grav)[:,0], np.array(self.t_int[0])), self.teff_1, method='linear',\
            bounds_error=False, fill_value=None)
        self.teff_07_rgi = RGI((np.array(self.grav)[:,0], np.array(self.t_int[0])), self.teff_07, method='linear',\
                         bounds_error=False, fill_value=None)
        self.t10_1_rgi = RGI((np.array(self.grav)[:,0], np.array(self.t_int[0])), self.t10_1, method='linear',\
                        bounds_error=False, fill_value=None)
        self.t10_07_rgi = RGI((np.array(self.grav)[:,0], np.array(self.t_int[0])), self.t10_07, method='linear',\
                         bounds_error=False, fill_value=None)

        self.t10_rgi = RGI((np.array([0, 4.56]), np.array(self.grav)[:,0], np.array(self.t_int[0])), [self.t10_07, self.t10_1], method='linear',\
                        bounds_error=False, fill_value=None)

        self.teff_rgi = RGI((np.array([0, 4.56]), np.array(self.grav)[:,0], np.array(self.t_int[0])), [self.teff_07, self.teff_1], method='linear',\
            bounds_error=False, fill_value=None)


    def _load_ice_giant_tables(self, flux_axis, col_labels):
        """Load the shared Fortney+2011 Uranus/Neptune table.

        Both ice giants read the SAME data file; they differ only in which
        insolation-flux pair they use — Uranus columns '1.0U'/'1.8U' on flux
        axis [1.0, 1.8], Neptune columns '0.12N'/'1.0N' on [0.12, 1.0]. The
        stack order of the grids follows col_labels, matching flux_axis.
        Builds three 3-D (flux, logg, Tint) RGIs.
        """
        self.data = pd.read_csv(
                    "atmospheres/fortney2011_uranus_neptune.txt",
                    sep='\t',
                    comment="#",
                    na_values=["", "NaN"])      # treat missing entries as real NaNs
        self.data["Gravity"] = (
                    self.data["Gravity"]
                    .ffill()                           # copy the preceding g-value downward
                    .astype(float) * 100               # m s⁻² → cm s⁻²
            ).pipe(np.log10) # log(g) in cm/s²

        self.data.drop('Unnamed: 14', axis=1, inplace=True)

        self.grav_vals = np.sort(self.data["Gravity"].unique())          # 6 gravities
        self.tint_vals = np.sort(self.data["T_int"].unique())            # 9 T_int values (27–217 K)

        # Build a (g, T_int) → value pivot for every column of interest
        twoD = {
            col: self.data.pivot(index="Gravity", columns="T_int", values=col)
                    .reindex(index=self.grav_vals, columns=self.tint_vals)
                    .to_numpy()
            for col in self.data.columns if col not in {"Gravity"}
        }

        g_axis    = self.grav_vals
        tint_axis = self.tint_vals
        lo, hi = col_labels

        T10_grid = np.stack([
                twoD[f"T_10, {lo}"],              # shape (6,9)
                twoD[f"T_10, {hi}"]               # shape (6,9)
            ], axis=0)                            # final shape (2,6,9)

        T1_grid = np.stack([
                twoD[f"T_1, {lo}"],               # shape (6,9)
                twoD[f"T_1, {hi}"]                # shape (6,9)
            ], axis=0)                            # final shape (2,6,9)

        Teff_grid = np.stack([
                twoD[f"T_eff, {lo}"],             # shape (6,9)
                twoD[f"T_eff, {hi}"]              # shape (6,9)
            ], axis=0)

        self.t10_rgi = RGI((flux_axis, g_axis, tint_axis), T10_grid,
                    method="linear", bounds_error=False, fill_value=None)

        self.t1_rgi = RGI((flux_axis, g_axis, tint_axis), T1_grid,
                    method="linear", bounds_error=False, fill_value=None)

        self.teff_rgi = RGI((flux_axis, g_axis, tint_axis), Teff_grid,
                    method="linear", bounds_error=False, fill_value=None)

    def get_teff(self, log_g, tint, flux=1.0):
        if np.isscalar(log_g):
            return self.teff_rgi(np.array([flux, log_g, tint]).T)[0]
        return self.teff_rgi(np.array([flux, log_g, tint]).T)

    def get_t10(self, log_g, tint, flux=1.0):
        if np.isscalar(log_g):
            return self.t10_rgi(np.array([flux, log_g, tint]).T)[0]
        return self.t10_rgi(np.array([np.full_like(log_g, flux), log_g, tint]).T)

    def _tint_root(self, tint, log_g, t10_target, flux):
        """
        Scalar residual for a single (log_g, flux) point.
        """
        return self.get_t10(log_g, tint, flux) - t10_target

    def get_tint(self, log_g, t10, flux=1.0, bracket=(0, 1000.0)):
        """
        Vectorised inversion: returns Tint such that
            T10(log_g, Tint, flux) == t10
        Works for scalars or arbitrary-shape arrays (broadcast rules).
        """
        flux, log_g, t10 = np.broadcast_arrays(flux, log_g, t10)
        out = np.empty_like(t10, dtype=float)

        # iterate over the broadcasted arrays with an nditer
        it = np.nditer([log_g, t10, flux, out],
                       flags=['refs_ok', 'multi_index'],
                       op_flags=[['readonly']]*3 + [['writeonly']])

        for g, t10_target, f, tint_slot in it:
            sol = root_scalar(self._tint_root,
                              args=(float(g), float(t10_target), float(f)),
                              bracket=bracket, method='brenth')
            tint_slot[...] = sol.root   # write into 'out'

        return out if out.size > 1 else out.item()

    def get_tint_teff(self, log_g, t10, flux=1.0):
        tint = self.get_tint(log_g, t10, flux)
        teff = self.get_teff(log_g, tint, flux)
        return tint, teff

    def get_dtintdt10(self, log_g, t10, flux=1.0, dt=0.1):

        #T0 = self.get_tint(log_g, t10, flux)
        T1 = self.get_tint(log_g, t10*(1+dt), flux)
        T2 = self.get_tint(log_g, t10*(1-dt), flux)

        return (T1 - T2)/(2*t10*dt)

class b97_atm:
    """
    Burrows et al. 1997 atmosphere models.

    Direct 2D lookup (logg, T_10) -> T_eff with no inversion needed.
    Loads Burrows data file with 31 gravity values x 501 temperature entries.
    """
    def __init__(self):
        col1, col2 = np.genfromtxt("atmospheres/Burrows_atms/Burrows_data.dat",unpack=True)

        gravity = col2[::502]
        Teffec = []
        T10 = []
        gr = []
        g_length = round(31)
        for i in range(g_length):
            Teffec.append(list(col1[i+1+i*501:502*(i+1)]))
            T10.append(list(col2[i+1+i*501:502*(i+1)]))
            gr.append(list(np.repeat(gravity[i],501)))

        self.log_g_grid = np.array(np.log10(gr))
        self.t10_grid = np.array(T10)
        self.teff_grid = np.array(Teffec)

        self.get_teff_rgi = RGI((self.log_g_grid[:,0], self.t10_grid[0]), self.teff_grid, method='linear',\
                bounds_error=False, fill_value=None)

    def get_teff(self, log_g, t10):
        if np.isscalar(log_g):
            return float(self.get_teff_rgi((log_g, t10)))
        else:
            return self.get_teff_rgi(np.array([log_g, t10]).T)

    def get_tint_teff(self, log_g, t10):
        tint = self.get_teff(log_g, t10)
        teff = self.get_teff(log_g, t10)
        return tint, teff

    def get_dtintdt10(self, log_g, t10, dt=0.05):

        T0 = self.get_teff(log_g, t10)
        T1 = self.get_teff(log_g, t10*(1+dt))
        T2 = self.get_teff(log_g, t10*(1-dt))

        return (T1 - T2)/(2*t10*dt)

class g75_atm:
    """
    Guillot 1995 analytical atmosphere model for Uranus and Neptune.

    Uses an analytical power-law relation between T_1bar, g, and T_eff:
        T_eff = (T_1bar * g^(1/6) / K)^(1/1.244)
        T_int = (T_eff^4 - T_eq^4)^(1/4)
    where K is a planet-specific constant derived from orbital parameters.
    No table loading required.
    """
    def __init__(self, planet):
        self.planet = planet

        if self.planet == 'Uranus':
            self.a = 19.191226393*u.au.to('cm')
            self.A = 0.3 # Uranus albedo
            self.Teq = (0.25*(1 - self.A) * (u.Rsun.to('cm')/self.a)**2 * 5776 ** 4)**(1/4)
            self.K = 1.481

        elif self.planet == 'Neptune':
            self.a = 30.06896348*u.au.to('cm')
            self.A = 0.29 # Neptune albedo
            self.Teq = (0.25*(1 - self.A) * (u.Rsun.to('cm')/self.a)**2 * 5776 ** 4)**(1/4)
            self.K = 1.451

        else:
            raise ValueError('Only Uranus and Neptune are supported for this atm model')

    def get_tint_teff(self, g, T1_bar):
        """Compute T_int and T_eff from surface gravity and T at 1 bar."""
        Teff = (1/self.K * T1_bar * g**(1/6))**(1/1.244)
        T_int = (Teff**4 - self.Teq**4) ** (1/4)

        if np.isnan(T_int):
            T_int = self.Teq

        return T_int, Teff

    def get_dtintdt1(self, g, T1_bar, dt=0.1):
        """dTint/dT_1bar via central finite differences."""
        T_eff_1 = (1/self.K * T1_bar*(1 - dt) * g**(1/6))**(1/1.244)
        T_eff_2 = (1/self.K * T1_bar*(1 + dt) * g**(1/6))**(1/1.244)

        T_int_1 = (T_eff_1**4 - self.Teq**4) ** (1/4)
        T_int_2 = (T_eff_2**4 - self.Teq**4) ** (1/4)

        if np.isnan(T_int_1):
            T_int_1 = self.Teq

        if np.isnan(T_int_2):
            T_int_2 = self.Teq

        return (T_int_2 - T_int_1)/(2*T1_bar*dt)

class f07_atm:
    """
    Fortney et al. 2007 metallicity-dependent atmosphere models.

    Loads pre-computed .npz tables and builds 4D RGI on (metallicity,
    incident_flux, logg, logT_1kbar) -> logT_int. Used primarily for
    sub-Neptune evolution with T_eq-driven irradiation.
    """
    def __init__(self, teq, A=0.3, atm_z=True):
        """
        teq : float or astropy.Quantity (K)
        A   : Bond albedo (0..1)
        atm_z : bool, use metallicity-dependent grid if True
        """
        self.atm_z = bool(atm_z)
        self.teq = float(teq)     # K
        self.A = float(A)

        # load tables
        if self.atm_z:
            self.data = np.load('atmospheres/modelAtmospheres_metallicity_yao_fortney.npz')
        else:
            self.data = np.load('atmospheres/modelAtmospheres_yaotang_forntey.npz')

        self.logT_1kbar = self.data['logT990']
        self.log_g_grid = self.data['logGravity']
        self.incident_flux_grid = self.data['flux']   # grid axis: flux (cgs)
        self.logT_int_grid = self.data['logTint']

        # build RGI -- ordering must match how you call it later:
        # axes = (metallicity, incident_flux, log_g, logT_1kbar)
        # NOTE: get_tint is responsible for clipping inputs to the grid before
        # calling the RGI. ~21% of grid cells contain sentinel logTint=-4054
        # values at unconverged corners, and the table's physical max is only
        # ~3.75 (T_int ~ 5600 K), so silent extrapolation produces nonsense.
        self.metallicity_grid = self.data['metallicity']
        self.logT_int_rgi = RGI(
            (self.metallicity_grid, self.incident_flux_grid, self.log_g_grid, self.logT_1kbar),
            self.logT_int_grid,
            method='linear', bounds_error=False, fill_value=None
        )

        # Physical max of logTint across non-sentinel cells (sentinels are <= -10).
        _phys = self.logT_int_grid[self.logT_int_grid > -10.0]
        self._logTint_phys_max = float(_phys.max()) if _phys.size else 4.0

        # Bounds used by get_tint to clamp inputs into the table.
        self._mh_lo, self._mh_hi = float(self.metallicity_grid.min()), float(self.metallicity_grid.max())
        self._lg_lo, self._lg_hi = float(self.log_g_grid.min()), float(self.log_g_grid.max())
        self._lt_lo, self._lt_hi = float(self.logT_1kbar.min()), float(self.logT_1kbar.max())
        self._oob_warned = False

        # compute stellar flux in cgs (erg / cm^2 / s)
        # absorbed/global-mean flux = 4 sigma Teq^4
        # incident (stellar) flux = absorbed/(1-A) = 4 sigma Teq^4 / (1 - A)
        # use astropy cgs value of sigma_sb to be explicit about units
        self.stellar_flux = 4.0 * const.sigma_sb * (self.teq ** 4) / (1.0 - self.A)

    def get_tint(self, log_g, logT_1kbar_, mh_):
        """
        log_g : scalar or array-like (same shape as logT_1kbar_ and mh_). Should match grid units.
        logT_1kbar_ : scalar or array-like (log T at 1kbar; same shape)
        mh_ : scalar or array-like (metallicity axis values; same shape)
        Returns: Tint in K (scalar if scalar inputs, array if array inputs)

        Inputs are clipped to the table grid bounds to avoid silent linear
        extrapolation of an RGI built with `bounds_error=False, fill_value=None`.
        Hot/young or strongly-irradiated planets routinely push logT_1kbar above
        the table's max of 4.0 (T = 1e4 K), and the unclamped RGI then returns
        unphysical T_int (e.g. >1e4 K), which blows up the surface BC Jacobian
        in transport.py via dL/dT_int ~ T_int^3.
        """
        mh_c = np.clip(mh_, self._mh_lo, self._mh_hi)
        lg_c = np.clip(log_g, self._lg_lo, self._lg_hi)
        lt_c = np.clip(logT_1kbar_, self._lt_lo, self._lt_hi)

        if not self._oob_warned:
            if (np.any(np.asarray(mh_) != mh_c) or
                np.any(np.asarray(log_g) != lg_c) or
                np.any(np.asarray(logT_1kbar_) != lt_c)):
                # Use the ORCHARD logger ('APPLE') so the message survives the
                # blanket warnings.filterwarnings("ignore") installed by parts
                # of the eos submodule. Logger goes to stderr + log file.
                logging.getLogger('APPLE').warning(
                    "f07_atm: input clipped to table grid "
                    "(metallicity in [%s, %s], log_g in [%s, %s], "
                    "logT_1kbar in [%s, %s]). T_int from f07 may be at the "
                    "edge of validity; consider bc_atm='gray'/'ideal_gray' "
                    "or a cooler S_ini.",
                    self._mh_lo, self._mh_hi,
                    self._lg_lo, self._lg_hi,
                    self._lt_lo, self._lt_hi,
                )
                self._oob_warned = True

        # quick scalar path (call RGI with a tuple)
        if np.isscalar(log_g) and np.isscalar(logT_1kbar_) and np.isscalar(mh_):
            val = self.logT_int_rgi((float(mh_c), float(self.stellar_flux), float(lg_c), float(lt_c)))
            val = min(float(val), self._logTint_phys_max)
            return float(10.0 ** val)

        # vectorized path -- broadcast all inputs to a common shape
        mh_a, flux_a, lg_a, lt_a = np.broadcast_arrays(mh_c, np.full_like(np.atleast_1d(mh_c), self.stellar_flux),
                                                       lg_c, lt_c)
        # flatten and stack as (N, 4) rows in the same axis order used by RGI
        pts = np.column_stack((mh_a.ravel(), flux_a.ravel(), lg_a.ravel(), lt_a.ravel()))
        out_logTint = np.minimum(self.logT_int_rgi(pts), self._logTint_phys_max)
        out_Tint = (10.0 ** np.array(out_logTint)).astype(float)
        return out_Tint.reshape(mh_a.shape)

    def get_tint_teff(self, log_g, logT_1kbar_, mh_):

        tint = self.get_tint(log_g, logT_1kbar_, mh_)
        teff = (tint ** 4 + self.teq **4) ** 0.25
        return tint, teff

    def get_dtintdt1000(self, log_g, logT_1kbar_, mh_, dt=0.1):

        T_1000 = 10 ** logT_1kbar_
        T_1000_1 = T_1000 * (1 - dt)
        T_1000_2 = T_1000 * (1 + dt)

        T_int_1 = self.get_tint(log_g, np.log10(T_1000_1), mh_)
        T_int_2 = self.get_tint(log_g, np.log10(T_1000_2), mh_)

        return (T_int_2 - T_int_1)/(T_1000_2 -T_1000_1)

class bare_atm:
    """
    Bare (airless / vanishingly thin) boundary condition.

    Convention matches your evolution.py:
      L_int = 4*pi*R^2*sigma*T_int^4

    We take:
      T_eff = T_surf
      T_int^4 = max(T_eff^4 - T_eq^4, 0)
    """

    def __init__(self, T_eq):
        self.Teq = float(T_eq)

    def get_intrinsic_flux(self, T_surf):
        """
        Intrinsic emergent flux (per unit area) in cgs:
            F_int = sigma * max(T_surf^4 - T_eq^4, 0)
        """
        T = np.asarray(T_surf, dtype=float)
        x = T**4 - self.Teq**4
        F_int = np.where(x > 0.0, const.sigma_sb * x, 0.0)
        return F_int.item() if F_int.size == 1 else F_int

    def get_tint_from_intrinsic_flux(self, F_int):
        """
        Convert intrinsic flux (cgs) to Tint via:
            F_int = sigma * Tint^4
        """
        F = np.asarray(F_int, dtype=float)
        Tint = np.zeros_like(F, dtype=float)
        mask = F > 0.0
        if np.any(mask):
            Tint[mask] = (F[mask] / const.sigma_sb) ** 0.25
        return Tint.item() if Tint.size == 1 else Tint

    def get_tint_teff(self, T_surf):
        T = np.asarray(T_surf, dtype=float)
        Teff = T
        F_int = np.asarray(self.get_intrinsic_flux(Teff), dtype=float)
        Tint = np.asarray(self.get_tint_from_intrinsic_flux(F_int), dtype=float)

        return (Tint.item(), Teff.item()) if Tint.size == 1 else (Tint, Teff)

class rock_gray_atm:
    """
    Semi-gray radiative atmosphere for rocky planets.

    Uses an Eddington-style relation at a matching pressure P_match:
        T_surf^4 = T_eq^4 + (3/4) * T_int^4 * (tau_match + 2/3)
        tau_match = kappa_ir * P_match / g

    Inputs:
      - T_surf: temperature at the top model cell [K]
      - g:      surface gravity [cm/s^2]

    Outputs:
      - T_int: intrinsic temperature used in L_int = 4*pi*R^2*sigma*T_int^4
      - T_eff: effective temperature including irradiation, (T_int^4 + T_eq^4)^(1/4)
    """

    def __init__(self, T_eq, kappa_ir=1e-3, P_match=1e6):
        self.Teq = float(T_eq)
        self.kappa_ir = float(kappa_ir)  # cm^2 / g
        self.P_match = float(P_match)    # dyn / cm^2
        if self.kappa_ir <= 0.0:
            raise ValueError("rock_gray_atm requires kappa_ir > 0.")
        if self.P_match <= 0.0:
            raise ValueError("rock_gray_atm requires P_match > 0.")

    def get_tint_teff(self, T_surf, g):
        T_arr = np.asarray(T_surf, dtype=float)
        intrinsic_term = np.maximum(T_arr**4 - self.Teq**4, 0.0)
        Tint = intrinsic_term ** 0.25
        Teff = (Tint**4 + self.Teq**4) ** 0.25

        if Tint.size == 1:
            return Tint.item(), Teff.item()
        return Tint, Teff

class ideal_gray_atm:
    """
    Ideal-gas dry-adiabat + gray photosphere boundary condition for rocky planets.

    Assumptions:
      - Matching temperature T_surf is at pressure P_match.
      - Atmosphere between P_match and photosphere follows a dry adiabat:
            T(P) = T_match * (P / P_match)^nabla_ad
        with nabla_ad = R_specific / c_p and R_specific = k_B / (mu * m_u).
      - Photosphere pressure in a gray atmosphere:
            P_ph = (2/3) * g / kappa_ir
      - Intrinsic flux emerges from the photosphere:
            F_int = sigma * max(T_ph^4 - T_eq^4, 0)
    """

    def __init__(self, T_eq, kappa_ir=1e-3, P_match=1e6, mu=28.0, cp=1.0e7):
        self.Teq = float(T_eq)
        self.kappa_ir = float(kappa_ir)  # cm^2 / g
        self.P_match = float(P_match)    # dyn / cm^2
        self.mu = float(mu)              # mean molecular weight in amu
        self.cp = float(cp)              # erg / (g K)

        if self.kappa_ir <= 0.0:
            raise ValueError("ideal_gray_atm requires kappa_ir > 0.")
        if self.P_match <= 0.0:
            raise ValueError("ideal_gray_atm requires P_match > 0.")
        if self.mu <= 0.0:
            raise ValueError("ideal_gray_atm requires mu > 0.")
        if self.cp <= 0.0:
            raise ValueError("ideal_gray_atm requires cp > 0.")

        self.r_specific = const.k / (self.mu * const.amu)
        if self.cp <= self.r_specific:
            raise ValueError("ideal_gray_atm requires cp > R_specific for a physical dry adiabat.")
        self.nabla_ad = self.r_specific / self.cp

    def get_photosphere_pressure(self, g):
        g_arr = np.asarray(g, dtype=float)
        p_ph = (2.0 / 3.0) * np.maximum(g_arr, 1e-30) / self.kappa_ir
        return p_ph.item() if p_ph.size == 1 else p_ph

    def get_t_photosphere(self, T_surf, g):
        T_arr = np.asarray(T_surf, dtype=float)
        g_arr = np.asarray(g, dtype=float)
        T_arr, g_arr = np.broadcast_arrays(T_arr, g_arr)

        p_ph = np.asarray(self.get_photosphere_pressure(g_arr), dtype=float)
        alpha = np.maximum(p_ph / self.P_match, 1e-30) ** self.nabla_ad
        T_ph = T_arr * alpha
        return T_ph.item() if T_ph.size == 1 else T_ph

    def get_intrinsic_flux(self, T_surf, g):
        T_ph = np.asarray(self.get_t_photosphere(T_surf, g), dtype=float)
        x = T_ph**4 - self.Teq**4
        F_int = np.where(x > 0.0, const.sigma_sb * x, 0.0)
        return F_int.item() if F_int.size == 1 else F_int

    def get_dintrinsic_flux_dtsurf(self, T_surf, g):
        T_arr = np.asarray(T_surf, dtype=float)
        g_arr = np.asarray(g, dtype=float)
        T_arr, g_arr = np.broadcast_arrays(T_arr, g_arr)

        p_ph = np.asarray(self.get_photosphere_pressure(g_arr), dtype=float)
        alpha = np.maximum(p_ph / self.P_match, 1e-30) ** self.nabla_ad
        T_ph = T_arr * alpha

        out = np.zeros_like(T_arr, dtype=float)
        mask = T_ph > self.Teq
        if np.any(mask):
            out[mask] = 4.0 * const.sigma_sb * (alpha[mask] ** 4) * T_arr[mask] ** 3

        return out.item() if out.size == 1 else out

    def get_tint_teff(self, T_surf, g):
        F_int = np.asarray(self.get_intrinsic_flux(T_surf, g), dtype=float)
        Tint = np.zeros_like(F_int, dtype=float)
        mask = F_int > 0.0
        if np.any(mask):
            Tint[mask] = (F_int[mask] / const.sigma_sb) ** 0.25
        Teff = (Tint**4 + self.Teq**4) ** 0.25
        if Tint.size == 1:
            return Tint.item(), Teff.item()
        return Tint, Teff


def create_atm(BC_atm, planet, *, cloud, irradiation, T_eq, bond_albedo,
               rock_gray_kappa_ir, rock_gray_p_match):
    """Construct the atmosphere boundary-condition object for `BC_atm`.

    Single factory shared by evolution.py and transport.py, replacing the
    twin 10-branch instantiation ladders those modules used to carry. Only
    the object for the selected BC is constructed (table loading happens in
    the class __init__, so building all ten would be wasteful). Branch order
    mirrors the historical ladders; an unknown BC_atm raises the same
    exception text the evaluation dispatch has always used.
    """
    if BC_atm in ('c23', 'c26'):
        return chen_atm(planet, bc_table=BC_atm, cloud=cloud, irradiation=irradiation)
    elif BC_atm == 'm21':
        return m21_atm()
    elif BC_atm == 'p20':
        return p20_atm()
    elif BC_atm == 'b97':
        return b97_atm()
    elif BC_atm == 'f11':
        return f11_atm(planet)
    elif BC_atm == 'g75':
        return g75_atm(planet)
    elif BC_atm == 'f07':
        return f07_atm(teq=T_eq, A=bond_albedo)
    elif BC_atm == 'gray':
        return rock_gray_atm(T_eq=T_eq, kappa_ir=rock_gray_kappa_ir,
                             P_match=rock_gray_p_match)
    elif BC_atm == 'ideal_gray':
        return ideal_gray_atm(T_eq=T_eq, kappa_ir=rock_gray_kappa_ir,
                              P_match=rock_gray_p_match)
    elif BC_atm == 'bare':
        return bare_atm(T_eq=T_eq)
    raise Exception('Atmospheric boundary condition name error.')