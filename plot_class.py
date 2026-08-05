"""
Module: plot_class.py — Visualization and analysis of ORCHARD planetary evolution models.

Provides the init_plot class for generating publication-quality multi-panel figures
of planetary interior structure, thermal evolution, and composition profiles. Supports
single-snapshot, multi-age comparison, and movie-frame generation modes.

Data loading is delegated to load_models.ModelSet, which reads text or HDF5 output
files and provides time-indexed profile access via AgeArray objects.

Plot modes:
    still_plot()              - Multi-panel snapshot at a single age (4, 6, or 9 panels)
    age_plot()                - Multi-age comparison with colormap encoding
    movie_evol_plots_by_age() - Sequential frames for animation (compiled via ffmpeg)
    debug_plot()              - 12-panel diagnostic for transport debugging
    mass_loss_plot()          - 4-panel atmospheric escape diagnostic

Standard panels include:
    - P-T diagram with silicate/iron melting curves and phase boundaries
    - Temperature, density, entropy vs. mass or radius coordinate
    - Composition profiles (Y, Z) with miscibility saturation curves
    - Radius and effective temperature evolution vs. age
    - Decomposed thermal fluxes (convective, radiative, semiconvective)
    - Brunt-Vaisala frequency and density ratio (1/R0) for stability analysis
    - Gravitational harmonics (J2, J4) for rotating models

Overlays:
    - He miscibility curves from misc/ module
    - Metal (rock/water) miscibility from Rogers or Gupta models
    - Silicate and iron melting curves from EOS modules
    - Observational constraints (Jupiter/Saturn radii, luminosities)
    - Solar System age markers (4.56 Gyr)

Developers:
    - Roberto Tejada Arevalo (APPLE+ORCHARD)

Affiliation:
    Department of Astrophysical Sciences, Princeton University
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from utils import const
import os
from astropy.constants import k_B
from astropy import units as u
from astropy.constants import u as amu
from matplotlib.ticker import MultipleLocator
kb = k_B.to('erg/K')
erg_to_kbbar = (u.erg/u.Kelvin/u.gram).to(k_B/amu)
kbbar_to_erg = 1/erg_to_kbbar
from scipy.interpolate import RegularGridInterpolator as RGI
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import AutoMinorLocator, LogLocator
from misc import misc
from scipy.interpolate import interp1d
import argparse
import pdb
import load_models
from atmospheres import spectra
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from utils.const import G
# from eos import mgsio3_JJ_eos, smooth
from misc import gupta_misc as gm
import shutil
from misc import gupta_misc_lowT as gm
from misc import gilmore_misc as rm

from eos import mgsio3_comb_eos
from eos import mg2sio4_aneos_eos
from eos import iron_comb_eos
from eos import gonzalez_iron_eos
from eos import aqua_core_eos

plt.style.use('default')
plt.rc('text', usetex=True)
plt.rc('font', family='serif',size=45)
plt.rc('axes', linewidth=3.5)
plt.rc('axes', labelsize=30)
plt.rc('xtick', labelsize=30, direction='in')
plt.rc('ytick', labelsize=30, direction='in')

plt.rc('xtick.major',size=15,pad=8)
plt.rc('xtick.minor',size=10,pad=8)
plt.rc('ytick.major',size=15)
plt.rc('ytick.minor',size=7)

plt.rcParams['axes.grid'] = True

mjup = 1.8981246e30
msat = 0.299409946*1.8981246e30

def get_arith_mean(x):
    return (x[1: ] + x[ :-1]) / 2
def get_geom_mean(x):
    return np.sqrt(x[1: ] * x[ :-1])

mgsio3_eos = mgsio3_comb_eos.MGSIO3_COMBINED_EOS()
# mgsio3_eos_JJ = mgsio3_JJ_eos.MGSIO3_JJ_EOS()
mg2sio4_eos = mg2sio4_aneos_eos.MG2SIO4_ANEOS_EOS()
h2o_eos = aqua_core_eos.AQUA_CORE_EOS()
D17_iron_eos = iron_comb_eos.Fe_COMBINED_EOS()
gonzalez_iron_eos = gonzalez_iron_eos.Fe_GONZALEZ_EOS()

class init_plot:
    """
    Main class for loading, organizing, and plotting ORCHARD model data.

    Data loading is delegated to load_models.ModelSet, which reads text or
    HDF5 output files and provides time-indexed profile access. This class
    then provides methods for generating multi-panel figures:

        still_plot()              - Snapshot at a single age (4, 6, or 9 panels)
        age_plot()                - Multi-age comparison with colormap encoding
        movie_evol_plots_by_age() - Sequential frames for animation
        debug_plot()              - 12-panel transport diagnostic
        mass_loss_plot()          - 4-panel atmospheric escape diagnostic

    Parameters
    ----------
    paths : list of str
        Paths to model directories or config .ini files.
    ym_plot : bool
        Overlay helium miscibility saturation curves on composition panels.
    zm_plot : bool
        Overlay metal miscibility curves (Rogers/Gupta) on P-T panels.
    plot_rock_phases : bool
        Overlay silicate/iron melting curves on temperature panels.
    plot_Js : bool
        Include gravitational harmonic (J2, J4) time-series panels.
    plot_fluxes : bool
        Include decomposed thermal flux panels.
    read_verbose : bool
        Load verbose diagnostic data (conductivities, gradients, derivatives).
    """

    def __init__(self,
                paths=[],
                ec=False,
                fit_mode=True,
                f_rock_profiles = False,
                ym_plot=False,
                zm_plot=False,
                plot_rock_phases=False,
                spectra=False,
                conv_zones=False,
                plot_Js=False,
                plot_fluxes=False,
                N_mode=False,
                read_verbose=None
                ):

        self.plot_Js = plot_Js
        self.fit_mode=fit_mode
        self.ym_plot = ym_plot
        self.zm_plot = zm_plot
        self.plot_rock_phases = plot_rock_phases
        self.f_rock_profiles = f_rock_profiles
        self.ec = ec
        self.spectra = spectra
        self.conv_zones = conv_zones
        self.plot_fluxes=plot_fluxes

        # Metadata containers (some are optional placeholders used by older plotting paths)
        self.misc_curve = []
        self.deltaT = []
        self.input_eos = []
        self.mcore_e = []
        self.m_factor = []
        self.mcore_fe_e = []
        self.mcore_fe = []
        self.planet = []
        self.mass_z = []
        self.fixed_y = []
        self.mcore = []
        self.Hr = []
        self.misc_pad = []
        self.R0 = []
        self.final_age = []

        self.mass_grid = []
        self.rad = []
        self.press = []
        self.rho = []
        self.temp = []
        self.S = []
        self.Y = []
        self.Z = []
        self.f_rock = []
        self.Ym = []
        self.Z1m = []
        self.Z2m = []
        #self.fluxes = []
        #self.pgaps = []
        self.conv_flux = []
        self.rad_flux = []
        self.semi_flux = []
        self.radiogenic_flux = []
        self.Lambda_cond_i = []
        self.Lambda_rad_i = []
        self.bracket_dsdr = []
        self.dsdy_prho = []
        self.dsdz_prho = []
        self.dsdy_pt = []
        self.dsdz_pt = []
        self.dsdr = []
        self.dydr = []
        self.dzdr = []
        self.brunt = []
        self.brunt_Prho = []
        self.brunt_PT = []
        self.brunt_transport = []
        self.R0_inv = []

        self.data_age = []
        self.data_tint = []
        self.data_teff = []
        self.data_I = []
        self.data_omega = []
        self.data_Y_atm = []
        self.data_Z_atm = []
        self.data_rad = []
        self.data_grav = []
        self.data_kcore = []
        self.data_kcore_fe = []

        self.J2 = []
        self.J2_TOF = []
        self.J6_TOF = []
        self.J4 = []
        self.J4_TOF = []
        self.oblateness = []
        self.oblateness_TOF = []
        self.moi = []
        self.Req = []
        self.Req_TOF = []
        self.Rpol = []
        self.Rpol_TOF = []
        self.Rvol = []
        self.Rvol_TOF = []

        self.deltaE = []
        self.dE_S = []
        self.energy = []
        self.mantle_phases = []

        self.luminosity = []

        self.Mdot = []
        self.M_env = []
        self.M_planet_current = []
        self.L_adv = []

        self.melt_fraction = []
        loader = load_models.ModelSet(paths=paths, verbose=False, read_verbose=read_verbose)

        # Keep plot_class attribute names/semantics, but source metadata from load_models.
        self.param_paths = list(loader.param_paths)
        self.model_metadata = list(loader.metadata)
        self.paths = list(loader.model_dirs)
        self.misc_curve = list(loader.misc_curve)
        self.deltaT = list(loader.deltaT)
        # getattr fallbacks keep this working if an older load_models module
        # (without the Z-misc metadata) is still loaded, e.g. in a notebook
        # kernel that reloaded plot_class but not load_models.
        _n = len(self.deltaT)
        self.zmisc_deltaT = list(getattr(loader, 'zmisc_deltaT', [None] * _n))
        self.zmisc_deltaP = list(getattr(loader, 'zmisc_deltaP', [0.0] * _n))
        self.zmisc_active = list(getattr(loader, 'zmisc_active', [False] * _n))
        self.input_eos = list(loader.input_eos)
        self.fixed_y = list(loader.fixed_y)
        self.misc_pad = list(loader.misc_pad)
        self.mcore_e = list(loader.mcore_e)
        self.mcore = list(loader.mcore)
        self.mcore_fe_e = list(loader.mcore_fe_e)
        self.mcore_fe = list(loader.mcore_fe)
        self.m_factor = list(loader.m_factor)
        self.planet = list(loader.planet)
        self.R0 = list(loader.R0)
        self.final_age = list(getattr(loader, "final_age", [None] * len(self.paths)))
        self.mantle_comp = list(getattr(loader, "mantle_comp", [None] * len(self.paths)))
        self.eos_mantle = list(getattr(loader, "eos_mantle", [None] * len(self.paths)))
        self.eos_core = list(getattr(loader, "eos_core", [None] * len(self.paths)))
        self.read_verbose_modes = list(getattr(loader, "read_verbose_modes", []))

        for d in loader:
            s_ref = np.asarray(d["S"])
            n_time = int(s_ref.shape[0]) if s_ref.ndim == 2 else None
            n_cell = int(s_ref.shape[-1]) if s_ref.ndim >= 1 else 1
            n_bound = int(np.asarray(d["mass_grid"]).shape[0])
            n_iface = max(n_cell - 1, 1)

            def _nan_time_profile(ncol):
                if n_time is None:
                    return np.full((ncol,), np.nan)
                return np.full((n_time, ncol), np.nan)

            self.mass_grid.append(d["mass_grid"])
            self.rad.append(d["rad"])
            self.press.append(d["press"])
            self.rho.append(d["rho"])
            self.temp.append(d["temp"])
            self.S.append(d["S"])
            self.Y.append(d["Y"])
            self.Z.append(d["Z"])

            self.Ym.append(d["Ym"] if d["Ym"] is not None else np.full_like(d["S"], np.nan))

            if self.zm_plot:
                if d["Z1m"] is not None: self.Z1m.append(d["Z1m"])
                if d["Z2m"] is not None: self.Z2m.append(d["Z2m"])

            if self.f_rock_profiles:
                if d["f_rock"] is not None: self.f_rock.append(d["f_rock"])

            self.conv_flux.append(d["conv_flux"] if d["conv_flux"] is not None else _nan_time_profile(n_bound))
            self.rad_flux.append(d["rad_flux"] if d["rad_flux"] is not None else _nan_time_profile(n_bound))
            self.semi_flux.append(d["semi_flux"] if d["semi_flux"] is not None else _nan_time_profile(n_bound))
            self.Lambda_cond_i.append(d["Lambda_cond_i"] if d["Lambda_cond_i"] is not None else _nan_time_profile(n_iface))
            self.Lambda_rad_i.append(d["Lambda_rad_i"] if d["Lambda_rad_i"] is not None else _nan_time_profile(n_iface))
            self.bracket_dsdr.append(d["bracket_dsdr"] if d["bracket_dsdr"] is not None else _nan_time_profile(n_bound))

            self.dsdy_prho.append(d["dsdy_prho"] if d["dsdy_prho"] is not None else _nan_time_profile(n_iface))
            self.dsdz_prho.append(d["dsdz_prho"] if d["dsdz_prho"] is not None else _nan_time_profile(n_iface))
            self.dsdy_pt.append(d["dsdy_pt"] if d["dsdy_pt"] is not None else _nan_time_profile(n_iface))
            self.dsdz_pt.append(d["dsdz_pt"] if d["dsdz_pt"] is not None else _nan_time_profile(n_iface))
            self.brunt.append(d["brunt"])
            self.R0_inv.append(d["R0_inv"])

            if self.plot_rock_phases:
                if d["mantle_phases"] is not None:
                    self.mantle_phases.append(d["mantle_phases"])
                self.melt_fraction.append(d["melt_fraction"])

            self.data_age.append(d["data_age"])
            self.data_tint.append(d["data_tint"])
            self.data_teff.append(d["data_teff"])
            self.data_rad.append(d["data_rad"])
            self.data_grav.append(d["data_grav"])
            self.data_omega.append(d["data_omega"])
            self.data_kcore.append(d.get("data_kcore"))
            self.data_kcore_fe.append(d.get("data_kcore_fe"))
            n_age_i = int(np.asarray(d["data_age"]).shape[0])
            self.data_Y_atm.append(d.get("data_Y_atm", np.full(n_age_i, np.nan, dtype=float)))
            self.data_Z_atm.append(d.get("data_Z_atm", np.full(n_age_i, np.nan, dtype=float)))
            self.luminosity.append(d["luminosity"])

            if self.plot_Js:
                n_age = int(np.asarray(d["data_age"]).shape[0])

                def _tof_series_or_nan(key):
                    arr = d.get(key)
                    if arr is None:
                        return np.full(n_age, np.nan, dtype=float)
                    arr_np = np.asarray(arr, dtype=float).reshape(-1)
                    if arr_np.size == n_age:
                        return arr_np
                    out = np.full(n_age, np.nan, dtype=float)
                    ncopy = min(n_age, arr_np.size)
                    if ncopy > 0:
                        out[:ncopy] = arr_np[:ncopy]
                    return out

                self.J2_TOF.append(_tof_series_or_nan("J2_TOF"))
                self.J4_TOF.append(_tof_series_or_nan("J4_TOF"))
                self.J6_TOF.append(_tof_series_or_nan("J6_TOF"))
                self.oblateness_TOF.append(_tof_series_or_nan("oblat_TOF"))
                self.Req_TOF.append(_tof_series_or_nan("Req_TOF"))
                self.Rpol_TOF.append(_tof_series_or_nan("Rpol_TOF"))
                self.Rvol_TOF.append(_tof_series_or_nan("Rvol_TOF"))
                self.data_I.append(_tof_series_or_nan("data_I"))

                # If this model was run with rotation = False (or rotation
                # never engaged in hydrostatic.py), data_omega is identically
                # zero and Theory-of-Figures was skipped, so Req_TOF here is
                # all-NaN.  In that spherical case the equatorial radius
                # equals the mean radius by definition, so substitute
                # data_rad in for Req_TOF (and the related polar / volumetric
                # ToF radii) so plot routines that key off Req_TOF for the
                # radius axis -- still_plot, age_plot, movie -- work
                # uniformly across rotating and non-rotating models without
                # any extra fallback logic.
                _omega_j = np.asarray(d["data_omega"], dtype=float).reshape(-1)
                if _omega_j.size == 0 or np.all(np.abs(_omega_j) < 1e-30):
                    _data_rad_j = np.asarray(d["data_rad"], dtype=float).reshape(-1)
                    if _data_rad_j.size == n_age:
                        self.Req_TOF[-1] = _data_rad_j.copy()
                        self.Rpol_TOF[-1] = _data_rad_j.copy()
                        self.Rvol_TOF[-1] = _data_rad_j.copy()

            if d["deltaE"] is not None: self.deltaE.append(d["deltaE"])
            if d["dE_S"]   is not None: self.dE_S.append(d["dE_S"])
            if d["energy"] is not None: self.energy.append(d["energy"])

            # Mass-loss data (optional)
            self.Mdot.append(d.get("Mdot"))
            self.M_env.append(d.get("M_env"))
            self.M_planet_current.append(d.get("M_planet_current"))
            self.L_adv.append(d.get("L_adv"))

        self.age_gyr = np.array(self.data_age[-1])*const.s_to_Gyr

        self.data_age = np.array(self.data_age, dtype=object)
        self.data_rad = np.array(self.data_rad, dtype=object)

        if N_mode:
            self.m_unit = [np.asarray(self.mass_grid[j] / self.mass_grid[j][0], dtype=float)
                        for j in range(len(self.paths))]
            self.m_mean = [np.asarray(get_arith_mean(self.m_unit[j]), dtype=float)
                        for j in range(len(self.paths))]

        else:
            self.m_unit = [np.asarray(self.mass_grid[j] / self.mass_grid[j][0], dtype=float)
                        for j in range(len(self.paths))]
            self.m_mean = [np.asarray(get_arith_mean(self.m_unit[j]), dtype=float)
                        for j in range(len(self.paths))]


        # Prefer kcore / kcore_fe from load_models (HDF5 metadata_json, or
        # fraction-resolved mass comparison inside the loader). Only fall back
        # to recomputing here when the loader did not provide a value.
        loader_kcore = list(getattr(loader, "kcore", []))
        loader_kcore_fe = list(getattr(loader, "kcore_fe", []))
        self.kcore = []
        self.kcore_fe = []
        for j in range(len(self.paths)):
            kc = loader_kcore[j] if j < len(loader_kcore) else None
            kcfe = loader_kcore_fe[j] if j < len(loader_kcore_fe) else None

            if kc is not None and int(kc) >= 0:
                self.kcore.append(int(kc))
                self.kcore_fe.append(int(kcfe) if (kcfe is not None and int(kcfe) >= 0) else -1)
                continue

            if self.mcore[j] is not None and self.mcore[j] > 0:
                mcore_cgs = self.mcore_e[j] * const.mearth
                mcore_fe_cgs = self.mcore_fe_e[j] * const.mearth

                idx_core = np.where(self.mass_grid[j] < mcore_cgs)[0]
                idx_fe   = np.where(self.mass_grid[j] < mcore_fe_cgs)[0]

                self.kcore.append(int(idx_core[0]) if idx_core.size else -1)
                self.kcore_fe.append(int(idx_fe[0]) if idx_fe.size else -1)
            else:
                self.kcore.append(-1)
                self.kcore_fe.append(-1)


        if len(self.paths) == 1:
            self.colors = ['k']
        elif len(self.paths) == 2:
            self.colors = ['royalblue', 'crimson']
        elif len(self.paths) == 3:
            self.colors = ['royalblue', 'forestgreen', 'crimson']
        else:
            self.colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][0:len(self.paths)]

        if self.R0[0] == 1.0:
            self.R0_header = 'Ledoux'
        elif self.R0[0] == 0.0:
            self.R0_header = 'Schwarzschild'


        self.intrinsic_flux = [self.luminosity[j] * 1e-7 / (4 * np.pi * (self.data_rad[j] * 1e-2)**2) for j in range(len(self.paths))] # W/m^2

    def silicate_meltcurve(self, P_GPa):
        """
        Returns the silicate melting curve in GPa and K.
        """
        P_GPa = np.array(P_GPa)
        T_si0 = 1831 # K

        return T_si0 * (P_GPa / 4.6 + 1) ** 0.33 # Belonoshko et al. 2005

    def iron_meltcurve(self, P_GPa):
        """
        Returns the iron melting curve in GPa and K.
        """
        P_GPa = np.array(P_GPa)
        T_fe0 = 1900 # K

        return T_fe0 * (P_GPa / 31.3 + 1) ** (1/1.99) # Zhang et al. 2015

    def energy_conservation(self, energy_arr):
        age = energy_arr[0]
        grav = energy_arr[2]
        u = energy_arr[1]
        loss = energy_arr[3]
        e = u - grav + loss - (u[0] - grav[0])
        return u, grav, loss, e,  age

    def _kcore_for_snapshot(self, static_val, series, i):
        """
        Per-snapshot envelope/core boundary index.

        kcore can migrate over time in some runs, so using the single
        static kcore cached at load time makes the Z=1 core-fill region
        in comp_ax look frozen across ages. Prefer the time-resolved
        kcore(t) series written by the evolution HDF5 saver; fall back to
        the static value for coreless models (sentinel <= 0) or outputs
        that don't carry the per-step series (e.g. text output, older runs).
        """
        if static_val is None or static_val <= 0 or series is None:
            return static_val
        arr = np.asarray(series)
        if i >= arr.size or i < -arr.size:
            return static_val
        val = arr[i]
        if not np.isfinite(val):
            return static_val
        return int(round(val))

    def still_plot(self, age,
                   output_file = 'test_plot',
                   labels = [],
                   leg_size=12,
                   save=False,
                   save_path=None,
                   plot_yz_time=False,
                   plot_water_misc=False,
                   rad_profiles=False,
                   ymisc_test = False,
                   rad_plot = False,
                   plot_he_rain = False,
                   panel_num=6,
                   rad_temp_plot=False,
                   movie_mode=False,
                   temp_ylim_kK=None,
                   log_age=False,
                   rad_target=None,
                   age_target=None,
                        ):

        def _time_len(arr):
            if arr is None:
                return None
            try:
                a = np.asarray(arr)
            except Exception:
                return None
            if a.ndim == 0:
                return None
            return a.shape[0]

        def _min_time_len(arrs):
            lens = [l for l in (_time_len(a) for a in arrs) if l is not None and l > 0]
            return min(lens) if lens else None

        age_floor_gyr = 1e-5

        def _age_for_plot(arr):
            a = np.asarray(arr, dtype=float)
            return np.maximum(a, age_floor_gyr) if log_age else a

        def _age_scalar_for_plot(val):
            v = float(val)
            return max(v, age_floor_gyr) if log_age else v

        if panel_num not in (4, 6, 9):
            raise ValueError(f"panel_num={panel_num} is not supported. Use 4, 6, or 9.")

        # Movie mode: larger than before but still smaller than publication,
        # with scaled-down font sizes to match.
        if movie_mode:
            _scale = 0.75
            _prev_rc = {k: plt.rcParams[k] for k in [
                'font.size', 'axes.labelsize', 'axes.linewidth',
                'xtick.labelsize', 'ytick.labelsize',
                'xtick.major.size', 'xtick.minor.size',
                'ytick.major.size', 'ytick.minor.size',
                'xtick.major.pad', 'xtick.minor.pad',
            ]}
            plt.rcParams.update({
                'font.size': 30,
                'axes.labelsize': 26,
                'axes.linewidth': 2.5,
                'xtick.labelsize': 24,
                'ytick.labelsize': 24,
                'xtick.major.size': 10,
                'xtick.minor.size': 6,
                'ytick.major.size': 10,
                'ytick.minor.size': 5,
                'xtick.major.pad': 5,
                'xtick.minor.pad': 5,
            })
            _marker_scale = 2.5  # enlarge scatter markers for movie
        else:
            _scale = 1.0
            _prev_rc = None
            _marker_scale = 1.0

        if panel_num == 9:
            fig = plt.figure(figsize=(35*_scale, 30*_scale))
            temp_log_ax = plt.subplot(3, 3, 1)
            flux_ax = plt.subplot(3, 3, 2)
            comp_ax = plt.subplot(3, 3, 3)
            temp_ax = plt.subplot(3, 3, 4)
            rad_ax = plt.subplot(3, 3, 5)
            teff_ax = plt.subplot(3, 3, 6)
            dens_ax = plt.subplot(3, 3, 7)
            brunt_ax = plt.subplot(3, 3, 8)
            miscellaneous_ax = plt.subplot(3, 3, 9)
            ax92 = miscellaneous_ax.twinx()
        elif panel_num == 6:
            fig = plt.figure(figsize=(35*_scale, 20*_scale))
            temp_log_ax = plt.subplot(2, 3, 1)
            comp_ax = plt.subplot(2, 3, 2)
            temp_ax = plt.subplot(2, 3, 3)
            rad_ax = plt.subplot(2, 3, 4)
            teff_ax = plt.subplot(2, 3, 5)
            dens_ax = plt.subplot(2, 3, 6)
            flux_ax = None
            brunt_ax = None
            miscellaneous_ax = None
            ax92 = None
        else:
            fig = plt.figure(figsize=(30*_scale, 20*_scale))
            comp_ax = plt.subplot(2, 2, 1)
            temp_ax = plt.subplot(2, 2, 2)
            rad_ax = plt.subplot(2, 2, 3)
            teff_ax = plt.subplot(2, 2, 4)
            temp_log_ax = None
            dens_ax = None
            flux_ax = None
            brunt_ax = None
            miscellaneous_ax = None
            ax92 = None

        entropy_ax = comp_ax.twinx()
        if self.plot_rock_phases:
            melt_frac_ax = temp_ax.twinx()
        lum_ax = teff_ax.twinx()
        yz_ax = rad_ax.twinx() if plot_yz_time else None
        if yz_ax is not None:
            yz_ax.grid(False)
        fig.subplots_adjust(wspace=0.33, hspace=0.24)

        # Movie mode: use tight_layout for faster save
        if movie_mode:
            fig.set_tight_layout(True)
        entropy_ax.yaxis.labelpad = 16
        lum_ax.yaxis.labelpad = 16

        linstyles = ['-']*len(self.paths)

        # colors = ['k', 'r', 'g', 'b', 'm']
        #colors = ['m', 'royalblue', 'forestgreen', 'darkorange', 'crimson', 'k']
        if len(self.paths) == 1:
            colors = ['k']
        else:
            colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][0:len(self.paths)]
        lw=4
        flux_color_handles = []

        rad_temp_ax = None   # twinx for radius profile (created on demand)
        max_temp_kK = 0.0

        planet_set = {self.planet[j] for j in range(len(self.paths))}

        _gas_giant_set = {'Jupiter', 'Saturn', 'Super_Jupiter'}
        _use_rjup = bool(planet_set) and planet_set.issubset(_gas_giant_set)
        _r_scale_ax = const.rjup if _use_rjup else const.rearth
        _r_ax_label = r'R$_{\rm Jup}$' if _use_rjup else r'R$_\oplus$'
        _rjup_per_rearth = const.rjup / const.rearth

        # Plot the model time-series tracks once per model (solid model colour)
        for j in range(len(self.paths)):
            times_gyr_raw = np.asarray(self.data_age[j]) * const.s_to_Gyr
            times_gyr = _age_for_plot(times_gyr_raw)

            if self.plot_Js and (j < len(self.Req_TOF)):
                req_series = np.asarray(self.Req_TOF[j], dtype=float)
                if req_series.size > 0 and np.any(np.isfinite(req_series)):
                    rad_series = req_series / _r_scale_ax
                else:
                    rad_series = np.asarray(self.data_rad[j], dtype=float) / _r_scale_ax
            else:
                rad_series = np.asarray(self.data_rad[j], dtype=float) / _r_scale_ax

            rad_ax.plot(times_gyr, rad_series, c=colors[j], lw=lw-1, ls='-', alpha=0.9, label=labels[j])
            if yz_ax is not None:
                y_series = np.asarray(self.data_Y_atm[j], dtype=float).reshape(-1)
                z_series = np.asarray(self.data_Z_atm[j], dtype=float).reshape(-1)
                ny = min(times_gyr.size, y_series.size)
                nz = min(times_gyr.size, z_series.size)
                if ny > 0:
                    yz_ax.plot(times_gyr[:ny], y_series[:ny], c=colors[j], lw=lw-1, ls='--', alpha=0.9, label='_nolegend_')
                if nz > 0:
                    yz_ax.plot(times_gyr[:nz], z_series[:nz], c=colors[j], lw=lw-1, ls=':', alpha=0.9, label='_nolegend_')
            teff_ax.plot(times_gyr, self.data_teff[j], c=colors[j], lw=lw-1, ls='-', alpha=0.9, label='_nolegend_')
            lum_ax.plot(times_gyr, self.luminosity[j], c=colors[j], lw=lw-1, ls='--', alpha=0.8, label='_nolegend_')

            # if (miscellaneous_ax is not None) and self.plot_Js and (j < len(self.J2_TOF)) and len(self.J2_TOF) > 0:
            #     if j < len(self.J2_TOF):
            #         miscellaneous_ax.plot(times_gyr, self.J2_TOF[j], color=colors[j], lw=lw-1, ls='-', alpha=0.9)
            #     if j < len(self.J4_TOF):
            #         ax92.plot(times_gyr, self.J4_TOF[j], color=colors[j], lw=lw-1, ls='-', alpha=0.9)

        for j in range(len(self.paths)):

            age_gyr_i = np.array(self.data_age[j])*const.s_to_Gyr

            if age == -1:
                i = -1

            elif age == 0:
                i = 0

            else:
                idx = np.where(age_gyr_i >= age)[0]
                i = idx[0] if idx.size else -1

            # skipping arrays that do not have complete time snapshots
            time_arrays = [
                self.data_age[j],
                self.rad[j],
                self.press[j],
                self.rho[j],
                self.temp[j],
                self.S[j],
                self.Y[j],
                self.Z[j],
                self.brunt[j],
                self.conv_flux[j],
                self.rad_flux[j],
                self.data_teff[j],
                self.data_rad[j],
            ]
            if self.plot_rock_phases and (j < len(self.melt_fraction)):
                time_arrays.append(self.melt_fraction[j])
            if self.plot_Js and hasattr(self, "Req_TOF") and (j < len(self.Req_TOF)):
                time_arrays.append(self.Req_TOF[j])

            min_len = _min_time_len(time_arrays)
            if min_len is None:
                print(f"Warning: no complete time snapshots for model index {j}; skipping.")
                continue
            if i < 0 or i >= min_len:
                i = min_len - 1

            age_gyr = age_gyr_i[i]
            age_gyr_plot = _age_scalar_for_plot(age_gyr)

            kcore = self._kcore_for_snapshot(
                self.kcore[j], self.data_kcore[j] if j < len(self.data_kcore) else None, i)
            kcore_fe = self._kcore_for_snapshot(
                self.kcore_fe[j], self.data_kcore_fe[j] if j < len(self.data_kcore_fe) else None, i)
            mass_norm = self.m_mean[j]
            rad_norm = self.rad[j][i]/const.rearth
            mass_b = self.m_unit[j]
            mass_i = mass_b[1:-1] # use for interface values
            S = self.S[j][i]
            logp = np.log10(self.press[j][i])
            press_Mbar = self.press[j][i]*1e-12
            press_GPa = self.press[j][i]*1e-10
            logrho = np.log10(self.rho[j][i])
            logt = np.log10(self.temp[j][i])
            Z = np.array(self.Z[j][i], dtype=float)
            if kcore > 0 and kcore < len(Z):
                Z[kcore-1:] = 1.0
            Y = self.Y[j][i][:kcore] # total Y fraction, not Y/(X + Y)

            if self.plot_rock_phases:
                melt_frac = self.melt_fraction[j][i]

            env_int_end = max(kcore - 1, 0)
            brunt = self.brunt[j][i][:env_int_end]

            convective_flux = self.conv_flux[j][i]
            semiconvective_flux = self.semi_flux[j][i]
            radiative_flux = self.rad_flux[j][i]

            omega_dyn = np.sqrt(G*self.mass_grid[j][0]/(self.rad[j][i][0]**3))

            max_temp_kK = max(max_temp_kK, float(np.nanmax((10**logt) * 1e-3)))

            if temp_log_ax is not None:
                temp_log_ax.loglog(press_GPa, 10 ** logt, color=colors[j], label=labels[j], lw=lw)

            temp_ax.plot(mass_norm, (10**(logt)) * 1e-3,color=colors[j], lw=lw, label=labels[j])

            if rad_temp_plot:
                if rad_temp_ax is None:
                    rad_temp_ax = temp_ax.twinx()
                    rad_temp_ax.set_ylabel(r'Normalized Radius [$R_P$]')
                    rad_temp_ax.set_ylim(0, 1)
                    temp_ax.tick_params(right=False, which='both')
                    rad_temp_ax.tick_params(left=False, which='both')
                _rrad = self.rad[j][i]
                _rp_raw = _rrad / _rrad[0]          # 1.0 at surface, ~0 at centre
                _mr_raw = self.m_mean[j]            # same grid, surface→centre
                # Build rad(M) interpolator (flip both to ascending order)
                _rad_of_m = interp1d(_mr_raw[::-1], _rp_raw[::-1],
                                     bounds_error=False,
                                     fill_value=(_rp_raw[-1], _rp_raw[0]))
                _m_fine = np.linspace(float(_mr_raw.min()), float(_mr_raw.max()), 500)
                rad_temp_ax.plot(_m_fine, _rad_of_m(_m_fine), color=colors[j],
                                 ls='--', lw=lw, alpha=0.4, label='_nolegend_')

            if plot_he_rain:
                pmisc, tmisc = misc.get_misc_p(logp[:kcore], Y/(1 - Z[:kcore]), misc=self.misc_curve[j],
                            delta_T=self.deltaT[j], pad=True)

                if temp_log_ax is not None:
                    pos = tmisc > 0
                    temp_log_ax.plot(pmisc[pos]*100, tmisc[pos]*1000, color=colors[j], ls='dotted', lw=lw+2)

                interp_pm = interp1d(
                            10 ** (logp[:kcore] - 12),   # P in Mbar (P/1e12 cgs)
                            mass_norm[:kcore],
                            kind='linear',
                            fill_value='extrapolate',
                        )

                mmisc = interp_pm(pmisc)
                temp_ax.plot(mmisc, tmisc, color=colors[j], ls='dotted', lw=lw+2)

                # Z-misc stand-in curve (misc_kind='hhe'): the H-He curve
                # evaluated with the Z-channel offsets and the local Z as the
                # composition. Same style family as the He curve but DASHED
                # so Z and Y curves are distinguishable.
                if getattr(self, 'zmisc_active', None) and self.zmisc_active[j]:
                    z_dT = self.zmisc_deltaT[j] if self.zmisc_deltaT[j] is not None else (self.deltaT[j] or 0.0)
                    z_dP = self.zmisc_deltaP[j] or 0.0  # Mbar
                    p_cgs = 10.0 ** logp[:kcore]
                    # The shifted curve is fully miscible below 1 Mbar + deltaP
                    zsel = p_cgs > (1.0 + z_dP) * 1e12
                    if np.count_nonzero(zsel) > 2:
                        logp_eff = np.log10(p_cgs[zsel] - z_dP * 1e12)
                        pmisc_z, tmisc_z = misc.get_misc_p(
                            logp_eff, Z[:kcore][zsel], misc=self.misc_curve[j],
                            delta_T=z_dT, pad=True)
                        if pmisc_z is not None:
                            pmisc_z = pmisc_z + z_dP  # back to local pressure [Mbar]
                            if temp_log_ax is not None:
                                posz = tmisc_z > 0
                                temp_log_ax.plot(pmisc_z[posz]*100, tmisc_z[posz]*1000,
                                                 color=colors[j], ls='--', lw=lw+2)
                            mmisc_z = interp_pm(pmisc_z)
                            temp_ax.plot(mmisc_z, tmisc_z, color=colors[j], ls='--', lw=lw+2)

            total_lum_out = self.luminosity[j]

            if flux_ax is not None:
                total_flux = convective_flux + radiative_flux + semiconvective_flux
                flux_ax.plot(mass_b, -convective_flux, color=colors[j], ls='--', lw=lw, alpha=0.3, label='_nolegend_')
                flux_ax.plot(mass_b, -radiative_flux, color=colors[j], ls='dotted', lw=lw, alpha=0.75, label='_nolegend_')
                flux_ax.plot(mass_b, -semiconvective_flux, color=colors[j], ls='dashdot', lw=lw+2, alpha=0.95, label='_nolegend_')
                total_flux_line, = flux_ax.plot(mass_b, -total_flux, color=colors[j], ls='-', lw=lw+5, label='_nolegend_')
                if j == 0:
                    flux_color_handles.append(total_flux_line)

            # Only build the P->m interpolator when we actually need it. This avoids
            # calling scipy.interp1d on empty arrays for bare-rock runs (kcore==0),
            # and also avoids extra work when plot_water_misc=False.
            interp_pm = None
            if plot_water_misc:
                # Requires at least 2 envelope points to construct an interpolator.
                if kcore >= 2:
                    interp_pm = interp1d(
                        10 ** (logp[:kcore] - 12),   # P in Mbar (P/1e12 cgs)
                        mass_norm[:kcore],
                        kind='linear',
                        fill_value='extrapolate',
                    )
                else:
                    # Bare rock (kcore==0) or a single-cell "envelope" (kcore==1):
                    # there is no meaningful H2-water miscibility curve to plot.
                    interp_pm = None
            if plot_water_misc and (interp_pm is not None):
                #valid_idx = (10**logt >= 750) & (10**logt <= 6000)
                if j == 0:
                    Pmisc, Tmisc, _, _, _ = gm.get_pmisc(10**logt, np.zeros(len(logt)), self.Z[j][i])

                    if temp_log_ax is not None:
                        temp_log_ax.plot(np.log10(Pmisc), np.log10(Tmisc), color='royalblue', ls='dotted', lw=lw+2, label=r'H$_2$-water miscibility')
                    mmisc = interp_pm(Pmisc*1e-2)
                    temp_ax.plot(mmisc, Tmisc*1e-3, color='royalblue', ls='dotted', lw=lw+2, label=r'H$_2$-water miscibility')

            if self.zm_plot:
                misc_colors1 = ['C0', 'C1', 'C4', 'b', 'm'] # for Z1m
                misc_colors2 = ['C0', 'C1', 'C5', 'm', 'cyan'] # for Z2m
                z1_misc = self.Z1m[j][i]
                z2_misc = self.Z2m[j][i]
                comp_ax.plot(mass_i, z1_misc, color=misc_colors1[j], ls='dashdot', lw=lw+5, alpha=0.5)
                comp_ax.plot(mass_i, z2_misc, color=misc_colors2[j], ls='dashdot', lw=lw, alpha=0.5)

            if self.plot_rock_phases:
                melt_frac_ax.plot(mass_norm, melt_frac, color=colors[j], lw=lw, label=labels[j], alpha=0.3, ls='--')

            if not rad_profiles:
                comp_ax.plot(mass_norm[:kcore], Y, color=colors[j], ls='dotted', lw=lw)
                comp_ax.plot(mass_norm, Z, color=colors[j], ls='--', lw=lw)

                if self.zm_plot:
                    entropy_ax.scatter(mass_norm, S, color=colors[j], s=150, label=labels[j])
                else:
                    entropy_ax.plot(mass_norm, S, color=colors[j], ls='-', lw=lw, label=labels[j])
                if self.f_rock_profiles:
                    comp_ax.plot(mass_norm[:kcore], self.f_rock[j][i][:kcore], color=colors[j], ls='dashdot', lw=lw)
            else:
                comp_ax.plot(rad_norm[:kcore], Y, color=colors[j], ls='dotted', lw=lw)
                comp_ax.plot(rad_norm, Z, color=colors[j], ls='--', lw=lw)

                entropy_ax.plot(rad_norm, S, color=colors[j], ls=linstyles[j], lw=lw, label=labels[j])


            if j == 0:

                comp_ax.plot(0,0, c='k', ls='-', label='$S$ Profle')
                comp_ax.plot(0,0, c='k', ls='dotted', label='$Y$ Profle')
                comp_ax.plot(0,0, c='k', ls='--', label='$Z$ Profle')
                comp_ax.legend(frameon=False, prop={'size':leg_size}, loc='upper left')

            entropy_ax.legend(frameon=False, prop={'size':leg_size}, loc='upper right')

            # Mark selected age on time-series axes with scatter markers
            if self.plot_Js:
                if self.planet[j] in ('Uranus', 'Neptune', 'Sub_Neptune', 'Super_Earth'):
                    rad_snap = self.Req_TOF[j][i] / _r_scale_ax
                else:
                    rad_snap = self.Req_TOF[j][i] / _r_scale_ax
                if miscellaneous_ax is not None:
                    if j < len(self.J2_TOF):
                        miscellaneous_ax.scatter(age_gyr_plot, self.J2_TOF[j][i], color=colors[j], s=150*_marker_scale, edgecolor='k', linewidth=0.6, zorder=4)
                    if j < len(self.J4_TOF):
                        ax92.scatter(age_gyr_plot, self.J4_TOF[j][i], color=colors[j], s=150*_marker_scale, edgecolor='k', linewidth=0.6, zorder=4)
                    if j < len(self.J6_TOF):
                        ax92.scatter(age_gyr_plot, self.J6_TOF[j][i], color=colors[j], s=150*_marker_scale, edgecolor='k', linewidth=0.6, zorder=4)
            else:
                if self.planet[j] in ('Uranus', 'Neptune', 'Sub_Neptune', 'Super_Earth'):
                    rad_snap = self.data_rad[j][i] / _r_scale_ax
                else:
                    rad_snap = self.data_rad[j][i] / _r_scale_ax
                if miscellaneous_ax is not None:
                    miscellaneous_ax.hist(mass_b, bins=50, color=colors[j], alpha=0.15, density=True, histtype='step')
                    ax92.plot(mass_norm, -np.diff(self.mass_grid[j]), color=colors[j], ls='-', lw=lw-1, alpha=0.7)

            rad_ax.scatter(age_gyr_plot, rad_snap, color=colors[j], s=110*_marker_scale, edgecolor='k', linewidth=0.6, zorder=4)
            if yz_ax is not None:
                y_series = np.asarray(self.data_Y_atm[j], dtype=float).reshape(-1)
                z_series = np.asarray(self.data_Z_atm[j], dtype=float).reshape(-1)
                if i < y_series.size and np.isfinite(y_series[i]):
                    yz_ax.scatter(age_gyr_plot, y_series[i], color=colors[j], s=110*_marker_scale, edgecolor='k', linewidth=0.6, zorder=4)
                if i < z_series.size and np.isfinite(z_series[i]):
                    yz_ax.scatter(age_gyr_plot, z_series[i], color=colors[j], s=110*_marker_scale, edgecolor='k', linewidth=0.6, zorder=4)
            teff_ax.scatter(age_gyr_plot, self.data_teff[j][i], color=colors[j], s=110*_marker_scale, edgecolor='k', linewidth=0.6, zorder=4)
            lum_ax.scatter(age_gyr_plot, total_lum_out[i], color=colors[j], s=90*_marker_scale, edgecolor='k', linewidth=0.6, zorder=4)

            if dens_ax is not None:
                dens_ax.plot(mass_norm, 10**logrho, color=colors[j], ls=linstyles[j], lw=lw, label=labels[j])
                dens_ax.legend(frameon=False, prop={'size':leg_size}, loc='upper right')


            if brunt_ax is not None:
                brunt_ax.plot(mass_i[:env_int_end], brunt/omega_dyn, color=colors[j], ls='-', lw=lw, alpha=0.9, label=labels[j])

            # if j == 0:
            #     ppv_phase = pd.read_csv('eos/rock_eos/phase_boundary_ppv_only.csv')
                # mgsio3_melt = pd.read_csv('eos/rock_eos/MgSiO3_melt_dong2025.csv')
                # fe_melt = pd.read_csv('eos/rock_eos/Fe_melt_dong2025.csv')

            #Tm_si = 6000 * (press_GPa / 140 ) ** 0.26 # Fei+ 2021
            #Tm_si2 = 6295 * (press_GPa / 140 ) ** 0.317
            _mantle_comp = self.mantle_comp[j] or 'mgsio3'
            _eos_mantle = self.eos_mantle[j] or 'comb'
            if _mantle_comp == 'mgsio3':
                # if _eos_mantle == 'JJ':
                #     Tm_si2 = mgsio3_eos_JJ.get_T_melt(press_GPa)
                #     melt_label = r'MgSiO$_3$ melt (JJ)'
                # else:
                Tm_si2 = mgsio3_eos.get_T_melt(press_GPa)
                melt_label = r'MgSiO$_3$ melt (comb)'
            elif _mantle_comp == 'mg2sio4':
                Tm_si2 = mg2sio4_eos.get_T_melt(press_GPa)
                melt_label = r'Mg$_2$SiO$_4$ melt'
            elif _mantle_comp == 'h2o':
                Tm_si2 = h2o_eos.get_T_melt(press_GPa)
                melt_label = r'H$_2$O freezing'

            _eos_core = self.eos_core[j] or 'D17_comb'
            if _eos_core == 'G23_comb':
                _iron_eos = gonzalez_iron_eos
            else:
                _iron_eos = D17_iron_eos
            Tm_fe = _iron_eos.get_T_melt(press_GPa)

            #Tm_fesi17 = 1500 * (1 + (press_GPa - 2.0) / (10.0+6.13)) ** (0.23+0.09) # Ezenwa et al. 2024

            mass_melt_interp = interp1d(press_GPa, mass_norm, kind='linear', fill_value='extrapolate')


            temp_h2_rock_misc = rm.get_tmisc(press_GPa[press_GPa < 200],
                                                np.zeros(len(press_GPa[press_GPa < 200])),
                                                self.Z[j][i][press_GPa < 200], Tc_mode='follow_melt')[0]
            #print(len(temp_h2_rock_misc), len(press_GPa[press_GPa < 200]))

            #ax1.plot(np.log10(self.P_GPa_SI), np.log10(self.T_K_SI), color='m', ls='-', lw=lw+2, label='Melting Curve')
            #ax1.plot(np.log10(self.P_GPa_IceX), np.log10(self.T_K_IceX), color='m', ls='dotted', lw=lw+2, label='IceX line')



            #ax1.plot(np.log10(ppv_phase['P(TPa)']*1000), np.log10(ppv_phase['T(K)']), color='brown', ls='--', lw=lw+2, label='PPV phase boundary')
            #ax1.plot(np.log10(press_GPa), np.log10(Tm_si), color='orange', ls='dashdot', lw=lw+2, label='MgSiO$_3$ PPV melt', alpha=0.5)
            #ax1.plot(np.log10(press_GPa), np.log10(Tm_fesi17), color='forestgreen', ls='dashdot', lw=lw+2, label='FeSi melt', alpha=0.5)
            #ax1.plot(np.log10(press_GPa), np.log10(Tm_fe2), color='red', ls='--', lw=lw+2, alpha=0.5)
            if self.zm_plot and (temp_log_ax is not None):
                temp_log_ax.plot(np.log10(press_GPa[press_GPa < 200]), np.log10(temp_h2_rock_misc), color='purple', ls='dashdot', lw=lw+2, label='H$_2$-Rock miscibility', alpha=0.75)

            #ax1.fill_between(np.log10(press_GPa[kcore:] ), np.log10(Tm_si_exp_lower), np.log10(Tm_si_exp_upper), color='orange', alpha=0.75)

            #ax3.plot(mass_melt_interp(press_GPa), Tm_si*1e-3, color='orange', ls='dashdot', lw=lw+2, label='MgSiO$_3$ PPV melt', alpha=0.5)
            if self.plot_rock_phases:
                if temp_log_ax is not None:
                    temp_log_ax.loglog(press_GPa, Tm_si2, color=colors[j], ls='--', lw=lw+2, alpha=0.5, label=melt_label)
                    temp_log_ax.loglog(press_GPa, Tm_fe, color=colors[j], ls='dotted', lw=lw+2, label='Fe melt', alpha=0.5)

                temp_ax.plot(mass_melt_interp(press_GPa), Tm_si2*1e-3, color=colors[j], ls='--', lw=lw+2, alpha=0.5, label=melt_label)
                temp_ax.plot(mass_melt_interp(press_GPa), Tm_fe*1e-3, color=colors[j], ls='dotted', lw=lw+2, label='Fe melt', alpha=0.5)
            #ax3.plot(mass_melt_interp(press_GPa), Tm_fesi17*1e-3, color='forestgreen', ls='dashdot', lw=lw+2, label='FeSi melt', alpha=0.5)
            #ax3.plot(mass_melt_interp(press_GPa), Tm_fe2*1e-3, color='red', ls='--', lw=lw+2, alpha=0.5)

            if self.zm_plot:
                temp_ax.plot(mass_melt_interp(press_GPa[press_GPa < 200]), temp_h2_rock_misc*1e-3, color='purple', ls='dashdot', lw=lw+2, label='H$_2$-Rock miscibility', alpha=0.75)

            if miscellaneous_ax is not None:
                if self.plot_Js:
                    _times_misc = _age_for_plot(np.asarray(self.data_age[j]) * const.s_to_Gyr)
                    miscellaneous_ax.plot(_times_misc, self.J2_TOF[j], color='forestgreen', lw=lw+1)
                    ax92.plot(_times_misc, self.J4_TOF[j], color='crimson', lw=lw+2, ls='dotted')
                    # J6 shares the J4 (right) axis: it lives near +34 for
                    # Jupiter, so the gas-giant ylim extends above zero.
                    ax92.plot(_times_misc, self.J6_TOF[j], color='darkorange', lw=lw+1, ls='dashdot')
                    miscellaneous_ax.plot(0,0, c='k', ls='-', label='$J_2$')
                    miscellaneous_ax.plot(0,0, c='k', ls='dotted', label='$J_4$')
                    miscellaneous_ax.plot(0,0, c='k', ls='dashdot', label='$J_6$')
                    miscellaneous_ax.legend(frameon=False, prop={'size':leg_size}, loc='upper right', ncol=3)
                else:
                    miscellaneous_ax.hist(mass_b, bins=50, color=colors[j], alpha=0.5, density=True, histtype='step', label=labels[j])
                    ax92.plot(mass_norm, -np.diff(self.mass_grid[j]), color=colors[j], ls='dashdot', lw=lw, )

        if temp_log_ax is not None:
            temp_log_ax.legend(frameon=False, prop={'size':leg_size}, loc='upper right', ncol=2)

        temp_ax.legend(frameon=False, prop={'size':leg_size}, loc='upper right', ncol=2)

        # Legends matching age_plot style
        comp_legend_handles = [
            Line2D([0], [0], c='k', ls='-', lw=lw, label='$S$ Profle'),
            Line2D([0], [0], c='k', ls='dotted', lw=lw, label='$Y$ Profle'),
            Line2D([0], [0], c='k', ls='--', lw=lw, label='$Z$ Profle'),
        ]
        comp_ax.legend(handles=comp_legend_handles, frameon=False, prop={'size':leg_size}, loc='upper left')

        if self.plot_rock_phases:
            melt_frac_ax.set_ylabel('Melt Fraction', fontsize=18 if movie_mode else 20)
            melt_frac_ax.set_ylim(0, 1.0)
            melt_frac_ax.set_yticks(np.arange(0, 1.1, 0.1))
            melt_frac_ax.grid(False)

        # ---------- Axis formatting / reference lines ----------
        def _final_age_or_saved(jj):
            if jj < len(self.final_age):
                try:
                    fa = float(self.final_age[jj])
                    if np.isfinite(fa) and fa > 0:
                        return fa
                except (TypeError, ValueError):
                    pass
            age_series = np.asarray(self.data_age[jj], dtype=float) * const.s_to_Gyr
            if age_series.size == 0:
                return age_floor_gyr
            return float(np.nanmax(age_series))

        max_age_x = max(_final_age_or_saved(jj) for jj in range(len(self.paths)))
        x_left = age_floor_gyr if log_age else -0.1
        x_right = max(max_age_x + max_age_x * 0.1, age_floor_gyr * 10.0) if log_age else max_age_x + max_age_x * 0.1

        if rad_target is not None and age_target is not None:
            rt = np.asarray(rad_target)
            at = np.asarray(age_target)
            if rt.ndim == 1:
                rt = rt.reshape(1, -1)
                at = at.reshape(1, -1)
            for ti in range(len(rt)):
                ci = colors[ti] if ti < len(colors) else colors[0]
                rad_ax.errorbar(at[ti][0], rt[ti][0],
                                xerr=[[at[ti][1]], [at[ti][2]]],
                                yerr=[[rt[ti][1]], [rt[ti][2]]],
                                fmt='o', color='none', markersize=10*_marker_scale,
                                markeredgecolor=ci, markeredgewidth=1.5,
                                ecolor='r', elinewidth=2, capsize=4, zorder=5)

        _rad_ylims = {'Jupiter': (9, 14), 'Saturn': (8, 14),
                       'Uranus': (1.0, 6.0), 'Neptune': (1.0, 6.0),
                       'Sub_Neptune': (1.0, 6.0), 'Super_Earth': (0.0, 5.0),
                       'Super_Jupiter': (5, 20), 'Exoplanet': (5, 20)}
        rad_ylo = min(_rad_ylims.get(p, (0, 4))[0] for p in planet_set)
        rad_yhi = max(_rad_ylims.get(p, (0, 4))[1] for p in planet_set)
        if _use_rjup:
            rad_ylo = rad_ylo / _rjup_per_rearth
            rad_yhi = rad_yhi / _rjup_per_rearth
        rad_ax.set_ylim(rad_ylo, rad_yhi)
        for _p in planet_set:
            if _p == 'Jupiter':
                _jup_rmean = 10.973 / _rjup_per_rearth if _use_rjup else 10.973
                _jup_req = 11.209 / _rjup_per_rearth if _use_rjup else 11.209
                rad_ax.axhline(y=_jup_rmean, c='r', ls='dashdot')
                rad_ax.axhline(y=_jup_req, c='k', ls='dashdot', zorder=1)
                lum_ax.axhline(y=(7.485+0.160) * 1e3 * 4 * np.pi * (const.rjup)**2, ls='--', c='m', lw=3)
                lum_ax.axhline(y=(5.444-0.425) * 1e3 * 4 * np.pi * (const.rjup)**2, ls='--', c='m', lw=3)
            elif _p == 'Saturn':
                _sat_rmean = 9.1402 / _rjup_per_rearth if _use_rjup else 9.1402
                _sat_req = 9.449 / _rjup_per_rearth if _use_rjup else 9.449
                rad_ax.axhline(y=_sat_rmean, c='r', ls='dashdot')
                rad_ax.axhline(y=_sat_req, c='k', ls='dashdot')
                lum_ax.axhline(y = 4.952 * 1e3 * 4 * np.pi * (const.rsat)**2, ls='--', c='m', lw=3)
            elif _p == 'Uranus':
                rad_ax.axhline(y=4.0, c='r', ls='dashdot')
                rad_ax.axhline(y=4.007, c='k', ls='dashdot')
                lum_ax.axhline(y=0.042 * 1e3 * 4 * np.pi * (4.0 * const.rearth)**2, ls='--', c='m', lw=3)
                lum_ax.axhline(y=0.089 * 1e3 * 4 * np.pi * (4.0 * const.rearth)**2, ls='--', c='m', lw=3)
            elif _p == 'Neptune':
                rad_ax.axhline(y=3.883, c='r', ls='dashdot')
                lum_ax.axhline(y=0.433 * 1e3 * 4 * np.pi * (3.883 * const.rearth)**2, ls='--', c='m', lw=3)

        rad_ax.set_xlim(x_left, x_right)
        if log_age:
            rad_ax.set_xscale('log')
        rad_ax.axvline(x = 4.56, color='k', ls='dashdot')
        rad_ax.set_ylabel(r'Mean Radius [' + _r_ax_label + r']')
        rad_ax.set_xlabel('Age [Gyr]')
        _rad_model_legend = rad_ax.legend(frameon=False, prop={'size':leg_size}, loc='upper right')
        if yz_ax is not None:
            yz_ax.set_xlim(x_left, x_right)
            if log_age:
                yz_ax.set_xscale('log')
            yz_ax.set_ylim(0.0, 0.3)
            yz_ax.set_ylabel(r'Atmosphere $Y,\ Z$')
            rad_ax.add_artist(_rad_model_legend)
            yz_ax.legend(
                handles=[
                    Line2D([0], [0], color='k', lw=lw, ls='-', label='Radius'),
                    Line2D([0], [0], color='k', lw=lw, ls='--', label='Y$_{\\rm atm}$'),
                    Line2D([0], [0], color='k', lw=lw, ls=':', label='Z$_{\\rm atm}$'),
                ],
                frameon=False,
                prop={'size': leg_size},
                loc='lower right',
            )

        teff_ax.set_ylim(0, 1000)
        teff_ax.set_xlim(0, x_right)
        if log_age:
            teff_ax.set_xscale('log')
            lum_ax.set_xscale('log')
        teff_ax.set_ylabel(r'T$_{\rm eff}$ [Kelvin]')
        teff_style_legend = [
            Line2D([0], [0], color='k', lw=lw, ls='-', label=r'$T_{\rm eff}$'),
            Line2D([0], [0], color='k', lw=lw, ls='--', label='Luminosity'),
        ]
        teff_ax.legend(handles=teff_style_legend, frameon=False, prop={'size':leg_size}, loc='upper right')

        lum_ax.set_yscale('log')
        lum_series = [
            np.asarray(self.luminosity[idx], dtype=float).ravel()
            for idx in range(len(self.paths))
            if np.size(self.luminosity[idx])
        ]
        lum_all = np.concatenate(lum_series) if lum_series else np.array([])
        lum_all = lum_all[np.isfinite(lum_all) & (lum_all > 0)]
        if lum_all.size:
            lum_min = float(np.nanmin(lum_all))
            lum_max = float(np.nanmax(lum_all))
            _lum_refs = []
            for _p in planet_set:
                if _p == 'Uranus':
                    _lum_refs.append(0.089 * 1e3 * 4 * np.pi * (4.0 * const.rearth)**2)
                elif _p == 'Neptune':
                    _lum_refs.append(0.433 * 1e3 * 4 * np.pi * (3.883 * const.rearth)**2)
            if _lum_refs:
                lum_min = min(lum_min, min(_lum_refs))
            lum_ax.set_ylim(max(lum_min * 1e-2, 1e-40), lum_max * 1.2)
        lum_ax.set_ylabel('Luminosity [erg/s]')
        lum_ax.grid(False)

        for _p in planet_set:
            if _p == 'Jupiter':
                teff_ax.axhline(y=125.6, color='r', ls='dashdot')
            elif _p == 'Saturn':
                teff_ax.axhline(y=96.5, color='r', ls='dashdot')
            elif _p == 'Uranus':
                teff_ax.axhline(y=59.1, color='r', ls='dashdot')
            elif _p == 'Neptune':
                teff_ax.axhline(y=59.3, color='r', ls='dashdot')
        teff_ax.axvline(x = 4.56, color='k', ls='dashdot')
        teff_ax.set_xlabel('Age [Gyr]')

        _age_fontsize = 25 if movie_mode else 25
        if temp_log_ax is not None:
            temp_log_ax.text(0.05, 0.15, f'Age: {age_gyr:.5f} Gyr', transform=temp_log_ax.transAxes,
                         ha='left', va='top', fontsize=_age_fontsize)
        else:
            temp_ax.text(0.03, 0.97, f'Age: {age_gyr:.5f} Gyr', transform=temp_ax.transAxes,
                         ha='left', va='top', fontsize=_age_fontsize)

        # Use fixed temperature y-limits if provided (e.g. for movies)
        _tlim_kK = temp_ylim_kK if temp_ylim_kK is not None else max_temp_kK

        if 'Jupiter' in planet_set:
            _jup_idx_still = next(jj for jj in range(len(self.paths)) if self.planet[jj] == 'Jupiter')
            comp_ax.fill_between(x=self.m_mean[_jup_idx_still], y1=(0.238+0.005)*(1-0.017), y2=(0.238-0.005)*(1-0.017), color='m', alpha=0.25)
            comp_ax.axhline(y = 3.16 * 0.017, c='b', ls='dashdot')

        if temp_log_ax is not None:
            temp_log_ax.set_xlabel(r'Pressure [GPa]')
            temp_log_ax.set_ylabel(r'Temperature [K]')
            temp_log_ax.set_xlim(1e-4, 1e4)
            temp_log_ax.set_ylim(100, max(_tlim_kK + 2.0, 2.0) * 1e3)
            temp_log_ax.invert_xaxis()

        if flux_ax is not None:
            flux_ax.set_xlabel('Mass [M$_P$]')
            flux_ax.set_ylabel('Fluxes')
            flux_ax.set_ylim(1e16,1e27)
            flux_ax.set_yscale('symlog')
            flux_ax.set_xlim(0.0, 1.01)

            if flux_color_handles:
                flux_style_legend = [
                    Line2D([0], [0], color='k', lw=lw, ls='-', label='Total Flux'),
                    Line2D([0], [0], color='k', lw=lw, ls='--', label='Conv. flux'),
                    Line2D([0], [0], color='k', lw=lw, ls='dotted', label='Cond+Rad. flux'),
                    Line2D([0], [0], color='k', lw=lw, ls='dashdot', label='Semi-conv. flux'),
                ]
                flux_ax.legend(
                    handles=flux_style_legend,
                    frameon=False,
                    prop={'size': leg_size},
                    loc='upper left',
                )

            if self.zm_plot:
                flux_ax.set_xlim(mass_norm[kcore+20], 1.0005)

        temp_ax.set_xlabel('Normalized Mass [M$_P$]')
        temp_ax.set_xlim(0, 1.01)
        temp_ax.set_ylim(0, max(_tlim_kK + 2.0, 2.0))
        temp_ax.set_ylabel('Temperature [$10^3$ K]')

        comp_ax.set_xlabel('Mass [M$_P$]' if not rad_profiles else r'Radius [R$_\oplus$]')
        comp_ax.set_ylabel(r'$Y,\ Z$')
        comp_ax.set_xlim(0, 1.05)

        if planet_set == {'Jupiter'}:
            comp_ax.set_ylim(0, 0.7)
        else:
            comp_ax.set_ylim(-0.01, 1.05)

        if self.zm_plot:
            comp_ax.set_xlim(mass_norm[kcore+20], 1.0005)

        entropy_ax.set_ylim(0, 12)
        entropy_ax.set_ylabel(r'$S$ [k$_b$/baryon]')
        entropy_ax.grid(False)

        if dens_ax is not None:
            dens_ax.set_ylabel(r'$\rho$ [g cm$^{-3}$]')
            dens_ax.set_xlabel('Mass [M$_P$]')

        if brunt_ax is not None:
            brunt_ax.set_xlabel('Mass [M$_P$]')
            brunt_ax.set_ylabel(r'$N/\omega_{\rm dyn}$')
            brunt_ax.set_ylim(-0.1, 5)
            brunt_ax.set_xlim(0, 1.05)

        if miscellaneous_ax is not None:
            if not self.plot_Js:
                miscellaneous_ax.set_xlabel('Mass [M$_P$]')
                miscellaneous_ax.set_ylabel(r'Mass Zone Count (N)')
                ax92.set_ylabel(r'$dm$')
            else:
                miscellaneous_ax.set_xlabel('Age [Gyr]')
                miscellaneous_ax.set_ylabel(r'$J_2$')
                ax92.set_ylabel(r'$J_4$, $J_6$')
                miscellaneous_ax.tick_params(axis='y', which='both', colors='g', right=False)
                miscellaneous_ax.yaxis.label.set_color('g')
                miscellaneous_ax.yaxis.get_offset_text().set_color('g')
                miscellaneous_ax.spines['left'].set_color('g')
                ax92.tick_params(axis='y', which='both', colors='r', left=False)
                ax92.yaxis.label.set_color('r')
                ax92.yaxis.get_offset_text().set_color('r')
                ax92.spines['right'].set_color('r')
                _gas_giants = planet_set & {'Jupiter', 'Saturn'}
                _ice_giants = planet_set & {'Uranus', 'Neptune'}
                if _gas_giants and not _ice_giants:
                    miscellaneous_ax.set_ylim(10000, 20000)
                    ax92.set_ylim(-1500, 120)
                elif _ice_giants and not _gas_giants:
                    miscellaneous_ax.set_ylim(0, 6000)
                    ax92.set_ylim(-100, 50)
                elif _gas_giants and _ice_giants:
                    miscellaneous_ax.set_ylim(0, 20000)
                    ax92.set_ylim(-1500, 50)
                for _p in planet_set:
                    if _p == 'Jupiter':
                        miscellaneous_ax.axhline(y=14696.572, color='g', lw=3, ls='dashdot')
                        ax92.axhline(y=-586.609, color='r', lw=3, ls='dashdot')
                        ax92.axhline(y=34.201, color='darkorange', lw=2, ls='dashdot')
                    elif _p == 'Saturn':
                        miscellaneous_ax.axhline(y=16290.573, color='g', lw=3, ls='dashdot')
                        ax92.axhline(y=-935.314, color='r', lw=3, ls='dashdot')
                        ax92.axhline(y=86.340, color='darkorange', lw=2, ls='dashdot')
                    elif _p == 'Uranus':
                        miscellaneous_ax.axhline(y=3509.709, color='g', lw=3, ls='dashdot')
                        ax92.axhline(y=-35.04, color='r', lw=3, ls='dashdot')
                    elif _p == 'Neptune':
                        miscellaneous_ax.axhline(y=3408.43, color='g', lw=3, ls='dashdot')
                        ax92.axhline(y=-33.40, color='r', lw=3, ls='dashdot')
                miscellaneous_ax.set_xlim(x_left, x_right)
                ax92.set_xlim(x_left, x_right)
                if log_age:
                    miscellaneous_ax.set_xscale('log')
                    ax92.set_xscale('log')

        ticks_axes = [comp_ax, temp_ax, rad_ax]
        if yz_ax is not None:
            ticks_axes.append(yz_ax)
        if temp_log_ax is not None:
            ticks_axes.append(temp_log_ax)
        if dens_ax is not None:
            ticks_axes.append(dens_ax)
        if flux_ax is not None:
            ticks_axes.append(flux_ax)
        if brunt_ax is not None:
            ticks_axes.append(brunt_ax)
        for ax in ticks_axes:
            ax.minorticks_on()
            ax.yaxis.set_ticks_position('both')
            ax.xaxis.set_ticks_position('both')

        teff_ax.tick_params(direction='in', length=10, width=2, grid_alpha=0.75, which='minor', right=False, top=True)
        lum_ax.tick_params(direction='in', length=17, width=2, grid_alpha=0.75, which='major', right=True, left=False)

        if save:
            _save_kw = {}
            if movie_mode:
                _save_kw['dpi'] = 100
            else:
                _save_kw['dpi'] = 200
                _save_kw['bbox_inches'] = 'tight'
            if save_path is not None:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, **_save_kw)
            else:
                os.makedirs('plots/plot_stills', exist_ok=True)
                plt.savefig('plots/plot_stills/{}.png'.format(output_file), **_save_kw)
            if not movie_mode:
                plt.show()
        else:
            plt.show()
        plt.close()

        # Restore rcParams if movie mode changed them
        if movie_mode and _prev_rc is not None:
            plt.rcParams.update(_prev_rc)

    def debug_plot(self, age=-1,
                   output_file='debug_plot',
                   labels=[],
                   leg_size=10,
                   save=False,
                   save_path=None,
                   ):
        """
        Dedicated diagnostic plot for thermal and compositional transport debugging.

        Produces a 3x4 (12-panel) figure:
          Row 1 — Thermal fluxes: decomposed fluxes, energy balance, conductivities, bracket_dsdr
          Row 2 — Composition transport: Y/Z with saturation, saturation excess, EOS derivs, composition gradients
          Row 3 — Stability: Brunt-Väisälä, 1/R0, entropy + dS/dr, P-T diagram

        Requires save_verbose=True in the parameter file for full diagnostics.
        Pass read_verbose=True to init_plot() constructor.
        """

        def _time_len(arr):
            if arr is None:
                return None
            try:
                a = np.asarray(arr)
            except Exception:
                return None
            if a.ndim == 0:
                return None
            return a.shape[0]

        def _min_time_len(arrs):
            lens = [l for l in (_time_len(a) for a in arrs) if l is not None and l > 0]
            return min(lens) if lens else None

        if len(labels) == 0:
            labels = [f'Model {j}' for j in range(len(self.paths))]

        # --- Figure setup: 3 rows x 4 columns ---
        fig, axes = plt.subplots(3, 4, figsize=(48, 32))
        fig.subplots_adjust(wspace=0.38, hspace=0.30)

        # Row 1: Thermal fluxes
        ax_flux       = axes[0, 0]
        ax_ebal       = axes[0, 1]
        ax_cond       = axes[0, 2]
        ax_bracket    = axes[0, 3]

        # Row 2: Composition transport
        ax_comp_sat   = axes[1, 0]
        ax_excess     = axes[1, 1]
        ax_eos_derivs = axes[1, 2]
        ax_comp_grad  = axes[1, 3]

        # Row 3: Stability diagnostics
        ax_brunt      = axes[2, 0]
        ax_r0         = axes[2, 1]
        ax_entropy    = axes[2, 2]
        ax_dsdr       = ax_entropy.twinx()
        ax_pt         = axes[2, 3]

        colors = self.colors
        lw = 4

        for j in range(len(self.paths)):

            # --- Age/timestep selection (same logic as still_plot) ---
            age_gyr_i = np.array(self.data_age[j]) * const.s_to_Gyr

            if age == -1:
                i = -1
            elif age == 0:
                i = 0
            else:
                idx = np.where(age_gyr_i >= age)[0]
                i = idx[0] if idx.size else -1

            time_arrays = [
                self.data_age[j],
                self.rad[j],
                self.press[j],
                self.temp[j],
                self.S[j],
                self.Y[j],
                self.Z[j],
                self.conv_flux[j],
                self.rad_flux[j],
            ]
            min_len = _min_time_len(time_arrays)
            if min_len is None:
                print(f"Warning: no complete time snapshots for model index {j}; skipping.")
                continue
            if i < 0 or i >= min_len:
                i = min_len - 1

            age_gyr = age_gyr_i[i]

            kcore = self.kcore[j]
            mass_norm = self.m_mean[j]       # cell centers (N)
            mass_b = self.m_unit[j]           # boundaries (N+1)
            mass_i = mass_b[1:-1]             # interface (N-1)
            env_int_end = max(kcore - 1, 0)

            # --- Extract profiles at this timestep ---
            S = self.S[j][i]
            Y_full = self.Y[j][i]
            Z_full = self.Z[j][i]
            Y = Y_full[:kcore]
            Z = Z_full[:kcore]
            logp = np.log10(self.press[j][i])
            logt = np.log10(self.temp[j][i])
            press_GPa = self.press[j][i] * 1e-10
            rad_profile = self.rad[j][i]

            convective_flux = self.conv_flux[j][i]
            semiconvective_flux = self.semi_flux[j][i]
            radiative_flux = self.rad_flux[j][i]
            total_flux = convective_flux + radiative_flux + semiconvective_flux

            omega_dyn = np.sqrt(G * self.mass_grid[j][0] / (rad_profile[0]**3))

            # Check if verbose data is available (not all NaN)
            has_verbose = not np.all(np.isnan(self.Lambda_cond_i[j]))

            # ====================================================
            # ROW 1: THERMAL FLUXES
            # ====================================================

            # --- Panel (1,1): Decomposed thermal fluxes ---
            ax_flux.plot(mass_b, -total_flux, color=colors[j], ls='-', lw=lw,
                         alpha=0.9, label='Total' if j == 0 else '_nolegend_')
            ax_flux.plot(mass_b, -convective_flux, color=colors[j], ls='--', lw=lw,
                         alpha=0.5, label='Convective' if j == 0 else '_nolegend_')
            ax_flux.plot(mass_b, -radiative_flux, color=colors[j], ls='dotted', lw=lw,
                         alpha=0.5, label='Radiative' if j == 0 else '_nolegend_')
            ax_flux.plot(mass_b, -semiconvective_flux, color=colors[j], ls='dashdot', lw=lw,
                         alpha=0.5, label='Semiconvective' if j == 0 else '_nolegend_')

            # --- Panel (1,2): Energy balance (F/F_surf) ---
            f_surf = total_flux[-1] if np.abs(total_flux[-1]) > 0 else np.nan
            ax_ebal.plot(mass_b, total_flux / f_surf, color=colors[j], ls='-', lw=lw,
                         alpha=0.9, label=labels[j])
            ax_ebal.axhline(y=1.0, color='gray', ls='--', lw=2, alpha=0.5)

            # --- Panel (1,3): Conductivities ---
            if has_verbose:
                lam_cond = self.Lambda_cond_i[j][i]
                lam_rad = self.Lambda_rad_i[j][i]
                n_lam = len(lam_cond)
                mass_iface = mass_i[:n_lam]
                ax_cond.semilogy(mass_iface, lam_cond, color=colors[j], ls='-', lw=lw,
                                 alpha=0.8, label='Conductive' if j == 0 else '_nolegend_')
                ax_cond.semilogy(mass_iface, lam_rad, color=colors[j], ls='--', lw=lw,
                                 alpha=0.8, label='Radiative' if j == 0 else '_nolegend_')

            # --- Panel (1,4): Entropy gradient bracket ---
            if has_verbose:
                bracket = self.bracket_dsdr[j][i]
                ax_bracket.plot(mass_b, bracket, color=colors[j], ls='-', lw=lw, alpha=0.8,
                                label=labels[j])
                ax_bracket.axhline(y=0, color='gray', ls='--', lw=2, alpha=0.5)

            # ====================================================
            # ROW 2: COMPOSITION TRANSPORT
            # ====================================================

            # --- Panel (2,1): Y, Z profiles with saturation curves ---
            ax_comp_sat.plot(mass_norm[:kcore], Y, color=colors[j], ls='-', lw=lw,
                             alpha=0.9, label=r'$Y$' if j == 0 else '_nolegend_')
            ax_comp_sat.plot(mass_norm[:kcore], Z, color=colors[j], ls='--', lw=lw,
                             alpha=0.9, label=r'$Z$' if j == 0 else '_nolegend_')

            # Ym saturation curve
            Ym_data = self.Ym[j][i]
            if not np.all(np.isnan(Ym_data)):
                n_ym = len(Ym_data)
                ax_comp_sat.plot(mass_i[:n_ym], Ym_data[:n_ym], color=colors[j],
                                 ls='dashdot', lw=lw+2, alpha=0.4,
                                 label=r'$Y_m$' if j == 0 else '_nolegend_')

            # Z1m, Z2m saturation curves (only if loaded via zm_plot=True)
            if len(self.Z1m) > j:
                Z1m_data = self.Z1m[j][i]
                if not np.all(np.isnan(Z1m_data)):
                    n_z1 = len(Z1m_data)
                    ax_comp_sat.plot(mass_i[:n_z1], Z1m_data[:n_z1], color=colors[j],
                                     ls='dotted', lw=lw+2, alpha=0.4,
                                     label=r'$Z_{1m}$' if j == 0 else '_nolegend_')
            if len(self.Z2m) > j:
                Z2m_data = self.Z2m[j][i]
                if not np.all(np.isnan(Z2m_data)):
                    n_z2 = len(Z2m_data)
                    ax_comp_sat.plot(mass_i[:n_z2], Z2m_data[:n_z2], color=colors[j],
                                     ls=':', lw=lw+2, alpha=0.4,
                                     label=r'$Z_{2m}$' if j == 0 else '_nolegend_')

            # --- Panel (2,2): Saturation excess (Y - Ym, Z - Z1m) ---
            if not np.all(np.isnan(Ym_data)):
                # Y is on cell centers (N), Ym is on interface (N-1)
                # Interpolate Y to interface grid, then align lengths
                Y_iface = get_arith_mean(Y_full[:kcore])
                n_excess = min(len(Y_iface), len(Ym_data), len(mass_i))
                delta_Y = Y_iface[:n_excess] - Ym_data[:n_excess]
                ax_excess.plot(mass_i[:n_excess], delta_Y, color=colors[j], ls='-', lw=lw,
                               alpha=0.8, label=r'$Y - Y_m$' if j == 0 else '_nolegend_')

            if len(self.Z1m) > j:
                Z1m_data = self.Z1m[j][i]
                if not np.all(np.isnan(Z1m_data)):
                    Z_iface = get_arith_mean(Z_full[:kcore])
                    n_excess_z = min(len(Z_iface), len(Z1m_data), len(mass_i))
                    delta_Z = Z_iface[:n_excess_z] - Z1m_data[:n_excess_z]
                    ax_excess.plot(mass_i[:n_excess_z], delta_Z, color=colors[j], ls='--', lw=lw,
                                   alpha=0.8, label=r'$Z - Z_{1m}$' if j == 0 else '_nolegend_')
            ax_excess.axhline(y=0, color='gray', ls='--', lw=2, alpha=0.5)

            # --- Panel (2,3): EOS derivatives ---
            if has_verbose:
                dsdy_prho = self.dsdy_prho[j][i]
                dsdz_prho = self.dsdz_prho[j][i]
                dsdy_pt = self.dsdy_pt[j][i]
                dsdz_pt = self.dsdz_pt[j][i]
                n_eos = len(dsdy_prho)
                m_eos = mass_i[:n_eos]

                ax_eos_derivs.plot(m_eos, dsdy_prho, color=colors[j], ls='-', lw=lw,
                                   alpha=0.8, label=r'$\partial S/\partial Y|_{P,\rho}$' if j == 0 else '_nolegend_')
                ax_eos_derivs.plot(m_eos, dsdz_prho, color=colors[j], ls='--', lw=lw,
                                   alpha=0.8, label=r'$\partial S/\partial Z|_{P,\rho}$' if j == 0 else '_nolegend_')
                ax_eos_derivs.plot(m_eos, dsdy_pt, color=colors[j], ls='dashdot', lw=lw-1,
                                   alpha=0.6, label=r'$\partial S/\partial Y|_{P,T}$' if j == 0 else '_nolegend_')
                ax_eos_derivs.plot(m_eos, dsdz_pt, color=colors[j], ls='dotted', lw=lw-1,
                                   alpha=0.6, label=r'$\partial S/\partial Z|_{P,T}$' if j == 0 else '_nolegend_')
                ax_eos_derivs.axhline(y=0, color='gray', ls='--', lw=2, alpha=0.5)

            # --- Panel (2,4): Composition gradients dY/dr, dZ/dr ---
            if kcore >= 3:
                r_env = rad_profile[:kcore]
                dY_dr = np.diff(Y) / np.diff(r_env)
                dZ_dr = np.diff(Z) / np.diff(r_env)
                # Normalize by planet radius for readable scale
                r_scale = rad_profile[0]
                mass_grad = get_arith_mean(mass_norm[:kcore])
                ax_comp_grad.plot(mass_grad, dY_dr * r_scale, color=colors[j], ls='-', lw=lw,
                                  alpha=0.8, label=r'$dY/dr \cdot R_P$' if j == 0 else '_nolegend_')
                ax_comp_grad.plot(mass_grad, dZ_dr * r_scale, color=colors[j], ls='--', lw=lw,
                                  alpha=0.8, label=r'$dZ/dr \cdot R_P$' if j == 0 else '_nolegend_')
                ax_comp_grad.axhline(y=0, color='gray', ls='--', lw=2, alpha=0.5)

            # ====================================================
            # ROW 3: STABILITY DIAGNOSTICS
            # ====================================================

            # --- Panel (3,1): Brunt-Väisälä frequency ---
            brunt = self.brunt[j][i][:env_int_end]
            if len(brunt) > 0 and not np.all(np.isnan(brunt)):
                ax_brunt.plot(mass_i[:env_int_end], brunt / omega_dyn, color=colors[j],
                              ls='-', lw=lw, alpha=0.8, label=labels[j])
            ax_brunt.axhline(y=0, color='gray', ls='--', lw=2, alpha=0.5)

            # --- Panel (3,2): 1/R0 (density ratio) ---
            R0_inv_data = self.R0_inv[j]
            if R0_inv_data is not None and not np.all(np.isnan(R0_inv_data)):
                r0_profile = R0_inv_data[i] if np.asarray(R0_inv_data).ndim > 1 else R0_inv_data
                n_r0 = len(r0_profile)
                ax_r0.plot(mass_b[:n_r0], r0_profile, color=colors[j], ls='-', lw=lw,
                           alpha=0.8, label=labels[j])
            ax_r0.axhline(y=1.0, color='red', ls='--', lw=2, alpha=0.6, label=r'Schwarzschild' if j == 0 else '_nolegend_')
            ax_r0.axhspan(0, 1.0, color='lightblue', alpha=0.1)

            # --- Panel (3,3): Entropy profile + dS/dr ---
            S_env = S[:kcore]
            S_kbbar = S_env * erg_to_kbbar
            ax_entropy.plot(mass_norm[:kcore], S_kbbar, color=colors[j], ls='-', lw=lw,
                            alpha=0.8, label=labels[j])

            if kcore >= 3:
                dS_dr = np.diff(S_env) / np.diff(rad_profile[:kcore])
                dS_dr_norm = dS_dr * rad_profile[0]
                mass_ds = get_arith_mean(mass_norm[:kcore])
                ax_dsdr.plot(mass_ds, dS_dr_norm * erg_to_kbbar, color=colors[j], ls='--', lw=lw-1,
                             alpha=0.5, label=r'$dS/dr \cdot R_P$' if j == 0 else '_nolegend_')

            # --- Panel (3,4): P-T diagram ---
            ax_pt.loglog(press_GPa, 10**logt, color=colors[j], lw=lw, alpha=0.9, label=labels[j])

        # ====================================================
        # AXIS LABELS, LIMITS, AND FORMATTING
        # ====================================================

        # Age annotation
        ax_flux.text(0.03, 0.97, f'Age: {age_gyr:.5f} Gyr', transform=ax_flux.transAxes,
                     ha='left', va='top', fontsize=leg_size + 8,
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        # Row 1
        ax_flux.set_xlabel(r'Mass [M$_P$]')
        ax_flux.set_ylabel(r'Flux [erg s$^{-1}$]')
        ax_flux.set_yscale('symlog')
        ax_flux.set_xlim(0.0, 1.01)
        ax_flux.legend(frameon=False, prop={'size': leg_size}, loc='upper left')

        ax_ebal.set_xlabel(r'Mass [M$_P$]')
        ax_ebal.set_ylabel(r'$F / F_{\rm surf}$')
        ax_ebal.set_xlim(0.0, 1.01)
        ax_ebal.set_ylim(-0.1, 1.5)

        ax_cond.set_xlabel(r'Mass [M$_P$]')
        ax_cond.set_ylabel(r'$\Lambda$ [erg cm$^{-1}$ s$^{-1}$ K$^{-1}$]')
        ax_cond.set_xlim(0.0, 1.01)
        ax_cond.legend(frameon=False, prop={'size': leg_size}, loc='upper left')

        ax_bracket.set_xlabel(r'Mass [M$_P$]')
        ax_bracket.set_ylabel(r'$[\nabla S]$ bracket')
        ax_bracket.set_yscale('symlog')
        ax_bracket.set_xlim(0.0, 1.01)

        # Row 2
        ax_comp_sat.set_xlabel(r'Mass [M$_P$]')
        ax_comp_sat.set_ylabel(r'$Y,\ Z,\ Y_m,\ Z_m$')
        ax_comp_sat.set_xlim(0.0, 1.05)
        ax_comp_sat.set_ylim(0, 1.0)
        ax_comp_sat.legend(frameon=False, prop={'size': leg_size}, loc='upper left')

        ax_excess.set_xlabel(r'Mass [M$_P$]')
        ax_excess.set_ylabel(r'Saturation excess')
        ax_excess.set_xlim(0.0, 1.05)
        ax_excess.legend(frameon=False, prop={'size': leg_size}, loc='upper left')

        ax_eos_derivs.set_xlabel(r'Mass [M$_P$]')
        ax_eos_derivs.set_ylabel(r'$\partial S / \partial Y,\ \partial S / \partial Z$')
        ax_eos_derivs.set_yscale('symlog')
        ax_eos_derivs.set_xlim(0.0, 1.01)
        ax_eos_derivs.legend(frameon=False, prop={'size': leg_size}, loc='upper left')

        ax_comp_grad.set_xlabel(r'Mass [M$_P$]')
        ax_comp_grad.set_ylabel(r'Composition gradient (normalized)')
        ax_comp_grad.set_yscale('symlog')
        ax_comp_grad.set_xlim(0.0, 1.05)
        ax_comp_grad.legend(frameon=False, prop={'size': leg_size}, loc='upper left')

        # Row 3
        ax_brunt.set_xlabel(r'Mass [M$_P$]')
        ax_brunt.set_ylabel(r'$N / \omega_{\rm dyn}$')
        ax_brunt.set_xlim(0.0, 1.05)
        ax_brunt.set_ylim(-0.5, 5)

        ax_r0.set_xlabel(r'Mass [M$_P$]')
        ax_r0.set_ylabel(r'$1/R_0$')
        ax_r0.set_xlim(0.0, 1.05)
        ax_r0.set_ylim(-0.5, 5)
        ax_r0.legend(frameon=False, prop={'size': leg_size}, loc='upper right')

        ax_entropy.set_xlabel(r'Mass [M$_P$]')
        ax_entropy.set_ylabel(r'$S$ [k$_b$/baryon]')
        ax_entropy.set_xlim(0.0, 1.05)
        ax_dsdr.set_ylabel(r'$dS/dr \cdot R_P$ [k$_b$/baryon]')
        ax_dsdr.set_yscale('symlog')
        ax_dsdr.grid(False)

        ax_pt.set_xlabel(r'Pressure [GPa]')
        ax_pt.set_ylabel(r'Temperature [K]')
        ax_pt.invert_xaxis()

        # Tick formatting
        for ax_row in axes:
            for ax in ax_row:
                ax.minorticks_on()
                ax.yaxis.set_ticks_position('both')
                ax.xaxis.set_ticks_position('both')

        # --- Save / show ---
        if save:
            if save_path is not None:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, dpi=200, bbox_inches='tight')
            else:
                os.makedirs('plots/plot_debug', exist_ok=True)
                plt.savefig('plots/plot_debug/{}.png'.format(output_file), dpi=200, bbox_inches='tight')
        plt.show()
        plt.close()

    def age_plot(self, ages,
                 output_file,
                 labels = [],
                 model_indices=None,
                 leg_size=12,
                 save=False,
                 plot_yz_time=False,
                 plot_water_misc=False,
                 rad_profiles=False,
                 plot_he_rain = False,
                 panel_num=6,
                 rad_temp_plot=False,
                 cmap='magma_r',
                 log_age=False,
                      ):
        """
        Generalization of still_plot():
        compare multiple requested ages (color-coded) for one or more models.

        Colors encode age. If multiple models are selected, line style / marker
        encode model identity.
        """

        def _time_len(arr):
            if arr is None:
                return None
            try:
                a = np.asarray(arr)
            except Exception:
                return None
            if a.ndim == 0:
                return None
            return a.shape[0]

        def _min_time_len(arrs):
            lens = [l for l in (_time_len(a) for a in arrs) if l is not None and l > 0]
            return min(lens) if lens else None

        age_floor_gyr = 1e-5

        def _age_for_plot(arr):
            a = np.asarray(arr, dtype=float)
            return np.maximum(a, age_floor_gyr) if log_age else a

        def _age_scalar_for_plot(val):
            v = float(val)
            return max(v, age_floor_gyr) if log_age else v

        def _resolve_snapshot_index(j, age_req):
            age_gyr_i = np.asarray(self.data_age[j]) * const.s_to_Gyr

            if age_req == -1:
                i = -1
            elif age_req == 0:
                i = 0
            else:
                idx = np.where(age_gyr_i >= age_req)[0]
                i = idx[0] if idx.size else -1

            time_arrays = [
                self.data_age[j],
                self.rad[j],
                self.press[j],
                self.rho[j],
                self.temp[j],
                self.S[j],
                self.Y[j],
                self.Z[j],
                self.brunt[j],
                self.conv_flux[j],
                self.rad_flux[j],
                self.data_teff[j],
                self.data_rad[j],
            ]
            if self.plot_rock_phases and (j < len(self.melt_fraction)):
                time_arrays.append(self.melt_fraction[j])
            if self.plot_Js and hasattr(self, "Req_TOF") and (j < len(self.Req_TOF)):
                time_arrays.append(self.Req_TOF[j])

            min_len = _min_time_len(time_arrays)
            if min_len is None:
                return None

            if i < 0 or i >= min_len:
                i = min_len - 1

            return i, age_gyr_i, float(age_gyr_i[i])

        if np.isscalar(ages):
            ages = [float(ages)]
        else:
            ages = [float(a) for a in ages]
        if len(ages) == 0:
            raise ValueError("age_plot requires at least one age (Gyr).")

        if model_indices is None:
            selected_models = list(range(len(self.paths)))
        elif np.isscalar(model_indices):
            selected_models = [int(model_indices)]
        else:
            selected_models = [int(i) for i in model_indices]

        # de-dup, preserve order
        seen_models = set()
        selected_models = [j for j in selected_models if not (j in seen_models or seen_models.add(j))]

        if len(selected_models) == 0:
            raise ValueError("No models selected for age_plot.")
        for j in selected_models:
            if j < 0 or j >= len(self.paths):
                raise IndexError(f"model_indices contains out-of-range index {j}.")

        if len(labels) == 0:
            model_labels = [os.path.basename(self.paths[j]) for j in selected_models]
        elif len(labels) == len(self.paths):
            model_labels = [labels[j] for j in selected_models]
        elif len(labels) == len(selected_models):
            model_labels = list(labels)
        else:
            raise ValueError(
                "labels must have length 0, len(self.paths), or len(model_indices)."
            )

        if panel_num not in (4, 6, 9):
            raise ValueError(f"panel_num={panel_num} is not supported. Use 4, 6, or 9.")

        if panel_num == 9:
            fig = plt.figure(figsize=(35, 30))
            temp_log_ax = plt.subplot(3, 3, 1)
            flux_ax = plt.subplot(3, 3, 2)
            comp_ax = plt.subplot(3, 3, 3)
            temp_ax = plt.subplot(3, 3, 4)
            rad_ax = plt.subplot(3, 3, 5)
            teff_ax = plt.subplot(3, 3, 6)
            dens_ax = plt.subplot(3, 3, 7)
            brunt_ax = plt.subplot(3, 3, 8)
            miscellaneous_ax = plt.subplot(3, 3, 9)
            ax92 = miscellaneous_ax.twinx()
        elif panel_num == 6:
            fig = plt.figure(figsize=(35, 20))
            temp_log_ax = plt.subplot(2, 3, 1)
            comp_ax = plt.subplot(2, 3, 2)
            temp_ax = plt.subplot(2, 3, 3)
            rad_ax = plt.subplot(2, 3, 4)
            teff_ax = plt.subplot(2, 3, 5)
            dens_ax = plt.subplot(2, 3, 6)
            flux_ax = None
            brunt_ax = None
            miscellaneous_ax = None
            ax92 = None
        else:
            fig = plt.figure(figsize=(30, 20))
            comp_ax = plt.subplot(2, 2, 1)
            temp_ax = plt.subplot(2, 2, 2)
            rad_ax = plt.subplot(2, 2, 3)
            teff_ax = plt.subplot(2, 2, 4)
            temp_log_ax = None
            dens_ax = None
            flux_ax = None
            brunt_ax = None
            miscellaneous_ax = None
            ax92 = None

        entropy_ax = comp_ax.twinx()
        if self.plot_rock_phases:
            melt_frac_ax = temp_ax.twinx()
        lum_ax = teff_ax.twinx()
        yz_ax = rad_ax.twinx() if plot_yz_time else None
        if yz_ax is not None:
            yz_ax.grid(False)
        
        fig.subplots_adjust(wspace=0.33, hspace=0.24)
        entropy_ax.yaxis.labelpad = 16
        lum_ax.yaxis.labelpad = 16

        # Each model gets a distinct color; ages are shown as fainter shades.
        model_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                        '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        model_markers = ['o', 's', '^', 'D', 'v', 'P', 'X', '*', '<', '>']

        def _fade_color(base, frac):
            """Blend *base* toward white by *frac* (0 = original, 1 = white)."""
            import matplotlib.colors as _mc
            r, g, b, a = _mc.to_rgba(base)
            return (r + (1 - r) * frac, g + (1 - g) * frac,
                    b + (1 - b) * frac, a)

        # Fade factors per age: earliest age = 0 (full colour), latest = 0.55
        if len(ages) == 1:
            age_fade = [0.0]
        else:
            age_fade = [i * 0.55 / (len(ages) - 1) for i in range(len(ages))]

        # Single model: use a colormap for age colours instead of dark-to-faint
        _single_model = (len(selected_models) == 1)
        if _single_model:
            _age_cmap = plt.get_cmap(cmap)
            if len(ages) == 1:
                _age_cmap_colors = [_age_cmap(0.5)]
            else:
                _age_cmap_colors = [_age_cmap(k / (len(ages) - 1)) for k in range(len(ages))]

        lw = 4
        flux_color_handles = []
        plotted_any = False
        rad_temp_ax = None   # twinx for radius profile (created on demand)
        plotted_age_used = []
        max_temp_kK = 0.0
        zm_xmins = []

        planet_set = {self.planet[j] for j in selected_models if j < len(self.planet)}

        # Plot the model time-series tracks once per selected model (solid model colour)
        for mpos, j in enumerate(selected_models):
            mcolor = model_colors[mpos % len(model_colors)]
            times_gyr_raw = np.asarray(self.data_age[j]) * const.s_to_Gyr
            times_gyr = _age_for_plot(times_gyr_raw)

            if self.plot_Js and (j < len(self.Req_TOF)):
                req_series = np.asarray(self.Req_TOF[j], dtype=float)
                if req_series.size > 0 and np.any(np.isfinite(req_series)):
                    rad_series = req_series / const.rearth
                else:
                    rad_series = np.asarray(self.data_rad[j], dtype=float) / const.rearth
            else:
                rad_series = np.asarray(self.data_rad[j], dtype=float) / const.rearth

            rad_ax.plot(times_gyr, rad_series, c=mcolor, lw=lw-1, ls='-', alpha=0.9, label='_nolegend_')
            if yz_ax is not None:
                y_series = np.asarray(self.data_Y_atm[j], dtype=float).reshape(-1)
                z_series = np.asarray(self.data_Z_atm[j], dtype=float).reshape(-1)
                ny = min(times_gyr.size, y_series.size)
                nz = min(times_gyr.size, z_series.size)
                if ny > 0:
                    yz_ax.plot(times_gyr[:ny], y_series[:ny], c=mcolor, lw=lw-1, ls='--', alpha=0.9, label='_nolegend_')
                if nz > 0:
                    yz_ax.plot(times_gyr[:nz], z_series[:nz], c=mcolor, lw=lw-1, ls=':', alpha=0.9, label='_nolegend_')
            teff_ax.plot(times_gyr, self.data_teff[j], c=mcolor, lw=lw-1, ls='-', alpha=0.9, label='_nolegend_')
            lum_ax.plot(times_gyr, self.luminosity[j], c=mcolor, lw=lw-1, ls='--', alpha=0.8, label='_nolegend_')

            if (miscellaneous_ax is not None) and self.plot_Js and (j < len(self.J2_TOF)) and len(self.J2_TOF) > 0:
                if j < len(self.J2_TOF):
                    miscellaneous_ax.plot(times_gyr, self.J2_TOF[j], color=mcolor, lw=lw-1, ls='-', alpha=0.9)
                if j < len(self.J4_TOF):
                    ax92.plot(times_gyr, self.J4_TOF[j], color=mcolor, lw=lw-1, ls='--', alpha=0.9)
                if j < len(self.J6_TOF):
                    ax92.plot(times_gyr, self.J6_TOF[j], color=mcolor, lw=lw-1, ls='dashdot', alpha=0.9)

        for mpos, j in enumerate(selected_models):
            mcolor = model_colors[mpos % len(model_colors)]
            marker = model_markers[mpos % len(model_markers)]

            for aidx, age_req in enumerate(ages):
                snap = _resolve_snapshot_index(j, age_req)
                if snap is None:
                    print(f"Warning: no complete time snapshots for model index {j}; skipping.")
                    continue

                i, age_gyr_i, age_gyr = snap
                age_gyr_plot = _age_scalar_for_plot(age_gyr)
                c_age = _age_cmap_colors[aidx] if _single_model else _fade_color(mcolor, age_fade[aidx])
                plotted_any = True
                plotted_age_used.append(age_gyr)

                kcore = self._kcore_for_snapshot(
                    self.kcore[j], self.data_kcore[j] if j < len(self.data_kcore) else None, i)
                kcore_fe = self._kcore_for_snapshot(
                    self.kcore_fe[j], self.data_kcore_fe[j] if j < len(self.data_kcore_fe) else None, i)
                mass_norm = self.m_mean[j]
                rad_norm = self.rad[j][i]/const.rearth
                mass_b = self.m_unit[j]
                mass_i = mass_b[1:-1]
                S = self.S[j][i]
                logp = np.log10(self.press[j][i])
                press_Mbar = self.press[j][i]*1e-12
                press_GPa = self.press[j][i]*1e-10
                logrho = np.log10(self.rho[j][i])
                logt = np.log10(self.temp[j][i])
                Z = np.array(self.Z[j][i], dtype=float)
                if kcore > 0 and kcore < len(Z):
                    Z[kcore:] = 1.0
                Y = self.Y[j][i][:kcore]

                if self.plot_rock_phases:
                    melt_frac = self.melt_fraction[j][i]

                env_int_end = max(kcore - 1, 0)
                brunt = self.brunt[j][i][:env_int_end]

                convective_flux = self.conv_flux[j][i]
                radiative_flux = self.rad_flux[j][i]
                total_lum_out = self.luminosity[j]

                omega_dyn = np.sqrt(G*self.mass_grid[j][0]/(self.rad[j][i][0]**3))

                max_temp_kK = max(max_temp_kK, float(np.nanmax((10**logt) * 1e-3)))

                # Temperature-pressure / temperature-mass
                if temp_log_ax is not None:
                    temp_log_ax.loglog(press_GPa, 10 ** logt, color=c_age, lw=lw, ls='-', label='_nolegend_')
                temp_ax.plot(mass_norm, (10**logt) * 1e-3, color=c_age, lw=lw, ls='-', label='_nolegend_')

                # Normalised radius profile — only at the last requested age
                if rad_temp_plot and aidx == len(ages) - 1:
                    if rad_temp_ax is None:
                        rad_temp_ax = temp_ax.twinx()
                        rad_temp_ax.set_ylabel(r'Normalized Radius [$R_P$]')
                        rad_temp_ax.set_ylim(0, 1)
                        temp_ax.tick_params(right=False, which='both')
                        rad_temp_ax.tick_params(left=False, which='both')
                    _rrad = self.rad[j][i]
                    _rp_raw = _rrad / _rrad[0]          # 1.0 at surface, ~0 at centre
                    _mr_raw = self.m_mean[j]            # same grid, surface→centre
                    # Build rad(M) interpolator (flip both to ascending order)
                    _rad_of_m = interp1d(_mr_raw[::-1], _rp_raw[::-1],
                                         bounds_error=False,
                                         fill_value=(_rp_raw[-1], _rp_raw[0]))
                    _m_fine = np.linspace(float(_mr_raw.min()), float(_mr_raw.max()), 500)
                    rad_temp_ax.plot(_m_fine, _rad_of_m(_m_fine), color=mcolor,
                                     ls='--', lw=lw, alpha=0.4, label='_nolegend_')

                if plot_he_rain:
                    pmisc, tmisc = misc.get_misc_p(
                        logp[:kcore], Y/(1 - Z[:kcore]), misc=self.misc_curve[j],
                        delta_T=self.deltaT[j], pad=True
                    )

                    if temp_log_ax is not None:
                        pos = tmisc > 0
                        temp_log_ax.plot(pmisc[pos]*100, tmisc[pos]*1000, color=c_age, ls='dotted', lw=lw+1, alpha=0.7)

                    interp_pm_he = interp1d(
                        10 ** (logp[:kcore] - 12),
                        mass_norm[:kcore],
                        kind='linear',
                        fill_value='extrapolate',
                    )
                    mmisc = interp_pm_he(pmisc)
                    temp_ax.plot(mmisc, tmisc, color=c_age, ls='dotted', lw=lw+1, alpha=0.7)

                    # Z-misc stand-in curve (misc_kind='hhe'): Z-channel offsets,
                    # local Z as composition; DASHED to distinguish from the
                    # dotted He curve.
                    if getattr(self, 'zmisc_active', None) and self.zmisc_active[j]:
                        z_dT = self.zmisc_deltaT[j] if self.zmisc_deltaT[j] is not None else (self.deltaT[j] or 0.0)
                        z_dP = self.zmisc_deltaP[j] or 0.0  # Mbar
                        p_cgs = 10.0 ** logp[:kcore]
                        zsel = p_cgs > (1.0 + z_dP) * 1e12
                        if np.count_nonzero(zsel) > 2:
                            logp_eff = np.log10(p_cgs[zsel] - z_dP * 1e12)
                            pmisc_z, tmisc_z = misc.get_misc_p(
                                logp_eff, Z[:kcore][zsel], misc=self.misc_curve[j],
                                delta_T=z_dT, pad=True)
                            if pmisc_z is not None:
                                pmisc_z = pmisc_z + z_dP  # back to local pressure [Mbar]
                                if temp_log_ax is not None:
                                    posz = tmisc_z > 0
                                    temp_log_ax.plot(pmisc_z[posz]*100, tmisc_z[posz]*1000,
                                                     color=c_age, ls='--', lw=lw+1, alpha=0.7)
                                mmisc_z = interp_pm_he(pmisc_z)
                                temp_ax.plot(mmisc_z, tmisc_z, color=c_age, ls='--', lw=lw+1, alpha=0.7)

                # Fluxes
                if flux_ax is not None:
                    total_flux = convective_flux + radiative_flux
                    flux_ax.plot(mass_b, -convective_flux, color=c_age, ls='--', lw=lw+4, alpha=0.45, label='_nolegend_')
                    flux_ax.plot(mass_b, -radiative_flux, color=c_age, ls='dotted', lw=lw, alpha=0.45, label='_nolegend_')
                    total_flux_line, = flux_ax.plot(mass_b, -total_flux, color=c_age, ls='-', lw=lw, label='_nolegend_')
                    if mpos == 0:
                        flux_color_handles.append(total_flux_line)

                interp_pm = None
                if plot_water_misc:
                    if kcore >= 2:
                        interp_pm = interp1d(
                            10 ** (logp[:kcore] - 12),
                            mass_norm[:kcore],
                            kind='linear',
                            fill_value='extrapolate',
                        )
                if plot_water_misc and (interp_pm is not None):
                    if mpos == 0 and aidx == 0:
                        Pmisc, Tmisc, _, _, _ = gm.get_pmisc(10**logt, np.zeros(len(logt)), self.Z[j][i])
                        if temp_log_ax is not None:
                            temp_log_ax.plot(np.log10(Pmisc), np.log10(Tmisc), color='royalblue', ls='dotted', lw=lw+2, label=r'H$_2$-water miscibility')
                        mmisc = interp_pm(Pmisc*1e-2)
                        temp_ax.plot(mmisc, Tmisc*1e-3, color='royalblue', ls='dotted', lw=lw+2, label=r'H$_2$-water miscibility')

                if self.zm_plot:
                    misc_colors1 = ['C0', 'C1', 'C4', 'b', 'm']
                    misc_colors2 = ['C0', 'C1', 'C5', 'm', 'cyan']
                    z1_misc = self.Z1m[j][i]
                    z2_misc = self.Z2m[j][i]
                    comp_ax.plot(mass_i, z1_misc, color=misc_colors1[min(aidx, len(misc_colors1)-1)], ls='dashdot', lw=lw+3, alpha=0.35)
                    comp_ax.plot(mass_i, z2_misc, color=misc_colors2[min(aidx, len(misc_colors2)-1)], ls='dashdot', lw=lw, alpha=0.35)

                if self.plot_rock_phases:
                    melt_frac_ax.plot(mass_norm, melt_frac, color=c_age, lw=lw, ls='-', alpha=0.22, label='_nolegend_')

                if not rad_profiles:
                    comp_ax.plot(mass_norm[:kcore], Y, color=c_age, ls='dotted', lw=lw, alpha=0.85)
                    comp_ax.plot(mass_norm, Z, color=c_age, ls='--', lw=lw, alpha=0.65)
                    if self.zm_plot:
                        entropy_ax.scatter(mass_norm, S, color=c_age, s=90, alpha=0.65, marker=marker)
                    else:
                        entropy_ax.plot(mass_norm, S, color=c_age, ls='-', lw=lw, alpha=0.85)
                    if self.f_rock_profiles:
                        comp_ax.plot(mass_norm[:kcore], self.f_rock[j][i][:kcore], color=c_age, ls='dashdot', lw=lw, alpha=0.6)
                else:
                    comp_ax.plot(rad_norm[:kcore], Y, color=c_age, ls='dotted', lw=lw, alpha=0.85)
                    comp_ax.plot(rad_norm, Z, color=c_age, ls='--', lw=lw, alpha=0.65)
                    entropy_ax.plot(rad_norm, S, color=c_age, ls='-', lw=lw, alpha=0.85)

                if dens_ax is not None:
                    dens_ax.plot(mass_norm, 10**logrho, color=c_age, ls='-', lw=lw, alpha=0.9)
                if brunt_ax is not None:
                    brunt_ax.plot(mass_i[:env_int_end], brunt/omega_dyn, color=c_age, ls='-', lw=lw, alpha=0.9)

                _mantle_comp = self.mantle_comp[j] or 'mgsio3'
                _eos_mantle = self.eos_mantle[j] or 'comb'
                if _mantle_comp == 'mgsio3':
                    # if _eos_mantle == 'JJ':
                    #     Tm_si2 = mgsio3_eos_JJ.get_T_melt(press_GPa)
                    #else:
                    Tm_si2 = mgsio3_eos.get_T_melt(press_GPa)
                elif _mantle_comp == 'mg2sio4':
                    Tm_si2 = mg2sio4_eos.get_T_melt(press_GPa)
                elif _mantle_comp == 'h2o':
                    Tm_si2 = h2o_eos.get_T_melt(press_GPa)
                _eos_core = self.eos_core[j] or 'D17_comb'
                if _eos_core == 'G23_comb':
                    _iron_eos = gonzalez_iron_eos
                else:
                    _iron_eos = D17_iron_eos
                Tm_fe = _iron_eos.get_T_melt(press_GPa)
                mass_melt_interp = interp1d(press_GPa, mass_norm, kind='linear', fill_value='extrapolate')

                temp_h2_rock_misc = rm.get_tmisc(
                    press_GPa[press_GPa < 200],
                    np.zeros(len(press_GPa[press_GPa < 200])),
                    self.Z[j][i][press_GPa < 200],
                    Tc_mode='follow_melt'
                )[0]

                if self.zm_plot and (temp_log_ax is not None):
                    temp_log_ax.plot(np.log10(press_GPa[press_GPa < 200]), np.log10(temp_h2_rock_misc),
                                     color='purple', ls='dashdot', lw=lw+1, alpha=0.45)

                if self.plot_rock_phases:
                    if temp_log_ax is not None:
                        temp_log_ax.loglog(press_GPa, Tm_si2, color=c_age, ls='--', lw=lw+1, alpha=0.35)
                        temp_log_ax.loglog(press_GPa, Tm_fe, color=c_age, ls='dotted', lw=lw+1, alpha=0.35)
                    temp_ax.plot(mass_melt_interp(press_GPa), Tm_si2*1e-3, color=c_age, ls='--', lw=lw+1, alpha=0.35)
                    temp_ax.plot(mass_melt_interp(press_GPa), Tm_fe*1e-3, color=c_age, ls='dotted', lw=lw+1, alpha=0.35)

                if self.zm_plot:
                    temp_ax.plot(mass_melt_interp(press_GPa[press_GPa < 200]), temp_h2_rock_misc*1e-3,
                                 color='purple', ls='dashdot', lw=lw+1, alpha=0.45)

                # Mark selected ages on time-series axes
                if self.plot_Js:
                    if self.planet[j] in ('Uranus', 'Neptune', 'Sub_Neptune', 'Super_Earth'):
                        rad_snap = self.Req_TOF[j][i] / const.rearth
                    else:
                        rad_snap = self.Req_TOF[j][i] / const.rearth
                    if miscellaneous_ax is not None:
                        if j < len(self.J2_TOF):
                            miscellaneous_ax.scatter(age_gyr_plot, self.J2_TOF[j][i], color=c_age, marker=marker, s=110, edgecolor='k', linewidth=0.6, zorder=4)
                        if j < len(self.J4_TOF):
                            ax92.scatter(age_gyr_plot, self.J4_TOF[j][i], color=c_age, marker=marker, s=110, edgecolor='k', linewidth=0.6, zorder=4)
                        if j < len(self.J6_TOF):
                            ax92.scatter(age_gyr_plot, self.J6_TOF[j][i], color=c_age, marker=marker, s=110, edgecolor='k', linewidth=0.6, zorder=4)
                else:
                    if self.planet[j] in ('Uranus', 'Neptune', 'Sub_Neptune', 'Super_Earth'):
                        rad_snap = self.data_rad[j][i] / const.rearth
                    else:
                        rad_snap = self.data_rad[j][i] / const.rearth
                    if miscellaneous_ax is not None:
                        miscellaneous_ax.hist(mass_b, bins=50, color=c_age, alpha=0.15, density=True, histtype='step')
                        ax92.plot(mass_norm, -np.diff(self.mass_grid[j]), color=c_age, ls='-', lw=lw-1, alpha=0.7)

                rad_ax.scatter(age_gyr_plot, rad_snap, color=c_age, marker=marker, s=110, edgecolor='k', linewidth=0.6, zorder=4)
                if yz_ax is not None:
                    y_series = np.asarray(self.data_Y_atm[j], dtype=float).reshape(-1)
                    z_series = np.asarray(self.data_Z_atm[j], dtype=float).reshape(-1)
                    if i < y_series.size and np.isfinite(y_series[i]):
                        yz_ax.scatter(age_gyr_plot, y_series[i], color=c_age, marker=marker, s=110, edgecolor='k', linewidth=0.6, zorder=4)
                    if i < z_series.size and np.isfinite(z_series[i]):
                        yz_ax.scatter(age_gyr_plot, z_series[i], color=c_age, marker=marker, s=110, edgecolor='k', linewidth=0.6, zorder=4)
                teff_ax.scatter(age_gyr_plot, self.data_teff[j][i], color=c_age, marker=marker, s=110, edgecolor='k', linewidth=0.6, zorder=4)
                lum_ax.scatter(age_gyr_plot, total_lum_out[i], color=c_age, marker=marker, s=90, edgecolor='k', linewidth=0.6, zorder=4)

                if self.zm_plot and (kcore + 20) < len(mass_norm):
                    zm_xmins.append(mass_norm[kcore+20])

        if not plotted_any:
            raise RuntimeError("No valid model-age snapshots could be plotted.")

        # ---------- Legends ----------
        # Age legend
        if _single_model:
            age_handles = [
                Line2D([0], [0], color=_age_cmap_colors[aidx], lw=lw, ls='-',
                       label=f"{ages[aidx]:.3f} Gyr")
                for aidx in range(len(ages))
            ]
        else:
            age_handles = [
                Line2D([0], [0], color=_fade_color('k', age_fade[aidx]), lw=lw, ls='-',
                       label=f"{ages[aidx]:.3f} Gyr")
                for aidx in range(len(ages))
            ]
        temp_ax.legend(handles=age_handles, frameon=False, prop={'size': leg_size},
                       loc='upper right', ncol=min(2, len(age_handles)))

        # Model legend: each model gets its own colour
        model_handles = []
        for mpos, lab in enumerate(model_labels):
            mcolor = model_colors[mpos % len(model_colors)]
            marker = model_markers[mpos % len(model_markers)]
            model_handles.append(
                Line2D([0], [0], color=mcolor, lw=lw, ls='-', marker=marker, label=lab)
            )
        _rad_model_legend = rad_ax.legend(handles=model_handles, frameon=False, prop={'size': leg_size}, loc='upper right')
        if yz_ax is not None:
            rad_ax.add_artist(_rad_model_legend)
            yz_ax.legend(
                handles=[
                    Line2D([0], [0], color='k', lw=lw, ls='-', label='Radius'),
                    Line2D([0], [0], color='k', lw=lw, ls='--', label='Y$_{\\rm atm}$'),
                    Line2D([0], [0], color='k', lw=lw, ls=':', label='Z$_{\\rm atm}$'),
                ],
                frameon=False,
                prop={'size': leg_size},
                loc='lower right',
            )

        comp_legend_handles = [
            Line2D([0], [0], c='k', ls='-', lw=lw, label='$S$ Profle'),
            Line2D([0], [0], c='k', ls='dotted', lw=lw, label='$Y$ Profle'),
            Line2D([0], [0], c='k', ls='--', lw=lw, label='$Z$ Profle'),
        ]
        comp_ax.legend(handles=comp_legend_handles, frameon=False, prop={'size':leg_size}, loc='upper left')

        if (flux_ax is not None) and flux_color_handles:
            # Replace labels with requested ages (colors map to ages)
            # for h, age_req in zip(flux_color_handles, ages):
            #     h.set_label(f"{age_req:.3f} Gyr")
            # flux_color_legend = flux_ax.legend(
            #     handles=flux_color_handles,
            #     frameon=False,
            #     prop={'size': leg_size},
            #     loc='upper left',
            # )
            # flux_ax.add_artist(flux_color_legend)

            flux_style_legend = [
                Line2D([0], [0], color='k', lw=lw, ls='-', label='Total Flux'),
                Line2D([0], [0], color='k', lw=lw, ls='--', label='Convec. flux'),
                Line2D([0], [0], color='k', lw=lw, ls='dotted', label='Cond+Rad. flux'),
            ]
            flux_ax.legend(
                handles=flux_style_legend,
                frameon=False,
                prop={'size': leg_size},
                loc='lower left',
            )

        if self.plot_rock_phases:
            melt_frac_ax.set_ylabel('Melt Fraction', fontsize=20)
            melt_frac_ax.set_ylim(0, 1.0)
            melt_frac_ax.set_yticks(np.arange(0, 1.1, 0.1))
            melt_frac_ax.grid(False)

        # ---------- Axis formatting / reference lines ----------
        def _final_age_or_saved(j):
            if j < len(self.final_age):
                try:
                    fa = float(self.final_age[j])
                    if np.isfinite(fa) and fa > 0:
                        return fa
                except (TypeError, ValueError):
                    pass
            age_series = np.asarray(self.data_age[j], dtype=float) * const.s_to_Gyr
            if age_series.size == 0:
                return age_floor_gyr
            return float(np.nanmax(age_series))

        max_age_x = max(_final_age_or_saved(j) for j in selected_models)
        x_left = age_floor_gyr if log_age else -0.1
        x_right = max(max_age_x + max_age_x * 0.1, age_floor_gyr * 10.0) if log_age else max_age_x + max_age_x * 0.1

        # Draw reference lines and compute ylim for all planet types present
        _rad_ylims = {'Jupiter': (9, 14), 'Saturn': (8, 14),
                       'Uranus': (1.0, 6.0), 'Neptune': (1.0, 6.0),
                       'Sub_Neptune': (1.0, 6.0), 'Super_Earth': (1.0, 6.0),
                       'Super_Jupiter': (5, 20), 'Exoplanet': (5, 20)}
        rad_ylo = min(_rad_ylims.get(p, (0, 5))[0] for p in planet_set)
        rad_yhi = max(_rad_ylims.get(p, (0, 5))[1] for p in planet_set)
        rad_ax.set_ylim(rad_ylo, rad_yhi)
        for _p in planet_set:
            if _p == 'Jupiter':
                rad_ax.axhline(y=10.973, c='r', ls='dashdot')
                rad_ax.axhline(y=11.209, c='k', ls='dashdot', zorder=1)
                lum_ax.axhline(y=(7.485+0.160) * 1e3 * 4 * np.pi * (const.rjup)**2, ls='--', c='m', lw=3) # Li 2018 value
                lum_ax.axhline(y=(5.444-0.425) * 1e3 * 4 * np.pi * (const.rjup)**2, ls='--', c='m', lw=3)
            elif _p == 'Saturn':
                rad_ax.axhline(y=9.1402, c='r', ls='dashdot')
                rad_ax.axhline(y=9.449, c='k', ls='dashdot')
                lum_ax.axhline(y = 4.952 * 1e3 * 4 * np.pi * (const.rsat)**2, ls='--', c='m', lw=3) # Li 2010 value
            elif _p == 'Uranus':
                rad_ax.axhline(y=4.0, c='r', ls='dashdot')
                rad_ax.axhline(y=4.007, c='k', ls='dashdot')
                lum_ax.axhline(y=0.042 * 1e3 * 4 * np.pi * (4.0 * const.rearth)**2, ls='--', c='m', lw=3) # P & C 1991 value
                lum_ax.axhline(y=0.089 * 1e3 * 4 * np.pi * (4.0 * const.rearth)**2, ls='--', c='m', lw=3) # Irwin 2024 value
            elif _p == 'Neptune':
                rad_ax.axhline(y=3.883, c='r', ls='dashdot')
                lum_ax.axhline(y=0.433 * 1e3 * 4 * np.pi * (3.883 * const.rearth)**2, ls='--', c='m', lw=3) # P & C 1991 value

        rad_ax.set_xlim(x_left, x_right)
        if log_age:
            rad_ax.set_xscale('log')
        rad_ax.axvline(x = 4.56, color='k', ls='dashdot')
        rad_ax.set_ylabel(r'Mean Radius [R$_\oplus$]')
        rad_ax.set_xlabel('Age [Gyr]')
        if yz_ax is not None:
            yz_ax.set_xlim(x_left, x_right)
            if log_age:
                yz_ax.set_xscale('log')
            yz_ax.set_ylim(0.0, 0.3)
            yz_ax.set_ylabel(r'Atmosphere $Y,\ Z$')

        teff_ax.set_ylim(0, 1000)
        teff_ax.set_xlim(x_left, x_right)
        if log_age:
            teff_ax.set_xscale('log')
            lum_ax.set_xscale('log')
        teff_ax.set_ylabel(r'T$_{\rm eff}$ [Kelvin]')
        teff_style_legend = [
            Line2D([0], [0], color='k', lw=lw, ls='-', label=r'$T_{\rm eff}$'),
            Line2D([0], [0], color='k', lw=lw, ls='--', label='Luminosity'),
        ]
        teff_ax.legend(handles=teff_style_legend, frameon=False, prop={'size': leg_size}, loc='upper right')

        lum_ax.set_yscale('log')
        lum_series = [
            np.asarray(self.luminosity[idx], dtype=float).ravel()
            for idx in selected_models
            if np.size(self.luminosity[idx])
        ]
        lum_all = np.concatenate(lum_series) if lum_series else np.array([])
        lum_all = lum_all[np.isfinite(lum_all) & (lum_all > 0)]
        if lum_all.size:
            lum_min = float(np.nanmin(lum_all))
            lum_max = float(np.nanmax(lum_all))
            # Include intrinsic flux reference values in ylim
            _lum_refs = []
            for _p in planet_set:
                if _p == 'Uranus':
                    _lum_refs.append(0.089 * 1e3 * 4 * np.pi * (4.0 * const.rearth)**2)
                elif _p == 'Neptune':
                    _lum_refs.append(0.433 * 1e3 * 4 * np.pi * (3.883 * const.rearth)**2)
            if _lum_refs:
                lum_min = min(lum_min, min(_lum_refs))
            lum_ax.set_ylim(max(lum_min * 1e-2, 1e-40), lum_max * 1.2)
        lum_ax.set_ylabel('Luminosity [erg/s]')
        lum_ax.grid(False)

        for _p in planet_set:
            if _p == 'Jupiter':
                teff_ax.axhline(y=125.6, color='r', ls='dashdot')
            elif _p == 'Saturn':
                teff_ax.axhline(y=96.5, color='r', ls='dashdot')
            elif _p == 'Uranus':
                teff_ax.axhline(y=59.1, color='r', ls='dashdot')
            elif _p == 'Neptune':
                teff_ax.axhline(y=59.3, color='r', ls='dashdot')
        teff_ax.axvline(x = 4.56, color='k', ls='dashdot')
        teff_ax.set_xlabel('Age [Gyr]')

        ages_text = ", ".join([f"{a:.3f}" for a in ages])
        if temp_log_ax is not None:
            temp_log_ax.text(0.35, 0.95, f'Ages [Gyr]: {ages_text}', transform=temp_log_ax.transAxes,
                         ha='left', va='top', fontsize=24)
        else:
            temp_ax.text(0.03, 0.97, f'Ages [Gyr]: {ages_text}', transform=temp_ax.transAxes,
                         ha='left', va='top', fontsize=24)

        if 'Jupiter' in planet_set:
            # Use the first Jupiter model for the fill_between x-axis
            _jup_idx = next(j for j in selected_models if self.planet[j] == 'Jupiter')
            comp_ax.fill_between(x=self.m_mean[_jup_idx],
                                 y1=(0.238+0.005)*(1-0.017),
                                 y2=(0.238-0.005)*(1-0.017),
                                 color='m', alpha=0.25)
            comp_ax.axhline(y = 3.16 * 0.017, c='b', ls='dashdot')

        if temp_log_ax is not None:
            temp_log_ax.set_xlabel(r'Pressure [GPa]')
            temp_log_ax.set_ylabel(r'Temperature [K]')
            temp_log_ax.set_xlim(1e-4, 1e4)
            temp_log_ax.set_ylim(100, max(max_temp_kK + 2.0, 2.0) * 1e3)
            temp_log_ax.invert_xaxis()

        if flux_ax is not None:
            flux_ax.set_xlabel('Mass [M$_P$]')
            flux_ax.set_ylabel('Fluxes')
            flux_ax.set_ylim(1e16,1e27)
            flux_ax.set_yscale('symlog')
            flux_ax.set_xlim(0.0, 1.01)

            if self.zm_plot and zm_xmins:
                flux_ax.set_xlim(min(zm_xmins), 1.0005)

        temp_ax.set_xlabel('Normalized Mass [M$_P$]')
        temp_ax.set_xlim(0, 1.01)
        temp_ax.set_ylim(0, max(max_temp_kK + 2.0, 2.0))
        temp_ax.set_ylabel('Temperature [$10^3$ K]')

        comp_ax.set_xlabel('Mass [M$_P$]' if not rad_profiles else r'Radius [R$_\oplus$]')
        comp_ax.set_ylabel(r'$Y,\ Z$')
        comp_ax.set_xlim(0, 1.05)
        if planet_set == {'Jupiter'}:
            comp_ax.set_ylim(0, 0.7)
        else:
            comp_ax.set_ylim(0, 1.0)
        if self.zm_plot and zm_xmins and (not rad_profiles):
            comp_ax.set_xlim(min(zm_xmins), 1.0005)

        entropy_ax.set_ylim(0, 12)
        entropy_ax.set_ylabel(r'$S$ [k$_b$/baryon]')
        entropy_ax.grid(False)

        if dens_ax is not None:
            dens_ax.set_ylabel(r'$\rho$ [g cm$^{-3}$]')
            dens_ax.set_xlabel('Mass [M$_P$]')

        if brunt_ax is not None:
            brunt_ax.set_xlabel('Mass [M$_P$]')
            brunt_ax.set_ylabel(r'$N/\omega_{\rm dyn}$')
            brunt_ax.set_ylim(-0.1, 5)
            brunt_ax.set_xlim(0, 1.05)

        if miscellaneous_ax is not None:
            if not self.plot_Js:
                miscellaneous_ax.set_xlabel('Mass [M$_P$]')
                miscellaneous_ax.set_ylabel(r'Mass Zone Count (N)')
                ax92.set_ylabel(r'$dm$')
            else:
                miscellaneous_ax.set_xlabel('Age [Gyr]')
                miscellaneous_ax.set_ylabel(r'$J_2$')
                ax92.set_ylabel(r'$J_4$, $J_6$')
                miscellaneous_ax.tick_params(axis='y', which='both', colors='g', right=False)
                miscellaneous_ax.yaxis.label.set_color('g')
                miscellaneous_ax.yaxis.get_offset_text().set_color('g')
                miscellaneous_ax.spines['left'].set_color('g')
                ax92.tick_params(axis='y', which='both', colors='r', left=False)
                ax92.yaxis.label.set_color('r')
                ax92.yaxis.get_offset_text().set_color('r')
                ax92.spines['right'].set_color('r')
                _gas_giants = planet_set & {'Jupiter', 'Saturn'}
                _ice_giants = planet_set & {'Uranus', 'Neptune'}
                if _gas_giants and not _ice_giants:
                    miscellaneous_ax.set_ylim(10000, 20000)
                    ax92.set_ylim(-1500, 120)
                elif _ice_giants and not _gas_giants:
                    miscellaneous_ax.set_ylim(0, 6000)
                    ax92.set_ylim(-100, 50)
                elif _gas_giants and _ice_giants:
                    miscellaneous_ax.set_ylim(0, 20000)
                    ax92.set_ylim(-1500, 50)
                for _p in planet_set:
                    if _p == 'Jupiter':
                        miscellaneous_ax.axhline(y=14696.572, color='g', lw=3, ls='dashdot')
                        ax92.axhline(y=-586.609, color='r', lw=3, ls='dashdot')
                        ax92.axhline(y=34.201, color='darkorange', lw=2, ls='dashdot')
                    elif _p == 'Saturn':
                        miscellaneous_ax.axhline(y=16290.573, color='g', lw=3, ls='dashdot')
                        ax92.axhline(y=-935.314, color='r', lw=3, ls='dashdot')
                        ax92.axhline(y=86.340, color='darkorange', lw=2, ls='dashdot')
                    elif _p == 'Uranus':
                        miscellaneous_ax.axhline(y=3509.709, color='g', lw=3, ls='dashdot')
                        ax92.axhline(y=-35.04, color='r', lw=3, ls='dashdot')
                    elif _p == 'Neptune':
                        miscellaneous_ax.axhline(y=3408.43, color='g', lw=3, ls='dashdot')
                        ax92.axhline(y=-33.40, color='r', lw=3, ls='dashdot')
                miscellaneous_ax.set_xlim(x_left, x_right)
                ax92.set_xlim(x_left, x_right)
                if log_age:
                    miscellaneous_ax.set_xscale('log')
                    ax92.set_xscale('log')

        ticks_axes = [comp_ax, temp_ax, rad_ax]
        if yz_ax is not None:
            ticks_axes.append(yz_ax)
        if temp_log_ax is not None:
            ticks_axes.append(temp_log_ax)
        if dens_ax is not None:
            ticks_axes.append(dens_ax)
        if flux_ax is not None:
            ticks_axes.append(flux_ax)
        if brunt_ax is not None:
            ticks_axes.append(brunt_ax)
        for ax in ticks_axes:
            ax.minorticks_on()
            ax.yaxis.set_ticks_position('both')
            ax.xaxis.set_ticks_position('both')

        teff_ax.tick_params(direction='in', length=10, width=2, grid_alpha=0.75, which='minor', right=False, top=True)
        lum_ax.tick_params(direction='in', length=17, width=2, grid_alpha=0.75, which='major', right=True, left=False)

        if save:
            os.makedirs('plots/plot_stills', exist_ok=True)
            plt.savefig('plots/plot_stills/{}.png'.format(output_file), bbox_inches='tight', dpi=200)
            plt.show()
        else:
            plt.show()
        plt.close()

    def movie_evol_plots_by_age(
            self,
            output_file,
            labels=[],
            leg_size=20,
            panel_num=6,
            dt_gyr=None,
            tmin_gyr=None,
            tmax_gyr=None,
            snap='nearest_left',
            clean_outdir=True,
            plot_water_misc=False,
            rad_profiles=False,
            plot_he_rain=False,
            rad_temp_plot=False,
            rad_target=None,
            age_target=None,
        ):
        """
        Generate movie frames by calling still_plot() at each time step.

        Uses the same panel layout, axes, and formatting as still_plot() and
        age_plot() so that movie frames are visually consistent with still
        plots.

        Parameters
        ----------
        output_file : str
            Name for the output directory under plots/.
        labels : list of str
            Legend labels, one per model path.
        leg_size : int
            Legend font size passed through to still_plot.
        panel_num : int
            Number of panels (4, 6, or 9) — same options as still_plot.
        dt_gyr : float or None
            Time spacing (Gyr) between frames.  If None, uses model 0's
            saved time snapshots.
        tmin_gyr, tmax_gyr : float or None
            Optional bounds on the frame time grid.
        snap : str
            Snapshot-matching strategy: 'nearest_left', 'nearest', 'floor',
            or 'ceil'.
        clean_outdir : bool
            If True, remove existing frame directory before generating.
        plot_water_misc, rad_profiles, plot_he_rain, rad_temp_plot : bool
            Feature toggles passed through to still_plot.
        """

        outdir = os.path.join('plots', output_file)

        if clean_outdir and os.path.isdir(outdir):
            outdir_abs = os.path.abspath(outdir)
            plots_abs  = os.path.abspath('plots')
            if not outdir_abs.startswith(plots_abs + os.sep):
                raise ValueError(f"Refusing to clean directory outside plots/: {outdir_abs}")
            shutil.rmtree(outdir)

        os.makedirs(outdir, exist_ok=True)

        if len(labels) == 0:
            raise ValueError("labels must not be empty.")
        if len(labels) != len(self.paths):
            raise ValueError("labels must have the same length as the number of model paths.")

        # ---------- snapshot index helpers ----------
        def _nearest_indices(arr, targets, how):
            arr = np.asarray(arr)
            targets = np.asarray(targets)

            idx_right = np.searchsorted(arr, targets, side='left')
            idx_left  = np.clip(idx_right - 1, 0, len(arr) - 1)
            idx_right = np.clip(idx_right,       0, len(arr) - 1)

            if how == 'floor':
                return idx_left
            if how == 'ceil':
                return idx_right

            dl = np.abs(arr[idx_left]  - targets)
            dr = np.abs(arr[idx_right] - targets)

            if how == 'nearest':
                pick_right = dr <= dl
                return np.where(pick_right, idx_right, idx_left)
            else:
                pick_right = dr < dl
                return np.where(pick_right, idx_right, idx_left)

        # ---------- build target time grid ----------
        times_gyr = [np.asarray(self.data_age[j]) * const.s_to_Gyr
                     for j in range(len(self.paths))]

        if dt_gyr is not None:
            starts = [t[0] for t in times_gyr]
            ends   = [t[-1] for t in times_gyr]

            t0 = tmin_gyr if tmin_gyr is not None else max(starts)
            t1_overlap = min(ends)
            t1 = min(t1_overlap, tmax_gyr) if tmax_gyr is not None else t1_overlap

            if t1 <= t0:
                raise ValueError(
                    f"No common time overlap across models in [{t0:.6f}, {t1:.6f}] Gyr."
                )

            target_times = np.arange(t0, t1 + 0.5 * dt_gyr, dt_gyr)
        else:
            target_times = times_gyr[0].copy()

        # ---------- compute global max temperature for fixed y-axis ----------
        _global_max_temp_kK = 0.0
        for j in range(len(self.paths)):
            _t_arr = np.asarray(self.temp[j])
            _tmax = float(np.nanmax(_t_arr)) * 1e-3  # K -> kK
            _global_max_temp_kK = max(_global_max_temp_kK, _tmax)

        # ---------- main frame loop ----------
        # Disable LaTeX for movie frames (much faster text rendering)
        _prev_usetex = plt.rcParams.get('text.usetex', False)
        plt.rc('text', usetex=False)

        try:
            from tqdm import tqdm
            frame_iter = tqdm(enumerate(target_times), total=len(target_times),
                              desc='Generating movie frames', unit='frame')
        except ImportError:
            frame_iter = enumerate(target_times)

        for k, T_gyr in frame_iter:
            frame_name = f"{(k+1):04d}_t{T_gyr:0.3f}Gyr"
            save_path = os.path.join(outdir, frame_name + '.png')

            self.still_plot(
                age=T_gyr,
                output_file=frame_name,
                labels=labels,
                leg_size=leg_size,
                save=True,
                save_path=save_path,
                plot_water_misc=plot_water_misc,
                rad_profiles=rad_profiles,
                plot_he_rain=plot_he_rain,
                panel_num=panel_num,
                rad_temp_plot=rad_temp_plot,
                movie_mode=True,
                temp_ylim_kK=_global_max_temp_kK,
                rad_target=rad_target,
                age_target=age_target,
            )

        # Restore LaTeX setting
        plt.rc('text', usetex=_prev_usetex)

    def mass_loss_plot(self, output_file='mass_loss', labels=None,
                       model_indices=None, log_age=True, save=True,
                       save_path=None, cmap=None, leg_size=12):
        """
        Diagnostic plot of atmospheric mass-loss evolution.

        Creates a 2x2 panel figure with:
        - Mdot(t)  [g/s]
        - M_env(t) [Earth masses]
        - M_planet(t) [Earth masses]
        - L_adv(t) [erg/s]
        """
        if model_indices is None:
            model_indices = range(len(self.paths))
        if labels is None:
            labels = [os.path.basename(p) for p in self.paths]
        if cmap is None:
            cmap = plt.cm.tab10

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        ax_mdot, ax_menv = axes[0]
        ax_mpl, ax_ladv = axes[1]

        for idx, j in enumerate(model_indices):
            if self.Mdot[j] is None:
                continue
            color = cmap(idx % 10)
            age_gyr = np.asarray(self.data_age[j]) * const.s_to_Gyr
            mdot = np.asarray(self.Mdot[j])
            m_env = np.asarray(self.M_env[j]) / const.mearth
            m_pl = np.asarray(self.M_planet_current[j]) / const.mearth
            l_adv = np.asarray(self.L_adv[j])

            # Ensure arrays match in length (trim to shortest)
            n = min(len(age_gyr), len(mdot), len(m_env), len(m_pl), len(l_adv))
            age_gyr = age_gyr[:n]
            mdot = mdot[:n]
            m_env = m_env[:n]
            m_pl = m_pl[:n]
            l_adv = l_adv[:n]

            lbl = labels[idx] if idx < len(labels) else None
            plot_fn = ax_mdot.loglog if log_age else ax_mdot.semilogy
            plot_fn(age_gyr, mdot, color=color, label=lbl)

            if log_age:
                ax_menv.semilogx(age_gyr, m_env, color=color)
                ax_mpl.semilogx(age_gyr, m_pl, color=color)
                ax_ladv.loglog(age_gyr, l_adv, color=color)
            else:
                ax_menv.plot(age_gyr, m_env, color=color)
                ax_mpl.plot(age_gyr, m_pl, color=color)
                ax_ladv.semilogy(age_gyr, l_adv, color=color)

        ax_mdot.set_ylabel(r'$\dot{M}$ [g/s]')
        ax_mdot.set_xlabel('Age [Gyr]')
        ax_mdot.legend(fontsize=leg_size)

        ax_menv.set_ylabel(r'$M_{\rm env}$ [$M_\oplus$]')
        ax_menv.set_xlabel('Age [Gyr]')

        ax_mpl.set_ylabel(r'$M_{\rm planet}$ [$M_\oplus$]')
        ax_mpl.set_xlabel('Age [Gyr]')

        ax_ladv.set_ylabel(r'$L_{\rm adv}$ [erg/s]')
        ax_ladv.set_xlabel('Age [Gyr]')

        fig.suptitle('Atmospheric Mass Loss Diagnostics', fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        if save:
            path = save_path or (output_file + '.pdf')
            fig.savefig(path, dpi=150)
            plt.close(fig)
        else:
            plt.show()
