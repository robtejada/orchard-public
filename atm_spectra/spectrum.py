"""
Atmospheric boundary-condition (BC) spectrum table interpolator for ORCHARD.

This module provides the :class:`BCtable` class, a 4-D quadlinear interpolator
over a grid of atmospheric model outputs spanning

    (T_eff, log g, Y, Z)

where ``T_eff`` is the effective temperature [K], ``log g`` is the surface
gravity (cgs, dex), ``Y`` is the helium mass fraction, and ``Z`` is the
metallicity in **solar units** (Z / Z_sun). For each grid point the table
stores the specific entropy, emergent frequency-flux spectrum F_nu
(erg cm^-2 s^-1 Hz^-1) on a common wavelength grid in microns, and the
temperature-pressure (T-P) profile of the atmosphere.

The table is loaded from a compressed ``.npz`` datacube (default:
``atm_spectra/BC_grid_YZmerged_FULL.npz``). Interpolation is linear in
``T_eff`` and ``log g``, and log-linear in ``Y`` and ``Z``; fluxes are
interpolated in log10-space to preserve their power-law character.

Unit conventions (see :meth:`BCtable.interpolate_spectrum_4D`):

- ``Y_in`` is a **mass fraction** (dimensionless).
- ``Z_in`` is in **solar units** (Z / Z_sun), not a mass fraction. Callers
  that hold Z as a mass fraction (as ORCHARD does internally) must divide by
  the chosen solar reference (e.g. ``Z_SOLAR_CHEN23 = 0.017``) first.
- Returned ``F_interp`` is a **frequency** flux F_nu in
  ``erg cm^-2 s^-1 Hz^-1``, not a wavelength flux.

Author
------
Yixian Chen (Princeton University) -- table construction and interpolator
implementation (Chen et al. 2026). Integrated into ORCHARD (Tejada Arevalo et al. 2026b).
"""

__author__ = "Yixian Chen"

import numpy as np


class spectrum:
    """
    Atmospheric boundary-condition table.
    Provides interpolation of entropy, spectra, and TP profiles
    over (Teff, logg, logY, logZ).

    Attributes:
        - Teff_grid: Teff[K] list with N_T
        - logg_grid: log g list wiht N_g
        - Y_grid: helium fraction data points with N_Y
        - Z_grid: metallicity data points with N_Z
        
        - entropy: S[k_B] per baryon data defined over grid  N_T x N_g x
N_Y x N_Z
        - lam: wavelength[micron] grid with N_lam
        - flux: Spectra data F_nu [erg/(cm^2 s Hz)] defined over grid N_T x
N_g x N_Y x N_Z x N_lam (frequency flux, not wavelength flux)
        - T_profile: list of temperature[K] data points for the TP profile 
over grid N_Teff x N_g x N_Y x N_Z x 99
        - P_profile: list of pressure[bar] data points for the TP profile 
over grid N_Teff x N_g x N_Y x N_Z x 99

    """

    def __init__(self, filename="atm_spectra/spectrum_data_YZ_C26.npz"):

        self.filename = filename
        data = np.load(self.filename)

        # Grids
        self.Teff_grid = data["T_eff"]
        self.logg_grid = data["grav"]
        self.logY_grid     = np.log10(data["Y"])
        self.logZ_grid     = np.log10(data["Z"])

        # Data
        self.entropy   = data["entropy"]
        self.lam       = data["lam"]
        self.flux      = data["flux"]
        self.T_profile = data["T_profile"]
        self.P_profile = data["P_profile"]

        # Dimensions
        self.N_T = len(self.Teff_grid)
        self.N_g = len(self.logg_grid)
        self.N_Y = len(self.logY_grid)
        self.N_Z = len(self.logZ_grid)

        # Sanity check
        assert self.entropy.shape == (
            self.N_T, self.N_g, self.N_Y, self.N_Z
        ), "Entropy grid shape mismatch"

    def _find_indices(self, Teff, logg, logY, logZ):
        """
        Locate lower grid indices for 4D interpolation.
        Returns (iT, ig, iY, iZ)
        """
    
        # ======================================================
        # 1. Teff (piecewise linear)
        # ======================================================
    
        if Teff < 500.0:
            x0 = 50.0
            dx = 50.0
            iT = int((Teff - x0) / dx)
            iT = max(0, min(iT, 8))
        else:
            x0 = 500.0
            dx = 100.0
            offset = 9
            iT = offset + int((Teff - x0) / dx)
            iT = max(offset, min(iT, 16))
    
        # ======================================================
        # 2. logg (uniform)
        # ======================================================
    
        Ng = self.N_g
        dg = (self.logg_grid[-1] - self.logg_grid[0]) / (Ng - 1)
        ig = int((logg - self.logg_grid[0]) / dg)
        ig = max(0, min(ig, Ng - 2))
    
        # ======================================================
        # 3. Y (log-linear)
        # ======================================================
    
        Ny = self.N_Y
        dY = (self.logY_grid[-1] - self.logY_grid[0]) / (Ny - 1)
        iY = int((logY - self.logY_grid[0]) / dY)
        iY = max(0, min(iY, Ny - 2))
    
        # ======================================================
        # 4. Z (log-linear)
        # ======================================================
    
        Nz = self.N_Z
        dZ = (self.logZ_grid[-1] - self.logZ_grid[0]) / (Nz - 1)
        iZ = int((logZ - self.logZ_grid[0]) / dZ)
        iZ = max(0, min(iZ, Nz - 2))
    
        return iT, ig, iY, iZ

    def _quadlinear_weights(self,
                            Teff, logg, logY, logZ,
                            T0, T1,
                            logg0, logg1,
                            logY0, logY1,
                            logZ0, logZ1):
        """
        Compute 16 quadlinear interpolation weights
        for 4D interpolation over:
            (Teff, logg, logY, logZ)
    
        Returns:
            w : array of length 16
        """
    
        tT = (Teff - T0) / (T1 - T0)
        tg = (logg - logg0) / (logg1 - logg0)
        tY = (logY - logY0) / (logY1 - logY0)
        tZ = (logZ - logZ0) / (logZ1 - logZ0)
    
        w = np.zeros(16)
    
        idx = 0
        for dT in [0, 1]:
            for dg in [0, 1]:
                for dY in [0, 1]:
                    for dZ in [0, 1]:
    
                        wT = tT if dT else (1 - tT)
                        wg = tg if dg else (1 - tg)
                        wY = tY if dY else (1 - tY)
                        wZ = tZ if dZ else (1 - tZ)
    
                        w[idx] = wT * wg * wY * wZ
                        idx += 1
    
        return w
    
    def interpolate_spectrum_4D(self, Teff_in, logg_in, Y_in, Z_in):
        """
        Quadlinear (4D) interpolation of the tabulated emergent spectrum at
        (Teff, logg, Y, Z). Interpolation is performed in log10(flux) with
        linear weights in (Teff, logg, log10 Y, log10 Z); inputs outside the
        grid are clipped to the nearest edge.

        Parameters
        ----------
        Teff_in : float
            Effective temperature [K].
        logg_in : float
            Surface gravity, log10(g / cm s^-2) (cgs).
        Y_in : float
            Helium mass fraction (dimensionless, must be > 0).
        Z_in : float
            Metallicity in SOLAR UNITS (i.e. Z / Z_sun, must be > 0).
            NOTE: this is NOT the mass fraction used internally by ORCHARD.
            Callers holding Z as a mass fraction must divide by the chosen
            solar reference (e.g. Z_SOLAR_CHEN23 = 0.017) before calling.

        Returns
        -------
        lam : ndarray, shape (N_lam,)
            Wavelength grid [micron].
        F_interp : ndarray, shape (N_lam,)
            Frequency flux F_nu [erg cm^-2 s^-1 Hz^-1] (NOT wavelength flux).
        """
        if Y_in <= 0 or Z_in <= 0:
            raise ValueError("Y and Z must be positive.")
            
        # --- clip to bounds ---
        Teff_in = np.clip(Teff_in,
                          self.Teff_grid[0],
                          self.Teff_grid[-1])
    
        logg_in = np.clip(logg_in,
                          self.logg_grid[0],
                          self.logg_grid[-1])
    
        logY_in = np.clip(np.log10(Y_in),
                          self.logY_grid[0],
                          self.logY_grid[-1])
    
        logZ_in = np.clip(np.log10(Z_in),
                          self.logZ_grid[0],
                          self.logZ_grid[-1])
    
        # --- grid indices ---
        i, j, k, l = self._find_indices(
            Teff_in, logg_in, logY_in, logZ_in
        )
    
        # --- local grid values ---
        T0, T1 = self.Teff_grid[i], self.Teff_grid[i+1]
        logg0, logg1 = self.logg_grid[j], self.logg_grid[j+1]
        logY0, logY1 = self.logY_grid[k], self.logY_grid[k+1]
        logZ0, logZ1 = self.logZ_grid[l], self.logZ_grid[l+1]
    
        # --- weights ---
        w = self._quadlinear_weights(
            Teff_in, logg_in, logY_in, logZ_in,
            T0, T1,
            logg0, logg1,
            logY0, logY1,
            logZ0, logZ1
        )
    
        # --- gather 16 corner spectra ---
        F = []
        for dT in [0, 1]:
            for dg in [0, 1]:
                for dY in [0, 1]:
                    for dZ in [0, 1]:
                        F.append(self.flux[
                            i+dT, j+dg, k+dY, l+dZ, :
                        ])
    
        F = np.array(F)  # shape (16, N_lam)
    
        # --- log-space interpolation ---
        logF_interp = np.sum(w[:, None] * np.log10(F), axis=0)
        F_interp = 10**logF_interp
    
        return self.lam, F_interp
        
    def spectrum_evolution(self,
                           N_frames,
                           age_path,
                           Teff_path,
                           logg_path,
                           Z_path,
                           Y_path,
                           plot=False,
                           figname="spectrum_evolution.pdf"):
        """
        Compute spectral evolution along a given trajectory.
    
        Parameters
        ----------
        N_frames : int
            Number of snapshots along the path.
    
        age_path : array (N_frames)
            Age in Gyr (recommended) or consistent time unit.
    
        Teff_path, logg_path, Z_path, Y_path : arrays (N_frames)
            Evolutionary trajectory.
    
        plot : bool
            If True, generate 3-panel visualization.
    
        Returns
        -------
        lam : array (N_lam)
        all_flux : array (N_frames, N_lam)
        """
        if np.isscalar(age_path):

            lam, F = self.interpolate_spectrum_4D(
                Teff_path, logg_path, Y_path, Z_path
            )
        
            if plot:
                print("Single snapshot detected. Skip plotting evolution tracks. ")
        
            return lam, F[None, :]   # keep consistent 2D shape
    
        # ------------------------------------------------------
        # Sanity checks
        # ------------------------------------------------------
        assert len(age_path) == N_frames
        assert len(Teff_path) == N_frames
        assert len(logg_path) == N_frames
        assert len(Z_path) == N_frames
        assert len(Y_path) == N_frames
    
        # ------------------------------------------------------
        # Colormap setup (log-time)
        # ------------------------------------------------------
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        from matplotlib import cm
    
        times = np.array(age_path)
        cmap = cm.magma_r
        norm = mpl.colors.LogNorm(vmin=times.min(), vmax=times.max())
    
        # ------------------------------------------------------
        # Interpolate spectra
        # ------------------------------------------------------
        all_flux = []
    
        for Teff_i, logg_i, Y_i, Z_i in zip(
                Teff_path, logg_path, Y_path, Z_path):
    
            lam, F_interp = self.interpolate_spectrum_4D(
                Teff_i, logg_i, Y_i, Z_i)
    
            all_flux.append(F_interp)
    
        all_flux = np.array(all_flux)  # shape (N_frames, N_lam)
    
        # ------------------------------------------------------
        # Plotting
        # ------------------------------------------------------
        if plot:
    
            fig = plt.figure(figsize=(13, 9))
    
            axY = fig.add_subplot(2, 2, 1, projection='3d')
            axZ = fig.add_subplot(2, 2, 2, projection='3d')
            ax2 = fig.add_subplot(2, 1, 2)
    
            # ==================================================
            # Y cube
            # ==================================================
            for Teff_i, logg_i, Y_i, t in zip(
                    Teff_path, logg_path, Y_path, times):
    
                axY.scatter(logg_i, Teff_i, Y_i,
                            s=60,
                            color=cmap(norm(t)),
                            edgecolor="k",
                            depthshade=True)
    
            axY.plot(logg_path, Teff_path, Y_path,
                     color="k", lw=1, ls=":")
    
            axY.set_xlabel(r"$\log_{10} g$", fontsize=12)
            axY.set_ylabel(r"$T_{\rm eff}$ [K]", fontsize=12)
            axY.set_zlabel(r"$Y$", fontsize=12)
            axY.tick_params(labelsize=8)
            axY.set_title("Helium Evolution")
            axY.zaxis.label.set_rotation(90)
            axY.zaxis.set_rotate_label(False)
            axY.view_init(elev=20, azim=-135)
    
            # ==================================================
            # Z cube
            # ==================================================
            for Teff_i, logg_i, Z_i, t in zip(
                    Teff_path, logg_path, Z_path, times):
    
                axZ.scatter(logg_i, Teff_i, Z_i,
                            s=60,
                            color=cmap(norm(t)),
                            edgecolor="k",
                            depthshade=True)
    
            axZ.plot(logg_path, Teff_path, Z_path,
                     color="k", lw=1, ls=":")
    
            axZ.set_xlabel(r"$\log_{10} g$", fontsize=12)
            axZ.set_ylabel(r"$T_{\rm eff}$ [K]", fontsize=12)
            axZ.set_zlabel(r"$Z/Z_\odot$", fontsize=12)
            axZ.tick_params(labelsize=8)
            axZ.zaxis.label.set_rotation(90)
            axZ.zaxis.set_rotate_label(False)
            axZ.set_title("Metallicity Evolution")
            axZ.view_init(elev=20, azim=-135)
    
            # ==================================================
            # Spectra
            # ==================================================
            for flux, t in zip(all_flux, times):
                ax2.plot(lam,
                         flux * 1e-6 / 5.371,  # keep your scaling
                         lw=1.,
                         color=cmap(norm(t)))
    
            ax2.set_xscale("log")
            ax2.set_yscale("log")
            ax2.set_xlabel(r"$\lambda$ [micron]", fontsize=14)
            ax2.set_ylabel(r"$F_\nu$ (erg cm$^{-2}$ s$^{-1}$ Hz$^{-1}$)", fontsize=14)
            ax2.set_title("Spectral Evolution")
            ax2.grid(True, which="both", ls="dotted", alpha=0.3)
            ax2.set_ylim(1e-15, 1e-5)
    
            # ==================================================
            # Colorbar
            # ==================================================
            sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
    
            cbar = fig.colorbar(sm, ax=ax2, pad=0.02)
            cbar.set_label("Time [Gyr]", fontsize=14)
    
            plt.tight_layout()
            plt.savefig(figname)
            plt.show()
    
        return lam, all_flux
    

