import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator as RGI
from scipy.interpolate import interp1d

class spectra:
    def __init__(self, planet, cloud=True, irradiation=True):
        """
        This class loads up to 3 DataFrames of fluxes at different metallicities:
          * self.spectra_df    ->  Z = 0   (i.e., 0 × solar)
          * self.spectra_df_z1 ->  Z = 1   (1 × solar metallicity)
          * self.spectra_df_z3 ->  Z = 3.16 (~3× solar)
        Then it merges them into a single 4D interpolation:
          (T_int, logg, lambda, Z) -> flux_nu.
        """
        # -------------------------------------------------
        # 1) Load data for your chosen planet configuration
        # -------------------------------------------------
        if planet == 'Jupiter':
            if cloud and irradiation:
                self.spectra_df = pd.read_csv('atm_spectra/jupiter_irradiated_clouds_CMSY025_spectrum_0solarZEOS.csv')
                self.spectra_df_z1 = pd.read_csv('atm_spectra/jupiter_irradiated_clouds_CMSY025_spectrum_1solarZEOS.csv')
                self.spectra_df_z3 = pd.read_csv('atm_spectra/jupiter_irradiated_clouds_CMSY025_spectrum_3.16solarZEOS.csv')
            elif not cloud and irradiation:
                raise Exception('No spectra tables for irradiated without cloud yet.')
            elif not irradiation and cloud:
                raise Exception('No spectra tables for isolated with cloud yet.')
            elif not (cloud or irradiation):
                # Only have a single table for 0× solar
                self.spectra_df = pd.read_csv('atm_spectra/jupiter_isolated_noclouds_CMSY025_spectrum.csv')
                self.spectra_df_z1 = None
                self.spectra_df_z3 = None
        elif planet == 'Saturn':
            # Just a placeholder example
            if cloud and irradiation:
                self.spectra_df = pd.read_csv('atm_spectra/saturn_irradiated_clouds_CMSY015_spectrum.csv')
                self.spectra_df_z1 = None
                self.spectra_df_z3 = None
                print('Warning: Saturn table only valid for T_int > 200 K.')
            else:
                raise Exception('No other Saturn spectra tables available yet.')
        else:
            raise Exception("Planet must be either 'Jupiter' or 'Saturn'.")

        # -------------------------------------------------
        # 2) Function to reshape a DataFrame into 3D arrays
        #    (T_int × logg × wavelength).
        # -------------------------------------------------
        def reshape_df(df):
            # Prepare dict of 3D arrays from DataFrame columns
            if df is None:
                return None
            # Determine shape from unique T_int, logg
            shape = (df['T_int'].nunique(), df['logg'].nunique(), -1)
            data_3d = {}
            for col in df.columns:
                data_3d[col] = np.reshape(df[col].values, shape)
            return data_3d

        self.spectra_grid_0  = reshape_df(self.spectra_df)
        self.spectra_grid_z1 = reshape_df(self.spectra_df_z1)
        self.spectra_grid_z3 = reshape_df(self.spectra_df_z3)

        # -------------------------------------------------
        # 3) Gather all lambda ranges across metallicities
        #    to build ONE universal lambda grid.
        # -------------------------------------------------
        all_lams = []
        for grid in [self.spectra_grid_0, self.spectra_grid_z1, self.spectra_grid_z3]:
            if grid is not None:
                all_lams.append(grid['lam'].min())
                all_lams.append(grid['lam'].max())

        lam_min = np.min(all_lams)
        lam_max = np.max(all_lams)
        self.lambda_grid = np.logspace(np.log10(lam_min), 
                                       np.log10(lam_max),
                                       10000)

        # -------------------------------------------------
        # 4) Build a 4D array: (nT, nG, nLam, nZ)
        #    for flux_nu, where nZ is how many metallicities
        #    we actually have (1, 2, or 3).
        # -------------------------------------------------
        # Identify T_int, logg from the "0× solar" grid (assuming they match for all).
        # If you truly have mismatched T/logg sets across metallicities,
        # you'd need more advanced logic to unify them.
        self.T_vals = self.spectra_grid_0['T_int'][:, 0, 0]
        self.logg_vals = self.spectra_grid_0['logg'][0, :, 0]

        # List metallicities that exist
        # (If only self.spectra_df is loaded, we might have just Z=[0].)
        Z_values = []
        grid_list = []
        if self.spectra_grid_0 is not None:
            Z_values.append(0.0)
            grid_list.append(self.spectra_grid_0)
        if self.spectra_grid_z1 is not None:
            Z_values.append(1.0)
            grid_list.append(self.spectra_grid_z1)
        if self.spectra_grid_z3 is not None:
            Z_values.append(3.16)
            grid_list.append(self.spectra_grid_z3)

        self.Z_vals = np.array(Z_values, dtype=float)  # e.g., [0, 1, 3.16]

        nT = len(self.T_vals)
        nG = len(self.logg_vals)
        nLam = len(self.lambda_grid)
        nZ = len(self.Z_vals)

        # Initialize a 4D array for flux:
        # shape = (nT, nG, nLam, nZ).
        log_flux_4d = np.full((nT, nG, nLam, nZ), -1.0, dtype=float)

        # -------------------------------------------------
        # 5) For each metallicity, loop over (T, G),
        #    interpolate onto self.lambda_grid, store in flux_4d[:,:,:,zIndex].
        # -------------------------------------------------
        for z_index, grid in enumerate(grid_list):
            # For each T, G in the 3D arrays:
            for i in range(nT):
                for j in range(nG):
                    # Original arrays for this T, G
                    lam_original = grid['lam'][i, j, :]
                    flux_original = grid['f_nu'][i, j, :]

                    # Make 1D interpolator
                    f_interp = interp1d(
                        np.log10(lam_original),
                        np.log10(flux_original),
                        kind='linear',
                        bounds_error=False,
                        fill_value=-100  # We'll overwrite out-of-bounds with -1
                    )

                    log_flux_interp = f_interp(np.log10(self.lambda_grid))

                    # Store in flux_4d
                    log_flux_4d[i, j, :, z_index] = log_flux_interp

        # Save these results
        self.log_flux_4d = log_flux_4d

        # -------------------------------------------------
        # 6) Finally, build a 4D RegularGridInterpolator
        #    so you can do f_nu(T, g, lam, Z).
        # -------------------------------------------------
        self.log_flux_nu_rgi_4d = RGI(
            (self.T_vals, self.logg_vals, self.lambda_grid, self.Z_vals),
            self.log_flux_4d,
            method='linear',
            bounds_error=False,
            fill_value=None
        )

    def get_log_flux_nu(self, tint, logg, lam, Z):
        """
        Evaluate flux at the given (tint, logg, lam, Z).
        If scalars are passed, returns a scalar.
        If arrays are passed, returns an array.
        """
        # shape-check
        tint, logg, lam, Z = np.broadcast_arrays(tint, logg, lam, Z)
        points = np.column_stack([tint.ravel(), logg.ravel(), lam.ravel(), Z.ravel()])
        flux_out = self.log_flux_nu_rgi_4d(points)
        return flux_out.reshape(tint.shape)