### ======================================================================================
#
#
#   Module: A set of choices for initial profiles of S, Y, and Z
#
#   Rob Tejada Arevalo (07/18/2024)
#
#
### ======================================================================================

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter1d

class profile_builder:
    def __init__(self, radius, m, Mcore, kcore):
        self.radius= radius
        self.rad_norm = self.radius/self.radius[0]
        self.m = m
        self.m_norm = self.m/self.m[0]

        self.Mcore = Mcore
        self.kcore = kcore
        self._coreless = (kcore >= len(m))

    def _n_point_profile(self, coord_norm,
                            s_coords,   y_coords,   z_coords,   f_rock_coords,
                            s_deltas,   y_deltas,   z_deltas,   f_rock_deltas,
                            s_arr,      y_arr,      z_arr,      f_rock_arr,
                            s_values=None, y_values=None, z_values=None, f_rock_values=None,
                            interp_method='linear',
                            smooth_method=None,
                            smooth_sigma=1.0,
                            smooth_local_fraction=0.02,
                            ):
        """
        Generate profiles for S, Y, Z, and f_rock using a defined set of
        breakpoints in the given normalized coordinate array (mass or radius
        fraction, see the n_point_profile_mass/_radius wrappers), with an
        option to smooth out sharp edges.

        Breakpoint values can be specified in two ways:
        - **Deltas mode**: provide deltas (changes between adjacent breakpoints,
          accumulated from the surface value). len(deltas) == len(coords) - 1.
        - **Values mode**: provide absolute values at each breakpoint.
          len(values) == len(coords). Deltas should be empty ([]) in this case.

        Parameters
        ----------
        s_coords, y_coords, z_coords, f_rock_coords : lists
            Lists of coordinate fractions (0 to 1) for the S, Y, Z, f_rock
            profiles. Each coordinate is a "knot" or breakpoint in the profile.

        s_deltas, y_deltas, z_deltas, f_rock_deltas : lists
            Deltas for S, Y, Z, f_rock. Each list length is exactly one less than
            the corresponding coordinate list. Each delta indicates how much
            the quantity changes between adjacent breakpoints (outer to inner).

        s_values, y_values, z_values, f_rock_values : lists or None
            Absolute values at each breakpoint coordinate. Each list length must
            equal the corresponding coordinate list. If provided (non-empty),
            these are used instead of deltas for that property.

        s_arr, y_arr, z_arr, f_rock_arr : np.ndarray
            Initial arrays for S, Y, Z, f_rock (1D). Must have the same length
            and ordering as coord_norm.

        interp_method : str
            Interpolation method between breakpoints.
            'linear' (default): piecewise-linear interpolation.
            'pchip': monotone PCHIP (Piecewise Cubic Hermite Interpolating
            Polynomial). Produces smooth C1 profiles that preserve monotonicity
            between adjacent control points, preventing unphysical overshoots.

        smooth_method : str or None
            If None, no smoothing is applied.
            If 'local_gaussian', applies Gaussian smoothing only around the breakpoints.
            If 'global_gaussian', applies a Gaussian filter to the entire profile.

        smooth_sigma : float
            Standard deviation for the Gaussian kernel (when a Gaussian smoothing
            method is selected).

        smooth_local_fraction : float
            Fraction of the domain used to define the half-window around each breakpoint
            for local smoothing. For example, 0.02 uses 2% of the profile length on
            each side of the breakpoint.

        Returns
        -------
        S, Y, Z, f_rock : np.ndarray
            Updated profiles for entropy (S), helium fraction (Y), metallicity (Z),
            and rock fraction (f_rock).
        """

        # Group the deltas, coords, values, and arrays for generic processing
        deltas_list = [s_deltas, y_deltas, z_deltas, f_rock_deltas]
        coords_list = [s_coords, y_coords, z_coords, f_rock_coords]
        values_list = [s_values or [], y_values or [], z_values or [], f_rock_values or []]

        # Make local copies so we don't overwrite the input arrays
        S       = s_arr.copy()
        Y       = y_arr.copy()
        Z       = z_arr.copy()
        f_rock  = f_rock_arr.copy()

        # We'll loop over each property in a generic way
        Q_arrays = [S, Y, Z, f_rock]
        names    = ['S', 'Y', 'Z', 'f_rock']

        for deltas_i, coords_i, values_i, Q, name in zip(deltas_list, coords_list, values_list, Q_arrays, names):
            # Skip if no coordinates, or if neither deltas nor values are provided
            if len(coords_i) == 0:
                continue
            if len(deltas_i) == 0 and len(values_i) == 0:
                continue

            use_values = len(values_i) > 0
            n_coords = len(coords_i)

            if use_values:
                if len(values_i) != n_coords:
                    raise ValueError(
                        f"Number of values must equal number of coordinates "
                        f"for {name}. Got {n_coords} coords, {len(values_i)} values."
                    )
            else:
                n_deltas = len(deltas_i)
                # We require #coords == (#deltas + 1)
                if n_coords != n_deltas + 1:
                    raise ValueError(
                        f"Number of coordinates must be one more than number of "
                        f"deltas for {name}. Got {n_coords} coords, {n_deltas} deltas."
                    )

            # Convert coords_i to mass coordinates
            # Assuming coords_i is fraction of coord_norm.max()
            r_coords = np.array(coords_i) * coord_norm.max()

            # Start from the outermost value in Q
            # (assuming coord_norm is sorted from outside -> inside)
            Q_out = Q[0]

            # Build up the Q_values at each coordinate
            if use_values:
                # Values mode: use absolute values directly at each breakpoint
                Q_values = list(values_i)
            else:
                # Delta mode: accumulate from surface value
                Q_values = [Q_out]
                for delta in reversed(deltas_i):
                    Q_values.append(Q_values[-1] - delta)
                Q_values = Q_values[::-1]

            # Interpolate between breakpoints
            r_arr = np.array(r_coords)
            q_arr = np.array(Q_values)

            # Determine which grid points fall within the breakpoint range
            r_min, r_max = r_arr.min(), r_arr.max()
            in_range = (coord_norm >= r_min) & (coord_norm <= r_max)

            if interp_method == 'pchip':
                # Sort breakpoints in ascending order for PCHIP
                sort_idx = np.argsort(r_arr)
                pchip = PchipInterpolator(r_arr[sort_idx], q_arr[sort_idx])
                Q[in_range] = pchip(coord_norm[in_range])
            else:
                # Linear: assign at breakpoints, then piecewise-linear between them
                for coord, Q_val in zip(r_coords, Q_values):
                    idx = np.argmin(np.abs(coord_norm - coord))
                    Q[idx] = Q_val

                for j in range(len(r_coords) - 1):
                    r1, r2 = r_coords[j], r_coords[j+1]
                    Q1, Q2 = Q_values[j], Q_values[j+1]

                    if r1 <= r2:
                        idx_range = np.where((coord_norm >= r1) & (coord_norm <= r2))[0]
                    else:
                        idx_range = np.where((coord_norm >= r2) & (coord_norm <= r1))[0]
                        r1, r2 = r2, r1
                        Q1, Q2 = Q2, Q1

                    if len(idx_range) == 0 or r1 == r2:
                        continue

                    Q[idx_range] = (
                        (coord_norm[idx_range] - r1) / (r2 - r1)
                    ) * (Q2 - Q1) + Q1

            # Optional smoothing step for each profile
            if smooth_method == 'local_gaussian' and interp_method != 'pchip':
                # Local smoothing only around each breakpoint (linear interp only;
                # PCHIP already eliminates kinks, and local smoothing introduces
                # discontinuities at window boundaries).
                edge_indices = [np.argmin(np.abs(coord_norm - rc)) for rc in r_coords]
                n_points = len(coord_norm)
                half_window = int(smooth_local_fraction * n_points)
                Q_smooth = Q.copy()

                for edge_idx in edge_indices:
                    i1 = max(0, edge_idx - half_window)
                    i2 = min(n_points, edge_idx + half_window + 1)
                    window_data = Q_smooth[i1:i2]
                    smoothed_window = gaussian_filter1d(window_data, sigma=smooth_sigma, mode='reflect')
                    Q_smooth[i1:i2] = smoothed_window

                Q[:] = Q_smooth

            elif smooth_method in ('global_gaussian', 'local_gaussian'):
                # Apply one Gaussian filter to the entire profile
                # This can be gentler if sigma is small
                Q_smooth = gaussian_filter1d(Q, sigma=smooth_sigma, mode='reflect')
                Q[:] = Q_smooth

        return S, Y, Z, f_rock

    def n_point_profile_mass(self, *args, **kwargs):
        """Piecewise profiles with breakpoints in normalized MASS coordinate.

        Thin wrapper around _n_point_profile (which holds the full parameter
        documentation), binding coord_norm = self.m_norm.
        """
        return self._n_point_profile(self.m_norm, *args, **kwargs)

    def n_point_profile_radius(self, *args, **kwargs):
        """Piecewise profiles with breakpoints in normalized RADIUS coordinate.

        Thin wrapper around _n_point_profile (which holds the full parameter
        documentation), binding coord_norm = self.rad_norm.
        """
        return self._n_point_profile(self.rad_norm, *args, **kwargs)

    def apply_outer_linear_ramp(
        self,
        q_arr,
        inner_coord,
        outer_coord,
        delta_outer,
        use_mass_coords=True,
        q_min=None,
        q_max=None,
        smooth_sigma=0.0,
    ):
        """
        Apply a linear ramp to the outer region of a profile.

        The ramp is defined in normalized mass/radius coordinates. A value of
        ``delta_outer`` is fully applied at ``outer_coord`` (and farther out),
        tapering linearly to zero at ``inner_coord``.

        Parameters
        ----------
        q_arr : np.ndarray
            Input profile (e.g., entropy).
        inner_coord, outer_coord : float
            Inner/outer coordinates in [0, 1], with outer_coord > inner_coord.
        delta_outer : float
            Additive change at the outer boundary. Negative values lower the
            outer profile.
        use_mass_coords : bool
            If True, use normalized mass coordinates; else normalized radius.
        q_min, q_max : float or None
            Optional hard bounds after applying the ramp.
        smooth_sigma : float
            Optional Gaussian smoothing sigma for the resulting profile.
        """
        if delta_outer == 0.0:
            return q_arr.copy()

        q = q_arr.copy()
        coord = self.m_norm if use_mass_coords else self.rad_norm

        inner = float(np.clip(inner_coord, 0.0, 1.0))
        outer = float(np.clip(outer_coord, 0.0, 1.0))
        if outer <= inner:
            raise ValueError(
                "outer_coord must be greater than inner_coord for an outer ramp "
                f"(got inner={inner_coord}, outer={outer_coord})."
            )

        # Outer-most cells have coord closest to 1.0 in this codebase.
        weight = np.clip((coord - inner) / (outer - inner), 0.0, 1.0)
        q += float(delta_outer) * weight

        if q_min is not None:
            q = np.maximum(q, float(q_min))
        if q_max is not None:
            q = np.minimum(q, float(q_max))

        if smooth_sigma is not None and smooth_sigma > 0.0:
            q = gaussian_filter1d(q, sigma=float(smooth_sigma), mode='reflect')

        return q

    def gaussian_profile(self, mean, std_dev, offset=0.0, amplitude=1.0):
        """
        Generates a Gaussian metallicity profile.

        Parameters:
        mean (float): The mean (center) of the Gaussian distribution.
        std_dev (float): The standard deviation (width) of the Gaussian distribution.
        offset (float): Baseline value in the outer region (wings of the Gaussian).
        amplitude (float): Height of the Gaussian above the offset.

        Returns:
        numpy array: The Gaussian metallicity profile.
        """
        profile = np.zeros_like(self.m)
        if self._coreless:
            env = np.s_[:]
        else:
            if mean < self.kcore:
                mean = self.m[self.kcore]/self.m[0]
            env = np.s_[:self.kcore]
        profile[env] = offset + amplitude * np.exp(-0.5 * ((self.m[env]/self.m[0] - mean) / std_dev) ** 2)
        if not self._coreless:
            profile[self.kcore:] = 0
        return profile

    def inverted_gaussian_profile(self, mean, std_dev, offset, amplitude=1.0):
        """
        Generates an inverted Gaussian entropy profile.

        Parameters:
        mass-normalized entropy profile (numpy array): Array of radial positions (normalized from 0 to 1, for example).
        mean (float): The mean (center) of the Gaussian distribution.
        std_dev (float): The standard deviation (width) of the Gaussian distribution.
        amplitude (float): The amplitude (height) of the Gaussian distribution.

        Returns:
        numpy array: The inverted Gaussian entropy profile.
        """
        profile = np.zeros_like(self.m)
        if self._coreless:
            env = np.s_[:]
        else:
            if mean < self.kcore:
                mean = self.m[self.kcore]/self.m[0]
            env = np.s_[:self.kcore]
        profile[env] = offset + amplitude * (1 - np.exp(-0.5 * ((self.m[env]/self.m[0] - mean) / std_dev) ** 2))
        if not self._coreless:
            profile[self.kcore:] = 1.0
        return profile

    def sigmoid_profile(self, midpoint, alpha, Z_ini, mass_profile=True):
        """
        Generates a sigmoid metallicity profile that is:
        - Flat at 1.0 in the core region up to kcore,
        - Transitions from ~1 to Z_ini beyond kcore via a logistic function.

        Parameters
        ----------
        midpoint : float
            The normalized coordinate at which the logistic function is centered
            (i.e., the point of 50% transition), 0 < midpoint < 1.
            Interpreted as mass fraction when mass_profile=True, or radius
            fraction when mass_profile=False.
        alpha : float
            Controls how sharp the transition is. Larger alpha -> sharper step.
        Z_ini : float
            The asymptotic value at the outer boundary (surface).
        mass_profile : bool
            If True (default), use normalized mass coordinates. The inner
            asymptote is 1.0.
            If False, use normalized radius coordinates. The inner asymptote
            is 0.985.

        Returns
        -------
        numpy.ndarray
            1D array of heavy-element abundances matching the logistic shape from the
            core boundary (kcore) outward, with constant value inside kcore (0.0).
        """



        # If desired, ensure the midpoint is not smaller than kcore in mass space
        # (similar logic to your 'mean < self.kcore' check).

        if mass_profile:
            profile = np.zeros_like(self.m)
            if self._coreless:
                env = np.s_[:]
            else:
                if midpoint < (self.m[self.kcore]/self.m[0]):
                    midpoint = self.m[self.kcore]/self.m[0]
                env = np.s_[:self.kcore]

            m_core_norm = self.m_norm[env]
            logistic_val = Z_ini + (0.985 - Z_ini ) / (1 + np.exp(alpha * (m_core_norm - midpoint)))

            profile[env] = logistic_val

        else:
            profile = np.zeros_like(self.radius)
            if self._coreless:
                env = np.s_[:]
            else:
                if midpoint < (self.radius[self.kcore]/self.radius[0]):
                    midpoint = self.radius[self.kcore]/self.radius[0]
                env = np.s_[:self.kcore]

            m_core_norm = self.rad_norm[env]
            logistic_val = Z_ini + (0.985 - Z_ini ) / (1 + np.exp(alpha * (m_core_norm - midpoint)))

            profile[env] = logistic_val

        # Outside kcore, set the profile to 0.0
        if not self._coreless:
            profile[self.kcore:] = 0.0

        return profile

    def exponential_profile(self, mean, rate, amplitude):
        """
        Generates an exponential metallicity profile.

        Parameters:
        rate (float): The rate parameter (lambda) of the exponential distribution.
        amplitude (float): The amplitude (height) of the distribution.

        Returns:
        numpy array: The exponential metallicity profile.
        """
        profile = np.zeros_like(self.m)
        if self._coreless:
            env = np.s_[:]
        else:
            if mean < self.kcore:
                mean = self.m[self.kcore]/self.m[0]
            env = np.s_[:self.kcore]
        profile[env] = amplitude * np.exp(-rate * (self.m[env]/self.m[0] - mean))
        if not self._coreless:
            profile[self.kcore:] = 0
        return profile

    def inverted_exponential_profile(self, mean, rate, offset, amplitude):
        """
        Generates an inverted exponential entropy profile.

        Parameters:
        rate (float): The rate parameter (lambda) of the exponential distribution.
        offset (float): The base value that the profile starts from.
        amplitude (float): The amplitude (height) of the distribution.

        Returns:
        numpy array: The inverted exponential entropy profile.
        """
        profile = np.zeros_like(self.m)
        if self._coreless:
            env = np.s_[:]
        else:
            if mean < self.kcore:
                mean = self.m[self.kcore]/self.m[0]
            env = np.s_[:self.kcore]
        profile[env] = offset + amplitude * (1 - np.exp(-rate * (self.m[env]/self.m[0] - mean)))
        if not self._coreless:
            profile[self.kcore:] = 1.0
        return profile


