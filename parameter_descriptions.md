#### Specific Parameter Reference (`parameters_default.ini`)

The list below provides brief descriptions of the model parameters in `parameters_default.ini`. Many parameters are advanced or experiment-specific, but they are included here so you can see what each one controls before changing it.

#### `[general]`

- `N` `[int]`: Sets the total number of mass zones used by the model, so increasing it usually improves spatial resolution at the cost of runtime and memory.
- `surf_width` `[float]`: Controls how strongly the mesh is concentrated toward the outer layers, so smaller values generally put more zones near the surface.
- `surf_width2` `[float]`: Controls how strongly the mesh is concentrated toward the deep interior/core region, which is useful when you need extra core or mantle resolution.
- `final_age` `[float]`: Stops the evolution run at this age in Gyr, so this is the main endpoint for the simulation timeline.
- `dt_` `[float]`: Sets the initial timestep in seconds, which only affects the starting step size before adaptive control takes over.
- `dt_abort_myr` `[float]`: Stall guard. If >0 and the adaptive timestep falls below this many Myr after t > 0.05 Gyr, the run ends cleanly with state saved through the last accepted step. Guards against dt-collapse stalls where a dt-independent per-step error (e.g. rain-zone activation flicker at sharp Z fronts) rejects every halving; such runs never recover and would otherwise burn cluster walltime at the dt floor. Default `0.0` (disabled).
- `z_atm_abort_solar` `[float]`: Rain-drain kill switch. If >0 and the committed outer-cell metal mass fraction (the quantity saved as `data_Z_atm`) falls below this many x solar (linear convention `Z / 0.017` with Chen+2023 Z_sun; NOTE the run summary's "Z atm ... x solar" line uses the Z/X-ratio convention instead, ~2% higher at few-times-solar) after t > 0.05 Gyr, the run ends cleanly with state saved through the last accepted step. Inactive on bare-rock (envelope-free) models. Guards against Z-miscibility rain corners (high `zmisc_deltaT`, low `R_rho`) that drain the outer envelope toward Z_out ~ 0 and can never recover the atmospheric-metallicity target. Default `0.0` (disabled).
- `error` `[float]`: Baseline timestep error tolerance for the adaptive timestep logic (smaller → more conservative stepping). The actual acceptance threshold is `error_smooth / 100`, where `error_smooth` adds early-time smoothing (≈5× stricter at t=0, relaxing to your value by ~0.1 Gyr) and a relaxation factor when helium/Z rain is active. Default `1.0`.
- `save_data` `[bool]`: Enables or disables writing model outputs, which is useful for quick performance/debug runs where you only care about logs.
- `early_save_interval_myr` `[float]`: Optional finer save cadence (Myr) applied at early times; set `0` to disable. Used with `early_save_until_gyr`.
- `early_save_until_gyr` `[float]`: Age cutoff (Gyr) for the early-time save interval; only active when `early_save_interval_myr > 0`.
- `save_interval` `[float]`: Sets how often results are saved in Myr; use `0` to save every accepted timestep instead of thinning the outputs.
- `verbose` `[bool]`: Turns on extra progress and convergence diagnostics in the terminal/logs, which helps with troubleshooting failed or slow runs.
- `energy_conservation_debug` `[bool]`: Enables detailed energy-conservation diagnostics (midpoint/HSE/transport residual comparisons and luminosity mismatch summaries); leave `False` for normal runs. The official `dE_G` ledger always integrates the transport solver's surface luminosity as a backward-Euler rectangle (`loss_lum * dt`), consistent with the heat the implicit step actually removes.
- `save_verbose` `[bool]`: Saves extra transport diagnostics (for example conductive/radiative coefficients and entropy-gradient brackets), which can be useful for analysis but increases output size.
- `early_max_step_myr` `[float]`: Optional timestep cap (in Myr) applied only at early times; leave `0` to disable.
- `early_max_step_until_gyr` `[float]`: Age cutoff in Gyr for `early_max_step_myr`; both values must be >0 for the early-time cap to be active.
- `late_max_step_myr` `[float]`: Optional late-time cap on the adaptive timestep (Myr). When set, dt is capped at this value once the model age reaches `late_max_step_from_gyr`. Useful for tightening dt during long-duration physics (e.g. He rain) without constraining early-time evolution. Set `0` to disable.
- `late_max_step_from_gyr` `[float]`: Age threshold (Gyr) at which the late-time timestep cap activates. Used only if `late_max_step_myr > 0`.
- `max_step` `[float]`: Caps the maximum allowed timestep in Myr.
- `restart_flag` `[bool]`: Must stay `False`. Restarting from saved output is not supported — the energy-conservation ledger cannot be reconstructed mid-run, so `True` raises an error at startup. Always run models fresh.
- `high_precision` `[bool]`: Switches timestep/convergence control into a stricter flux-based mode, and it uses `tol_flux` in `[transport]` to decide acceptable flux mismatch.
- `save_hdf5` `[bool]`: Chooses HDF5 output (`True`) versus text output (`False`), trading human readability for smaller/faster files and easier post-processing.
- `hdf5_filename` `[str]`: Sets the HDF5 filename used when `save_hdf5 = True`, so you can give different runs distinct output files.
- `mesh_mode` `[str]`: Chooses how the mass mesh is built (`legacy` for a single mesh or `partitioned` for separate envelope and mantle/core meshes), which changes how resolution is distributed through the planet. The default setting is `legacy`, while `partitioned` is experimental.
- `N_env` `[int | None]`: Sets the number of envelope zones when using `mesh_mode = partitioned`; leave it as `None` if you want ORCHARD to infer it from `N_core`.
- `N_core` `[int | None]`: Sets the number of mantle+core zones when using `mesh_mode = partitioned`; leave it as `None` if you want ORCHARD to infer it from `N_env`.

#### `[boundary_condition]`

- `bc_atm` `[str]`: Selects the atmosphere boundary-condition model/table group (for example `c23`, `c26`, `f11`, or `f07`), which is one of the most important choices for cooling/evolution behavior. For Jupiter and Saturn, either `c23` (Chen et al. 2023) or `f11` (Fortney et al. 2011) should be used. For Uranus and Neptune, either `g75` (Graboske 1975), `f11`, or `f07` (Fortney et al. 2005, 2007, 2020; Ohno \& Fortney 2023) should be used. For gas giant exoplanets, `c26` (Chen et al. 2026, **in prep**) should be used. For sub-Neptune exoplanets, `f07` should be used. For super-Earth exoplanets, `bare` ($\sigma T^4$) or `rock_gray` (semi-gray atmosphere) should be used.
- `planet` `[str]`: Selects the planet label used to choose the appropriate atmosphere table within the chosen BC family (for example Jupiter, Saturn, or a sub-Neptune/super-Jupiter case). These are `Jupiter`, `Saturn`, `Uranus`, `Neptune`, `Sub_Neptune`, `Super_Earth`, and `Super_Jupiter`.
- `irradiation` `[bool]`: Enables atmospheric irradiation when supported by the selected BC. This feature only applies to `c23`.
- `cloud` `[bool]`: Enables ammonia cloud effects in the `c23` atmospheric boundary condition tables.
- `y_atm_fixed` `[bool]`: If `True`, uses a fixed atmospheric helium value (`Y_atm`) instead of interpolating across the helium dimension of the BC tables. Applies to `c23` and `c26`.
- `z_atm_fixed` `[bool]`: If `True`, uses a fixed atmospheric metallicity value (`Z_atm`) rather than metallicity interpolation in BCs that support it. Applies to `c26`.
- `Y_atm` `[float]`: Sets the atmosphere helium value used when `y_atm_fixed = True`, so it controls which helium slice of the BC tables is applied. Applies to `c23` and `c26`.
- `Z_atm` `[float]`: Atmosphere metallicity used when `z_atm_fixed = True`, given as an enhancement factor relative to solar (default `3.16` = 3.16× solar). This is used to select individual tables, not set the metallicity of the interior. Internally multiplied by the solar reference `Z_SOLAR_CHEN23` to obtain a mass fraction. Selects which metallicity slice of the `c26` BC tables is applied.
- `T_eq` `[float]`: Sets the equilibrium temperature in K for supported sub-Neptune / gray-atmosphere boundary conditions, and it directly affects the outer thermal boundary. Active only when `bc_atm = f07`, `gray`, or `ideal_gray`.
- `bond_albedo` `[float]`: Bond albedo for BC options that use irradiation/insolation; higher values lower the absorbed stellar power. Active only when `bc_atm = f07`, `g75`, `gray`, or `ideal_gray`. For `g75` (Uranus/Neptune) it enters the equilibrium-temperature calculation; for the others it scales the incident stellar flux.
- `rock_gray_kappa_ir` `[float]`: Sets the IR opacity (cm^2/g) for the semi-gray rock/ideal-gray atmosphere options, which changes the atmospheric insulation strength. Used when `bc_atm = gray` or `ideal_gray`.
- `rock_gray_p_match_bar` `[float]`: Pressure (bar) at which the semi-gray rock / ideal-gray atmosphere is matched to the interior model. Active only when `bc_atm = gray` or `ideal_gray`.
- `extrap_endpoint` `[bool]`: Enables the extrapolated endpoint method, which adds a per-cell stellar irradiation source $Q_{\rm irr}(k) = (F_{\rm inc}/\mu)\,\kappa_v\,e^{-\tau_v(k)/\mu}\,dm(k)$ to the entropy equation. The optical depth $\tau_v \approx \kappa_v\,P/g$ comes from a hydrostatic estimate, so the heating decays exponentially inward from the surface. Mirrors how radiogenic and latent heat enter the residual. Off by default; intended for close-in / strongly irradiated planets where the stellar light penetrates below the atmosphere boundary. Independent of `bc_atm` choice — works alongside any BC.
- `ee_kappa_v` `[float]`: Visible-band gray opacity in cm^2/g used to compute $\tau_v$ and the heating prefactor. Sets the photosphere depth at $\tau_v = 2/3 \Rightarrow P \approx (2/3)\,g/\kappa_v$.
- `ee_mu` `[float]`: Cosine of the incidence angle. Default `0.5` (Hansen 2008 hot-Jupiter convention). Diffuse-limit value is $1/\sqrt{3} \approx 0.577$.
- `ee_F_inc_override` `[float]`: Override for the incident flux $F_{\rm inc}$ in erg/s/cm^2. Negative (default `-1.0`) means use $\sigma T_{\rm eq}^4$ from the existing `T_eq` parameter.
- `ee_apply_theta` `[bool]`: If `True` (default), multiplies $Q_{\rm irr}$ by the same early-time ramp $\Theta(t) = (2/\pi)\arctan(t/0.5\,{\rm Gyr})$ used by the radiative diffusion conductivity, to avoid a $t=0$ surface heating spike during the first few Newton timesteps.

#### `[equation_of_state]`

- `eos_version` `[float]`: Selects the EOS interface version. `1.0` is the original mixtures class (rectangular S-P tables, old derivatives). `2.0` is the `hhe_z_mixtures` class (smoothed tables, rhomboid xi-mapping, P-T dependent mu_H).
- `hhe_eos` `[str]`: Selects the H-He equation of state (`cd` for Chabrier & Debras 2021, `cms` for Chabrier et al. 2019, or `scvh` for Saumon et al. 1995).
- `hg` `[bool]`: Enables Howard & Guillot (2023) non-ideal H-He mixing corrections, but only for `hhe_eos = cms`.
- `z_eos` `[str]`: Selects the heavy-element mixture EOS used for the H-He-metal mixture (for example `aqua` or `ice_mixture`).
- `water_eos` `[str]`: Chooses the water EOS variant (e.g. `aqua`, or the corrected `aqua_mlcp`) used when `rock_mixtures = True`. Affects the composition mixture regardless of the `z_eos` value.
- `eos_tab` `[bool]`: Uses tabulated EOS lookups (`True`) instead of direct EOS inversions (`False`), and the tabulated mode is the recommended fast/default workflow.
- `rock_mixtures` `[bool]`: Enables loading H-He-water-rock mixture tables, which is required if you want nonzero envelope rock fraction (`f_rock_ini`) to affect the model.
- `z_eos2` `[str]`: Placeholder for a second heavy-element EOS component (e.g. post-perovskite `ppv2`) intended for flexible multi-component Z mixtures. Currently read but not used by the code.
- `z_eos3` `[str]`: Placeholder for a third heavy-element EOS component (H-He-Z_water-Z_rock-Z_fe mixture); currently unavailable.

#### `[initial]`

- `M_planet_unit_cgs` `[float]`: Sets the reference planet mass unit to 1 Earth mass in grams. This is the internal CGS conversion constant and should not be changed.
- `M_Mearth` `[float]`: Sets the planet mass in Earth masses (e.g., `317.907` for Jupiter, `95.159` for Saturn, `14.536` for Uranus, `17.147` for Neptune). This is the primary mass parameter to set for your model.
- `M_MJup` `[float]`: Multiplier applied to the reference structure mass when set to a value other than `1.0` (it scales `M_Mearth` at the point of isentrope/profile interpolation). Default `1.0` uses `M_Mearth` directly; e.g. `2.0` or `0.5` scale the reference profile up or down.
- `rk4_initial` `[bool]`: If `True`, uses the RK4 shooting method to build the initial planetary structure from scratch, computing self-consistent (P, r, rho, T) profiles for any mass and entropy. When `False`, uses an isentrope interpolation. Default `True`.
- `S_ini` `[float]`: Sets the initial entropy ($k_B$ baryon$^{-1}$) for the envelope when no custom entropy profile is provided, so it is the main hot-start/cold-start control. Typical hot start is 10–12 $k_B$ baryon$^{-1}$; default is 10.
- `Y_ini` `[float]`: Sets the initial hydrogen-helium ratio parameter (`Y' = Y/(X+Y)`), which determines the baseline helium abundance in the envelope. The protosolar value is 0.277 (Bahcall et al. 2006).
- `Z_ini` `[float]`: Sets the initial envelope heavy-element fraction at the surface (or the homogeneous value if no Z profile is specified).
- `f_rock_ini` `[float]`: Sets the initial rock fraction within the envelope heavy-element budget `Z`, and it only affects the model when `rock_mixtures = True`. Default is `0.0` (pure water).
- `f_fe_ini` `[float]`: Reserved — intended initial iron fraction within Z in the envelope. Currently a placeholder: it is not read or used by the evolution code and has no effect.
- `mz_struct` `[bool]`: Chooses whether Z-profile deltas are interpreted relative to total metal mass (`M_z`) or relative to the surface `Z_ini`, which changes how custom Z gradients are normalized.
- `M_z` `[float]`: Sets the total heavy-element mass in Earth masses used when `mz_struct = True`, so ORCHARD can build an initial Z profile with the requested metal inventory.
- `initial_profiles_S` `[str]`: Selects the initial entropy profile construction method (for example `struct`, `gaussian`, `exponential`, or `sigmoid`).
- `initial_profiles_Y` `[str]`: Selects the initial helium-profile construction method, though in practice Y is often derived from the Z profile to keep `Y'` flat.
- `initial_profiles_Z` `[str]`: Selects the initial heavy-element profile construction method and is the main switch for custom envelope composition gradients.
- `initial_profiles_f_rock` `[str]`: Selects the initial rock-fraction profile construction method within the heavy-element component.
- `mass_struct` `[bool]`: If `True`, uses fractional mass coordinates (0 = center, 1 = surface) for `struct` profile inputs; if `False`, the same profile arrays are interpreted in fractional radius.
- `initial_profile_name` `[str | None]`: Provides an optional named profile/preset identifier for profile-loading workflows; can usually be left `None`.
- `z_init_coords` `[list[float]]`: Defines the coordinate breakpoints for the `struct` Z profile, where each entry is a fractional mass (or radius if `mass_struct = False`) vertex.
- `y_init_coords` `[list[float]]`: Defines the coordinate breakpoints for the `struct` Y-profile input (really the Y-prime profile), following the same coordinate convention as Z and S.
- `s_init_coords` `[list[float]]`: Defines the coordinate breakpoints for the `struct` entropy profile used to build piecewise S gradients.
- `f_rock_init_coords` `[list[float]]`: Defines the coordinate breakpoints for the `struct` rock-fraction profile within the envelope heavy-element component.
- `z_init_deltas` `[list[float]]`: Defines Z-profile segment deltas between consecutive `z_init_coords`; length must be exactly one less than the coordinate array length.
- `y_init_deltas` `[list[float]]`: Defines Y/Y-prime profile segment deltas between consecutive `y_init_coords`, mainly for advanced setups because Y is often inferred from Z.
- `s_init_deltas` `[list[float]]`: Defines entropy-profile segment deltas between consecutive `s_init_coords`, letting you build piecewise entropy gradients relative to the outer value.
- `f_rock_init_deltas` `[list[float]]`: Defines rock-fraction profile segment deltas between consecutive `f_rock_init_coords` for custom interior rock gradients.
- `z_init_values` `[list[float]]`: Provides explicit Z profile values at each breakpoint coordinate (values mode); length must equal `z_init_coords`.
- `y_init_values` `[list[float]]`: Provides explicit Y/Y-prime profile values at each breakpoint coordinate (values mode); length must equal `y_init_coords`.
- `s_init_values` `[list[float]]`: Provides explicit entropy profile values at each breakpoint coordinate (values mode); length must equal `s_init_coords`.
- `f_rock_init_values` `[list[float]]`: Provides explicit rock-fraction profile values at each breakpoint coordinate (values mode); length must equal `f_rock_init_coords`.
- `struct_interp` `[str]`: Interpolation method for struct profiles: `'linear'` (piecewise-linear, sharp kinks at breakpoints) or `'pchip'` (monotone cubic, smooth C1 profiles that pass through control points without unphysical overshoots).
- `struct_smooth` `[bool]`: Enables Gaussian smoothing of the constructed `struct` profile after the piecewise profile is built, which can reduce sharp numerical features.
- `struct_sigma` `[float]`: Sets the Gaussian smoothing width in grid cells used when `struct_smooth = True`, with larger values producing smoother and broader transitions.
- `init_gauss_center` `[float]`: Sets the normalized-mass location of the Gaussian profile center (for Gaussian profile options), marking where the Z peak or S trough/gradient is placed.
- `init_gauss_stdev_z` `[float]`: Sets the standard deviation of the Gaussian used for the initial Z profile when a Gaussian Z profile is selected.
- `init_gauss_stdev_s` `[float]`: Sets the standard deviation of the Gaussian used for the initial entropy profile when a Gaussian S profile is selected.
- `minimum_init_s` `[float]`: Sets the minimum entropy reached at the bottom of the inverted Gaussian entropy profile, while `S_ini` still controls the outer/maximum level.
- `maximum_init_z` `[float]`: Sets the maximum deep-interior Z value used as the amplitude for the Gaussian heavy-element profile.
- `init_exp_rate` `[float]`: Sets the decay rate for the exponential initial profile option; larger values produce steeper radial/mass gradients.
- `alpha` `[float]`: Sets the sigmoid steepness parameter for sigmoid initial-profile options; larger values make the transition sharper.
- `midpoint` `[float]`: Sets the sigmoid transition midpoint (in normalized coordinate space) for sigmoid initial-profile options.
- `outer_entropy_gradient` `[bool]`: Enables an additional outer-envelope entropy ramp after the base entropy profile is built, mainly useful for bare super-Earth initializations where the outer regions may be too hot at the start.
- `outer_entropy_use_mass` `[bool]`: Chooses whether the outer-entropy ramp coordinates are interpreted in mass (`True`) or radius (`False`) space.
- `outer_entropy_inner_coord` `[float]`: Sets the inner coordinate where the optional outer-envelope entropy ramp begins.
- `outer_entropy_outer_coord` `[float]`: Sets the outer coordinate where the optional outer-envelope entropy ramp ends (usually near the surface).
- `outer_entropy_delta` `[float]`: Sets the entropy change added across the optional outer-envelope ramp; negative values make the outer envelope cooler/lower entropy.
- `outer_entropy_smooth_sigma` `[float]`: Sets the smoothing width applied to the optional outer-envelope entropy ramp to avoid sharp transitions.
- `outer_entropy_min` `[float]`: Sets a floor on the entropy value after applying the outer-entropy ramp; usually safe to leave at default.
- `enforce_no_temp_inversion` `[bool]`: Reserved for future use. Intended to adjust the initial entropy profile to avoid temperature inversions, but it is currently read and **not applied** by the code (not yet implemented). Leave at the default `False`.

#### `[hydrostatic_equilibrium]`

- `tol_hydro` `[float]`: Sets the Newton-Raphson/Henyey convergence tolerance for the hydrostatic solve, with smaller values demanding a tighter equilibrium before the step is accepted.
- `p_atm` `[float]`: Sets the outer boundary pressure in bar for the hydrostatic model, which effectively defines the pressure level of the radius/atmosphere match point.
- `hse_alpha_init` `[float]`: Initial damping factor for the HSE Newton-Raphson solver (`0 < alpha <= 1`). Lower values are more stable but converge slower. Default `0.4`.
- `hse_adaptive_alpha` `[bool]`: Enables adaptive damping, ramping alpha toward 1.0 as the error decreases. When `False`, alpha stays at `hse_alpha_init` throughout. Default `False`.
- `hse_gravity_centering` `[str]`: Centering of the pressure factor in the Henyey gravity term. `legacy` (default) is the historical one-sided form, kept bit-identical with all prior results; `midpoint` is the second-order symmetric form, which removes the O(1/N) work-vs-gravity energy-conservation defect and converges observables at much lower N, but changes results (~1% radius at N=300).
- `time_centered_heat_factor` `[bool]`: (in `[transport]`) Temperature weighting of the heat terms in the entropy equation. `False` (default) keeps the legacy old-state 1/T factor, bit-identical with prior results; `True` meters heat at the time-centered temperature `T_old + (dt/2)*(dT/dt)_prev` (lagged-rate midpoint, second-order in time), removing the eps_S conservation term (~+1.6e-3 of E_rad). Changes results slightly; opt-in until validated. Combined with `hse_gravity_centering = midpoint`, the conservation residual reduces to the pure EOS-table term.
- `isothermal_compact_core` `[bool]`: If `True`, initializes the compact core as isothermal at the core-mantle boundary temperature; if `False`, the core starts with a constant entropy and evolves thermally. `False` is recommended for gas giants and sub-Neptunes/super-Earths.
- `rotation` `[bool]`: Includes rigid-body rotation in the hydrostatic equilibrium calculation.
- `tof_calc` `[bool]`: Enables the full Theory-of-Figures (ToF4 by default) calculation in the hydrostatic solve, computing `J_2..J_8`, the figure shapes, and the structural moment of inertia; the resulting MoI ratio rescales the angular velocity entering the HSE equations. Legacy name `I_correction` is still accepted for backward compatibility.
- `C_MoI` `[float]`: Sets the dimensionless moment-of-inertia factor used by the ToF correction (`I = C_MoI * M * R^2`). Default is Jupiter's value (0.26393) from Militzer & Hubbard (2023). Saturn (0.2181) and other planets have their own values in the literature.
- `period` `[int]`: Sets the rotation period in seconds, which matters when `rotation` and/or `tof_calc` are enabled because it determines the angular velocity. Jupiter: 35730 s, Saturn: 38014 s, Uranus: 62064 s, Neptune: 57996 s.
- `env_s_start` `[bool]`: If `True`, the mantle entropy starts continuously from the envelope (recommended for gas giants); if `False`, the mantle entropy is set explicitly by `[core].mantle_entropy` (recommended for sub-Neptunes and super-Earths).

#### `[transport]`

- `convection` `[bool]`: Enables convective heat transport; turning it off is mostly useful for controlled numerical tests rather than realistic models.
- `radiation` `[bool]`: Enables radiative heat transport in the envelope using Rosseland mean opacities, which is important when radiative zones are expected.
- `opac` `[str]`: Rosseland opacity table for the radiative envelope (only used when `radiation = True`). `'default'` uses a single 3.16x solar table. `'metal_rich'` uses a 5-point grid in fzsol. `'water_rich'` uses a water-rich opacity module.
- `losses` `[bool]`: Enables luminosity loss/cooling; if set to `False`, the planet is prevented from cooling and the run becomes a debugging/sanity-check setup.
- `tol_NR` `[float]`: Sets the Newton-Raphson convergence tolerance for the transport solver, with tighter values improving solver precision but often increasing iteration cost.
- `repair_T_inversions` `[bool]`: Post-step repair of unphysical envelope temperature inversions (T decreasing with depth — Schwarzschild-unstable layers that real convection would erase, injected by under-converged transport-NR entropy updates during He/Z-rain eras and artificially preserved by composition-gradient gating). On each accepted step the envelope T profile is made monotone by mass-weighted isotonic regression (pool-adjacent-violators — the minimal monotone adjustment, equivalent to mixing each inverted run to a mass-weighted isothermal patch); repaired cells get S recomputed by inverting `get_logt_sp` at fixed (P, Y, Z, f_rock) and rho refreshed. The internal-energy residual of each repair is computed and logged (`T-INVERSION REPAIR` lines). Default `False` (exact no-op).
- `hinge_smooth` `[bool]`: Replaces the hard `max(x, 0)` in the convective-velocity expression with a differentiable hinge. With `hinge_func = 'softplus'` it uses the softplus function; with `hinge_func = 'classic'` it uses `max(x**3 / (1 + x**2), 0)`. Sharpness is set by `hinge_k`. Default `False`.
- `hinge_func` `[str]`: Selects the differentiable hinge function used when `hinge_smooth = True` (for example `softplus`), affecting how sharply convection turns on.
- `hinge_k` `[float]`: Sharpness parameter for the softplus hinge function. Lower `k` gives a smoother transition; default `1e7` is very sharp (nearly identical to hard max).
- `tol_flux` `[float]`: Sets the relative flux-convergence tolerance used by the high-precision transport mode (`[general].high_precision = True`), where smaller values demand closer flux matching.
- `debug_transport_NR` `[bool]`: Prints per-iteration NR diagnostics (step error, residual norms, worst variable) and other transport Jacobian debugging output. Default `False`.
- `lag_bc_jacobian` `[bool]`: Treats the surface boundary-condition Jacobian coupling (`dTint/dS`, `dTint/dY`, `dTint/dZ`) as lagged/Picard. The surface-loss residual uses the step-start `T_int` (explicit within each transport solve), so the consistent Jacobian contribution is zero and this never changes a converged solution — only the Newton trajectory. Set `True` when the accelerated implicit coupling destabilizes the solver, e.g. `radiation = False` ice-giant models at the surface-convection shutoff. Default `False`.
- `eos_deriv_z_cap` `[float]`: High-Z sampling cap for the EOS composition derivatives entering the Ledoux bracket and transport Jacobian (evaluated at `min(Z, cap)`). Protects against the divergent `dS/dZ` table-edge tail beyond Z ≈ 0.98. Models whose deep envelopes exceed the cap (e.g. the inhomogeneous ice-giant examples with Z ≈ 0.98) should set `1.0` (cap disabled) — frozen derivatives at the cap make convective-overturn epochs numerically intractable. Default `0.97`.
- `smooth_convection_criterion` `[bool]`: Replaces the hard "if bracket > 0 then `D_mlt` else `D_micro`" gate with a smooth sigmoid blend `D_b = σ_conv · D_mlt + D_micro`, where `σ_conv` is built from the raw (pre-softhinge) bracket. Eliminates the discrete-cell "staircase" artifact in composition gradients at convective/stable boundaries. Default `False`.
- `smooth_conv_k` `[float]`: Sharpness of the smooth-convection sigmoid. Large `k` (>>1) recovers the hard-step limit; small `k` (~1) gives a very gradual transition. Default `10` is moderately sharp.
- `smooth_conv_scale` `[float]`: Bracket scale used to non-dimensionalize the smooth-convection sigmoid argument. If ≤ 0, auto-calibrated per timestep as the 25th percentile of the positive bracket distribution.
- `smooth_conv_bracket_sigma` `[float]`: Gaussian smoothing width (in cells) applied to `bracket_raw` before computing `σ_conv`. This is the key knob that breaks the composition-gradient staircase: spatial smoothing of 1.5–3 cells makes adjacent cells get similar diffusivity values so the transition is graded. Set to `0` to disable. Default `2.0`.
- `semiconvection` `[bool]`: Enables semiconvective transport in Ledoux-stable but Schwarzschild-unstable regions. Numerically sensitive/WIP; use with caution.
- `sc_zones_min` `[int]`: Minimum number of consecutive Schwarzschild-unstable / Ledoux-stable zones required for a region to qualify as semiconvective; filters single-zone noise.
- `Nu_T` `[float]`: Thermal Nusselt number for semiconvective regions (used only when `semiconvection = True`), setting the strength of semiconvective heat transport. Default `10`.
- `Nu_X` `[float]`: Compositional Nusselt number for semiconvective regions (used only when `semiconvection = True`), setting the strength of semiconvective composition transport. Default `100`.
- `Pr` `[float]`: Prandtl number used to compute the critical density ratio `R_crit^{-1} = (Pr + 1)/(Pr + tau)` that bounds semiconvective regions.
- `tau` `[float]`: Diffusivity ratio used alongside `Pr` to set the semiconvective threshold `R_crit`. Under `semi_model = spruit` it additionally sets the layered-state ceiling `R_rho < tau^{-1/2}` and enters the effective diffusivities directly.
- `semi_model` `[str]`: Semiconvection closure choice, gated by the same master `semiconvection` flag. `nusselt` (default): legacy constant-Nusselt chi-ramp over the full instability window (exact pre-existing behavior). `spruit`: Spruit (2013) / Fuentes et al. (2026) layered double-diffusive closure — `Nu_T = 1 + tau^{-1/2}/R_rho`, `D_eff = kappa_T tau^{1/2}/R_rho` (same for He and Z; `kappa_T = Lambda/(rho c_p)` is the local microscopic conductive+radiative thermal diffusivity) — active only in the narrower window `1 < R_rho < tau^{-1/2}` (no layered state above the ceiling; microscopic transport only outside). `R_rho` is the code's `R0_inv` (= Wood 2013's `R_0^{-1}`, = Fuentes 2026's `R_rho`). Reuses the existing detection, `sc_zones_min` filter, Nusselt time-lag, and lagged-coefficient Jacobian machinery. Incompatible with `semi_shoot = True` (raises).
- `spruit_edge_frac` `[float]`: Spruit model only. Fractional width (relative to the window span `tau^{-1/2} - 1`) of a smoothstep taper bringing the enhancement continuously to zero at the `R_rho = tau^{-1/2}` ceiling, softening Spruit's discontinuous shutoff for the Newton solver. `0` recovers the strict discontinuity. Default `0.1`.
- `spruit_tau_local` `[bool]`: Spruit model only. Computes `tau = D_micro/kappa_T` per boundary instead of using the constant `tau`, making the ceiling a per-zone profile. Default `False`.
- `spruit_nu_x_max` `[float]`: Spruit model only. Safety cap on the compositional target ratio `D_eff/D_micro` fed to the `Nu_X` lag machinery; binding undershoots `D_eff` (conservative). Default `1e10`.
- `smooth_D_spatial` `[int]`: Half-width (in neighboring boundaries) of the log-space running geometric mean applied to the composition diffusion coefficients `D_b`, suppressing grid-scale oscillations at convective brackets. E.g. `3` is a 7-point window. Default `0` (off).
- `use_local_Hp` `[bool]`: Uses the local pressure scale height (`P/ρg`) for the mixing-length and rain advection scale heights instead of the constant `Hr_y`. Default `True` (set in `parameters_default.ini`); the code fallback is `False` for backward compatibility if the key is absent.
- `two_stage_NR` `[bool]`: Reserved for future testing — intended to solve entropy first, then the coupled S–Y–Z system. Currently read but **not implemented** (no effect). Default `False`.
- `flux_relax` `[bool]`: Enables convective flux bracket time-relaxation to damp oscillations at He rain onset. Default `False`.
- `flux_relax_tau` `[float]`: Relaxation timescale in units of `t_Db_relax`. Used only when `flux_relax = True`.
- `radioactive` `[bool]`: Enables radioactive heating in the mantle/core region, which can matter in sub-Neptune and rocky-interior cases.
- `latent_heat_effect` `[bool]`: Enables latent-heat terms in mantle/core evolution for sub-Neptune and super-Earth cases.
- `latent_heat_factor` `[float]`: Multiplies the latent-heat contribution for sensitivity studies; values other than `1.0` are best treated as exploratory.
- `latent_delta_t` `[float]`: Temperature width (K) of the smooth melt-fraction transition `χ(T)` used by latent heating and viscous/inviscid mantle-transport blending. Default `200.0` (per `parameters_default.ini`; the code fallback is `300.0` if the key is unset).
- `t_chi_relax` `[float]`: Sets the χ-relaxation timescale (Myr), limiting how much melt fraction (and therefore `Δχ`) can change in one timestep.
- `chi_transport_relax` `[bool]`: Applies the same χ-relaxation limiter to the mantle transport blend, reducing abrupt inviscid-to-viscous jumps in one timestep.
- `implicit_viscous_transition` `[bool]`: Rebuilds the mantle viscous/inviscid blend from trial-state `χ(T)` inside Newton iterations (semi-implicit), so the solve can anticipate phase-transition handoffs.
- `max_delta_chi` `[float]`: Maximum allowed melt-fraction change per timestep. The chi limiter halves dt when any mantle cell exceeds this threshold. Default `0.2`.
- `eddy_overshoot` `[bool]`: Enables an artificial eddy-conductivity overshoot near the envelope-mantle boundary (EMB), mainly for exploratory sub-Neptune/super-Earth cooling experiments. Should not be used for gas giants.
- `eddy_overshoot_factor` `[float]`: Overshoot diffusivity prefactor (`f_ov`) applied when `eddy_overshoot = True`; larger values increase extra transport near the envelope–mantle boundary. Default `0.1`.
- `eddy_overshoot_fhp` `[float]`: Sets the overshoot exponential scale length in units of the local pressure scale height, controlling how far the EMB enhancement extends.
- `eddy_overshoot_lambda_cap` `[float]`: Caps the overshoot-enhanced conductivity relative to the local baseline conductivity; nonpositive values disable the cap.
- `semi_shoot` `[bool]`: Enables an optional semiconvective-style EMB extension that enhances transport near the EMB without directly changing the deeper core thermal structure.
- `semi_shoot_len_env` `[int]`: Sets how many boundaries outward into the envelope are included in the `semi_shoot` enhancement region.
- `semi_shoot_len_mantle` `[int]`: Sets how many boundaries inward into the mantle are included in the `semi_shoot` enhancement region.
- `semi_shoot_fhp` `[float]`: Sets the exponential taper scale (in EMB pressure scale heights) for the `semi_shoot` transport enhancement.
- `semi_shoot_cap_frac` `[float]`: Caps the fractional strength of the `semi_shoot` enhancement; `1.0` allows full enhancement, `0.0` disables it.
- `p_err_floor` `[float]`: Pressure floor (GPa) for the adaptive error metric. Zones with P below this threshold are excluded from the S/T/rho error so rapidly-cooling surface zones don't force tiny timesteps. Raise for lower-mass planets.

#### `[diffusion]`

- `composition_change` `[bool]`: Enables time evolution of composition (`Y` and `Z`) instead of keeping composition fixed throughout the run.
- `D_MLT` `[bool]`: Uses mixing-length-theory diffusivity for composition transport instead of relying only on user-specified diffusion coefficients.
- `under_relax` `[float]`: Sets the under-relaxation factor for the transport Newton-Raphson solve; reducing it can help stabilize difficult runs at the expense of speed.
- `lam_factor` `[float]`: Multiplies conductivity by an exploration factor; mainly a sensitivity-testing knob rather than a standard physical parameter.
- `water_cond_extrapolation_mode` `[int]`: Chooses how the French (2019) water conductivity model behaves outside its calibrated range. `0` = legacy/raw extrapolation, `1` = clipping at rho≤6 g/cc and T≤50,000 K, `2` = log-space extrapolation (default).
- `R_rho` `[float]`: Sets the stability criterion interpolation between Schwarzschild (`0`) and Ledoux (`1`), so it changes when layers are treated as convective vs. compositionally stabilized. Default `1.0` (full Ledoux).
- `Dmicro_y` `[float]`: Sets the baseline microscopic self-diffusion coefficient for helium composition transport (cm^2/s), which is added to any MLT-based diffusivity.
- `Dmicro_z` `[float]`: Sets the baseline microscopic self-diffusion coefficient for metals (cm^2/s), which is added to convective/MLT mixing when enabled.
- `Dmicro_z_in` `[float]`: Sets an alternate inner-region microscopic Z diffusivity (cm^2/s) used below the enclosed-mass threshold set by `z_diff_inner_mfrac`.
- `pressure_dependent_Dmicro` `[bool]`: Uses pressure-dependent micro-diffusion coefficients from ab initio data (French 2012 + Preising 2023). When `True`, `Dmicro_y` is replaced by D_He(P,Y). When `False`, the constant `Dmicro_y` value is used.
- `y_dependent_conductivity` `[bool]`: Blends thermal conductivity and diffusion coefficients between French (2012) and Preising (2023) based on local helium mass fraction Y. When `False`, only the French/Becker pressure-only baseline is used.
- `env_metal_cond_frock` `[bool]`: f_rock-weighted envelope metal conductivity. The metal term of the envelope conduction blend becomes `(1 - f_rock)*Lambda_water + f_rock*Lambda_silicate` (Stamenkovic electronic + PD25 lattice), recovering rocky-mantle conductivities in the `f_rock -> 1` limit. Default `False` (legacy water-only metal term).
- `z_diff_inner_mfrac` `[float]`: Sets the enclosed-mass-fraction cutoff below which `Dmicro_z_in` is used instead of the default `Dmicro_z`.
- `smooth_ledoux` `[bool]`: Smoothly ramps `D_MLT` from its Schwarzschild value down to `D_micro` in Schwarzschild-unstable but Ledoux-stable zones, reducing artificial composition-gradient staircase artifacts. Active regardless of the `semiconvection` flag. Default `False` for backward compatibility.
- `smooth_ledoux_n` `[float]`: Power-law exponent controlling the steepness of the smooth Ledoux transition. `D_transition = D_mlt_schwarz * (1 - chi)^n`. `n=1` is linear; `n=2` is a smooth quadratic ramp. Higher `n` → sharper cutoff.
- `smooth_eos_derivs` `[bool]`: Enables Gaussian smoothing of dS/dY and dS/dZ EOS derivatives to suppress grid-scale noise.
- `smooth_eos_sigma` `[float]`: Sigma for Gaussian smoothing in grid cells; only used when `smooth_eos_derivs = True`.
- `helium_rain` `[bool]`: Enables helium phase separation/rain in the diffusion-advection treatment; only has an effect when `composition_change = True`.
- `metal_rain` `[bool]`: Enables analogous rain/separation treatment for metals (water or rock), primarily intended for sub-Neptune-type evolution studies.
- `evol_Z` `[bool]`: Enables evolution of the heavy-element profile `Z`, including convective mixing and/or Z-rain effects when those processes are active.
- `misc_curve` `[str]`: Selects the miscibility curve model (`l` for Lorenzen, `s` for Schoettler/Redmer), which directly changes where phase separation is predicted.
- `misc_deltaT` `[float]`: Applies a temperature offset to the selected miscibility curve, which is useful for sensitivity tests around the nominal phase boundary.
- `misc_deltaP` `[float]`: Applies a pressure offset (in Mbar) to the selected miscibility curve. Positive values displace the curve to higher pressure — the curve is evaluated at `P - misc_deltaP` — moving the rain region deeper into the planet. Default `0.0`. The high-temperature validity ceiling of the tabulated curve shifts with `misc_deltaT` (e.g. 14 kK + deltaT for Lorenzen).
- `zmisc_hhe` `[bool]`: Selects the kind of Z (metal) rain. `True` (default) uses the modified H-He miscibility diagram as a stand-in for H-Z miscibility — the only metal-rain mode currently available (a numerical test, not a physical curve), and the channel that `zmisc_deltaT`/`zmisc_deltaP` apply to. `False` uses the physical curve selected by `misc_kind` (`water` = Gupta, `mgsio3` = Gilmore/silicate), which are not yet released. The legacy `misc_kind = hhe` is equivalent to `zmisc_hhe = True`.
- `zmisc_deltaT` `[float]`: Temperature offset for the metal-rain channel when `zmisc_hhe = True` uses the H-He curve as the H-Z miscibility stand-in. Defaults to `misc_deltaT`. Set independently when helium rain and the Z stand-in run together (He at its physical calibration, Z strongly displaced). Has no effect when `zmisc_hhe = False`.
- `zmisc_deltaP` `[float]`: Pressure offset (Mbar) for the metal-rain channel when `zmisc_hhe = True`. Defaults to `misc_deltaP`. Has no effect when `zmisc_hhe = False`.
- `zmisc_p_max` `[float]`: Deep-pressure cap (Mbar) on the Z-rain activation window. `0.0` (default) is legacy behavior: the window runs from `1 Mbar + zmisc_deltaP` down to the envelope base. When set, cells with `P > zmisc_p_max` never activate metal rain — the miscibility stand-in is not meant to apply at envelope-base depths, and the cap keeps envelope-base depths out of the activation window. Applies to all `misc_kind` branches.
- `Hr_y` `[float]`: Sets the pressure scale height parameter for rain transport (both the helium and metal rain channels), controlling how rain redistribution is spatially distributed. With `use_local_Hp = True` the local envelope pressure scale height is used instead.
- `alpha_herain` `[float]`: Mixing-length multiplier for the rain pressure scale height. Scales Hp only in the rain advection velocity `v = D2 / (alpha_herain * Hp)` — applied identically to the helium and metal rain channels — leaving the MLT convective scale height unchanged. Values < 1 accelerate sedimentation; values > 1 slow it; `1.0` is no scaling. Default `0.3`.
- `rain_model` `[str]`: Selects the rain advection model. `'legacy'` uses the D/H scale-height pseudo-velocity (original ORCHARD formulation). `'bottom_up'` uses implicit advection with D2/Hp velocity (Helled et al. 2025). Convective-diffusion suppression is controlled separately by `rain_conv_supp_k` and applies to both models.
- `rain_error_factor` `[float]`: Multiplicative relaxation factor for the adaptive timestep error threshold when rain is active (≥ 1, default `1.0`). With the implicit rain solver, no relaxation is needed.
- `rain_conv_supp_k` `[float]`: Convective composition-diffusion suppression strength in the rain zone. Higher values suppress convective diffusion more aggressively where Y exceeds the miscibility curve, letting advective sedimentation dominate. Default `0.0` (suppression off). Typical values when enabling: 1–100.
- `innermisc_thesh` `[float]`: Cap on the helium (Y) abundance used in the miscibility treatment, and the denominator of the He-rain advection velocity `v = D2 / (Hp * innermisc_thesh)`. Helps avoid unphysical helium values in deep layers. Default `0.99`.
- `t_Db_relax` `[float]`: Sets the timescale (Myr) over which the composition diffusion coefficient is relaxed, smoothing abrupt diffusivity changes.
- `max_Db_change` `[float]`: Maximum allowed ratio of D_b change from its initial-timestep value. D_b is clipped to `[D_b_first / max_Db_change, D_b_first * max_Db_change]`. Larger values allow D_b to vary more freely at the Ledoux transition, improving Newton convergence for inhomogeneous profiles.
- `misc_kind` `[str]`: Selects the *physical* Z-miscibility species/curve used **only when `zmisc_hhe = False`**: `mgsio3` (Gilmore/Stixrude silicate rain) or `water` (Gupta water rain). Ignored when `zmisc_hhe = True` (the H-He stand-in is used instead). The legacy value `hhe` is still honored and is equivalent to setting `zmisc_hhe = True`.
- `pmisc_min` `[float]`: Currently read but not applied by the code. Intended as a minimum pressure threshold for miscibility handling; the implementation is incomplete. Default `1.0`.

#### `[core]`

- `mass_core` `[float]`: Sets the total compact interior mass (mantle + iron core) in Earth masses. Set to `0.0` for coreless gas giants.
- `mass_core_fe` `[float]`: Sets the iron-core mass portion (in Earth masses) within `mass_core`, so it controls the mantle/core split. Set to `0.0` for no iron core.
- `use_mass_fractions` `[bool]`: When `True`, `mass_core` and `mass_core_fe` are computed from the total planet mass as fractions: `mass_core = mantle_fraction * M_factor` and `mass_core_fe = iron_core_fraction * mass_core`. When `False` (default), the absolute `mass_core` / `mass_core_fe` values above are used directly.
- `mantle_fraction` `[float]`: Fractional (mantle + iron core) mass of the planet. Used only when `use_mass_fractions = True`.
- `iron_core_fraction` `[float]`: Fractional iron-core mass of the (mantle + iron core). Used only when `use_mass_fractions = True`.
- `mantle_comp` `[str]`: Selects the mantle composition/EOS: `mg2sio4` (ANEOS, Stewart et al. 2020), `mgsio3` (Luo & Deng 2023), or `h2o` (water), which sets the rocky-interior thermodynamics.
- `eos_mantle` `[str]`: Selects the mantle EOS implementation. For `mantle_comp = mgsio3`: `comb` (combined), `JJ` (JJ Dong), or `PPV_2` (post-perovskite). For `mantle_comp = h2o`: `revised` selects the revised AQUA water EOS (`aqua_revised_core_eos`, matching a `z_eos = aqua_revised` envelope); any other value (e.g. the default `comb`) selects the original AQUA (`aqua_core_eos`, Haldemann et al. 2020). Ignored for `mantle_comp = mg2sio4`.
- `f_rock_core` `[float]`: Rock mass fraction of the core metal when `mantle_comp = aquarock`. `0.5` (default) keeps the validated `AQUAROCK_CORE_EOS` VAL mixture with its cached f_rock=0.50 S–P inversion table — exact legacy behavior. Any other value in [0, 1] switches to `ROCKWATER_INTERP_CORE_EOS`, which interpolates between the two validated endpoint core EOSes (revised-AQUA water ≡ `mantle_comp=h2o, eos_mantle=revised`; mg2sio4 ANEOS ≡ `mantle_comp=mg2sio4`): additive-volume density, mass-weighted S/U/Cp/Cv, volume-weighted alpha, bisection T(S,P), forsterite melt curve/latent heat (the AQUAROCK convention). No per-f_rock tables required. Pair with a matching envelope `[initial] f_rock_ini` so the core and envelope stay EOS-continuous across the boundary.
- `core_comp` `[str]`: Selects the core composition model (`Fe_pure` for pure iron or `Fe_alloy` for iron alloy FeSi16), which affects deep-interior density/thermal behavior.
- `eos_core` `[str]`: Selects the core EOS for `core_comp = Fe_pure`: `I14` (Ichikawa et al. 2014, liquid), `D17` (Dorogokupets et al. 2017, liquid), `D17_comb` (combined), `Fe_2`, or `G23_comb` (Gonzalez-Cataldo & Militzer 2023, combined). Ignored for `core_comp = Fe_alloy`.
- `mantle_thermal_conductivity` `[float | None]`: Sets a constant mantle thermal conductivity value in W/m/K (converted internally to cgs in `transport.py`). Set to `None` to use the Stamenkovic et al. (2011) rho/T-dependent model instead.
- `core_thermal_conductivity` `[float]`: Sets the core thermal conductivity in W/m/K (currently treated as a constant); affects how efficiently the core conducts heat.
- `mantle_convection` `[bool]`: Enables convection in the mantle using the inviscid MLT formulation. Should be `True` for sub-Neptunes and super-Earths, `False` for gas giants.
- `viscous_mantle_convection` `[bool]`: Enables viscous mantle convection, extending the inviscid MLT with a viscosity term in the convective velocity calculation. Requires `mantle_convection = True`. Recommended for sub-Neptunes and super-Earths.
- `iron_core_convection` `[bool]`: Enables convection in the iron core using inviscid MLT. Convection is halted when the iron core solidifies (determined by the core material's melting curve). Recommended for sub-Neptunes and super-Earths; `False` for gas giants.
- `mantle_entropy` `[float]`: Sets the initial mantle entropy (k_B per baryon) when `env_s_start = False`, allowing a discontinuous envelope-to-mantle entropy setup.
- `core_entropy` `[float]`: Sets the initial core entropy (k_B per baryon). If no mantle layer exists (i.e., `mass_core_fe == mass_core`), this value is also applied to the core.
- `fe_core_offset` `[float]`: Adds an entropy offset between mantle and core (k_B per baryon); a small positive value seeds extra initial heat in the iron core.

#### `[regrid]`

> **Note:** The active regridder (`utils/regrid.py`) implements a simple scheme — it splits the largest cells in/near the helium- or metal-rain gap until the rain zone holds `max_rain_cells`. Only `regrid_enabled`, `regrid_every`, `t_regrid_start_gyr`, `n_splits_per_step`, and `max_rain_cells` are read by the current code. The remaining keys (`alpha_osc`, `w_floor_rain`, `w_min_global`, `w_smooth_sigma`, `dm_min_frac`, `remap_limiter`) are placeholders for a weight-function / conservative-remap regridder that is not active in the current implementation; they are retained for forward compatibility.

- `regrid_enabled` `[bool]`: Enables adaptive mesh regridding, which redistributes mass cells to concentrate resolution where composition gradients are steepest (primarily the He or metal rain zone). Default `False`.
- `regrid_every` `[int]`: Triggers a regrid every this many accepted timesteps.
- `t_regrid_start_gyr` `[float]`: Minimum age (Gyr) before regridding activates; avoids disrupting the early-time evolution.
- `alpha_osc` `[float]`: Weight-function parameter (grouped under "weight function parameters" in the ini). Not read by the current regrid implementation.
- `w_floor_rain` `[float]`: Weight-function floor for the rain zone. Not read by the current regrid implementation.
- `w_min_global` `[float]`: Global weight-function floor. Not read by the current regrid implementation.
- `w_smooth_sigma` `[float]`: Gaussian smoothing width (in cells) for the weight function. Not read by the current regrid implementation.
- `dm_min_frac` `[float]`: Intended minimum cell mass as a fraction of `M_planet / N`. Not read by the current regrid implementation.
- `n_splits_per_step` `[int]`: Number of cells to split in/near the rain zone per regrid step.
- `max_rain_cells` `[int]`: Maximum number of cells in the rain zone; once reached, the regridder stops splitting.
- `remap_limiter` `[str]`: Intended slope limiter for conservative profile remapping (`minmod`). Not read by the current regrid implementation.
