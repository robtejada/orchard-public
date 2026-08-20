"""
Module: static.py — one-call static (non-evolving) planetary structures.

Builds a converged hydrostatic structure for a given mass, entropy, and
composition without running the evolution loop. This is a thin wrapper around
`initial.initialize_grid()` and `hydrostatic.hydrostatic_equilibrium(init=True)`
that handles the library-mode bookkeeping for you:

  * config overrides are applied BEFORE the solver modules read them,
  * import-frozen parameters (core masses, profile shapes, rotation, ...)
    trigger an automatic module reload instead of a kernel restart,
  * the coreless / no-iron-core index sentinels are normalized,
  * the result comes back as a single object with named fields and a
    quick-look plot.

Typical use, from a notebook or script:

    from static import build

    jup = build(M_MJup=1.0, S=6.5, Y=0.277, Z=0.02, mass_core=10.0)
    print(jup)                  # one-line summary
    jup.plot()                  # 2x2 overview figure
    jup.radius_rj, jup.p, jup.rho, jup.temp, ...

    # scans over mass/S/Y/Z/f_rock reuse the loaded EOS (fast):
    radii = [build(M_MJup=m, S=6.5, Y=0.277, Z=0.02).radius_rj
             for m in (0.3, 1.0, 3.0)]

    # changing an import-frozen parameter (e.g. mass_core) is detected and
    # the solver modules are reloaded automatically (quick: the EOS objects
    # are cached across reloads):
    subnep = build(M_Mearth=6.0, S=6.0, Y=0.277, Z=0.05,
                   mass_core=5.4, mass_core_fe=1.8)

Composition profiles for inhomogeneous (fuzzy-core) models are passed as
(coords, values) breakpoint pairs in normalized mass, center (0) -> surface (1),
exactly like `z_init_coords` / `z_init_values` in a config file:

    fuzzy = build(M_MJup=1.0, S=7.0, Y=0.277, Z=0.0156,
                  z_profile=([0, 0.2, 0.5, 1.0], [0.18, 0.18, 0.04, 0.0156]),
                  rotation=True, tof=True)
    print(fuzzy.j2_ppm, fuzzy.j4_ppm)

Everything runs in memory; nothing is written to models/.
"""

import os
os.environ.setdefault("ORCHARD_LIBRARY_MODE", "1")

import importlib
import json

import numpy as np

from utils import common, const

config = common.config

# Config keys that the solver modules read at import time, grouped by the
# cheapest reload that makes a change take effect. Keys not listed here are
# either function arguments (mass, S, Y, Z, f_rock, N) or read at call time.
_INITIAL_FROZEN = {
    ("core", "mass_core"), ("core", "mass_core_fe"),
    ("core", "mantle_comp"), ("core", "core_comp"),
    ("core", "eos_core"), ("core", "eos_mantle"),
    ("core", "mantle_entropy"), ("core", "core_entropy"),
    ("core", "fe_core_offset"),
    ("initial", "rk4_initial"), ("initial", "mz_struct"), ("initial", "m_z"),
    ("initial", "initial_profiles_s"), ("initial", "initial_profiles_y"),
    ("initial", "initial_profiles_z"), ("initial", "initial_profiles_f_rock"),
    ("initial", "struct_interp"), ("initial", "struct_smooth"),
    ("initial", "struct_sigma"),
    ("initial", "z_init_coords"), ("initial", "z_init_values"),
    ("initial", "s_init_coords"), ("initial", "s_init_values"),
    ("initial", "z_init_deltas"), ("initial", "s_init_deltas"),
    ("initial", "f_rock_init_coords"), ("initial", "f_rock_init_deltas"),
    ("equation_of_state", "hhe_eos"), ("equation_of_state", "z_eos"),
    ("equation_of_state", "rock_mixtures"), ("equation_of_state", "eos_version"),
}
_HYDRO_FROZEN = {
    ("hydrostatic_equilibrium", "rotation"),
    ("hydrostatic_equilibrium", "tof_calc"),
    ("hydrostatic_equilibrium", "c_moi"),
    ("hydrostatic_equilibrium", "period"),
    ("hydrostatic_equilibrium", "tol_hydro"),
    ("hydrostatic_equilibrium", "hse_alpha_init"),
    ("hydrostatic_equilibrium", "hse_adaptive_alpha"),
    ("hydrostatic_equilibrium", "hse_gravity_centering"),
    ("hydrostatic_equilibrium", "isothermal_compact_core"),
    ("hydrostatic_equilibrium", "env_s_start"),
}

_initial_mod = None
_hydro_mod = None


def _set(section, key, value):
    """Write one override into the in-memory config; return True if changed."""
    sval = str(value)
    if config.has_option(section, key) and config.get(section, key) == sval:
        return False
    config[section][key] = sval
    return True


def _reload(level):
    """(Re)import the solver modules. level: 'initial' reloads both initial
    and hydrostatic (hydrostatic imports from initial); 'hydro' reloads only
    hydrostatic; 'none' imports them if this is the first call."""
    global _initial_mod, _hydro_mod
    if _initial_mod is None or _hydro_mod is None:
        importlib.import_module("utils.profiles")
        _initial_mod = importlib.import_module("initial")
        _hydro_mod = importlib.import_module("hydrostatic")
        return
    if level == "initial":
        importlib.reload(importlib.import_module("utils.profiles"))
        _initial_mod = importlib.reload(_initial_mod)
        _hydro_mod = importlib.reload(_hydro_mod)
    elif level == "hydro":
        _hydro_mod = importlib.reload(_hydro_mod)


class StaticModel:
    """A converged static structure. Arrays run surface (index 0) to center
    (index N-1). Cell-centered arrays (p, rho, temp, S, Y, Z, f_rock) have
    length N; boundary arrays (r_b, m_b) have length N+1."""

    def __init__(self, **kw):
        self.__dict__.update(kw)

    # -- convenience scalars -------------------------------------------------
    @property
    def radius(self):
        """Outer (1-bar-level) radius [cm]."""
        return float(self.r_b[0])

    @property
    def radius_rj(self):
        return self.radius / const.rjup

    @property
    def radius_re(self):
        return self.radius / const.rearth

    @property
    def mass_me(self):
        return float(self.m_b[0]) / const.mearth

    @property
    def j2_ppm(self):
        """J2 in ppm (x 1e6). Zero unless built with rotation=True, tof=True."""
        return float(self.j2n[0])

    @property
    def j4_ppm(self):
        return float(self.j2n[1])

    def __repr__(self):
        s = (f"StaticModel: M = {self.mass_me:.2f} M_E "
             f"({self.mass_me * const.mearth / const.mjup:.3f} M_J), "
             f"R = {self.radius_rj:.4f} R_J ({self.radius_re:.3f} R_E), "
             f"P_c = {self.p[-1] * 1e-12:.1f} Mbar, "
             f"T_c = {self.temp[-1]:.0f} K, {self.niter} iterations")
        if abs(self.j2_ppm) > 0:
            s += f", J2 = {self.j2_ppm:.1f} ppm"
        return s

    # -- quick-look figure ---------------------------------------------------
    def plot(self, show=True):
        """2x2 overview: rho(P), T(P), S(m), m(r). Returns the figure."""
        import matplotlib.pyplot as plt

        N = len(self.p)
        P_Mbar = self.p * 1e-12
        m_norm = self.m_b[:-1] / self.m_b[0]
        kc = self.kcore
        env, core = slice(0, kc), slice(max(kc - 1, 0), N)
        r_unit, r_lab = ((const.rjup, r"$r$ [R$_{\rm Jup}$]")
                         if self.radius > 2 * const.rearth
                         else (const.rearth, r"$r$ [R$_\oplus$]"))

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
            2, 2, figsize=(11, 7.5), constrained_layout=True)
        ax1.plot(P_Mbar[env], self.rho[env], label="envelope")
        if kc < N:
            ax1.plot(P_Mbar[core], self.rho[core], label="mantle/core")
            ax1.legend(frameon=False)
        ax1.set_xscale("log"); ax1.set_yscale("log")
        ax1.set_xlabel(r"$P$ [Mbar]"); ax1.set_ylabel(r"$\rho$ [g/cm$^3$]")

        ax2.plot(P_Mbar[env], self.temp[env] / 1e3)
        if kc < N:
            ax2.plot(P_Mbar[core], self.temp[core] / 1e3)
        ax2.set_xscale("log")
        ax2.set_xlabel(r"$P$ [Mbar]"); ax2.set_ylabel(r"$T$ [$10^3$ K]")

        # Show Z = 1 in the compact core region for plotting purposes (the
        # stored envelope Z array does not cover the rock/iron core), and fix
        # the axis to 0-0.3 so the Z and S curves do not overlap.
        z_plot = np.array(self.Z, dtype=float, copy=True)
        if kc < N:
            z_plot[kc:] = 1.0
        ax3.plot(m_norm, z_plot, color="C0")
        ax3.set_ylim(0.0, 0.1)
        ax3.set_xlabel(r"$m/M$")
        ax3.set_ylabel(r"$Z$", color="C0")
        ax3.tick_params(axis="y", labelcolor="C0")
        ax3b = ax3.twinx()
        ax3b.plot(m_norm, self.S, color="C1", ls="--")
        ax3b.set_ylabel(r"$S$ [$k_B$/baryon]", color="C1")
        ax3b.tick_params(axis="y", labelcolor="C1")

        ax4.plot(self.r_b / r_unit, self.m_b / self.m_b[0])
        ax4.set_xlabel(r_lab); ax4.set_ylabel(r"$m(r)/M$")

        for ax in (ax1, ax2):
            if 0 < kc < N:
                ax.axvline(P_Mbar[kc - 1], color="0.6", lw=0.8, ls=":")
        if show:
            plt.show()
        return fig


def build(M_Mearth=None, M_MJup=None, S=None, Y=None, Z=None, f_rock=None,
          N=None, mass_core=None, mass_core_fe=None, mantle_comp=None,
          core_comp=None, z_profile=None, s_profile=None, rotation=None,
          tof=None, period=None, c_moi=None, rk4_initial=None,
          overrides=None):
    """Build one converged static structure and return a StaticModel.

    Mass, S, Y, Z, f_rock, and N are passed straight to the solver, so
    scanning them is cheap. The remaining keyword arguments are config
    overrides; when one of them is import-frozen (core masses, compositions,
    profiles, rotation, ...) the solver modules are reloaded automatically.

    Parameters
    ----------
    M_Mearth, M_MJup : float
        Total planet mass (give one of the two).
    S : float
        Envelope entropy [k_B/baryon] (outer value if s_profile is given).
    Y, Z : float
        Helium and heavy-element mass fractions (outer values if a profile
        is given). Y is Y/(X+Y).
    f_rock : float, optional
        Envelope rock fraction (default: config f_rock_ini).
    N : int, optional
        Number of mass zones (default: config [general] N).
    mass_core, mass_core_fe : float, optional
        Compact core and iron sub-core masses [M_Earth].
    mantle_comp, core_comp : str, optional
        e.g. 'mg2sio4' / 'mgsio3' / 'h2o' and 'Fe_pure' / 'Fe_alloy'.
    z_profile, s_profile : (coords, values), optional
        Piecewise ('struct') profiles in normalized mass, center (0) ->
        surface (1), same convention as z_init_coords / z_init_values.
    rotation, tof : bool, optional
        Rigid-body rotation and the ToF4 gravity-harmonics calculation.
        With both True, the returned j2n carries J2..J8 (x 1e6), the
        oblateness, R_eq, R_pol, and the moment of inertia.
    period : float, optional
        Rotation period [s]. c_moi : moment-of-inertia factor for ToF4.
    rk4_initial : bool, optional
        Build the first-guess structure by RK4 shooting (recommended for
        rocky planets and cored gas giants).
    overrides : dict, optional
        Raw escape hatch: {section: {key: value}} applied verbatim to the
        config. Unknown keys force a full module reload to be safe.
    """
    if (M_Mearth is None) == (M_MJup is None):
        raise ValueError("give exactly one of M_Mearth or M_MJup")

    changed = set()

    def setk(section, key, value):
        if value is not None and _set(section, key, value):
            changed.add((section, key.lower()))

    setk("core", "mass_core", mass_core)
    setk("core", "mass_core_fe", mass_core_fe)
    setk("core", "mantle_comp", mantle_comp)
    setk("core", "core_comp", core_comp)
    setk("initial", "rk4_initial", rk4_initial)
    setk("hydrostatic_equilibrium", "rotation", rotation)
    setk("hydrostatic_equilibrium", "tof_calc", tof)
    setk("hydrostatic_equilibrium", "period", period)
    setk("hydrostatic_equilibrium", "c_moi", c_moi)

    if z_profile is not None:
        coords, values = z_profile
        setk("initial", "initial_profiles_Z", "struct")
        setk("initial", "z_init_coords", json.dumps(list(coords)))
        setk("initial", "z_init_values", json.dumps(list(values)))
    if s_profile is not None:
        coords, values = s_profile
        setk("initial", "initial_profiles_S", "struct")
        setk("initial", "s_init_coords", json.dumps(list(coords)))
        setk("initial", "s_init_values", json.dumps(list(values)))

    force_full = False
    if overrides:
        for section, kv in overrides.items():
            for key, value in kv.items():
                if _set(section, key, value):
                    ck = (section, key.lower())
                    changed.add(ck)
                    if ck not in _INITIAL_FROZEN and ck not in _HYDRO_FROZEN:
                        force_full = True

    if force_full or (changed & _INITIAL_FROZEN):
        _reload("initial")
    elif changed & _HYDRO_FROZEN:
        _reload("hydro")
    else:
        _reload("none")

    ini, hyd = _initial_mod, _hydro_mod

    n_zones = int(N) if N is not None else int(config["general"]["N"])
    M_planet = (float(M_Mearth) * const.mearth if M_Mearth is not None
                else float(M_MJup) * const.mjup)
    S_val = float(S) if S is not None else float(config["initial"]["S_ini"])
    Y_val = float(Y) if Y is not None else float(config["initial"]["Y_ini"])
    Z_val = float(Z) if Z is not None else float(config["initial"]["Z_ini"])
    fr_val = (float(f_rock) if f_rock is not None
              else float(config["initial"]["f_rock_ini"]))

    r_b0, p0, m_b, dm, S_arr, Y_arr, Z_arr, fr_arr, kcore, kcore_fe = \
        ini.initialize_grid(n_zones, M_planet, S_val, Y_val, Z_val, fr_val)

    # -1 sentinels mean "no boundary" (coreless / no iron sub-core); map to N
    # in the surface->center index convention, as evolution.py does.
    kcore = n_zones if kcore == -1 else kcore
    kcore_fe = n_zones if kcore_fe == -1 else kcore_fe

    r_b, p, rho, temp, S_arr, g, omega, j2n, shapes, niter = \
        hyd.hydrostatic_equilibrium(
            S_arr, np.log(r_b0), np.log(p0), m_b, Y_arr, Z_arr, fr_arr,
            kcore, kcore_fe, omega_prev=0.0, time_sc=0.0,
            logt_old=np.zeros_like(S_arr), init=True)

    if not np.isfinite(r_b[0]) or r_b[0] <= 0:
        raise RuntimeError(
            "hydrostatic solve did not converge (non-finite radius). "
            "Common fixes: lower/raise S toward a physical value, set "
            "rk4_initial=True for rocky or cored planets, or keep "
            "hse_adaptive_alpha = False for cold static builds.")

    return StaticModel(
        r_b=r_b, p=p, rho=rho, temp=temp, S=S_arr, Y=Y_arr, Z=Z_arr,
        f_rock=fr_arr, m_b=m_b, dm=dm, kcore=kcore, kcore_fe=kcore_fe,
        g=g, omega=omega, j2n=np.asarray(j2n, dtype=float), shapes=shapes,
        niter=int(niter))
