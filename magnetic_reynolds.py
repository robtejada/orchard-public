"""
Magnetic Reynolds number (dynamo criterion) for ORCHARD rocky-planet models.

Follows the CMAPPER method (Zhang 2025, "The CMAPPER Terrestrial Planet
Evolution Code", Section 3.3):

    Re_m = mu_0 * v * L_c * sigma                       (Eq. 14)   dynamo if Re_m > ~50
    v    = ( L_c * F_conv / (rho * H_T) )^(1/3)         (Eq. 16, Christensen 2010)
    H_T  = c_P / (alpha * g)                            temperature scale height

where
    L_c    = radial thickness of a contiguous LIQUID (convecting) region,
    F_conv = convective heat FLUX (erg/cm^2/s),
    sigma  = electrical conductivity (S/m).

ORCHARD outputs everything needed.  IMPORTANT subtleties baked in here:
  * ORCHARD's saved `conv_flux` is a convective LUMINOSITY (erg/s), so the
    per-area flux is F_conv = conv_flux / (4 pi r_b^2).
  * EOS C_P / alpha take pressure in GPa (P_dyn * dyn_to_GPa) and return CGS.
  * Electrical conductivity of the magma ocean is obtained from the SAME
    combined thermal conductivity ORCHARD uses for the mantle
    (transport.py:2108-2111): k = get_k_el(T) + get_k_lat_liq(rho, T),
    inverted to sigma via Wiedemann-Franz, sigma = k / (L0 * T).
    The liquid iron core uses CMAPPER Eq. 18: sigma = k_c / (L0 * T).

This is a prototype analysis tool (L_c via a melt-fraction threshold, default
0.75).  Run with the orchard_env interpreter:

    /opt/anaconda3/envs/orchard_env/bin/python magnetic_reynolds.py

Developer: prototype for ORCHARD dynamo diagnostics.
"""

from __future__ import annotations

import configparser
import glob
import os

import h5py
import numpy as np

import load_models
from utils import const
from utils import mantle_core_props as mcp
from eos import mg2sio4_aneos_eos, gonzalez_iron_eos


# --------------------------------------------------------------------------- #
# Physical constants and parameters
# --------------------------------------------------------------------------- #
MU0 = 4.0 * np.pi * 1.0e-7      # vacuum permeability [SI, H/m]
L0_SILICATE = 2.7e-8           # Lorenz number used by ORCHARD (get_k_elec_liq)  [W ohm/K^2]
L0_IRON = 2.44e-8              # Sommerfeld Lorenz number for the metallic core   [W ohm/K^2]
REM_CRIT = 50.0               # critical magnetic Reynolds number (Christensen & Aubert 2006)
MELT_THRESHOLD = 0.75         # melt fraction above which a cell is "liquid" (dynamo-capable)
G_CGS = const.G               # gravitational constant [cm^3 g^-1 s^-2]

# W/m/K  ->  erg/(s cm K)   (1 W/m/K = 1e5 erg/(s cm K))
WmK_to_cgs = 1.0e5
ERG_PER_S_TO_W = 1.0e-7
ERGCM2S_TO_WM2 = 1.0e-3
CM_TO_M = 1.0e-2


# --------------------------------------------------------------------------- #
# EOS singletons (mirror initial.py for mantle_comp=mg2sio4, eos_core=G23_comb)
# --------------------------------------------------------------------------- #
_MANTLE_EOS = None
_CORE_EOS = None


def _get_eos():
    """Instantiate (once) the Mg2SiO4 mantle EOS and Gonzalez iron core EOS."""
    global _MANTLE_EOS, _CORE_EOS
    if _MANTLE_EOS is None:
        _MANTLE_EOS = mg2sio4_aneos_eos.MG2SIO4_ANEOS_EOS()
    if _CORE_EOS is None:
        _CORE_EOS = gonzalez_iron_eos.Fe_GONZALEZ_EOS()
    return _MANTLE_EOS, _CORE_EOS


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _find_h5(model_dir: str) -> str:
    for name in ("evolution.h5", "outputs.h5", "profiles.h5"):
        fp = os.path.join(model_dir, name)
        if os.path.isfile(fp):
            return fp
    cand = sorted(glob.glob(os.path.join(model_dir, "*.h5")) +
                  glob.glob(os.path.join(model_dir, "*.hdf5")))
    if not cand:
        raise FileNotFoundError(f"No HDF5 output found in {model_dir}")
    return cand[0]


def _read_boundary_arrays(h5path: str, idx: int):
    """Read time-resolved boundary radius r_b and enclosed mass m_b [CGS]."""
    with h5py.File(h5path, "r", swmr=True, libver="latest") as f:
        r_b = np.asarray(f["r_b"][idx], dtype=float)   # (N+1) cm
        m_b = np.asarray(f["m_b"][idx], dtype=float)    # (N+1) g (enclosed mass at boundary)
        cp_saved = np.asarray(f["cp"][idx], dtype=float) if "cp" in f else None  # (N-1) erg/g/K
    return r_b, m_b, cp_saved


def _read_core_k(param_path: str) -> float:
    """core_thermal_conductivity [W/m/K] from the .ini (fallback to default 40)."""
    cfg = configparser.ConfigParser()
    cfg.read(param_path)
    base = configparser.ConfigParser()
    base.read("parameters_default.ini")
    for c in (cfg, base):
        if "core" in c and "core_thermal_conductivity" in c["core"]:
            try:
                return float(c["core"]["core_thermal_conductivity"])
            except ValueError:
                pass
    return 40.0


def _contiguous_runs(mask: np.ndarray, lo: int, hi: int):
    """Yield (a, b) inclusive index ranges of contiguous True runs in mask[lo:hi]."""
    runs = []
    i = lo
    while i < hi:
        if mask[i]:
            j = i
            while j + 1 < hi and mask[j + 1]:
                j += 1
            runs.append((i, j))
            i = j + 1
        else:
            i += 1
    return runs


# --------------------------------------------------------------------------- #
# Core computation
# --------------------------------------------------------------------------- #
def compute_rem(ms: load_models.ModelSet, model_idx: int, age_gyr: float,
                sigma_mode: str = "combined", melt_threshold: float = MELT_THRESHOLD):
    """
    Compute the per-cell magnetic Reynolds number at a given age.

    Parameters
    ----------
    ms : load_models.ModelSet
    model_idx : int
    age_gyr : float
    sigma_mode : {"combined", "electronic"}
        Silicate electrical conductivity from the combined (electronic+lattice)
        or electronic-only thermal conductivity, inverted via Wiedemann-Franz.

    Returns
    -------
    dict with per-cell arrays (length N): Re_m, v, sigma, L_c, F_conv, g, H_T,
    cp, alpha, P, T, rho, melt, region ('env'/'mantle'/'iron'), plus the matched
    age and region indices.
    """
    model = ms[model_idx]
    snap = model.snapshot(age_gyr, units="Gyr")
    idx = int(snap["snapshot_index"])
    matched_age = float(snap["matched_age"])

    kcore = int(ms.kcore[model_idx])
    kcore_fe = int(ms.kcore_fe[model_idx])
    core_k = _read_core_k(ms.param_paths[model_idx])

    r = np.asarray(snap["rad"], dtype=float)          # (N) cm, cell-centered
    rho = np.asarray(snap["rho"], dtype=float)        # (N) g/cm^3
    T = np.asarray(snap["temp"], dtype=float)         # (N) K
    P = np.asarray(snap["press"], dtype=float)        # (N) dyn/cm^2
    melt = np.asarray(snap["melt_fraction"], dtype=float)   # (N)
    cf = np.asarray(snap["conv_flux"], dtype=float)   # (N+1) erg/s  (LUMINOSITY)
    N = r.size

    h5path = _find_h5(ms.model_dirs[model_idx])
    r_b, m_b, cp_saved = _read_boundary_arrays(h5path, idx)

    # --- F_conv: luminosity -> per-area flux at boundaries, then to centers ---
    A_b = 4.0 * np.pi * r_b ** 2
    F_b = cf / np.maximum(A_b, 1e-99)                 # erg/cm^2/s at boundaries
    F_cell = 0.5 * (F_b[:-1] + F_b[1:])               # (N) at cell centers
    F_cell = np.abs(F_cell)

    # --- gravity at cell centers: g = G M_enc / r^2 ---
    m_cell = 0.5 * (m_b[:-1] + m_b[1:])               # (N) enclosed mass at cell center
    g = G_CGS * m_cell / np.maximum(r, 1e-99) ** 2    # cm/s^2

    # --- c_P, alpha from the relevant EOS (P in GPa, CGS out) ---
    mantle_eos, core_eos = _get_eos()
    cp = np.full(N, np.nan)
    alpha = np.full(N, np.nan)

    m_sl = slice(max(kcore, 0), max(kcore_fe, 0))     # mantle cells
    f_sl = slice(max(kcore_fe, 0), N)                 # iron cells
    if m_sl.stop > m_sl.start:
        cp[m_sl] = mantle_eos.get_cp_pt(P[m_sl] * mantle_eos.dyn_to_GPa, T[m_sl])
        alpha[m_sl] = mantle_eos.get_alpha_pt(P[m_sl] * mantle_eos.dyn_to_GPa, T[m_sl])
    if f_sl.stop > f_sl.start:
        cp[f_sl] = core_eos.get_CP_pt(P[f_sl] * core_eos.dyn_to_GPa, T[f_sl])
        alpha[f_sl] = core_eos.get_alpha_pt(P[f_sl] * core_eos.dyn_to_GPa, T[f_sl])

    # --- temperature scale height H_T = c_P / (alpha g) ---
    with np.errstate(divide="ignore", invalid="ignore"):
        H_T = cp / (alpha * g)                         # cm

    # --- electrical conductivity sigma [S/m] ---
    sigma = np.zeros(N)
    if m_sl.stop > m_sl.start:
        k_el = mcp.get_k_el(T[m_sl])                              # W/m/K (electronic)
        if sigma_mode == "combined":
            k_lat = mcp.get_k_lat_liq(rho[m_sl], T[m_sl])         # W/m/K (lattice)
            k_comb = k_el + k_lat
        elif sigma_mode == "electronic":
            k_comb = k_el
        else:
            raise ValueError("sigma_mode must be 'combined' or 'electronic'")
        # melt-fraction weighting of the silicate conductivity (CMAPPER)
        sigma[m_sl] = melt[m_sl] * k_comb / (L0_SILICATE * T[m_sl])
    if f_sl.stop > f_sl.start:
        sigma[f_sl] = core_k / (L0_IRON * T[f_sl])               # CMAPPER Eq. 18

    # --- liquid mask and per-region L_c (thickness of contiguous liquid runs) ---
    liquid = melt >= melt_threshold
    liquid[:kcore] = False                            # ignore the H/He envelope
    L_c = np.zeros(N)                                 # cm
    region = np.array(["env"] * N, dtype=object)
    region[m_sl] = "mantle"
    region[f_sl] = "iron"
    for (lo, hi) in ((max(kcore, 0), max(kcore_fe, 0)), (max(kcore_fe, 0), N)):
        for (a, b) in _contiguous_runs(liquid, lo, hi):
            # boundary radii bracket the run: r_b[a] (outer) .. r_b[b+1] (inner)
            L_c[a:b + 1] = abs(r_b[a] - r_b[b + 1])

    # --- velocity and magnetic Reynolds number ---
    valid = liquid & (L_c > 0) & (F_cell > 0) & np.isfinite(H_T) & (H_T > 0)
    v = np.zeros(N)                                   # cm/s
    v[valid] = (L_c[valid] * F_cell[valid] / (rho[valid] * H_T[valid])) ** (1.0 / 3.0)

    Re_m = np.zeros(N)
    Re_m[valid] = MU0 * (v[valid] * CM_TO_M) * (L_c[valid] * CM_TO_M) * sigma[valid]

    return dict(
        age=matched_age, idx=idx, N=N, kcore=kcore, kcore_fe=kcore_fe, core_k=core_k,
        Re_m=Re_m, v=v, sigma=sigma, L_c=L_c, F_conv=F_cell, g=g, H_T=H_T,
        cp=cp, alpha=alpha, P=P, T=T, rho=rho, melt=melt, region=region,
        valid=valid, cp_saved=cp_saved, r=r, r_b=r_b,
    )


# --------------------------------------------------------------------------- #
# Validation: EOS c_P vs ORCHARD's saved `cp` (confirms GPa unit convention)
# --------------------------------------------------------------------------- #
def validate_cp(res):
    """Compare EOS-derived c_P (cell-centered) to ORCHARD's saved cp (interfaces)."""
    cp_saved = res["cp_saved"]
    if cp_saved is None:
        print("  [validate] no saved cp dataset; skipping.")
        return
    kc, kcfe = res["kcore"], res["kcore_fe"]
    cp = res["cp"]
    # saved cp has length N-1 (interfaces between cells i and i+1); compare on
    # the mantle interfaces against the mean of the two bracketing cell values.
    cp_cell_iface = 0.5 * (cp[:-1] + cp[1:])          # (N-1) cell->interface
    sl = slice(kc, kcfe - 1)
    a = cp_cell_iface[sl]
    b = cp_saved[sl]
    good = np.isfinite(a) & np.isfinite(b) & (b > 0)
    if good.sum() == 0:
        print("  [validate] no overlapping finite cp values.")
        return
    ratio = a[good] / b[good]
    print(f"  [validate] mantle c_P  EOS/saved  ratio: "
          f"mean={np.mean(ratio):.4f}  median={np.median(ratio):.4f}  "
          f"5-95%={np.percentile(ratio,5):.3f}-{np.percentile(ratio,95):.3f}")


# --------------------------------------------------------------------------- #
# Reporting + plotting
# --------------------------------------------------------------------------- #
def _region_summary(res, region_name):
    sel = (res["region"] == region_name) & res["valid"]
    if sel.sum() == 0:
        return None
    Re = res["Re_m"][sel]
    P_GPa = res["P"][sel] * 1e-10
    # mass-weighted mean over the liquid region (weight by cell mass ~ uniform here)
    return dict(n=int(sel.sum()), peak=float(np.nanmax(Re)),
                mean=float(np.nanmean(Re)), Pmin=float(P_GPa.min()),
                Pmax=float(P_GPa.max()),
                Lc_km=float(np.nanmax(res["L_c"][sel]) * 1e-5),
                v_mean_ms=float(np.nanmean(res["v"][sel]) * CM_TO_M),
                sig_mean=float(np.nanmean(res["sigma"][sel])))


def report(ms, model_idx, ages):
    print("=" * 92)
    print("MAGNETIC REYNOLDS NUMBER  (CMAPPER Section 3.3 recipe, applied to ORCHARD)")
    print(f"  model: {ms.param_paths[model_idx]}")
    print(f"  kcore={ms.kcore[model_idx]}  kcore_fe={ms.kcore_fe[model_idx]}  "
          f"core_k={_read_core_k(ms.param_paths[model_idx])} W/m/K   "
          f"Re_m,crit={REM_CRIT:.0f}   melt threshold={MELT_THRESHOLD}")
    print("=" * 92)
    for ag in ages:
        res = compute_rem(ms, model_idx, ag)
        print(f"\n--- t = {res['age']:.4f} Gyr  (snapshot {res['idx']}) ---")
        validate_cp(res)
        for rn, label in (("mantle", "magma ocean"), ("iron", "iron core")):
            s = _region_summary(res, rn)
            if s is None:
                print(f"  {label:12s}: no dynamo-capable (liquid) cells")
                continue
            print(f"  {label:12s}: Re_m peak={s['peak']:.3e}  mean={s['mean']:.3e}  "
                  f"| {s['n']:3d} cells, P={s['Pmin']:.1f}-{s['Pmax']:.1f} GPa, "
                  f"L_c={s['Lc_km']:.0f} km, <v>={s['v_mean_ms']:.2e} m/s, "
                  f"<sigma>={s['sig_mean']:.2e} S/m")


def plot_rem_vs_pressure(ms, model_idx, ages, outfile="plots/magnetic_reynolds_1Mearth.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm

    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.0, 5.6))
    colors = cm.viridis(np.linspace(0, 0.92, len(ages)))

    for ag, col in zip(ages, colors):
        res = compute_rem(ms, model_idx, ag)
        sel = res["valid"]
        if sel.sum() == 0:
            continue
        P_GPa = res["P"][sel] * 1e-10
        Re = res["Re_m"][sel]
        order = np.argsort(P_GPa)
        ax.plot(P_GPa[order], Re[order], "-", lw=1.8, color=col,
                label=f"{res['age']:.3f} Gyr")
        # mark the iron-core sub-segment with a heavier line
        iron = sel & (res["region"] == "iron")
        if iron.sum() > 0:
            Pi = res["P"][iron] * 1e-10
            Rei = res["Re_m"][iron]
            oi = np.argsort(Pi)
            ax.plot(Pi[oi], Rei[oi], "-", lw=3.4, color=col, alpha=0.55)

    ax.axhline(REM_CRIT, color="k", ls="--", lw=1.2)
    ax.text(0.98, REM_CRIT * 1.3, r"$Re_{m,\rm crit}\approx50$", ha="right",
            va="bottom", transform=ax.get_yaxis_transform(), fontsize=9)
    ax.axhline(1e4, color="firebrick", ls=":", lw=1.2)
    ax.text(0.98, 1e4 * 1.3, r"Earth core $\sim10^4$", ha="right", va="bottom",
            color="firebrick", transform=ax.get_yaxis_transform(), fontsize=9)

    ax.set_yscale("log")
    ax.set_xlabel("Pressure [GPa]")
    ax.set_ylabel(r"Magnetic Reynolds number  $Re_m$")
    ax.set_title("ORCHARD 1 M$_\\oplus$ super-Earth: dynamo criterion\n"
                 "(thin lines = full liquid profile; thick = liquid iron core)")
    ax.legend(title="age", fontsize=8, ncol=2)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    print(f"\nSaved plot -> {outfile}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    PARAM = "parameter_examples/parameter_user_super_earth_1Mearth_thinenv_example.ini"
    ms = load_models.ModelSet(paths=[PARAM], read_verbose=True)

    example_ages = [0.01, 0.1, 1.0, 2.0, 4.56]
    report(ms, 0, example_ages)

    plot_ages = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 4.56]
    plot_rem_vs_pressure(ms, 0, plot_ages)
