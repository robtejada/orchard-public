#!/usr/bin/env python3
"""
Figures for the MH24 reproducibility study. Saves to results/figs/.
    /opt/anaconda3/envs/orchard_env/bin/python validation/mh24/make_plots.py
"""
import os
import sys
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)
import mh24_io
FIG = os.path.join(HERE, "results", "figs")
os.makedirs(FIG, exist_ok=True)


def fig_step1():
    """TOF7-vs-CMS relative |diff| per model (corrected, self-consistent input)."""
    R = json.load(open(os.path.join(HERE, "results", "step1_gravity.json")))
    labels = [r["label"] for r in R]
    x = np.arange(len(R))
    fig, ax = plt.subplots(figsize=(11, 5))
    for n, mk in zip((2, 4, 6, 8), ("o", "s", "^", "d")):
        rel = [abs(r["tof7"][str(n)] - r["cms"][str(n)]) / abs(r["cms"][str(n)]) for r in R]
        ax.semilogy(x, rel, mk + "-", label=f"J{n}")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("|J_n(ToF7) - J_n(CMS)| / |J_n(CMS)|")
    ax.set_title("Step 1: ORCHARD ToF7 vs MH24 CMS on MH24's own density (EOS-independent)")
    ax.axhline(0.0006/14696, ls=":", c="gray", label="Juno sigma/J2")
    ax.grid(True, which="both", alpha=0.3); ax.legend(ncol=5, fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "step1_tof7_vs_cms.png"), dpi=130)
    print("wrote step1_tof7_vs_cms.png")


def fig_structure(mid="10969"):
    """Reproduce MH24 Fig 5 (Z,Y) + Fig 6 (rho,T) with ORCHARD HSE overlay."""
    rows = {r["mid"]: r for r in json.load(open(os.path.join(HERE, "results", "step2_hse_summary.json")))}
    r = rows[mid]
    mf = np.array(r["mf_cell"]); rho_or = np.array(r["rho_or"]); T_or = np.array(r["T_or"])
    rho_mh = np.array(r["rho_mh"]); T_mh = np.array(r["T_mh"])
    L = mh24_io.load_layers(mid); mfL = L["CMass"]

    # x-axis = fractional mass, 0 = CENTER (left), 1 = SURFACE (right), as in
    # MH24 Fig 5/6 (do NOT invert): Z is flat-high in the interior, drops to
    # flat-low in the exterior (negative slope inside->outside).
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    (a, b), (cc, d) = axes
    a.plot(mfL, L["Z"], "k-", lw=2, label="MH24 Z (=1-X-Y)")
    a.plot(mfL, L["Y"], "b-", lw=2, label="MH24 Y")
    a.set_ylabel("mass fraction"); a.set_title(f"Fig 5 reproduction: composition ({r['label']})")
    a.legend(); a.set_xlabel("fractional mass  (0=center, 1=surface)")
    b.plot(mf, rho_mh, "k-", lw=2, label="MH24 rho"); b.plot(mf, rho_or, "r--", lw=1.5, label="ORCHARD cd+aqua")
    b.set_ylabel("rho [g/cc]"); b.set_title("Fig 6 reproduction: density"); b.legend()
    cc.plot(mf, T_mh, "k-", lw=2, label="MH24 T"); cc.plot(mf, T_or, "r--", lw=1.5, label="ORCHARD")
    cc.set_xlabel("fractional mass  (0=center, 1=surface)"); cc.set_ylabel("T [K]")
    cc.set_title("temperature"); cc.legend()
    res = (rho_or - rho_mh) / rho_mh * 100
    d.plot(mf, res, "g-"); d.axhline(0, c="k", lw=0.5)
    d.set_xlabel("fractional mass  (0=center, 1=surface)"); d.set_ylabel("(rho_OR - rho_MH)/rho_MH [%]")
    d.set_title("density residual"); d.set_ylim(-5, 5); d.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, f"step2_structure_{mid}.png"), dpi=130)
    print(f"wrote step2_structure_{mid}.png")


def fig_step2():
    """End-to-end J_n relative diff per model."""
    R = json.load(open(os.path.join(HERE, "results", "step2_hse_summary.json")))
    labels = [r["label"] for r in R]; x = np.arange(len(R))
    fig, ax = plt.subplots(figsize=(11, 5))
    for n, mk in zip((2, 4, 6, 8), ("o", "s", "^", "d")):
        rel = [(r["tof7"][str(n)] - r["cms"][str(n)]) / abs(r["cms"][str(n)]) * 100 for r in R]
        ax.plot(x, rel, mk + "-", label=f"J{n}")
    ax.axhline(0, c="k", lw=0.5)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("(ORCHARD - CMS)/CMS  [%]")
    ax.set_title("Step 2: end-to-end ORCHARD HSE(cd+aqua)+ToF7 vs MH24 CMS (same Y,Z)")
    ax.grid(True, alpha=0.3); ax.legend(ncol=4)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "step2_endtoend_Jn.png"), dpi=130)
    print("wrote step2_endtoend_Jn.png")


if __name__ == "__main__":
    fig_step1(); fig_structure("10969"); fig_step2()
    print(f"figs in {FIG}")
