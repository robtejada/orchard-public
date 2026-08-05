#!/usr/bin/env python3
"""
Offline ToF7 post-processing of saved ORCHARD evolution profiles.

Recomputes J2..J8, oblateness, R_eq/R_pol and MoI from a model's FINAL saved
profile (mean radius, density, enclosed mass, omega) using the validated
ToF7 solver (utils.TOF7.get_moments7), regardless of the tof_order the run
was evolved with.

This isolates the harmonic-METHOD difference (ToF4 vs ToF7) on a FIXED
structure. It does NOT re-equilibrate HSE under ToF7 (that is a separate,
smaller, structure-level effect). The structure was already iterated to
self-consistency under its evolution-time gravity at every timestep, so its
final (r, rho, m, omega) is a valid fixed input.

Usage:
    python validation/offline_tof7.py models/<run1> models/<run2> ...
    python validation/offline_tof7.py --glob 'models/exp_*'
"""
import os
import sys
import glob
import argparse

import numpy as np
import h5py

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from utils.TOF7 import get_moments7  # noqa: E402

# Juno reference (Durante et al. 2020), x1e6
JUNO = dict(J2=14696.57, J4=-586.61, J6=34.20)


def tof7_from_h5(h5_path, snapshot=-1):
    """Return ToF7 j2n for the requested snapshot of an evolution.h5 file.

    j2n = [J2*1e6, J4*1e6, J6*1e6, J8*1e6, oblateness, R_eq, R_pol, I_MoI].
    Returns None if the run is non-rotating (omega == 0) or fields are absent.
    """
    with h5py.File(h5_path, "r", swmr=True) as f:
        if "omega" not in f or "r" not in f:
            return None
        omega = float(np.asarray(f["omega"])[snapshot])
        if not np.isfinite(omega) or omega == 0.0:
            return None
        r = np.asarray(f["r"][snapshot], dtype=float)       # surface->center
        rho = np.asarray(f["rho"][snapshot], dtype=float)
        m_b = np.asarray(f["m_b"], dtype=float)
        # m_b may be (T, N+1) time-series or static (N+1,)
        if m_b.ndim == 2:
            m_b = m_b[snapshot]
    mass_mid = 0.5 * (m_b[1:] + m_b[:-1])                    # surface->center
    # ToF7 wants center->surface ordering
    j2n, _ = get_moments7(r[::-1], rho[::-1], mass_mid[::-1], omega)
    return np.asarray(j2n, dtype=float)


def tof4_saved(h5_path, snapshot=-1):
    """Read the evolution-time (ToF4 or whatever tof_order) saved harmonics."""
    with h5py.File(h5_path, "r", swmr=True) as f:
        if "j2" not in f:
            return None
        get = lambda k: float(np.asarray(f[k])[snapshot]) if k in f else np.nan
        return dict(J2=get("j2"), J4=get("j4"), J6=get("j6"),
                    Req=get("eq_rad") if "eq_rad" in f else np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*", help="model output dirs (containing evolution.h5)")
    ap.add_argument("--glob", default=None, help="glob pattern for model dirs")
    args = ap.parse_args()

    dirs = list(args.runs)
    if args.glob:
        dirs += sorted(glob.glob(args.glob))
    if not dirs:
        print("no runs given")
        return

    hdr = f"{'model':46s} {'J2e6':>9} {'J4e6':>8} {'J6e6':>6} {'Req km':>7} | {'dJ2/Juno':>8} {'dJ4/Juno':>8}"
    print(hdr)
    print("-" * len(hdr))
    rows = []
    for d in dirs:
        h5 = os.path.join(d, "evolution.h5")
        if not os.path.exists(h5):
            continue
        name = os.path.basename(d.rstrip("/"))
        try:
            j = tof7_from_h5(h5)
        except Exception as e:
            print(f"{name:46s}  ERR {repr(e)[:60]}")
            continue
        if j is None:
            print(f"{name:46s}  (non-rotating / no harmonics)")
            continue
        J2, J4, J6, Req = j[0], j[1], j[2], j[5] / 1e5
        dJ2 = J2 - JUNO["J2"]
        dJ4 = J4 - JUNO["J4"]
        print(f"{name:46s} {J2:9.1f} {J4:8.2f} {J6:6.2f} {Req:7.0f} | {dJ2:8.1f} {dJ4:8.2f}")
        rows.append((name, J2, J4, J6, Req, dJ2, dJ4))
    print("-" * len(hdr))
    print(f"{'Juno (Durante+2020)':46s} {JUNO['J2']:9.1f} {JUNO['J4']:8.2f} {JUNO['J6']:6.2f} {71492:7.0f}")
    return rows


if __name__ == "__main__":
    main()
