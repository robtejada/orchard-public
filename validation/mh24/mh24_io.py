#!/usr/bin/env python3
"""
Parsers and a model registry for the Militzer & Hubbard (2024) Jupiter data
(Zenodo doi:10.5281/zenodo.10471389).

Two machine-readable table types per model:

layers (2048 spheroids, ordered surface -> center, j=0 at surface):
    j, rE, rS, CMass, P, rho, rho_P, T, S, X, Y, YXY, Z
      rE    fractional EQUATORIAL radius   [units of 71492 km]
      rS    fractional VOLUMETRIC radius   [units of 71492 km], 4/3 pi rS^3 = vol
      CMass cumulative mass               [units of Mjup], computed inside-out
      P     pressure                      [GPa]
      rho   spheroid density              [g/cm^3]  (mass-consistent; use for ToF)
      rho_P density at P from MH24 EOS     [g/cm^3]  (use for EOS comparison)
      T     temperature                   [K]
      S     entropy                       [kb/electron, H:He=110:9 ref]  (NOT kb/baryon)
      X,Y,Z mass fractions               [-]
      YXY   Y/(X+Y)                       [-]

harmonics (n = 2..10):
    n, J_Int (static interior), J_Wind (winds), J_Total (= Int + Wind)
    ORCHARD's ToF7 computes the STATIC field -> compare against J_Int.
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")

# 0-based column indices in a layers data row
LCOL = dict(j=0, rE=1, rS=2, CMass=3, P=4, rho=5, rho_P=6, T=7,
            S=8, X=9, Y=10, YXY=11, Z=12)

# Model registry. M_core (Earth masses) and labels from MH24 Table 1 / Zenodo
# description; M_core is re-derived from the data in measure_core() and
# cross-checked against these expected values.
MODELS = {
    "10972": dict(label="2 layer",                 n_layers=2, mcore_exp=0.0),
    "10973": dict(label="3 layer",                 n_layers=3, mcore_exp=4.0),
    "10970": dict(label="4 layer A (core abrupt)", n_layers=4, mcore_exp=0.0),
    "10971": dict(label="4 layer B (rain abrupt)", n_layers=4, mcore_exp=0.0),
    "10969": dict(label="5 layer A (reference)",   n_layers=5, mcore_exp=0.0),
    "10975": dict(label="5 layer B (compact core)",n_layers=5, mcore_exp=2.5),
    "10976": dict(label="5 layer C (no He rain)",  n_layers=5, mcore_exp=2.5),
    "10977": dict(label="5 layer D (no core trans)",n_layers=5, mcore_exp=2.5),
    "10967": dict(label="6 layer",                 n_layers=6, mcore_exp=2.0),
}
# Display order: simplest -> most complex
ORDER = ["10972", "10973", "10970", "10971", "10969",
         "10975", "10976", "10977", "10967"]


def _data_after_last_dashes(path):
    """Return list of data lines: everything after the last all-dashes line."""
    with open(path) as fh:
        lines = fh.readlines()
    last_sep = -1
    for i, ln in enumerate(lines):
        s = ln.strip()
        if s and set(s) <= set("-"):
            last_sep = i
    return [ln for ln in lines[last_sep + 1:] if ln.strip()]


def layers_path(mid):
    return os.path.join(DATA_DIR, f"out{mid}_layers_mrt_with_header.txt")


def harmonics_path(mid):
    return os.path.join(DATA_DIR, f"out{mid}_harmonics_mrt_with_header.txt")


def load_layers(mid):
    """Load a layers table as a dict of float arrays (surface -> center order)."""
    rows = _data_after_last_dashes(layers_path(mid))
    arr = np.array([[float(x) for x in ln.split()] for ln in rows])
    if arr.shape[1] != len(LCOL):
        raise ValueError(f"{mid}: expected {len(LCOL)} cols, got {arr.shape[1]}")
    return {name: arr[:, idx] for name, idx in LCOL.items()}


def load_harmonics(mid):
    """Load harmonics as dict: {n: {'int':.., 'wind':.., 'total':..}}."""
    rows = _data_after_last_dashes(harmonics_path(mid))
    out = {}
    for ln in rows:
        p = ln.split()
        n = int(p[0])
        out[n] = dict(int=float(p[1]), wind=float(p[2]), total=float(p[3]))
    return out


def measure_core(mid, z_thresh=0.999):
    """Locate the compact core (innermost cells with Z >= z_thresh) and return
    its mass in Mjup and Earth masses, plus the boundary CMass. Returns
    mcore_mjup=0 if no compact core (no cells reach Z>=z_thresh)."""
    from utils import const
    L = load_layers(mid)
    Z = L["Z"]
    CMass = L["CMass"]  # Mjup, =1 at surface (j=0), ->0 at center
    core_mask = Z >= z_thresh
    if not core_mask.any():
        return dict(mcore_mjup=0.0, mcore_mearth=0.0, cmass_boundary=0.0,
                    k_core_boundary=None)
    # cells are surface->center; compact core is the trailing block (center side)
    k0 = np.argmax(core_mask & (np.arange(len(Z)) >= 0))
    # boundary = outermost core cell index (smallest index in the core block)
    core_idx = np.where(core_mask)[0]
    k_b = core_idx.min()
    # mass of the compact core = CMass at the boundary cell (mass enclosed below)
    mcore_mjup = float(CMass[k_b])
    return dict(
        mcore_mjup=mcore_mjup,
        mcore_mearth=mcore_mjup * const.M_jup / const.mearth,
        cmass_boundary=mcore_mjup,
        k_core_boundary=int(k_b),
    )


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))  # repo root
    from utils import const
    print(f"{'id':>6} {'label':<26} {'Nrows':>5} {'Mtot/Mj':>8} "
          f"{'Mcore_E(data)':>13} {'exp':>5} {'J2_Int*1e6':>12}")
    for mid in ORDER:
        L = load_layers(mid)
        H = load_harmonics(mid)
        core = measure_core(mid)
        m = MODELS[mid]
        print(f"{mid:>6} {m['label']:<26} {len(L['Z']):>5} "
              f"{L['CMass'].max():>8.4f} {core['mcore_mearth']:>13.3f} "
              f"{m['mcore_exp']:>5.1f} {H[2]['int']*1e6:>12.4f}")
