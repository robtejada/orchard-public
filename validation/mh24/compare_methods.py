#!/usr/bin/env python3
"""
Compare J_2..J_8 from MH24's models computed by THREE ORCHARD gravity solvers
-- TOF4 (utils.TOF), TOF7 (utils.TOF7), and the optimized CMS (utils.cms_hubbard)
-- against MH24's published CMS interior harmonics (J_Int), with relative errors.

All three are fed MH24's OWN structure (full 2049 spheroids) so the comparison
isolates the gravity method:
  TOF4/TOF7 : mean (volumetric) radii rS, density = dCMass/dV (self-consistent)
  CMS       : equatorial radii rE, spheroid density (col 5), qrot

Run:
    /opt/anaconda3/envs/orchard_env/bin/python validation/mh24/compare_methods.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from utils import const                       # noqa: E402
from utils.TOF import get_moments as tof4      # noqa: E402
from utils.TOF7 import get_moments7 as tof7    # noqa: E402
from utils import cms_hubbard as C             # noqa: E402
import mh24_io                                 # noqa: E402

OMEGA = 2 * np.pi / 35730.0


def methods(mid):
    L = mh24_io.load_layers(mid)
    H = mh24_io.load_harmonics(mid)
    # ToF inputs: mean radii (center->surface), self-consistent density from CMass
    rS = L["rS"][::-1] * const.rj_eq
    m = L["CMass"][::-1] * const.M_jup
    rho_tof = np.gradient(m, rS) / (4 * np.pi * rS ** 2)
    j4 = tof4(rS, rho_tof, m, OMEGA)[0]
    j7 = tof7(rS, rho_tof, m, OMEGA)[0]
    # CMS inputs: equatorial radii (surface->center), spheroid density
    jc = C.get_moments_cms(L["rE"] * const.rj_eq, L["rho"], const.M_jup, OMEGA,
                           G=const.G)[0]
    cms_Jint = {n: H[n]["int"] * 1e6 for n in (2, 4, 6, 8)}
    val = {"TOF4": {2: j4[0], 4: j4[1], 6: j4[2], 8: j4[3]},
           "TOF7": {2: j7[0], 4: j7[1], 6: j7[2], 8: j7[3]},
           "CMS": {2: jc[0], 4: jc[1], 6: jc[2], 8: jc[3]}}
    return cms_Jint, val


def main():
    ref = "10969"
    cms_Jint, val = methods(ref)
    print("=" * 100)
    print(f"Reference 5-layer model (out{ref}): MH24 CMS J_Int vs ORCHARD TOF4 / TOF7 / "
          f"optimized CMS  (x1e6)")
    print("=" * 100)
    print(f"{'n':>3} | {'MH24 J_Int':>13} | {'TOF4':>13} {'err':>10} | "
          f"{'TOF7':>13} {'err':>10} | {'CMS':>13} {'err':>10}")
    print("-" * 100)
    for n in (2, 4, 6, 8):
        mh = cms_Jint[n]
        row = f"{n:>3} | {mh:>13.4f} |"
        for meth in ("TOF4", "TOF7", "CMS"):
            v = val[meth][n]
            row += f" {v:>13.4f} {(v - mh) / mh:>+10.1e} |"
        print(row)

    print("\n" + "=" * 100)
    print("All 9 models: relative error in J2 / J4 / J6 vs MH24 J_Int  (signed)")
    print("=" * 100)
    print(f"{'id':>6} {'model':<24} | {'TOF4 J2/J4/J6':>26} | {'TOF7 J2/J4/J6':>26} "
          f"| {'CMS J2/J4/J6':>26}")
    print("-" * 100)
    for mid in mh24_io.ORDER:
        cj, vv = methods(mid)
        cells = []
        for meth in ("TOF4", "TOF7", "CMS"):
            e = [ (vv[meth][n] - cj[n]) / cj[n] for n in (2, 4, 6) ]
            cells.append(f"{e[0]:>+7.0e}{e[1]:>+9.0e}{e[2]:>+9.0e}")
        print(f"{mid:>6} {mh24_io.MODELS[mid]['label']:<24} | {cells[0]:>26} | "
              f"{cells[1]:>26} | {cells[2]:>26}")
    print("\nAll fed MH24's own 2049-spheroid structure -> residual = gravity-method")
    print("truncation only. TOF4 (4th order) and TOF7 (7th order) are perturbative;")
    print("CMS is nonperturbative (matches MH24's method) and is discretization-limited.")


if __name__ == "__main__":
    main()
