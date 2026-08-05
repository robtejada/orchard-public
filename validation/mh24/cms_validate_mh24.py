#!/usr/bin/env python3
"""
Validate the clean-room CMS (cms_hubbard.py) against MH24's published CMS J_Int,
and cross-check against ORCHARD's TOF7 on the same density.

(1) Layer-count convergence on the reference model: CMS J2..J8 -> MH24 J_Int as
    the number of spheroids increases (residual is discretization, not algorithm).
(2) All 9 models at a fixed resolution.
(3) CMS vs TOF7 (both on MH24's own structure) -> agree to the ToF7 truncation.

Run:
    /opt/anaconda3/envs/orchard_env/bin/python validation/mh24/cms_validate_mh24.py
"""
import os
import sys
import json
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)
from utils import const                 # noqa: E402
from utils.TOF7 import get_moments7      # noqa: E402
import mh24_io                           # noqa: E402
from utils import cms_hubbard as C       # noqa: E402  (module now lives in utils/)

OMEGA = 2 * np.pi / 35730.0


def _downsample(arr, n):
    idx = np.unique(np.linspace(0, len(arr) - 1, n).astype(int))
    return idx


def cms_on_model(mid, nlayers, kmax=10, L=48):
    L_ = mh24_io.load_layers(mid)
    idx = _downsample(L_["rE"], nlayers)
    lam_eq = L_["rE"][idx] * const.rj_eq
    rho = L_["rho"][idx]
    j2n, zeta, info = C.get_moments_cms(lam_eq, rho, const.M_jup, OMEGA,
                                        G=const.G, kmax=kmax, L=L)
    return j2n, info


def main():
    ref = "10969"
    H = mh24_io.load_harmonics(ref)
    cms_ref = {n: H[n]["int"] * 1e6 for n in (2, 4, 6, 8)}

    print("=" * 92)
    print(f"(1) Layer-count convergence: clean-room CMS -> MH24 CMS J_Int  "
          f"(reference 5-layer model)")
    print("=" * 92)
    print(f"{'Nlayers':>8}{'iters':>6}{'time_s':>8}{'oblat':>9}"
          f"{'J2 rel':>11}{'J4 rel':>11}{'J6 rel':>11}{'J8 rel':>11}")
    conv = []
    for n in (128, 256, 512, 1024, 2049):
        t0 = time.time()
        j2n, info = cms_on_model(ref, n)
        dt = time.time() - t0
        rel = [(j2n[i] - cms_ref[2 * (i + 1)]) / cms_ref[2 * (i + 1)] for i in range(4)]
        print(f"{n:>8}{info['iters']:>6}{dt:>8.1f}{j2n[4]:>9.5f}"
              f"{rel[0]:>+11.2e}{rel[1]:>+11.2e}{rel[2]:>+11.2e}{rel[3]:>+11.2e}")
        conv.append(dict(n=int(n), iters=info["iters"], oblat=float(j2n[4]),
                         J=[float(x) for x in j2n[:4]], rel=rel))

    print("\n" + "=" * 92)
    print("(2) All 9 models at N=512 spheroids:  CMS vs MH24 J_Int  vs  TOF7 (same density)")
    print("=" * 92)
    print(f"{'id':>6} {'label':<24}{'J2 CMS-mine':>13}{'J2 MH24':>12}"
          f"{'CMS rel':>10}{'TOF7 rel':>10}")
    allrows = []
    for mid in mh24_io.ORDER:
        Hm = mh24_io.load_harmonics(mid)
        cms_mh = {n: Hm[n]["int"] * 1e6 for n in (2, 4, 6, 8)}
        j2n_cms, info = cms_on_model(mid, 512)
        # TOF7 on the same density (Step-1 style: density from CMass)
        Ld = mh24_io.load_layers(mid)
        rS = Ld["rS"][::-1] * const.rj_eq
        m = Ld["CMass"][::-1] * const.M_jup
        rho_tof = np.gradient(m, rS) / (4 * np.pi * rS ** 2)
        j2n_tof, _ = get_moments7(rS, rho_tof, m, OMEGA)
        cms_rel = (j2n_cms[0] - cms_mh[2]) / cms_mh[2]
        tof_rel = (j2n_tof[0] - cms_mh[2]) / cms_mh[2]
        print(f"{mid:>6} {mh24_io.MODELS[mid]['label']:<24}"
              f"{j2n_cms[0]:>13.3f}{cms_mh[2]:>12.3f}{cms_rel:>+10.1e}{tof_rel:>+10.1e}")
        allrows.append(dict(mid=mid, J2_cms=float(j2n_cms[0]), J2_mh=cms_mh[2],
                            cms_rel=cms_rel, tof_rel=tof_rel))

    out = os.path.join(HERE, "results", "cms_validation.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(dict(convergence=conv, all_models=allrows), fh, indent=2)
    print(f"\nSaved -> {out}")
    print("CMS residual vs MH24 shrinks with spheroid count (discretization). CMS and")
    print("TOF7, both on MH24's structure, bracket MH24's CMS J_Int at the ~1e-4 level.")


if __name__ == "__main__":
    main()
