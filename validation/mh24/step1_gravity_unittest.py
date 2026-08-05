#!/usr/bin/env python3
"""
Step 1 - Gravity-solver unit test: ORCHARD ToF7 vs MH24 CMS, EOS-independent.

For each MH24 model we feed MH24's OWN density profile into ORCHARD's ToF7
(utils.TOF7.get_moments7) and compare the resulting J_2..J_8 against MH24's
published CMS interior harmonics (J_Int). The ONLY difference is then the
gravity method (perturbative ToF7 vs nonperturbative CMS): this isolates the
gravity solver with zero EOS dependence.

IMPORTANT CONSISTENCY POINT (root-caused during development):
ToF requires the density and enclosed-mass arrays to be mutually consistent,
i.e. m(l) = integral 4 pi l'^2 rho dl'. MH24's 'rho' column (Note 3) carries a
~0.2% CMS *discretization error*: spherically integrating it gives 1.0021 Mjup,
not the accurate 'CMass' = 1.0 Mjup. Feeding ToF that inconsistent (rho, CMass)
pair produces a spurious ~0.47% J2 offset (ToF4 and ToF7 agree on it -> it is an
input bug, not a solver bug). The fix: treat CMass as authoritative and DERIVE
the ToF-ready density from it, rho = dM/dV = (1/4 pi l^2) dCMass/dl. With that,
ToF7 reproduces CMS to ~1e-5 on J2.

Inputs to get_moments7 (center -> surface):
    l   = rS * R_eq_jup       (cm)  volumetric radius (MH24 normalizes rS to 71492 km)
    m   = CMass * M_jup       (g)   authoritative cumulative mass
    rho = dM/dV from CMass    (g/cc) density consistent with m (NOT col 5 directly)
    omega = 2 pi / 35730 s          Jupiter System III (same as ORCHARD default)

Built-in cross-check: ToF7's returned R_eq must come out ~71492 km.

Run:
    /opt/anaconda3/envs/orchard_env/bin/python validation/mh24/step1_gravity_unittest.py
"""
import os
import sys
import json

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from utils import const                       # noqa: E402
from utils.TOF7 import get_moments7           # noqa: E402
import mh24_io                                # noqa: E402

PERIOD_S = 35730.0                            # Jupiter System III (ORCHARD default)
OMEGA = 2.0 * np.pi / PERIOD_S
R_EQ_JUP = const.rj_eq                        # 7.1492e9 cm = 71492 km

# Juno 1-sigma (x1e6), Durante et al. 2020 (J2 tide-removed, as MH24 match it)
JUNO_SIGMA = {2: 0.0006, 4: 0.0024, 6: 0.0067}


def density_from_cmass(l, m):
    """ToF-consistent density from the cumulative-mass profile:
    rho(l) = (1/4 pi l^2) dM/dl, so that integral 4 pi l^2 rho dl == m exactly."""
    return np.gradient(m, l) / (4.0 * np.pi * l ** 2)


def run_model(mid):
    L = mh24_io.load_layers(mid)
    H = mh24_io.load_harmonics(mid)
    # reverse surface->center  ==>  center->surface for ToF7
    l = L["rS"][::-1] * R_EQ_JUP
    m = L["CMass"][::-1] * const.M_jup        # authoritative cumulative mass
    rho = density_from_cmass(l, m)            # consistent density (see module docstring)
    j2n, _ = get_moments7(l, rho, m, OMEGA)
    J = {2: j2n[0], 4: j2n[1], 6: j2n[2], 8: j2n[3]}     # x1e6
    R_eq, R_pol, oblat = j2n[5], j2n[6], j2n[4]
    cms = {n: H[n]["int"] * 1e6 for n in (2, 4, 6, 8)}    # J_Int x1e6
    # diagnostic: naive pairing of col-5 rho with CMass (documents the pitfall)
    rho_naive = L["rho"][::-1]
    j_naive, _ = get_moments7(l, rho_naive, m, OMEGA)
    return dict(mid=mid, label=mh24_io.MODELS[mid]["label"],
                tof7=J, cms=cms, R_eq=R_eq, R_pol=R_pol, oblat=oblat,
                j2_naive=j_naive[0])


def main():
    results = []
    print("=" * 96)
    print("STEP 1  ToF7 (ORCHARD) vs CMS (MH24)  on MH24's own density "
          "- gravity solver only, no EOS")
    print("=" * 96)
    for mid in mh24_io.ORDER:
        r = run_model(mid)
        results.append(r)
        print(f"\n{mid}  {r['label']}")
        print(f"   R_eq(ToF7) = {r['R_eq']/1e5:9.2f} km   "
              f"(target 71492.00 km, diff {r['R_eq']/1e5 - 71492:.2f} km)   "
              f"oblat = {r['oblat']:.5f}")
        print(f"   [naive col-5 rho pairing would give J2={r['j2_naive']:.4f} "
              f"-> {(r['j2_naive']/r['cms'][2]-1)*100:+.3f}% (inconsistent input)]")
        print(f"   {'n':>3} {'ToF7 x1e6':>16} {'CMS x1e6':>16} "
              f"{'abs diff':>12} {'rel':>11} {'|diff|/Juno-sig':>16}")
        for n in (2, 4, 6, 8):
            t, c = r["tof7"][n], r["cms"][n]
            d = t - c
            rel = d / c if c != 0 else float("nan")
            sig = JUNO_SIGMA.get(n)
            sigstr = f"{abs(d)/sig:>16.2f}" if sig else f"{'(n/a)':>16}"
            print(f"   {n:>3} {t:>16.5f} {c:>16.5f} {d:>12.2e} "
                  f"{rel:>11.2e} {sigstr}")

    out = os.path.join(HERE, "results", "step1_gravity.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
