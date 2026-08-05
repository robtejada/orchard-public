#!/usr/bin/env python3
"""
Print the J_n comparison tables: MH24 (CMS) vs ORCHARD (TOF7).
    /opt/anaconda3/envs/orchard_env/bin/python validation/mh24/make_j_table.py
"""
import os
import sys
import json

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mh24_io  # noqa: E402

RES = os.path.join(HERE, "results")


def main():
    s1 = {r["mid"]: r for r in json.load(open(os.path.join(RES, "step1_gravity.json")))}
    s2 = {r["mid"]: r for r in json.load(open(os.path.join(RES, "step2_hse_summary.json")))}

    # Juno (Durante+2020) for context, x1e6
    juno = {2: 14696.5063, 4: -586.6085, 6: 34.2007}

    print("=" * 100)
    print("ABSOLUTE J_n (x1e6):  MH24 CMS (J_Int, static interior)  vs  ORCHARD TOF7")
    print("  [A] TOF7 on MH24's OWN density (same structure -> pure gravity-solver comparison)")
    print("=" * 100)
    print(f"{'model':<24}{'n':>3}{'MH24 CMS J_Int':>16}{'ORCHARD TOF7':>15}"
          f"{'diff':>12}{'rel':>11}")
    for mid in mh24_io.ORDER:
        lab = mh24_io.MODELS[mid]["label"]
        H = mh24_io.load_harmonics(mid)
        r = s1[mid]
        for k, n in enumerate((2, 4, 6, 8)):
            cms = H[n]["int"] * 1e6
            tof = r["tof7"][str(n)]
            print(f"{lab if k == 0 else '':<24}{n:>3}{cms:>16.4f}{tof:>15.4f}"
                  f"{tof-cms:>12.2e}{(tof-cms)/cms:>+11.2e}")
        print("-" * 81)

    print("\n" + "=" * 100)
    print("  [B] END-TO-END: ORCHARD EOS(cd+aqua) + hydrostatic.py + TOF7 (same Y,Z as MH24)")
    print("=" * 100)
    print(f"{'model':<24}{'n':>3}{'MH24 CMS J_Int':>16}{'ORCHARD e2e':>15}"
          f"{'diff':>12}{'rel':>11}")
    for mid in mh24_io.ORDER:
        lab = mh24_io.MODELS[mid]["label"]
        r = s2[mid]
        for k, n in enumerate((2, 4, 6, 8)):
            cms = r["cms"][str(n)]
            tof = r["tof7"][str(n)]
            print(f"{lab if k == 0 else '':<24}{n:>3}{cms:>16.4f}{tof:>15.4f}"
                  f"{tof-cms:>12.2e}{(tof-cms)/cms:>+11.2e}")
        print("-" * 81)

    print("\nFor reference, Juno (Durante+2020, x1e6, tide-removed): "
          f"J2={juno[2]}, J4={juno[4]}, J6={juno[6]}")
    print("MH24's J_Total (interior+winds) matches Juno; we compare TOF7 to their static J_Int.")


if __name__ == "__main__":
    main()
