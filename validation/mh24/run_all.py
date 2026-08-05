#!/usr/bin/env python3
"""
Step 2 master - reproduce all 9 MH24 models through ORCHARD HSE+ToF7.

Runs run_one_model.py as a fresh SUBPROCESS per model (initial.py reads core
params at import time), collects results/hse_<id>.json, and prints an aggregate
J_n comparison table (ORCHARD HSE+ToF7 vs MH24 CMS J_Int).

Run:
    /opt/anaconda3/envs/orchard_env/bin/python validation/mh24/run_all.py [mantle_comp]
"""
import os
import sys
import json
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import mh24_io  # noqa: E402

PY = sys.executable
MANTLE = sys.argv[1] if len(sys.argv) > 1 else "mgsio3"
JUNO_SIGMA = {2: 0.0006, 4: 0.0024, 6: 0.0067}   # x1e6, Durante+2020


def main():
    env = dict(os.environ, ORCHARD_LIBRARY_MODE="1")
    rows = []
    for mid in mh24_io.ORDER:
        print(f"--- running {mid} ({mh24_io.MODELS[mid]['label']}) ---", flush=True)
        r = subprocess.run([PY, os.path.join(HERE, "run_one_model.py"), mid, MANTLE],
                           env=env, capture_output=True, text=True)
        jpath = os.path.join(HERE, "results", f"hse_{mid}.json")
        if r.returncode != 0 or not os.path.exists(jpath):
            print(f"  FAILED (rc={r.returncode}):\n{r.stderr[-1500:]}")
            continue
        rows.append(json.load(open(jpath)))
        last = rows[-1]
        print(f"  ok: rot_iters={last['n_rot_iter']} R_eq={last['R_eq_km']:.1f}km "
              f"J2 rel={ (last['tof7']['2']-last['cms']['2'])/last['cms']['2']:+.2e}")

    # ---- aggregate table ----
    print("\n" + "=" * 104)
    print("STEP 2  ORCHARD HSE(cd+aqua, mgsio3 core) + ToF7  vs  MH24 CMS J_Int "
          "(end-to-end, same Y,Z)")
    print("=" * 104)
    hdr = (f"{'id':>6} {'label':<24} {'Mcore':>5} {'R_eq km':>9} {'dR%':>6}  "
           f"{'J2 rel':>9} {'J4 rel':>9} {'J6 rel':>9} {'J8 rel':>9}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        c = r["cms"]; t = r["tof7"]
        def rel(n): return (t[str(n)] - c[str(n)]) / c[str(n)]
        dR = (r["R_eq_km"] - 71492.0) / 71492.0 * 100
        print(f"{r['mid']:>6} {r['label']:<24} {r['mcore_Me']:>5.1f} "
              f"{r['R_eq_km']:>9.1f} {dR:>6.2f}  "
              f"{rel(2):>+9.2e} {rel(4):>+9.2e} {rel(6):>+9.2e} {rel(8):>+9.2e}")

    out = os.path.join(HERE, "results", "step2_hse_summary.json")
    with open(out, "w") as fh:
        json.dump(rows, fh, indent=2)
    print(f"\nSaved -> {out}")
    print("\nJ_n rel = (ORCHARD - CMS)/CMS. Residual is dominated by the cd+aqua")
    print("vs Militzer&Hubbard-2013 EOS difference (Step 3), propagated through")
    print("the self-consistent rotating structure; ToF7 itself matches CMS to")
    print("~1e-5 on the same density (Step 1).")


if __name__ == "__main__":
    main()
