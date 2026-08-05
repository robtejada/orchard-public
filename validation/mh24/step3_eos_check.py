#!/usr/bin/env python3
"""
Step 3 - EOS spot-check: ORCHARD cd+aqua vs MH24's tabulated rho(P).

MH24 uses the Militzer & Hubbard (2013) ab-initio EOS; ORCHARD's default
`cd` (Chabrier & Debras 2021) H-He EOS is built on the same ab-initio data,
so densities should agree closely *provided the same Y and Z are used*. MH24
publishes a 'rho(P)' column (Note 4: "Combine with pressure to compare the
equation of state") expressly for this check.

We compute rho_ORCHARD = 10**get_logrho_pt(log10 P[dyn], log10 T, Y, Z, frock=0)
with the default mixtures(hhe_eos='cd', z_eos='aqua') and compare to MH24's
rho(P) along each profile. Expect ~%-level agreement in the H-He envelope +
dilute core (Z<1). Compact-core cells (Z~1, pure rock/ice) are reported
separately - aqua-water is not the right EOS there (those are handled by the
silicate mantle EOS in Step 2).

Run:
    ORCHARD_LIBRARY_MODE=1 /opt/anaconda3/envs/orchard_env/bin/python \
        validation/mh24/step3_eos_check.py
"""
import os
import sys
import json

os.environ.setdefault("ORCHARD_LIBRARY_MODE", "1")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from eos import eos_class      # noqa: E402
import mh24_io                 # noqa: E402

GPA_TO_DYN = 1e10              # 1 GPa = 1e10 dyn/cm^2


def main():
    mix = eos_class.mixtures(hhe_eos="cd", z_eos="aqua", hg=True, y_prime=False)
    print("=" * 92)
    print("STEP 3  EOS density spot-check: ORCHARD cd+aqua rho(P,T,Y,Z) vs "
          "MH24 rho(P)")
    print("=" * 92)
    print(f"{'id':>6} {'label':<26} {'envelope (Z<1)':>22} {'compact core (Z>=.999)':>24}")
    print(f"{'':>6} {'':<26} {'med|res|   max|res|':>22} {'med|res|   N':>24}")
    results = []
    for mid in mh24_io.ORDER:
        L = mh24_io.load_layers(mid)
        P, T, Y, Z = L["P"], L["T"], L["Y"], L["Z"]
        rho_mh = L["rho_P"]                       # col 6, EOS density
        logP = np.log10(P * GPA_TO_DYN)
        logT = np.log10(T)
        # guard against EOS table range: compute, mark non-finite
        with np.errstate(all="ignore"):
            rho_or = 10 ** mix.get_logrho_pt(logP, logT, Y, Z, 0.0)
        res = (rho_or - rho_mh) / rho_mh
        env = (Z < 0.999) & np.isfinite(res)
        core = (Z >= 0.999) & np.isfinite(res)
        env_med = np.median(np.abs(res[env])) if env.any() else np.nan
        env_max = np.max(np.abs(res[env])) if env.any() else np.nan
        core_med = np.median(np.abs(res[core])) if core.any() else np.nan
        n_core = int(core.sum())
        n_bad = int((~np.isfinite(res)).sum())
        print(f"{mid:>6} {mh24_io.MODELS[mid]['label']:<26} "
              f"{env_med*100:>9.3f}% {env_max*100:>9.3f}% "
              f"{core_med*100:>11.3f}% {n_core:>6d}"
              + (f"   ({n_bad} non-finite)" if n_bad else ""))
        results.append(dict(mid=mid, env_med=float(env_med), env_max=float(env_max),
                            core_med=float(core_med), n_core=n_core, n_bad=n_bad))
    out = os.path.join(HERE, "results", "step3_eos.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nSaved -> {out}")
    print("\nNote: residual blends EOS difference AND Z-composition difference")
    print("(ORCHARD aqua = pure water; MH24 Z = H2O:CH4:NH3 ice + rock).")


if __name__ == "__main__":
    main()
