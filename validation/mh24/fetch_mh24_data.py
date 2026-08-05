#!/usr/bin/env python3
"""
Download the Militzer & Hubbard (2024) Jupiter interior models from Zenodo.

MH24: "Study of Jupiter's Interior: Comparison of 2, 3, 4, 5, and 6 Layer Models"
      Icarus 411 (2024) 115955.
Data: Militzer (2024), "Models for Jupiter's Interior Structure", Zenodo,
      doi:10.5281/zenodo.10471389.

For each of the 9 models there are two machine-readable tables:
  * <id>_layers_mrt_with_header.txt    full radius/density/P/T/composition profile
  * <id>_harmonics_mrt_with_header.txt CMS-computed gravity harmonics J_n

Run from anywhere:
    /opt/anaconda3/envs/orchard_env/bin/python validation/mh24/fetch_mh24_data.py
"""
import os
import sys
import urllib.request

RECORD = "10471389"
BASE = f"https://zenodo.org/api/records/{RECORD}/files"
MODEL_IDS = ["10967", "10969", "10970", "10971", "10972", "10973",
             "10975", "10976", "10977"]
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")


def fetch(force=False):
    os.makedirs(DATA_DIR, exist_ok=True)
    n_ok = 0
    for mid in MODEL_IDS:
        for kind in ("layers", "harmonics"):
            fname = f"out{mid}_{kind}_mrt_with_header.txt"
            dst = os.path.join(DATA_DIR, fname)
            if os.path.exists(dst) and not force:
                print(f"  [skip] {fname} ({os.path.getsize(dst)} B)")
                n_ok += 1
                continue
            url = f"{BASE}/{fname}/content"
            print(f"  [get ] {fname} ...", end=" ", flush=True)
            urllib.request.urlretrieve(url, dst)
            print(f"{os.path.getsize(dst)} B")
            n_ok += 1
    print(f"Done: {n_ok}/{len(MODEL_IDS) * 2} files in {DATA_DIR}")


if __name__ == "__main__":
    fetch(force="--force" in sys.argv)
