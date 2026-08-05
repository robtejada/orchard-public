# Validating ORCHARD's gravity harmonics against Militzer & Hubbard (2024)

This folder reproduces the Jupiter gravity-harmonic results of Militzer &
Hubbard (2024), *Study of Jupiter's interior: Comparison of 2, 3, 4, 5, and 6
layer models*, Icarus 411, 115955 ("MH24"), using ORCHARD's two gravity
backends:

- **ToF7** (`utils/TOF7.py`) — 7th-order Theory of Figures, the fast
  perturbative solver used inside the evolution loop (`tof_order = 7`);
- **CMS** (`utils/cms_hubbard.py`) — a clean-room implementation of the
  nonperturbative Concentric Maclaurin Spheroid method (Hubbard 2013) that
  MH24 themselves used, available as `gravity_method = cms`.

The MH24 model tables (9 Jupiter models × {layers, harmonics}) are included
in `data/` and come from Militzer (2024), *Models for Jupiter's Interior
Structure*, Zenodo, doi:10.5281/zenodo.10471389 — cite both the paper and the
dataset if you use them. `fetch_mh24_data.py` re-downloads them from the
source if you prefer.

## Headline results (reference outputs in `results/`)

1. **Gravity solvers are exact.** Feeding MH24's own density profiles to
   ORCHARD's ToF7 reproduces their published CMS `J_Int` to ~1e-5 (J2,
   coreless models; ~1e-4 with a compact core — genuine perturbative
   truncation). ORCHARD's own CMS matches to the spheroid-count
   discretization limit. (`step1_gravity_unittest.py`,
   `cms_validate_mh24.py`)
2. **EOS agreement.** ORCHARD's `cd + aqua` EOS matches the MH24
   (Militzer & Hubbard 2013) density to ~0.5% (median) over the envelope.
   (`step3_eos_check.py`)
3. **End-to-end.** Rebuilding each MH24 model through ORCHARD's own EOS +
   Henyey solver + ToF7 (same Y, Z) lands within ~0.8% (J2) to ~2% (J6) of
   the published values, with the residual dominated by the EOS
   difference in (2), not the gravity method. (`run_one_model.py`,
   `run_all.py`)

## Reproducing

```bash
conda activate orchard_env
python validation/mh24/step1_gravity_unittest.py   # ToF7 vs published CMS, seconds
python validation/mh24/cms_validate_mh24.py        # our CMS vs published CMS
python validation/mh24/run_all.py                  # end-to-end, ~minutes/model
```

See `tutorials/tutorial_gravity_harmonics.ipynb` for a guided walkthrough of
all of this, including how to use both gravity backends on your own models.

## Data-consistency gotcha (important if you reuse `data/`)

MH24's `rho` column carries their CMS discretization (~0.2% high when
spherically integrated). For Theory-of-Figures input, treat the cumulative
mass `CMass` as authoritative and derive the density as rho = dM/dV
(`step1_gravity_unittest.py:density_from_cmass`); pairing the raw `rho`
column with `CMass` produces a spurious ~0.5% J2 offset in *any* ToF solver.
