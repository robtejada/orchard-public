# ORCHARD: A General Planetary Evolution Code [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19829061.svg)](https://doi.org/10.5281/zenodo.19829061)

[![ORCHARD logo](orchard_logo.png)](orchard_logo.png)

ORCHARD (Tejada Arevalo et al. 2026b) is a public, general-purpose planetary interior structure and evolution code derived from APPLE ([Sur et al. 2024](https://iopscience.iop.org/article/10.3847/1538-4357/ad57c3)). It models the thermal and compositional evolution of planets over Gyr timescales, solving for hydrostatic structure via Henyey relaxation and for coupled thermal/compositional transport with helium rain modeled with diffusion-advection. While the gas-giant evolution part of the code is inherited from APPLE, the rocky-planet evolution code is inspired by the CMAPPER evolution code ([Zhang et al. 2022](https://iopscience.iop.org/article/10.3847/1538-4357/ac8e65)). CMAPPER is found [here](https://github.com/zhangjis/CMAPPER_rock).

**Current release: v0.2.0** — the first full public release. See [What's new in v0.2.0](#whats-new-in-v020).

## What's new in v0.2.0

v0.2.0 is the first fully open release of ORCHARD. The previous version (v0.1.0, the version submitted with the ORCHARD paper) was deposited on Zenodo with restricted file access during peer review. Changes since v0.1.0:

- **Metal (Z) rain**: a heavy-element miscibility channel paralleling helium rain — selectable miscibility curves, independent temperature/pressure offsets, supersaturation-responsive rain-zone diffusion, and a deep-pressure activation cap (`zmisc_p_max`).
- **Gravity-harmonics improvements**: corrected moment-of-inertia integral in TOF4.
- **Rock-fraction-aware EOS support ([EOS v2.0](https://doi.org/10.5281/zenodo.21812109))**: aquarock water-rock core EOS, continuous water/rock core composition (`f_rock_core`), rock-mixture envelope options; fixed core EOS evaluation for rock fractions other than 0.5. THIS FEATURE IS ONLY AVAILABLE THROUGH THE ZENODO EOS MODULE. It's too large to include in the GitHub repository.
- **More reliable initial models**: the RK4 initial-structure builder handles cored gas giants; fixed an RK4 substep density bug; fixed the rotating-initialization breakup check.
- **Energy-accounting overhaul**: consistent lost-energy ledger, rotational kinetic energy tracked, optional midpoint gravity centering in the hydrostatic solver, optional second-order time-centered heat metering, plus an energy-conservation methods writeup.
- **Solver robustness**: adaptive Henyey under-relaxation, density- and temperature-inversion guards with optional post-step repair, a timestep stall guard (`dt_abort_myr`), fixed early-age high-resolution timestepping, fixed retry accounting in the adaptive timestepper.
- **Documentation and support**: an FAQ, Windows install notes, a fully audited `parameter_descriptions.md`, published methods writeups, and an experimental documentation assistant (`docs_bot`) with a GitHub Actions issue responder.
- **Two new tutorials** — a getting-started on-ramp (`tutorial_getting_started.ipynb`) and static structure models without evolution (`tutorial_static_structures.ipynb`) — bringing the tutorial set to eight, with expanded struct-profile and EOS-comparison material and exercises throughout.

## Getting Started

### Prerequisites

Install [Conda](https://docs.conda.io/) (Miniconda or Anaconda).

**Windows users:** we recommend the Zenodo install path below (the GitHub developer path uses bash-only commands). See [Windows notes](#windows-notes) at the end of this section.

### Install from Zenodo (recommended)

We highly recommend installing `orchard` v0.2.0 from the Zenodo release archive from [Zenodo](https://doi.org/10.5281/zenodo.19829061). This version is the same as the one released here under orhcard-public. Zenodo is recommended because the equation of state (EOS) submodule is quite large (~30 GB, compressed), so cloning through GitHub consumes Git LFS space, which is limited. 

xtract orchard and set up the environment likeso:

```bash
unzip orchard-v0.2.0.zip
cd orchard-v0.2.0
conda env create -f environment.yaml
conda activate orchard_env
python setup_eos.py
```

`setup_eos.py` downloads the ~30 GB of EOS tables from the [EOS Zenodo record](https://doi.org/10.5281/zenodo.10659248). That link is the permanent *concept* DOI. `setup_eos.py` will download the newest published version at download time, currently **EOS v2.0** ([10.5281/zenodo.21812109](https://doi.org/10.5281/zenodo.21812109)). No additional manual downloads are required, and the command does not change when a new EOS version is deposited.

### Install from GitHub (developers; skip if installed from Zenodo)

To clone the source repository directly instead:

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone --recurse-submodules https://github.com/robtejada/orchard-public.git
cd orchard-public
conda env create -f environment.yaml
conda activate orchard_env
python setup_eos.py
```

`GIT_LFS_SKIP_SMUDGE=1` prevents git from pulling the ~27 GB of git-tracked EOS tables through LFS at clone time; `setup_eos.py` then fetches the complete ~30 GB set (including tables too large for LFS) from Zenodo.

#### Windows notes

Both install paths work on Windows 10 (build 17063+) and Windows 11, with two small adjustments:

1. **Conda activation.** In a stock PowerShell or `cmd.exe`, the `conda activate` command works only after running `conda init` once for your shell. The simplest option is to use **Anaconda Prompt** (installed alongside Miniconda/Anaconda), where `conda activate` works out of the box. Alternatively, run once per shell:
   - PowerShell: `conda init powershell`, then restart the shell
   - `cmd.exe`: `conda init cmd.exe`, then restart the shell

2. **Install to a short root path.** The EOS data unpacks into nested directories (~30 GB extracted). Windows imposes a 260-character path limit by default, which can cause silent extraction failures for deeply nested files. To avoid this, install ORCHARD to a short root such as `C:\orchard\` rather than under directories like `C:\Users\<name>\OneDrive\…\Code\`. Alternatively, enable long-path support (`LongPathsEnabled = 1` in the Windows registry) before running `setup_eos.py`.

The **GitHub developer install** uses `GIT_LFS_SKIP_SMUDGE=1 git clone …`, which is bash-only syntax. Windows users wanting a clone-based install should run the command in **Git Bash** (bundled with [Git for Windows](https://git-scm.com/download/win)), or substitute the cmd/PowerShell equivalent:

- `cmd.exe`:
  ```
  set GIT_LFS_SKIP_SMUDGE=1 && git clone --recurse-submodules https://github.com/robtejada/orchard-public.git
  ```
- PowerShell:
  ```
  $env:GIT_LFS_SKIP_SMUDGE=1; git clone --recurse-submodules https://github.com/robtejada/orchard-public.git
  ```

### Verify the install

```bash
python evolution.py
```

The run should finish in about a minute with `Evolution complete. Final data saved to: models/parameters_default`.

## Running ORCHARD

### 1. Quick Start by Planet Type

Each command below runs a pre-configured example. Start here and adjust parameters as needed.

**Gas giant (1 $M_J$ Jupiter):**
```bash
python evolution.py --config parameter_examples/parameter_user.ini
```

**Sub-Neptune (10 $M_\oplus$):**
```bash
python evolution.py --config parameter_examples/parameter_user_sub_neptune_10Mearth_example.ini
```

**Super-Earth (3 $M_\oplus$, bare):**
```bash
python evolution.py --config parameter_examples/parameter_user_super_earth_3Mearth_bare_example.ini
```

Upon starting a run, ORCHARD prints an initialization summary:

```
======================================================================
ORCHARD — Initial Model Summary
======================================================================
  Planet type           : Jupiter
  Total mass            : 317.907 M_Earth (1.00025 M_Jup)
  Mantle + core mass    : 0.000 M_Earth (iron core: 0.000 M_Earth)
  Total Z mass          : 16.213 M_Earth (envelope: 16.213, mantle+core: 0.000)
  Surface metallicity   : 3.107 x solar (Chen+2023)
  Initial radius        : 1.758 R_Jup
  T_eff / T_int         : 457.42 / 453.21 K
  Energy budget (U-E_g) : -8.5543e+42 erg
----------------------------------------------------------------------
  H-He EOS              : cd
  Z EOS                 : aqua
  Boundary condition    : Chen+2023
======================================================================
```

### 2. Configure a Run

Start from any example in `parameter_examples/` and edit the sections you need. The most commonly changed values are:

| Section | Key parameters |
|---------|---------------|
| `[general]` | Grid size (`N`), final age, save interval, tolerances |
| `[initial]` | Planetary mass (`M_Mearth`), initial entropy/composition |
| `[boundary_condition]` | Planet type, atmosphere model (`bc_atm`), irradiation, clouds |
| `[equation_of_state]` | H/He EOS (`hhe_eos`), heavy-element EOS (`z_eos`) |
| `[diffusion]` / `[transport]` | Miscibility, diffusion, convection, conductivity |
| `[core]` | Core mass, mantle/core composition and conductivity |

For a full description of every parameter, see `parameter_descriptions.md`.

#### Example configurations

| Planet type | Example config | Tutorial |
|------------|---------------|----------|
| 1 $M_J$ Jupiter | `parameter_examples/parameter_user.ini` | `tutorial_model_plotting.ipynb` |
| Jupiter (He rain) | `parameter_examples/parameter_user_jup_herain.ini` | `tutorial_solarsystem_planets.ipynb` |
| Saturn (He rain) | `parameter_examples/parameter_user_saturn_herain.ini` | `tutorial_solarsystem_planets.ipynb` |
| Super-Jupiter | `parameter_examples/parameter_user_super_jupiter.ini` | `tutorial_superjupiters.ipynb` |
| 10 $M_\oplus$ sub-Neptune | `parameter_examples/parameter_user_sub_neptune_10Mearth_example.ini` | `tutorial_subneptunes_superearths.ipynb` |
| 3 $M_\oplus$ super-Earth | `parameter_examples/parameter_user_super_earth_3Mearth_bare_example.ini` | `tutorial_subneptunes_superearths.ipynb` |

#### Setting planet mass

Set `M_Mearth` in the `[initial]` section. Units are Earth masses.

For gas giants, you can alternatively set `M_MJup` (Jupiter masses) if the mass exceeds 1 $M_J$.

For super-Earths, set `M_Mearth` equal to `mass_core` in the `[core]` section. For example, for a 3 $M_\oplus$ bare rocky planet: `M_Mearth = 3` and `mass_core = 3`.

#### Equation of state choices

**H-He EOS** (`hhe_eos` in `[equation_of_state]`):
- `cd` — Chabrier & Debras 2021 (recommended default)
- `cms` — Chabrier et al. 2019, with Howard & Guillot 2023 corrections
- `scvh` — Saumon et al. 1995 (legacy, largely superseded)

**Heavy-element EOS** (`z_eos`):
- `aqua` — AQUA water tables (Haldemann et al. 2020)
- `ice_mixture` — CH4 + NH3 + H2O ices (4:1:7 ratio)
- `ppv` — Post-perovskite

**Mantle composition** (`mantle_comp` in `[core]`):
- `mg2sio4` — Mg2SiO4 (Stewart et al. 2020, default)
- `mgsio3` — MgSiO3 (Luo & Deng 2023)
- `h2o` — AQUA tables (Haldemann et al. 2020)

### 3. Plotting

You can produce live plots during the evolution:
```bash
python evolution.py --config parameter_examples/parameter_user.ini -plot_end     # movie at end
python evolution.py --config parameter_examples/parameter_user.ini -plot_active  # live plotting
```

If you ran a model without `-plot_end` or `-plot_active`, you can generate plots from saved data:

**Single snapshot** at the last saved timestep:
```bash
python plots.py --config parameter_examples/parameter_user.ini
```

**Snapshot at a specific age** (in Gyr):
```bash
python plots.py --config parameter_examples/parameter_user.ini --age 4.56
```

**Compare multiple ages** side-by-side:
```bash
python plots.py --config parameter_examples/parameter_user.ini --mode age --ages 1.0 4.56 10.0
```

**Generate a movie** from saved snapshots:
```bash
python plots.py --config parameter_examples/parameter_user.ini --mode movie --dt_gyr 0.05
```

**Compare two models** on the same plot:
```bash
python plots.py --config parameter_examples/param_A.ini parameter_examples/param_B.ini \
                --labels "Model A" "Model B"
```

Run `python plots.py --help` for the full list of options, including panel layout, helium rain overlays, and radius-based profiles.

### 4. Interactive CLI

When using rock mixture EOS tables (`rock_mixtures = True`), the initial EOS loading can take several minutes due to table sizes (~15 GB). The interactive CLI pre-loads these tables so subsequent runs start immediately:

```bash
python orchard_cli.py
```

Once loaded, enter a parameter file path to run:
```
(orchard_env) user@computer orchard % python orchard_cli.py

Name of params file (with path) [type "help" for usage]: parameter_examples/parameter_user.ini
```

The CLI also supports hot-reloading modules with `reload <module>` without restarting.

### Output locations

- `models/<config-name>/`: Saved model data (text files per timestep, or a single `evolution.h5` if `save_hdf5 = True`)
- `logs/<config-name>.log`: Log file with the output displayed during the run
- `plots/<config-name>/`: Default plot output folder

## Tutorials

The tutorial notebooks live at the repository root (so they can be run in place without path juggling). We recommend working through them in this order:

1. `tutorial_getting_started.ipynb` — Start here: first run, console output, loading results
2. `tutorial_model_plotting.ipynb` — Plotting and visualization
3. `tutorial_eos.ipynb` — Equation of state exploration
4. `tutorial_static_structures.ipynb` — Static structure models (no evolution)
5. `tutorial_inhomogeneous_evolution.ipynb` — Non-uniform composition profiles
6. `tutorial_solarsystem_planets.ipynb` — Jupiter, Saturn, Uranus, Neptune
7. `tutorial_subneptunes_superearths.ipynb` — Sub-Neptune and super-Earth workflows
8. `tutorial_superjupiters.ipynb` — Exoplanet super-Jupiter modeling

## Repository Layout

Core solver modules:

- `evolution.py` — Top-level driver: config, initialization, adaptive timestepping, output
- `initial.py` — Builds the starting model: mass mesh, EOS initialization, S/Y/Z/f_rock profiles
- `hydrostatic.py` — Henyey relaxation solver for hydrostatic equilibrium
- `transport.py` — Coupled Newton-Raphson solver for thermal and compositional transport
- `atm_bc.py` — Atmosphere boundary condition classes (Chen+23, Chen+26, Fortney+07/+11, gray, bare, etc.)

Supporting directories:

- `eos/` — Equation-of-state module (git submodule)
- `misc/` — Miscibility models and tables (H-He, water/rock)
- `conductivities/` — Thermal conductivity models
- `atmospheres/` — Atmosphere boundary condition tables
- `opacities/` — Opacity tables for transport and transit-radius utilities
- `utils/` — Shared utilities: constants, config parsing, TOF, diffusion coefficients, profiles, adaptive regridding
- `parameter_examples/` — Example run configurations
- `tutorial_*.ipynb` — Jupyter notebook tutorials (repository root)

## Support and License

- License: see `LICENSE`
- For questions or issues, contact Roberto (Rob) Tejada Arevalo (arevalo@princeton.edu) or Ankan Sur (ankan.sur@princeton.edu), and/or open a GitHub issue (recommended for reproducible bug reports).

## Project Team

- **Evolution Code:** Roberto Tejada Arevalo (ORCHARD), Ankan Sur (APPLE), Yubo Su (APPLE), 
- **Equation of State (EOS):** Roberto Tejada Arevalo, Yubo Su
- **Miscibility:** Roberto Tejada Arevalo, Yubo Su
- **Boundary Conditions:** Yixian Chen, Roberto Tejada Arevalo
- **Principal Investigator (PI):** Adam Burrows

**Affiliation:** Department of Astrophysical Sciences, Princeton University

**Funding:** This project is funded by the Center for Matter at Atomic Pressures (CMAP), sponsored by the National Science Foundation (NSF), United States. Roberto Tejada Arevalo was partially supported by the Ford Foundation, and Yubo Su was supported by the Lyman Spitzer Fellowship during his time at Princeton University.

**APPLE Created:** June 2023 | **ORCHARD Created:** February 2026

## Funding Support

ORCHARD is made possible thanks to

[![Centre for Matter at Atomic Pressures](logos/cmap_logo.jpg)](https://cmap.princeton.edu/)

[![Princeton University](logos/princeton_logo.jpg)](https://www.princeton.edu/)

## Citation

If you find `ORCHARD` useful in your work, please star the repository and cite the papers in our [citation file](CITATIONS.bib).

In addition, please cite the module- and planet-specific references that apply to your calculation:

- **Atmospheric boundary conditions:** cite [Chen et al. 2023](https://ui.adsabs.harvard.edu/abs/2023ApJ...957...36C/abstract) paper when using atmosphere tables and irradiated boundary conditions
- **Gas giant planets:** cite [Sur et al. 2025](https://ui.adsabs.harvard.edu/abs/2025ApJ...980L...5S/abstract) and [Sur et al. 2026](https://ui.adsabs.harvard.edu/abs/2026ApJ...998..305S/abstract) papers when modeling Jupiter, Saturn, or gas-giant exoplanets.
- **Uranus and Neptune:** cite [Tejada Arevalo 2025](https://ui.adsabs.harvard.edu/abs/2025ApJ...989L..40T/abstract) paper when modeling ice giants or Uranus/Neptune-like planets.
- **Sub-Neptunes:** cite [Tejada Arevalo et al. 2026](https://ui.adsabs.harvard.edu/abs/2026arXiv260100059T/abstract) when modeling sub-Neptunes or super-Earths.


---

[![Department of Astrophysical Sciences](logos/astrophysical_sciences_banner.jpg)](https://web.astro.princeton.edu/)
