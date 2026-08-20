# ORCHARD FAQ

Frequently asked questions for new users of **ORCHARD**, a public planetary
interior-structure and evolution code (Tejada Arevalo et al. 2026b). This page is
written for newcomers from four communities — **gas-giant exoplanets**,
**sub-Neptune exoplanets**, **super-Earth / terrestrial planets**, and
**Solar-System interior evolution** — but most of the "Getting started",
"Configuration", "Outputs", and "Troubleshooting" material applies to everyone.

> **Can't find your answer here?** See [Getting help](#getting-help) at the
> bottom. In short: search/ask the docs assistant, browse `parameter_examples/`
> and the `tutorial_*.ipynb` notebooks (repository root), read `parameter_descriptions.md`, and if you're still stuck,
> open a **GitHub Issue** (bug/feature) or a **GitHub Discussion** (usage question).

---

## Contents
- [Getting started](#getting-started)
- [Gas-giant exoplanets](#gas-giant-exoplanets)
- [Sub-Neptune exoplanets](#sub-neptune-exoplanets)
- [Super-Earths & terrestrial planets](#super-earths--terrestrial-planets)
- [Solar-System interior evolution](#solar-system-interior-evolution)
- [Configuration basics](#configuration-basics)
- [Outputs & analysis](#outputs--analysis)
- [Troubleshooting](#troubleshooting)
- [Getting help](#getting-help)

---

## Getting started

**Q: What is ORCHARD, and what can it model?**
ORCHARD evolves the thermal and compositional structure of planets from
~0.5 Earth masses to 10+ Jupiter masses over gigayear timescales. It solves
hydrostatic equilibrium (Henyey relaxation) coupled to thermal/compositional
transport, and supports helium rain, heavy-element (Z) miscibility, fuzzy/dilute
cores, viscous mantle convection, radiogenic heating, and rotation with
gravitational harmonics. It covers gas giants, ice giants, sub-Neptunes, and
super-Earths.

**Q: How do I install it?**
Two paths are described in the `README.md`: (1) **install from the Zenodo release
tarball** (recommended for users), or (2) the **GitHub developer path**. Both
create a Conda environment (`orchard_env`) from `environment.yaml`. The GitHub
path additionally requires **Git LFS** (for large data files) and initializing the
**EOS submodule** (`git submodule update --init --recursive`). See the README's
"Getting Started" section for the exact commands.

**Q: How do I run my first model?**
Activate the environment and point `evolution.py` at a config file:
```bash
conda activate orchard_env
python evolution.py --config parameter_examples/parameter_user_1mj.ini
```
Running `python evolution.py` with no `--config` uses `parameters_default.ini`.
Add `-plot_end` for a movie at the end, or `-plot_active` for live plotting.

**Q: Where do results go?**
Into `models/<config_name>/` (a folder named after your `.ini`). Output is either
one text file per saved timestep or a single `evolution.h5` (HDF5) if `save_hdf5`
is enabled. See [Outputs & analysis](#outputs--analysis).

**Q: I just want a template for my planet — where do I start?**
Copy the closest file in `parameter_examples/` and edit it. The naming is
self-explanatory (e.g., `parameter_user_2mj.ini`,
`parameter_user_sub_neptune_6Mearth_example.ini`,
`parameter_user_super_earth_3Mearth_bare_example.ini`,
`parameter_user_saturn_herain.ini`). Every parameter is documented in
`parameter_descriptions.md`, and the `tutorial_*.ipynb` notebooks (repository root) walk through full
workflows.

---

## Gas-giant exoplanets

**Q: I work on hot Jupiters / super-Jupiters / brown dwarfs. Can ORCHARD model an
irradiated giant, and how do I set the mass and irradiation?**
Yes. Set the mass in `[initial]` with `M_MJup` (Jupiter masses) or `M_Mearth`
(Earth masses; 1 M_Jup ≈ 317.9 M_Earth). Choose an atmosphere model in
`[boundary_condition]` with `bc_atm` — `c23` for Jupiter/Saturn-like, `c26` for
super-Jupiters — and set `planet` accordingly (`Jupiter`, `Super_Jupiter`).
Irradiation/clouds are toggles for the `c23`/`c26` models. Start from
`parameter_examples/parameter_user_super_jupiter*.ini` or the `*mj.ini` series
(`parameter_user_1mj.ini` … `parameter_user_10mj.ini`), and see the
**`tutorial_superjupiters.ipynb`** notebook.

**Q: What's a "hot start" vs "cold start", and how do I set the initial entropy?**
Initial specific entropy is `S_ini` in `[initial]`, in units of k_B/baryon.
Hot-start giants are ~9–11; lower values are colder/older-looking initial states.
You can also start from non-uniform entropy profiles (see
`initial_profiles_S`).

**Q: How do I model helium rain (e.g., to study cooling or atmospheric He
depletion)?**
In `[diffusion]`, set `composition_change = True` and `helium_rain = True`, pick a
miscibility curve with `misc_curve` (e.g., `lorenzen`, `schoettler`), and control
the rain vigor with `alpha_herain`. See `parameter_user_jup_herain.ini` and
`parameter_user_saturn_herain.ini`, and the
**`tutorial_inhomogeneous_evolution.ipynb`** notebook.

**Q: Can I add a heavy-element core or a dilute/fuzzy core?**
Yes. A compact core mass is set with `mass_core` in `[core]`. For dilute cores and
composition gradients, use non-uniform Z profiles via `initial_profiles_Z`
(`struct`, `gaussian`, etc.); see the `*inhomog*` examples
(`parameter_user_jup_inhomog_gaussian_example1.ini`).

**Q: Which H–He equation of state is used, and can I change it?**
`[equation_of_state] hhe_eos` selects it: `cd` (Chabrier & Debras 2021, default),
`cms` (Chabrier et al. 2019, with optional Howard & Guillot 2023 corrections via
`hg`), or `scvh` (legacy). Heavy-element EOS is `z_eos`.

**Q: How do I get the cooling curve / radius contraction?**
Set `final_age` (Gyr) and read `T_eff`, `T_int`, luminosity, and the total
radius from the outputs (see [Outputs & analysis](#outputs--analysis)).

---

## Sub-Neptune exoplanets

**Q: I study the radius valley. Can ORCHARD model a sub-Neptune (H/He envelope on
a rocky core)?**
Yes — this is a core use case. Set the total mass with `M_Mearth`, the rocky
interior with `mass_core` and the iron core with `mass_core_fe` (both Earth
masses), and use `bc_atm = f07` (Fortney et al. 2007) with an equilibrium
temperature `T_eq`. Start from
`parameter_examples/parameter_user_sub_neptune_6Mearth_example.ini` and see
**`tutorial_subneptunes_superearths.ipynb`**.

**Q: How do I set the envelope mass fraction?**
The envelope mass is the total minus the core: (`M_Mearth` − `mass_core`). The
example files encode common fractions in their names, e.g.
`parameter_user_sub_neptune_10Mearth_env_10pcent_example.ini` and
`parameter_user_sub_neptune_5Mearth_thinenv_10pcent_example.ini`.

**Q: Can the envelope or interior be water/ice (water worlds)?**
Yes. Heavy-element/water options are set with `[equation_of_state] z_eos`
(`aqua` for AQUA water, `ice_mixture` for a CH4/NH3/H2O mix), and a water mantle
via `[core] mantle_comp = h2o`.

**Q: Does ORCHARD include atmospheric mass loss / photoevaporation?**
**Not in the current public version.** Mass loss is experimental and currently
**disabled** — enabling it raises an error. If your science depends on
escape-driven evolution, please note this limitation and watch the repo / open a
Discussion to ask about status.

**Q: How do I set irradiation for a close-in planet?**
Use `T_eq` (and `bond_albedo`) with the `f07` or gray atmosphere models. The
equilibrium temperature controls the irradiated boundary.

---

## Super-Earths & terrestrial planets

**Q: I study rocky-planet thermal evolution. Can ORCHARD do a bare rocky
super-Earth?**
Yes. Set `bc_atm = bare` (a σT⁴ balance, no atmosphere) and make the planet all
core: `M_Mearth = mass_core` (both equal the planet mass). Choose the mantle with
`[core] mantle_comp` (`mg2sio4` default, `mgsio3`, or `h2o`) and the iron core with
`core_comp` / `mass_core_fe`. See
`parameter_examples/parameter_user_super_earth_3Mearth_bare_example.ini`.

**Q: Does it include radiogenic heating?**
Yes — set `[transport] radioactive = True`. Heating from ⁴⁰K, ²³²Th, ²³⁵U, ²³⁸U is
applied to mantle cells. (Note: the present-day isotope abundances follow
McDonough & Sun 1995.)

**Q: Can I model mantle melting / a magma ocean?**
Yes. Enable `latent_heat_effect` and `viscous_mantle_convection` (in `[core]`);
the melt fraction is an output field. The mantle conductivity/viscosity follow
standard models (see `parameter_descriptions.md`).

**Q: Can a rocky planet have a thin atmosphere?**
Yes — use `bc_atm = rock_gray` or `ideal_gray` and a small envelope. See the
`*thinenv*` examples (e.g.,
`parameter_user_super_earth_3Mearth_thinenv_example.ini`).

**Q: How do I set the mantle and iron-core sizes?**
`mass_core` is the mantle+core mass and `mass_core_fe` is the iron-core mass (both
Earth masses, with `mass_core_fe ≤ mass_core`). For a coreless or ironless body
use the documented sentinel values (see `parameter_descriptions.md`).

---

## Solar-System interior evolution

**Q: Can I model Jupiter, Saturn, Uranus, and Neptune?**
Yes. Set `planet` and `bc_atm` (`c23` for Jupiter/Saturn; `g75`, Guillot 1995, for
Uranus/Neptune). See `parameter_user_saturn_homog_noherain.ini`,
`parameter_user_neptune.ini`, and **`tutorial_solarsystem_planets.ipynb`**.

**Q: How do I get gravitational harmonics (J2, J4) to compare with Juno/Cassini?**
Enable rotation; the Theory-of-Figures machinery then outputs `J2, J4, J6, J8`,
oblateness, and equatorial/polar radii. See `parameter_descriptions.md` for the
`[hydrostatic_equilibrium]` rotation settings (`rotation`, `tof_calc`, `C_MoI`,
`period`). Note: for accurate standalone
gravitational moments the rotating hydrostatic solution should be iterated to
self-consistency (a single pass over-inflates J2 by ~7%).

**Q: How do I study helium rain in Saturn (its luminosity excess) or in Jupiter?**
Enable `helium_rain` in `[diffusion]` and choose `misc_curve`
(`lorenzen` / `schoettler`). The He-rain layer, atmospheric He depletion, and the
resulting luminosity/age are all in the outputs. See
`parameter_user_saturn_herain.ini`.

**Q: Can I model a dilute/fuzzy core (as Juno suggests for Jupiter)?**
Yes — use non-uniform composition profiles (`initial_profiles_Z = struct` or
`gaussian`) to set up a gradient that erodes over time. See the `*inhomog*`
examples.

**Q: How do I reproduce a specific planet's present-day state?**
Set the mass, composition, and `final_age` (4.56 Gyr for the Solar System) and
compare the model's present-day radius, T_eff, J2/J4, and atmospheric He against
observations. The Solar-System tutorial demonstrates this.

---

## Configuration basics

**Q: How are config files structured?**
INI format with sections: `[general]` (grid `N`, `final_age`, timestep `dt_`,
`save_interval`, `save_hdf5`), `[boundary_condition]` (`bc_atm`, `planet`, `T_eq`,
irradiation), `[equation_of_state]` (`hhe_eos`, `z_eos`), `[initial]` (mass,
`S_ini`, composition profiles), `[core]` (`mass_core`, `mass_core_fe`,
`mantle_comp`, `core_comp`), `[diffusion]` (`composition_change`, `helium_rain`,
`metal_rain`, `misc_curve`), `[transport]` (`convection`, `radiation`,
`radioactive`, `semiconvection`). **Every parameter is documented in
`parameter_descriptions.md`** — that file is the authoritative reference.

**Q: Do I have to set every parameter?**
No. `parameters_default.ini` supplies defaults; your `--config` file only needs to
list what differs.

**Q: What units does ORCHARD use?**
Internally everything is **CGS**. Config files use mixed convenient units (Gyr,
Earth/Jupiter masses, bar) that are converted on read. Entropy is in k_B/baryon.

---

## Outputs & analysis

**Q: How do I load and plot a finished model?**
Use the plotting CLI or the loader:
```bash
python plots.py --config <your.ini>                          # snapshot at last age
python plots.py --config <your.ini> --mode age --ages 1.0 4.56
python plots.py --config A.ini B.ini --labels "A" "B"        # compare models
```
```python
import load_models
m = load_models.ModelSet(paths=["<your.ini>"])[0]
T = m['temp'].age(4.56, units='Gyr')   # temperature profile at 4.56 Gyr
snap = m.snapshot(age=4.0, units='Gyr')
```
See **`tutorial_model_plotting.ipynb`**.

**Q: What quantities are available?**
Profiles (radius, pressure, density, temperature, entropy `S`, `Y`, `Z`,
`f_rock`, fluxes, melt fraction, …) and scalars per age (`T_int`, `T_eff`,
luminosity, `J2`–`J8`, oblateness, atmospheric Y/Z, …). The model-plotting and
EOS tutorials show how to access them.

**Q: Text output or HDF5 — which should I use?**
HDF5 (`save_hdf5 = True`, one `evolution.h5`) is compact and convenient for
analysis; text output writes one file per saved timestep. `save_interval` (Myr)
controls how often models are saved.

---

## Troubleshooting

**Q: It can't find the EOS / crashes on import (developer/GitHub install).**
Initialize the submodule: `git submodule update --init --recursive`, and make sure
**Git LFS** pulled the large data files. The Zenodo tarball already bundles these.

**Q: My run is rejected/halving the timestep a lot, or stalls early.**
Some physics (especially helium rain crossing the miscibility curve) legitimately
forces small steps and many retries — a run that keeps making progress is fine.
If it truly stalls, loosen tolerances or check that your initial composition/mass
is physical. Early-time convergence is stricter and usually relaxes as the model
settles.

**Q: I get NaNs / the structure looks unphysical.**
Most often an EOS evaluation outside its valid P–T–composition range, or an
inconsistent setup (e.g., `mass_core` larger than the planet mass). Re-check the
masses and composition against `parameter_descriptions.md` and the closest
example file.

**Q: Which example is closest to my planet?**
Browse `parameter_examples/` — files are named by planet type, mass, and physics
(rain, inhomogeneous, bare/thin envelope, etc.).

---

## Getting help

If this FAQ, `parameter_descriptions.md`, the `parameter_examples/`, and the
the tutorial notebooks don't resolve your question:

1. **Ask the docs assistant** (if deployed on the ORCHARD website / repo) — it
   answers from this FAQ, the parameter docs, the tutorials, and the examples, and
   links you to the source.
2. **Search existing GitHub Issues and Discussions** — your question may already
   be answered.
3. **Open a GitHub Discussion** for usage questions ("how do I model X?").
4. **Open a GitHub Issue** for bugs or feature requests — include your `.ini`,
   the command you ran, the full error, and your OS / install method.

When reporting a problem, the most helpful thing you can include is your **config
file** and the **exact command and error message**.
