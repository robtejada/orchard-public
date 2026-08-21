#!/usr/bin/env python
"""
ORCHARD regression tests.

A short suite meant as a reference point when changing the code: it pins the
numbers that a handful of representative calculations produce, so an
unintended change shows up immediately.

Usage (from the repository root, in the orchard_env environment):

    python tests/run_tests.py             # everything (~2 minutes)
    python tests/run_tests.py --quick     # skip the evolution runs (~20 s)
    python tests/run_tests.py --update    # regenerate the reference values

What is covered:

  1. Static structures   seven models spanning gas giants, a sub-Neptune, a
                         bare super-Earth, and an MH24-like fuzzy core with
                         rotation and ToF4. Exercises initialize_grid, both
                         first-guess builders, the H-He-Z mixture EOS, the
                         mantle/core EOSes, struct composition profiles, the
                         Henyey solver, and the gravity harmonics.
  2. EOS spot values     direct table lookups for the H-He-Z mixture, the
                         mantle, and the core EOSes. These catch a changed or
                         re-downloaded EOS module before it reaches a model.
  3. Invariants          checks that need no stored reference: mass
                         conservation, inward monotonicity of P/rho, local
                         hydrostatic balance, and repeatability.
  4. Evolution tracks    a homogeneous gas giant and a bare super-Earth, run
                         through evolution.py and compared at fixed ages,
                         plus a bound on the energy-conservation residual.

Reference values live in tests/reference_values.json and are compared with
RELATIVE tolerances. Results are not bit-identical across machines, numpy
builds, or BLAS versions, so exact equality is deliberately not required. If
a change is meant to move the numbers, rerun with --update and commit the
new file alongside the change.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF_PATH = os.path.join(REPO, "tests", "reference_values.json")

os.environ.setdefault("ORCHARD_LIBRARY_MODE", "1")
sys.path.insert(0, REPO)
os.chdir(REPO)

import numpy as np                                            # noqa: E402

# Relative tolerances. Static structures are a single converged solve and
# reproduce tightly; evolution endpoints accumulate thousands of adaptive
# steps and drift more between platforms.
TOL_STATIC = 1e-6
TOL_EOS = 1e-8
TOL_EVOL = 1e-4

# Energy-conservation residual |dE/E_rad| accepted for an evolution run. This
# is a physics bound, not a pinned value: the residual is dominated by the
# EOS-table Maxwell defect (see writeups/energy_conservation).
MAX_ENERGY_RESIDUAL = 5e-2

MH24_COORDS = [0.0, 0.10, 0.20, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60,
               0.70, 0.85, 1.0]
MH24_Z = [0.1831, 0.1831, 0.1825, 0.1383, 0.1152, 0.0905, 0.0657, 0.0392,
          0.0156, 0.0156, 0.0156, 0.0156, 0.0156]

STATIC_CASES = {
    "jupiter_10ME_core": dict(M_MJup=1.0, S=6.5, Y=0.277, Z=0.02,
                              mass_core=10.0),
    "jupiter_coreless": dict(M_MJup=1.0, S=6.5, Y=0.277, Z=0.02,
                             mass_core=0.0, mass_core_fe=0.0),
    "saturn_mass_0.3MJ": dict(M_MJup=0.3, S=6.5, Y=0.277, Z=0.02,
                              mass_core=10.0),
    # Coreless on purpose: at 5 M_J a compact core pushes the rock EOS past
    # its tabulated range (see tutorial_superjupiters.ipynb).
    "super_jupiter_5MJ": dict(M_MJup=5.0, S=6.5, Y=0.277, Z=0.02,
                              mass_core=0.0, mass_core_fe=0.0),
    "sub_neptune_6ME": dict(M_Mearth=6.0, S=6.0, Y=0.277, Z=0.05,
                            mass_core=5.4, mass_core_fe=1.8),
    "bare_super_earth_3ME": dict(
        M_Mearth=3.0, S=6.0, mass_core=3.0, mass_core_fe=1.0,
        overrides={"hydrostatic_equilibrium": {"env_s_start": "False"},
                   "core": {"mantle_entropy": "0.55"}}),
    "fuzzy_core_mh24": dict(M_MJup=1.0, S=7.0, Y=0.277, Z=0.0156,
                            mass_core=0.0, mass_core_fe=0.0,
                            z_profile=(MH24_COORDS, MH24_Z),
                            rotation=True, tof=True, period=35730,
                            c_moi=0.26393),
}

# Evolution tracks: (config file to start from, overrides, ages [Gyr] to
# compare at). Kept short on purpose; both finish in about a minute.
EVOLUTION_CASES = {
    "gas_giant_track": dict(
        base="parameters_default.ini",
        overrides={"general": {"N": "300", "final_age": "1.0",
                               "save_hdf5": "True"}},
        ages=[0.1, 0.5, 1.0]),
    "bare_super_earth_track": dict(
        base="parameter_examples/parameter_user_super_earth_1Mearth_bare_example.ini",
        overrides={"general": {"N": "300", "final_age": "0.2",
                               "save_hdf5": "True"}},
        ages=[0.05, 0.1, 0.2]),
}


class Results:
    def __init__(self):
        self.passed = self.failed = 0
        self.failures = []

    def check(self, name, value, reference, tol):
        """Compare one number against its reference within a relative tol."""
        if reference is None:
            print(f"    {name:38s} {value:>14.7g}   (no reference)")
            return
        denom = abs(reference) if abs(reference) > 0 else 1.0
        rel = abs(value - reference) / denom
        if rel <= tol:
            self.passed += 1
            print(f"    {name:38s} {value:>14.7g}   ok ({rel:.1e})")
        else:
            self.failed += 1
            msg = (f"{name}: got {value:.10g}, expected {reference:.10g} "
                   f"(relative {rel:.2e} > {tol:.0e})")
            self.failures.append(msg)
            print(f"    {name:38s} {value:>14.7g}   FAIL ({rel:.1e})")

    def assert_true(self, name, condition, detail=""):
        if condition:
            self.passed += 1
            print(f"    {name:38s} {'ok':>14s}")
        else:
            self.failed += 1
            self.failures.append(f"{name}: {detail}")
            print(f"    {name:38s} {'FAIL':>14s}  {detail}")


def section(title):
    print(f"\n{title}\n{'-' * len(title)}")


# --------------------------------------------------------------------------
# 1. Static structures
# --------------------------------------------------------------------------
def run_static(ref, res, update):
    section("1. Static structures")
    import static

    out = {}
    for name, kwargs in STATIC_CASES.items():
        t0 = time.time()
        model = static.build(**kwargs)
        dt = time.time() - t0
        values = {
            "radius_rj": model.radius_rj,
            "p_center_mbar": float(model.p[-1]) * 1e-12,
            "t_center_k": float(model.temp[-1]),
        }
        if kwargs.get("tof"):
            values["j2_ppm"] = model.j2_ppm
            values["j4_ppm"] = model.j4_ppm
        out[name] = values
        print(f"  {name}  ({dt:.2f} s, {model.niter} iterations)")
        res.assert_true(
            f"{name}_physical",
            bool(np.all(model.temp > 0) and np.all(model.rho > 0)
                 and np.all(np.isfinite(model.temp))),
            "converged structure has non-positive or non-finite T/rho "
            "(EOS evaluated outside its tables)")
        for key, value in values.items():
            res.check(key, value, ref.get(name, {}).get(key) if not update
                      else None, TOL_STATIC)
    return out


# --------------------------------------------------------------------------
# 2. EOS spot values
# --------------------------------------------------------------------------
def run_eos(ref, res, update):
    section("2. EOS spot values")
    import initial

    out = {}
    log10e = np.log10(np.e)

    # H-He-Z mixture: T(S, P) and rho(P, T) at envelope conditions.
    s_kb = np.array([6.5, 8.0])
    log10p = np.array([9.0, 12.0])            # 1 kbar and 1 Mbar, log10 cgs
    y = np.array([0.277, 0.277])
    z = np.array([0.02, 0.10])
    fr = np.zeros(2)
    logt = initial.mixtures_eos.get_logt_sp(s_kb, log10p, y, z, fr,
                                            tab=True)
    logrho = initial.mixtures_eos.get_logrho_pt(log10p, logt, y, z, fr)
    out["mixture_logt_sp"] = [float(v) for v in logt]
    out["mixture_logrho_pt"] = [float(v) for v in logrho]

    # Mantle and core EOSes at their own conditions.
    p_gpa = np.array([100.0, 500.0])
    out["mantle_rho_sp"] = [
        float(v) for v in initial.mantle_eos.get_rho_sp(
            np.array([0.55, 0.55]), p_gpa)]
    out["core_t_sp"] = [
        float(v) for v in initial.core_eos.get_T_sp(
            np.array([0.22, 0.22]), p_gpa)]

    for key, values in out.items():
        reference = ref.get(key) if not update else None
        for i, value in enumerate(values):
            res.check(f"{key}[{i}]", value,
                      reference[i] if reference else None, TOL_EOS)
    return out


# --------------------------------------------------------------------------
# 3. Invariants (no stored reference needed)
# --------------------------------------------------------------------------
def run_invariants(res):
    section("3. Invariants")
    import static
    from utils import const

    model = static.build(M_MJup=1.0, S=6.5, Y=0.277, Z=0.02, mass_core=10.0)

    total = float(model.m_b[0])
    res.check("mass_sum_over_total", float(np.sum(model.dm)) / total, 1.0,
              1e-12)

    res.assert_true("pressure_increases_inward",
                    bool(np.all(np.diff(model.p) > 0)),
                    "pressure is not monotonic from surface to center")
    res.assert_true("density_increases_inward",
                    bool(np.all(np.diff(model.rho) > 0)),
                    "density is not monotonic from surface to center")
    res.assert_true("all_finite",
                    bool(np.all(np.isfinite(model.p))
                         and np.all(np.isfinite(model.rho))
                         and np.all(np.isfinite(model.temp))),
                    "non-finite values in the converged structure")

    # Local hydrostatic balance: dP/dm = -G m / (4 pi r^4) between cell
    # centers, evaluated on the envelope where the EOS is smoothest.
    k = slice(5, model.kcore - 5)
    m_mid = 0.5 * (model.m_b[:-1] + model.m_b[1:])
    dP_dm = np.gradient(model.p, m_mid)
    r_mid = 0.5 * (model.r_b[:-1] + model.r_b[1:])
    expected = -const.G * m_mid / (4.0 * np.pi * r_mid ** 4)
    rel = np.abs(dP_dm[k] - expected[k]) / np.abs(expected[k])
    # Coarse check: a centered difference on a non-uniform mass grid carries
    # its own discretization error (a few percent where the spacing changes),
    # so compare the median rather than the worst cell. This is a guard
    # against gross errors such as a sign flip or a missing factor, not a
    # measure of the solver's accuracy.
    res.assert_true("hydrostatic_balance_envelope",
                    float(np.median(rel)) < 2e-2,
                    f"median relative residual {float(np.median(rel)):.2e}")

    # Repeatability: the same inputs must give the same structure.
    again = static.build(M_MJup=1.0, S=6.5, Y=0.277, Z=0.02, mass_core=10.0)
    res.assert_true("repeatable",
                    bool(np.array_equal(model.p, again.p)
                         and np.array_equal(model.r_b, again.r_b)),
                    "two identical build() calls disagree")


# --------------------------------------------------------------------------
# 4. Evolution tracks
# --------------------------------------------------------------------------
def run_evolution(ref, res, update):
    section("4. Evolution tracks")
    import configparser

    import load_models
    from utils import const

    out = {}
    workdir = tempfile.mkdtemp(prefix="orchard_tests_")
    try:
        for name, case in EVOLUTION_CASES.items():
            config = configparser.ConfigParser()
            config.read(os.path.join(REPO, case["base"]))
            for sec, entries in case["overrides"].items():
                if not config.has_section(sec):
                    config.add_section(sec)
                for key, value in entries.items():
                    config.set(sec, key, value)
            cfg_name = f"_test_{name}"
            cfg_path = os.path.join(workdir, cfg_name + ".ini")
            with open(cfg_path, "w") as handle:
                config.write(handle)

            # evolution.py must run in CLI mode: with ORCHARD_LIBRARY_MODE
            # set it skips argument parsing, ignores --config, and exits
            # without evolving anything.
            env = dict(os.environ)
            env.pop("ORCHARD_LIBRARY_MODE", None)

            t0 = time.time()
            proc = subprocess.run(
                [sys.executable, "evolution.py", "--config", cfg_path],
                cwd=REPO, capture_output=True, text=True, env=env)
            dt = time.time() - t0
            print(f"  {name}  ({dt:.0f} s)")
            if proc.returncode != 0:
                res.assert_true(f"{name}_completed", False,
                                (proc.stdout + proc.stderr).strip()[-300:])
                continue
            model_dir = os.path.join(REPO, "models", cfg_name)
            if not os.path.isdir(model_dir) or not os.listdir(model_dir):
                res.assert_true(f"{name}_completed", False,
                                f"no output written to models/{cfg_name}")
                continue
            res.assert_true(f"{name}_completed", True)

            model = load_models.load([cfg_path])[0]
            age_gyr = model["data_age"] * const.s_to_Gyr
            values = {}
            for age in case["ages"]:
                i = int(np.argmin(np.abs(age_gyr - age)))
                tag = f"{age:g}gyr".replace(".", "p")
                values[f"radius_re_{tag}"] = float(
                    model["data_rad"][i]) / const.rearth
                values[f"teff_{tag}"] = float(model["data_teff"][i])
            out[name] = values
            for key, value in values.items():
                res.check(key, value,
                          ref.get(name, {}).get(key) if not update else None,
                          TOL_EVOL)

            # Energy conservation: a bound, not a pinned number.
            match = re.search(r"Energy nonconservation:\s*([-\d.eE+]+)",
                              proc.stdout)
            if match:
                residual = abs(float(match.group(1)))
                res.assert_true(
                    f"{name}_energy_residual",
                    residual < MAX_ENERGY_RESIDUAL,
                    f"|dE/E_rad| = {residual:.3e} exceeds "
                    f"{MAX_ENERGY_RESIDUAL:.0e}")

            shutil.rmtree(os.path.join(REPO, "models", cfg_name),
                          ignore_errors=True)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update", action="store_true",
                        help="regenerate tests/reference_values.json")
    parser.add_argument("--quick", action="store_true",
                        help="skip the evolution runs")
    args = parser.parse_args()

    ref = {}
    if os.path.exists(REF_PATH) and not args.update:
        with open(REF_PATH) as handle:
            ref = json.load(handle)
    elif not args.update:
        print(f"No reference file at {REF_PATH}; run with --update first.")
        return 1

    print("ORCHARD regression tests")
    print(f"  repository : {REPO}")
    print(f"  mode       : {'UPDATE references' if args.update else 'check'}")

    started = time.time()
    res = Results()
    new = {}
    new.update(run_static(ref, res, args.update))
    new.update(run_eos(ref, res, args.update))
    run_invariants(res)
    if not args.quick:
        new.update(run_evolution(ref, res, args.update))

    if args.update:
        merged = dict(ref)
        merged.update(new)
        merged["_note"] = ("Reference values for tests/run_tests.py. "
                           "Regenerate with: python tests/run_tests.py --update")
        with open(REF_PATH, "w") as handle:
            json.dump(merged, handle, indent=2, sort_keys=True)
            handle.write("\n")
        print(f"\nWrote {REF_PATH}")
        return 0

    elapsed = time.time() - started
    print(f"\n{'=' * 60}")
    print(f"  {res.passed} passed, {res.failed} failed  ({elapsed:.0f} s)")
    if res.failures:
        print(f"{'=' * 60}")
        for failure in res.failures:
            print(f"  - {failure}")
    print(f"{'=' * 60}")
    return 1 if res.failed else 0


if __name__ == "__main__":
    sys.exit(main())
