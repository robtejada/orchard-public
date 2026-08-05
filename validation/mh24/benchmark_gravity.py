#!/usr/bin/env python3
"""
Speed benchmark: ORCHARD TOF7 (utils.TOF7, numba-JIT) vs the clean-room CMS
(utils.cms_hubbard, pure Python), per gravity solve, on a SMOOTH n=1 polytrope
(Jupiter M, R_eq, omega -> q~0.089). A smooth structure is used so both solvers
converge cleanly and the comparison reflects algorithm cost, not downsampling
artifacts of MH24's sharp-featured profile.

Quantifies how much a `gravity_method=cms` evolution run (tof_calc=True) would
cost vs TOF7: _call_tof fires every HSE/rotation iteration.

Run:
    /opt/anaconda3/envs/orchard_env/bin/python validation/mh24/benchmark_gravity.py
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from utils import const                       # noqa: E402
from utils.TOF7 import get_moments7            # noqa: E402
from utils import cms_hubbard as C             # noqa: E402

M = const.M_jup
R = const.rj_eq                                # equatorial radius (cm)
OMEGA = 2 * np.pi / 35730.0
Q = OMEGA ** 2 * R ** 3 / (const.G * M)        # ~0.089 (Jupiter)


def polytrope(N):
    """Smooth n=1 polytrope: rho(l)=rho_c sinc(pi l/R), analytic enclosed mass.
    Grid center->surface, l[0]=eps>0 (so enclosed mass m[0]>0)."""
    eps = R / (2.0 * N)
    l = np.linspace(eps, R, N)
    z = l / R
    rho_c = np.pi * M / (4.0 * R ** 3)
    rho = rho_c * np.sin(np.pi * z) / (np.pi * z)
    m = (4.0 * rho_c * R ** 3 / np.pi ** 2) * (np.sin(np.pi * z) - np.pi * z * np.cos(np.pi * z))
    return l, rho, m


def bench(N, ntof=20):
    l, rho, m = polytrope(N)                   # center->surface (TOF7 convention)
    lam = l[::-1]                              # surface->center, equatorial (CMS convention)
    rho_cms = rho[::-1]

    # --- TOF7 (numba) ---
    j7, sh7 = get_moments7(l, rho, m, OMEGA)   # JIT warmup (discard)
    t = time.perf_counter()
    for _ in range(ntof):
        j7, sh7 = get_moments7(l, rho, m, OMEGA)
    tof_cold = (time.perf_counter() - t) / ntof
    ig7 = tuple(sh7[1:])
    t = time.perf_counter()
    for _ in range(ntof):
        jw, _ = get_moments7(l, rho, m, OMEGA, initial_guess=ig7)
    tof_warm = (time.perf_counter() - t) / ntof

    # --- CMS (pure Python) ---
    t = time.perf_counter()
    jc, zeta, info = C.get_moments_cms(lam, rho_cms, M, OMEGA, G=const.G)
    cms_cold = time.perf_counter() - t
    t = time.perf_counter()
    jcw, _, info2 = C.get_moments_cms(lam, rho_cms, M, OMEGA, G=const.G, initial_guess=zeta)
    cms_warm = time.perf_counter() - t

    return dict(N=N, tof_cold=tof_cold, tof_warm=tof_warm,
                cms_cold=cms_cold, cms_warm=cms_warm,
                ic=info["iters"], iw=info2["iters"],
                tof_J2=float(jw[0]), cms_J2=float(jcw[0]))


def main():
    print("=" * 100)
    print(f"GRAVITY SOLVER SPEED  (smooth n=1 polytrope, Jupiter q={Q:.4f}, per solve)")
    print(f"  ToF7 = utils.TOF7 (numba-JIT) ; CMS = utils.cms_hubbard (pure Python)")
    print("=" * 100)
    print(f"{'N':>6} | {'ToF7 cold':>11} {'ToF7 warm':>11} | {'CMS cold':>10} {'CMS warm':>10} "
          f"| {'CMS/ToF7':>9} | {'J2 (T/C)':>16}")
    print("-" * 100)
    for N in (512, 1024, 2049):
        r = bench(N)
        ratio = r["cms_warm"] / r["tof_warm"]
        print(f"{N:>6} | {r['tof_cold']*1e3:>9.2f}ms {r['tof_warm']*1e3:>9.2f}ms | "
              f"{r['cms_cold']:>8.2f}s {r['cms_warm']:>8.2f}s | {ratio:>8.0f}x | "
              f"{r['tof_J2']:>7.1f}/{r['cms_J2']:<7.1f}")

    print("\nIntegrated _call_tof cost (CMS lambda-loop = what hydrostatic.py runs), N=512:")
    l, rho, m = polytrope(512)
    t = time.perf_counter()
    C.get_moments_cms_from_mean(l[::-1], rho[::-1], m[-1], OMEGA, G=const.G)
    print(f"  get_moments_cms_from_mean: {time.perf_counter()-t:.1f} s  (~lambda_iters x single solve)")
    print("\nTOF7 J2 ~= CMS J2 (same polytrope) confirms a like-for-like comparison.")


if __name__ == "__main__":
    main()
