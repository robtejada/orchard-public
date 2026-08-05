"""
Validate utils/TOF7.py against Nettelmann+2021 Table 1 (n=1 polytrope).

Setup (Wisdom & Hubbard 2016 / N21 Sect. 3):
    GM       = 12.6686536e16 m^3/s^2
    R_eq     = 71492 km
    q_rot    = omega^2 R_eq^3 / GM = 0.089195487

N21's polytrope validation uses a fully self-consistent outer loop that solves
hydrostatic equilibrium with P = K rho^2 and re-updates rho(l) at each pass.
That outer loop is NOT implemented here yet.

This script does three validations:

  1. Non-rotating limit (omega = 0): all J_{2n}, oblateness should vanish.
  2. Consistency between our Eq. 5 and Eq. 7 paths for J_{2n}.
  3. Cross-check our TOF7 against the existing TOF.py (ToF4) on the same
     non-self-consistent polytrope density. Low-order J_{2n} should agree to
     ~1e-5 relative; the two should diverge at the ToF4 truncation level for
     J_6, J_8 (roughly 1% and 10% per N21 Fig. 1, for a Jupiter-q polytrope).

Only (1)-(3) are run; the absolute J_{2n} will differ from N21 Table 1 by
~several percent because we feed in the non-rotating polytrope density
rho(l) = rho_c sin(pi l/R_m)/(pi l/R_m) rather than the self-consistent
rotating polytrope density. To hit N21's ~1e-5 accuracy target, we'd need
the self-consistent outer loop — a follow-up.

Run from repo root:
    python utils/TOF7/test_tof7_polytrope.py
"""

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _ROOT)

from utils import const  # noqa: E402
from utils.TOF import get_moments as get_moments_tof4  # noqa: E402
from utils.TOF7 import (  # noqa: E402
    _build_f,
    _build_Sn_Snp,
    get_moments7,
)

GM = 12.6686536e16 * 1e6  # cm^3/s^2
R_eq_target = 7.1492e9  # cm
q_rot = 0.089195487
M_target = GM / const.G
omega_target = np.sqrt(q_rot * GM / R_eq_target ** 3)

# N21 Table 1 targets, all scaled by 1e-6
BESSEL = {2: 13988.51, 4: -531.8281, 6: 30.11832, 8: -2.13212, 10: 0.17407}
EQ7 = {2: 13988.54, 4: -531.8292, 6: 30.11989, 8: -2.13384, 10: 0.17426}


def polytrope_grid(R_m, N):
    """Grid split per N21 3.1: N/2 in [0, 0.95 R_m], rest in [0.95 R_m, R_m]."""
    eps = R_m / (2.0 * N)
    l_in = np.linspace(eps, 0.95 * R_m, N // 2, endpoint=False)
    l_out = np.linspace(0.95 * R_m, R_m, N - N // 2)
    return np.concatenate([l_in, l_out])


def polytrope_rho_m(l, R_m, M):
    """Non-rotating n=1 polytrope density and enclosed mass on the grid."""
    rho_c = np.pi * M / (4.0 * R_m ** 3)
    z = l / R_m
    pz = np.pi * z
    sinc = np.where(z > 0, np.sin(pz) / np.where(z > 0, pz, 1.0), 1.0)
    rho = rho_c * sinc
    m = (4.0 * rho_c * R_m ** 3 / np.pi ** 2) * (np.sin(pz) - pz * np.cos(pz))
    return rho, m


def test_1_nonrotating():
    print("=" * 72)
    print("Test 1: omega = 0  ->  J_{2n}, oblateness should vanish")
    print("=" * 72)
    N = 500
    R_m = R_eq_target
    l = polytrope_grid(R_m, N)
    rho, m = polytrope_rho_m(l, R_m, M_target)
    j2n, _ = get_moments7(l, rho, m, 0.0)
    J2, J4, J6, J8, J10, J12, J14, oblat, Req, Rpol, I = j2n
    print(f"  J_2 * 1e6  = {J2:.3e}   (target 0)")
    print(f"  J_4 * 1e6  = {J4:.3e}   (target 0)")
    print(f"  oblateness = {oblat:.3e} (target 0)")
    print(f"  R_eq - R_m = {(Req - R_m):.3e} cm (target 0)")
    passed = max(abs(J2), abs(J4), abs(J6), abs(oblat)) < 1e-6
    print(f"  => {'PASS' if passed else 'FAIL'}")
    return passed


def test_2_eq5_vs_eq7():
    print()
    print("=" * 72)
    print("Test 2: Eq. 5 vs Eq. 7  (low-order J_{2n}: tight agreement;")
    print("                         high-order: Eq.5 drops accuracy by design)")
    print("=" * 72)
    N = 1000
    R_m = R_eq_target
    l = polytrope_grid(R_m, N)
    rho, m = polytrope_rho_m(l, R_m, M_target)
    j2n, shapes = get_moments7(l, rho, m, omega_target)
    R_eq = j2n[8]

    # Compute J_{2i} via Eq. 5: J_{2i} = -(R_m/R_eq)^{2i} S_{2i}(1)
    s_list = list(shapes[1:])  # drop s_0
    f, fp = _build_f(s_list)
    S, Sp = _build_Sn_Snp(l, rho, m, f, fp)
    # Expected Eq.5 vs Eq.7 divergence (from N21 Table 1 ratios):
    #   J_2: ~1e-7,  J_4: ~1e-6,  J_6: ~1e-4,  J_8: ~1e-3,
    #   J_10:~1e-1 (Eq.5 deteriorates at high orders -- the f_n tables are
    #               truncated at 7th order in m_rot)
    tol = {2: 1e-5, 4: 1e-5, 6: 1e-4, 8: 1e-3, 10: 1, 12: 1, 14: 1}
    print(f"  {'':6s} {'Eq. 7':>14s} {'Eq. 5':>14s} {'diff/|Eq.7|':>12s} "
          f"{'tol':>10s}")
    ok = True
    for ii, label in enumerate([2, 4, 6, 8, 10, 12, 14]):
        J_eq7 = j2n[ii]
        J_eq5 = -1e6 * (R_m / R_eq) ** (2 * (ii + 1)) * S[2 * (ii + 1)][-1]
        rel = abs(J_eq7 - J_eq5) / max(abs(J_eq7), 1e-30)
        flag = "" if rel < tol[label] else "  <-- over tol"
        print(f"  J_{label:<2d}  {J_eq7:14.5f} {J_eq5:14.5f} {rel:12.2e} "
              f"{tol[label]:10.0e}{flag}")
        if rel > tol[label]:
            ok = False
    print(f"  => {'PASS' if ok else 'FAIL'}")
    return ok


def test_3_tof4_vs_tof7():
    print()
    print("=" * 72)
    print("Test 3: TOF4 (existing utils/TOF.py) vs TOF7  on the same density")
    print("=" * 72)
    N = 1000
    R_m = R_eq_target
    l = polytrope_grid(R_m, N)
    rho, m = polytrope_rho_m(l, R_m, M_target)
    j2n4, _ = get_moments_tof4(l, rho, m, omega_target)
    j2n7, _ = get_moments7(l, rho, m, omega_target)
    # TOF.py layout: [J2, J4, J6, J8, oblat, Req, Rpol, I]  (J scaled by 1e6)
    # TOF7    layout: [J2, J4, J6, J8, J10, J12, J14, oblat, Req, Rpol, I]
    #
    # Expected TOF4 truncation (vs ToF7) scales as m_rot^(4-order/2) ~
    # q_rot^{1,2,3,4} at J_2,J_4,J_6,J_8 respectively.  Using q = 0.089,
    # rough predictions:  J_2: ~9e-2*1e-4 = 1e-5,  J_4: 8e-3,  J_6: 7e-2,
    # J_8: 0.6 (order-of-magnitude).  We use 2x these as loose bounds.
    print(f"  {'':6s} {'TOF4':>14s} {'TOF7':>14s} {'|diff|/|TOF7|':>14s}"
          f"  {'tol':>10s}")
    tol = {2: 2e-5, 4: 2e-3, 6: 5e-2, 8: 5e-1}
    ok = True
    for k, tof4_i, tof7_i in [(2, 0, 0), (4, 1, 1), (6, 2, 2), (8, 3, 3)]:
        a, b = j2n4[tof4_i], j2n7[tof7_i]
        rel = abs(a - b) / max(abs(b), 1e-30)
        flag = "" if rel < tol[k] else "  <-- over tol"
        print(f"  J_{k:<2d}  {a:14.3f} {b:14.3f} {rel:14.2e}  {tol[k]:10.0e}{flag}")
        if rel > tol[k]:
            ok = False
    # MoI agreement
    I4 = j2n4[7]
    I7 = j2n7[10]
    rel_I = abs(I4 - I7) / I7
    print(f"  I     {I4:14.4e} {I7:14.4e} {rel_I:14.2e}  (MoI, loose tol 5e-2)")
    if rel_I > 5e-2:
        ok = False
    print(f"  => {'PASS' if ok else 'FAIL'}")
    return ok


def test_4_absolute_values():
    print()
    print("=" * 72)
    print("Test 4: Absolute J_{2n} values (non-self-consistent polytrope density)")
    print("=" * 72)
    print("  Uses rho(l) = rho_c sin(pi l/R_m)/(pi l/R_m) — the NON-rotating")
    print("  polytrope profile on the rotating grid. Expected deviation from")
    print("  N21 Bessel/Eq.7 targets: a few % (self-consistent density needed")
    print("  for the ~1e-5 precision claim).")
    N = 2000
    R_m = R_eq_target
    l = polytrope_grid(R_m, N)
    rho, m = polytrope_rho_m(l, R_m, M_target)
    j2n, _ = get_moments7(l, rho, m, omega_target)
    print(f"  {'':6s} {'Bessel':>14s} {'N21 Eq.7':>14s} {'OURS':>14s}"
          f" {'|diff/Bessel|':>14s}")
    for ii, label in enumerate([2, 4, 6, 8, 10]):
        J = j2n[ii]
        b, e = BESSEL[label], EQ7[label]
        rel = abs(J - b) / abs(b)
        print(f"  J_{label:<2d}  {b:14.5f} {e:14.5f} {J:14.5f} {rel:14.2e}")
    print(f"  R_eq   = {j2n[8]/1e5:.3f} km  (target {R_eq_target/1e5:.3f}; grid R_m = R_eq)")
    print(f"  oblat  = {j2n[7]:.5f}")
    print("  NOTE: this is an *internal sanity* check; tight Bessel agreement")
    print("  requires the self-consistent density outer loop (deferred).")


def main():
    print("TOF7 validation against Nettelmann+2021 (n=1 polytrope setup)\n")
    p1 = test_1_nonrotating()
    p2 = test_2_eq5_vs_eq7()
    p3 = test_3_tof4_vs_tof7()
    test_4_absolute_values()
    print()
    print("=" * 72)
    print("Summary: " + " ".join(
        [f"{name}: {'PASS' if ok else 'FAIL'}"
         for name, ok in [("nonrotating", p1), ("eq5_vs_eq7", p2),
                          ("tof4_vs_tof7", p3)]]
    ))
    print("=" * 72)
    if p1 and p2 and p3:
        print("All internal-consistency checks pass. The residual offset from")
        print("N21 Bessel in Test 4 is due to the non-self-consistent density")
        print("profile, not the TOF7 algorithm. Self-consistent polytrope loop")
        print("(rho <- sqrt(P/K) + hydrostatic integration) is the next step")
        print("for precision validation at the N21-advertised ~1e-5 level.")


if __name__ == "__main__":
    main()
