"""
Self-consistent rotating n=1 polytrope validation of utils/TOF7.py.

Outer loop (N21 Sect. 3.2):
    Given (GM, R_eq_target, q_rot):
    1. run TOF7 on current (l, rho, m)   ->  shape functions, R_eq
    2. compute U(l) = -(GM/R_m) * z^2 * A_0(z)                 (stable form)
    3. update rho(l) via polytropic P = K rho^2 in hydrostatic:
         rho(l) = rho_c * (U(R_m) - U(l)) / (U(R_m) - U(0))
    4. renormalize rho_c so total mass == M_target
    5. rescale grid so R_eq_computed == R_eq_target (preserves mass)
    6. recompute m(l) = 4*pi * int_0^l rho(l') l'^2 dl'
    7. iterate until max|delta rho/rho_c| < tol

A_0 includes a contribution from S'_0, whose standard form has a 1/z^2
singularity at z->0. We avoid this by computing
    (z^2 * S'_0)(z)  directly via integration by parts,
which is finite everywhere.  All other A_0 contributions are multiplied
by z^2 in the potential formula, so we work with
    tilde_A_0(z) := z^2 * A_0(z)
which is smooth at the center.

Expected targets (N21 Table 1, scaled by 1e-6):
    J_2 = 13988.54 (Eq.7),  J_4 = -531.83,  J_6 = 30.12,
    J_8 = -2.13,  J_10 = 0.174.
Bessel-exact (Wisdom & Hubbard 2016):
    J_2 = 13988.51,  J_4 = -531.83,  J_6 = 30.12,
    J_8 = -2.13,  J_10 = 0.174.
"""

import os
import sys

import numpy as np
from scipy.integrate import cumulative_trapezoid as cumtrapz

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _ROOT)

from utils import const  # noqa: E402
from utils.TOF7 import (  # noqa: E402
    _C7,
    _N_VALUES,
    _build_f,
    _build_Sn_Snp,
    _eval_block,
    get_moments7,
)


GM = 12.6686536e16 * 1e6
R_eq_target = 7.1492e9
q_rot = 0.089195487
M_target = GM / const.G
omega_target = np.sqrt(q_rot * GM / R_eq_target ** 3)

BESSEL = {2: 13988.51, 4: -531.8281, 6: 30.11832, 8: -2.13212, 10: 0.17407}
EQ7_N21 = {2: 13988.54, 4: -531.8292, 6: 30.11989, 8: -2.13384, 10: 0.17426}


# ---------------------------------------------------------------------------
# Grid / initial polytrope density
# ---------------------------------------------------------------------------
def polytrope_grid(R_m, N, split=0.95, frac_inner=0.5):
    eps = R_m / (2.0 * N)
    n_in = int(round(N * frac_inner))
    n_out = N - n_in
    l_in = np.linspace(eps, split * R_m, n_in, endpoint=False)
    l_out = np.linspace(split * R_m, R_m, n_out)
    return np.concatenate([l_in, l_out])


def initial_polytrope(l, R_m, M):
    rho_c = np.pi * M / (4.0 * R_m ** 3)
    z = l / R_m
    pz = np.pi * z
    sinc = np.where(z > 0, np.sin(pz) / np.where(z > 0, pz, 1.0), 1.0)
    rho = rho_c * sinc
    m = (4.0 * rho_c * R_m ** 3 / np.pi ** 2) * (np.sin(pz) - pz * np.cos(pz))
    return rho, m


# ---------------------------------------------------------------------------
# Compute z^2 S'_0 (stable at z->0) and tilde_A_0 = z^2 A_0
# ---------------------------------------------------------------------------
def _z2_Sp0(l, rho, m, fp0):
    """
    Compute z^2 * S'_0(z) everywhere on the grid, via
        z^2 S'_0 = rho(1)/rhobar * f'_0(1)
                  - z^2 * (rho/rhobar) * f'_0
                  - integral_z^1  z'^2 f'_0 d(rho/rhobar).

    Finite at z=0 because the first term is a constant and the other two vanish.
    """
    M_tot = m[-1]
    R_m = l[-1]
    z = l / R_m
    rhobar = 3.0 * M_tot / (4.0 * np.pi * R_m ** 3)

    integrand = (z ** 2) * fp0 / rhobar  # to be integrated over d rho
    cum = cumtrapz(integrand, x=rho, initial=0.0)
    int_z_to_1 = cum[-1] - cum
    surface = rho[-1] / rhobar * fp0[-1]
    return surface - (z ** 2) * rho / rhobar * fp0 - int_z_to_1


def tilde_A0(l, rho, m, shapes, m_rot):
    """
    tilde_A_0(z) = z^2 A_0(z), smooth at z=0.

    Builds the A_0 expression from _C7['A']['A0'] but treats the S'_0 term
    as (z^2 S'_0)(z) / z^2 cancelled against the outer z^2 factor. All other
    terms pick up a z^2 factor from tilde_A_0 = z^2 * A_0.
    """
    s0, s2, s4, s6, s8, s10, s12, s14 = shapes
    s_list = [s2, s4, s6, s8, s10, s12, s14]
    f, fp = _build_f(s_list)
    S, Sp = _build_Sn_Snp(l, rho, m, f, fp)

    R_m = l[-1]
    z = l / R_m
    A0 = _C7["A"]["A0"]

    # Regular S_n and S'_n contributions (no singularity) -- multiplied by z^2
    tildeA = np.zeros_like(z)
    for n in _N_VALUES:
        tildeA = tildeA + _eval_block(A0[f"S{n}"], s_list) * S[n] * (z ** 2)
    # S'_0 contribution: coefficient * (z^2 S'_0), with the 1/z^2 cancelled
    z2Sp0 = _z2_Sp0(l, rho, m, fp[0])
    tildeA = tildeA + _eval_block(A0["S0p"], s_list) * z2Sp0
    # Other S'_n (n >= 2) contributions: no singularity
    for n in _N_VALUES[1:]:
        tildeA = tildeA + _eval_block(A0[f"S{n}p"], s_list) * Sp[n] * (z ** 2)
    # Centrifugal (m_rot / 3) contribution: multiplied by z^2
    tildeA = tildeA + (m_rot / 3.0) * _eval_block(A0["m"], s_list) * (z ** 2)
    return tildeA


def U_of_l(l, rho, m, shapes, m_rot, GM):
    """Effective potential on level surfaces.  U(l) = -(GM/R_m) * tilde_A_0(z)."""
    R_m = l[-1]
    return -(GM / R_m) * tilde_A0(l, rho, m, shapes, m_rot)


# ---------------------------------------------------------------------------
# Outer self-consistent iteration
# ---------------------------------------------------------------------------
def self_consistent(
    N=2000,
    max_outer=100,
    tol_rho=1e-10,
    mu_nodes=64,
    verbose=True,
):
    R_m = 0.978 * R_eq_target  # reasonable first guess
    l = polytrope_grid(R_m, N)
    rho, m = initial_polytrope(l, R_m, M_target)
    shapes_prev = None

    j2n = None
    for it in range(max_outer):
        # 1. Run TOF7
        j2n, shapes, diag = get_moments7(
            l, rho, m, omega_target,
            initial_guess=shapes_prev,
            mu_nodes=mu_nodes,
            return_diagnostics=True,
        )
        R_eq = j2n[8]
        shapes_prev = tuple(sk.copy() for sk in shapes[1:])

        # 2. Effective potential
        m_rot = diag["m_rot"]
        U = U_of_l(l, rho, m, shapes, m_rot, GM)

        # 3. Polytropic density: rho = rho_c * (U(R_m) - U(l)) / (U(R_m) - U(0))
        U_surf = U[-1]
        U_ctr = U[0]
        rho_shape = (U_surf - U) / (U_surf - U_ctr)  # 1 at center, 0 at surface

        # 4. Renormalize rho_c to match total mass
        I_norm = 4.0 * np.pi * np.trapezoid(rho_shape * l ** 2, x=l)
        rho_c_new = M_target / I_norm
        rho_new = rho_c_new * rho_shape

        # 5. Rescale grid to match R_eq_target.  Mass labels m(l) are invariant
        #    under uniform radial rescaling (ρ scales as 1/alpha^3, volume as
        #    alpha^3, so M stays fixed), so we just scale l by alpha and rho by
        #    1/alpha^3.
        alpha = R_eq_target / R_eq
        l_next = l * alpha
        rho_next = rho_new / alpha ** 3

        # 6. Recompute m(l) cumulative-mass profile.
        #    Seed m[0] with the analytic mass of a sphere of density rho[0] and
        #    radius l[0] — not zero — so that S_0(l[0]) = m[0]/(M z[0]^3) stays
        #    finite instead of 0/0.
        m_next = cumtrapz(4.0 * np.pi * rho_next * l_next ** 2,
                          x=l_next, initial=0.0)
        m_next = m_next + (4.0 / 3.0) * np.pi * rho_next[0] * l_next[0] ** 3

        # 7. Convergence check on rho profile (normalized)
        delta_rho = np.max(np.abs(rho_next - rho) / max(np.max(rho_next), 1e-30))

        if verbose:
            print(
                f"  it {it:3d}  R_m={l_next[-1]/1e5:.4f} km  "
                f"R_eq={R_eq/1e5:.4f}  "
                f"J_2={j2n[0]:.4f}  J_4={j2n[1]:.4f}  "
                f"J_10={j2n[4]:.5f}  "
                f"d_rho={delta_rho:.2e}  tof_iter={diag['iterations']}"
            )

        l, rho, m = l_next, rho_next, m_next

        if delta_rho < tol_rho and it > 2:
            break

    return j2n, shapes, l, rho, m


def report(j2n, label="ours"):
    J2, J4, J6, J8, J10, J12, J14, oblat, Req, Rpol, I = j2n
    print()
    print(f"{label} vs N21 Table 1 (all J * 1e6):")
    print(f"  {'':6s} {'Bessel':>13s} {'N21 Eq.7':>13s} {'OURS':>13s} "
          f"{'|Δ/Bessel|':>12s} {'|Δ/Eq.7|':>12s}")
    ours = {2: J2, 4: J4, 6: J6, 8: J8, 10: J10}
    for k in (2, 4, 6, 8, 10):
        b = BESSEL[k]
        e = EQ7_N21[k]
        o = ours[k]
        print(f"  J_{k:<2d}  {b:13.5f} {e:13.5f} {o:13.5f}"
              f"   {abs(o-b)/abs(b):12.2e} {abs(o-e)/abs(e):12.2e}")
    print(f"  J_12 * 1e6 = {J12:.5f}")
    print(f"  J_14 * 1e6 = {J14:.5f}")
    print(f"  R_eq  = {Req/1e5:.4f} km (target {R_eq_target/1e5:.4f})")
    print(f"  R_pol = {Rpol/1e5:.4f} km")
    print(f"  oblat = {oblat:.6f}")
    print(f"  I/(M R_eq^2) = {I/(M_target*R_eq_target**2):.6f}")


def main():
    print("Self-consistent n=1 polytrope vs Nettelmann+2021 Table 1")
    print("=" * 72)
    print(f"  GM = {GM:.6e} cm^3/s^2,  M = {M_target:.6e} g")
    print(f"  R_eq_target = {R_eq_target/1e5:.3f} km,  q_rot = {q_rot:.9f}")
    print(f"  omega = {omega_target:.4e} rad/s  (P = {2*np.pi/omega_target/3600:.4f} hr)")
    print()
    print("Iterating (rho, R_m) until self-consistent...")
    j2n, shapes, l, rho, m = self_consistent(N=2000, mu_nodes=48)
    report(j2n, label="self-consistent polytrope")


if __name__ == "__main__":
    main()
