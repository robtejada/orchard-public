"""
Generate validation plots for utils/TOF7.py.

Produces four PNGs in utils/:
    tof7_convergence.png      — outer-loop convergence of J_2 … J_10
                                  vs Bessel targets
    tof7_accuracy.png         — final |Delta J / J_Bessel| bar chart
                                  comparing Ours, N21 Eq.7, N21 Eq.5
    tof7_shape_and_rho.png    — shape functions s_2k(z) + density rho(z)
    tof7_vs_tof4.png          — TOF.py (ToF4) vs TOF7 divergence on same
                                  density, vs predicted truncation floor

Run from repo root:
    python utils/TOF7/plot_validation.py
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import cumulative_trapezoid as cumtrapz

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)

from utils import const  # noqa: E402
from utils.TOF import get_moments as tof4_get_moments  # noqa: E402
from utils.TOF7 import get_moments7  # noqa: E402
from selfconsistent_polytrope import (  # noqa: E402
    GM,
    M_target,
    R_eq_target,
    initial_polytrope,
    omega_target,
    polytrope_grid,
    q_rot,
    tilde_A0,
)

OUT_DIR = os.path.join(_ROOT, "utils")

BESSEL = {2: 13988.51, 4: -531.8281, 6: 30.11832, 8: -2.13212, 10: 0.17407}
EQ7_N21 = {2: 13988.54, 4: -531.8292, 6: 30.11989, 8: -2.13384, 10: 0.17426}
EQ5_N21 = {2: 13988.55, 4: -531.8207, 6: 30.13506, 8: -2.10486, 10: 0.19555}


# ---------------------------------------------------------------------------
# Re-run self-consistent loop with history tracking
# ---------------------------------------------------------------------------
def run_selfconsistent_with_history(N=2000, max_outer=30, tol_rho=1e-10, mu_nodes=48):
    R_m = 0.978 * R_eq_target
    l = polytrope_grid(R_m, N)
    rho, m = initial_polytrope(l, R_m, M_target)
    shapes_prev = None
    history = []

    for it in range(max_outer):
        j2n, shapes, diag = get_moments7(
            l, rho, m, omega_target,
            initial_guess=shapes_prev,
            mu_nodes=mu_nodes,
            return_diagnostics=True,
        )
        R_eq = j2n[8]
        shapes_prev = tuple(sk.copy() for sk in shapes[1:])

        history.append({
            "iter": it,
            "J": [j2n[0], j2n[1], j2n[2], j2n[3], j2n[4], j2n[5], j2n[6]],
            "R_eq": R_eq,
            "R_m": l[-1],
            "oblat": j2n[7],
            "I_MoI": j2n[10],
            "l": l.copy(),
            "rho": rho.copy(),
            "shapes": tuple(sk.copy() for sk in shapes),
            "tof_iter": diag["iterations"],
        })

        # Update rho via polytropic relation in hydrostatic equilibrium
        m_rot = diag["m_rot"]
        U = -(GM / l[-1]) * tilde_A0(l, rho, m, shapes, m_rot)
        rho_shape = (U[-1] - U) / (U[-1] - U[0])
        I_norm = 4.0 * np.pi * np.trapezoid(rho_shape * l ** 2, x=l)
        rho_c_new = M_target / I_norm
        rho_new = rho_c_new * rho_shape

        alpha = R_eq_target / R_eq
        l_next = l * alpha
        rho_next = rho_new / alpha ** 3
        m_next = cumtrapz(4.0 * np.pi * rho_next * l_next ** 2, x=l_next, initial=0.0)
        m_next += (4.0 / 3.0) * np.pi * rho_next[0] * l_next[0] ** 3

        delta_rho = np.max(np.abs(rho_next - rho) / max(np.max(rho_next), 1e-30))
        l, rho, m = l_next, rho_next, m_next

        if delta_rho < tol_rho and it > 2:
            break

    return history


# ---------------------------------------------------------------------------
# Plot 1: outer-loop convergence of J_2 through J_10
# ---------------------------------------------------------------------------
def plot_convergence(history):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    axes = axes.ravel()

    iters = np.arange(len(history))
    J_arr = np.array([h["J"] for h in history])  # shape (n_it, 7)

    for idx, harm in enumerate([2, 4, 6, 8, 10]):
        ax = axes[idx]
        ours = J_arr[:, idx]
        bessel = BESSEL[harm]
        eq7 = EQ7_N21[harm]
        ax.axhline(bessel, color="C0", linestyle="-", lw=1.2, label="Bessel exact")
        ax.axhline(eq7, color="C1", linestyle="--", lw=1.2, label="N21 Eq. 7")
        ax.plot(iters, ours, "o-", color="C3", ms=5, lw=1.3, label="Ours")
        ax.set_xlabel("outer iteration")
        ax.set_ylabel(f"$J_{{{harm}}} \\times 10^6$")
        ax.set_title(f"$J_{{{harm}}}$ convergence")
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)

    # Panel 6: R_eq vs target (absolute error in cm)
    ax = axes[5]
    R_eq_hist = np.array([h["R_eq"] for h in history]) - R_eq_target
    ax.semilogy(iters, np.abs(R_eq_hist), "o-", color="C2")
    ax.axhline(1.0, color="gray", ls=":", lw=1, label="1 cm")
    ax.set_xlabel("outer iteration")
    ax.set_ylabel(r"$|R_\mathrm{eq} - R_\mathrm{eq,target}|$ [cm]")
    ax.set_title("Geometry convergence")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)

    fig.suptitle(
        "Self-consistent n=1 polytrope (Jupiter-q), N21 Sect. 3.2 setup\n"
        f"q_rot = {q_rot:.6f},  target R_eq = 71492 km,  target M = {M_target:.3e} g",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = os.path.join(OUT_DIR, "tof7_convergence.png")
    fig.savefig(out, dpi=150)
    print(f"  wrote {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 2: final relative error bar chart
# ---------------------------------------------------------------------------
def plot_accuracy(history):
    final = history[-1]["J"]
    ks = [2, 4, 6, 8, 10]
    ours_err = [abs(final[i] - BESSEL[k]) / abs(BESSEL[k]) for i, k in enumerate(ks)]
    n21_eq7_err = [abs(EQ7_N21[k] - BESSEL[k]) / abs(BESSEL[k]) for k in ks]
    n21_eq5_err = [abs(EQ5_N21[k] - BESSEL[k]) / abs(BESSEL[k]) for k in ks]

    fig, ax = plt.subplots(figsize=(8.5, 5))
    x = np.arange(len(ks))
    width = 0.27
    ax.bar(x - width, ours_err, width, label="Ours (ToF7 + Eq. 7)", color="C3")
    ax.bar(x, n21_eq7_err, width, label="N21 Eq. 7", color="C1")
    ax.bar(x + width, n21_eq5_err, width, label="N21 Eq. 5", color="C0")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"$J_{{{k}}}$" for k in ks])
    ax.set_ylabel(r"$|\Delta J_{2n} / J_{2n}^\mathrm{Bessel}|$")
    ax.set_title("Accuracy vs Wisdom-Hubbard Bessel exact solution (N=2000)")
    ax.legend()
    ax.grid(True, which="both", axis="y", alpha=0.3)

    # annotate ours above each bar
    for i, (k, v) in enumerate(zip(ks, ours_err)):
        ax.text(
            x[i] - width, v * 1.2, f"{v:.1e}",
            ha="center", va="bottom", fontsize=8, color="C3",
        )

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "tof7_accuracy.png")
    fig.savefig(out, dpi=150)
    print(f"  wrote {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 3: shape functions s_2k(z) and density profile
# ---------------------------------------------------------------------------
def plot_shape_and_rho(history):
    last = history[-1]
    l = last["l"]
    rho = last["rho"]
    s0, s2, s4, s6, s8, s10, s12, s14 = last["shapes"]
    z = l / l[-1]
    rho_c = rho[0]

    # Non-rotating polytrope for comparison
    nr_rho, _ = initial_polytrope(l, l[-1], M_target)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: shape functions
    ax = axes[0]
    for label, sk, color in [
        ("$s_0$", s0, "k"),
        ("$s_2$", s2, "C0"),
        ("$s_4$", s4, "C1"),
        ("$s_6$", s6, "C2"),
        ("$s_8$", s8, "C3"),
        ("$s_{10}$", s10, "C4"),
        ("$s_{12}$", s12, "C5"),
        ("$s_{14}$", s14, "C6"),
    ]:
        ax.plot(z, np.abs(sk), label=label, color=color, lw=1.4)
    ax.set_yscale("log")
    ax.set_xlabel(r"$z = l / R_m$")
    ax.set_ylabel(r"$|s_{2k}(z)|$")
    ax.set_title("Converged figure functions (magnitude)")
    ax.legend(ncol=2, fontsize=9, loc="lower left")
    ax.grid(True, which="both", alpha=0.3)
    ax.set_ylim(1e-18, 1e0)

    # Right: density profile (ours vs non-rotating reference)
    ax = axes[1]
    ax.plot(z, rho / rho_c, "C3-", lw=1.6, label="Self-consistent (rotating)")
    ax.plot(z, nr_rho / nr_rho[0], "C0--", lw=1.3, label="Non-rotating polytrope")
    ax.set_xlabel(r"$z = l / R_m$")
    ax.set_ylabel(r"$\rho(z) / \rho_c$")
    ax.set_title("Density profile")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "tof7_shape_and_rho.png")
    fig.savefig(out, dpi=150)
    print(f"  wrote {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 4: TOF4 vs TOF7 comparison on same non-self-consistent density
# ---------------------------------------------------------------------------
def plot_tof4_vs_tof7():
    N = 1000
    R_m = R_eq_target
    l = polytrope_grid(R_m, N)
    rho, m = initial_polytrope(l, R_m, M_target)
    j2n4, _ = tof4_get_moments(l, rho, m, omega_target)
    j2n7, _ = get_moments7(l, rho, m, omega_target)

    ks = [2, 4, 6, 8]
    # TOF.py j2n layout: [J2, J4, J6, J8, oblat, ...]
    # TOF7    j2n layout: [J2, J4, J6, J8, J10, J12, J14, oblat, ...]
    rels = []
    for i, k in enumerate(ks):
        a = j2n4[i]
        b = j2n7[i]
        rels.append(abs(a - b) / max(abs(b), 1e-30))

    # Predicted ToF4 truncation: J_{2k} leading-order scales as m_rot^k; ToF at
    # order O captures up to m_rot^O, so relative truncation is m_rot^{O+1-k}.
    # For ToF4 (O=4) with q_rot ~ m_rot:  q^4, q^3, q^2, q^1 for J_2..J_8.
    q = q_rot
    # ks = [2, 4, 6, 8]  ->  index k_idx = 1..4
    predicted = [q ** (5 - k_idx) for k_idx in (1, 2, 3, 4)]

    fig, ax = plt.subplots(figsize=(7.5, 5))
    x = np.arange(len(ks))
    width = 0.38
    ax.bar(x - width / 2, rels, width,
           label=r"Observed $|J^\mathrm{TOF4} - J^\mathrm{TOF7}| / |J^\mathrm{TOF7}|$",
           color="C3")
    ax.bar(x + width / 2, predicted, width,
           label=r"Predicted $m_\mathrm{rot}^{\,O+1-k}$ truncation (O=4)",
           color="C0")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"$J_{{{k}}}$" for k in ks])
    ax.set_ylabel("relative difference")
    ax.set_title("ToF4 (utils/TOF.py) vs ToF7 on identical density input\n"
                 r"diverges at predicted truncation floor $\sim m_\mathrm{rot}^{\,5-k}$")
    ax.legend()
    ax.grid(True, which="both", axis="y", alpha=0.3)
    # annotate observed numbers
    for i, v in enumerate(rels):
        ax.text(x[i] - width / 2, v * 1.3, f"{v:.1e}",
                ha="center", va="bottom", fontsize=8, color="C3")
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "tof7_vs_tof4.png")
    fig.savefig(out, dpi=150)
    print(f"  wrote {out}")
    plt.close(fig)


def main():
    print("Running self-consistent loop to collect history ...")
    history = run_selfconsistent_with_history(N=2000, max_outer=30)
    print(f"  converged in {len(history)} outer iterations")
    print(f"  final J_2 * 1e6 = {history[-1]['J'][0]:.4f}  (Bessel: {BESSEL[2]})")

    print("Writing plots to utils/ ...")
    plot_convergence(history)
    plot_accuracy(history)
    plot_shape_and_rho(history)
    plot_tof4_vs_tof7()
    print("done.")


if __name__ == "__main__":
    main()
