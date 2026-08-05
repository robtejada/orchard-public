#!/usr/bin/env python3
"""
Step 2 worker - reproduce ONE MH24 model through ORCHARD's hydrostatic.py + ToF7.

Run as a fresh subprocess (initial.py reads core params at import time):
    ORCHARD_LIBRARY_MODE=1 python run_one_model.py <out-id> [mantle_comp]

Procedure
---------
1. Load default config (1 Mjup), override: mass_core (per model), mantle_comp,
   rotation/tof_calc/tof_order=7/period/p_atm.  THEN import initial/hydrostatic.
2. initialize_grid -> take only the MASS GRID (m_b, kcore, kcore_fe); the default
   hot-start structure is discarded.
3. Inject MH24 composition Y(m), Z(m) interpolated onto ORCHARD's mass grid.
4. Derive ORCHARD entropy S(m) from MH24 (P,T,Y,Z) via mixtures_eos.get_s_pt
   (envelope) and mantle_eos inversion (compact core).
5. Use MH24's own (r, P, T) as the HSE initial guess (keeps q<1, fast convergence).
6. hydrostatic_equilibrium(init=True) -> structure + ToF7 j2n.
7. Compare J2..J8 to MH24 CMS J_Int; dump JSON to results/hse_<id>.json.
"""
import os
os.environ.setdefault("ORCHARD_LIBRARY_MODE", "1")
import sys
import json

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import mh24_io  # noqa: E402


def main():
    mid = sys.argv[1]
    mantle_comp = sys.argv[2] if len(sys.argv) > 2 else "mgsio3"
    meta = mh24_io.MODELS[mid]
    core = mh24_io.measure_core(mid)
    mcore_Me = core["mcore_mearth"]      # from data (matches Table 1)

    # ---- config overrides (BEFORE importing initial/hydrostatic) ----
    from utils import common
    c = common.config
    from utils import const
    c["initial"]["M_Mearth"] = f"{const.M_jup / const.mearth:.10g}"   # exactly 1 Mjup
    c["core"]["mass_core"] = f"{mcore_Me:.10g}"
    c["core"]["mass_core_fe"] = "0.0"
    c["core"]["mantle_comp"] = mantle_comp
    # Z EOS: aqua (pure water, default) or ice_mixture (H2O:CH4:NH3, closer to MH24).
    z_eos = os.environ.get("Z_EOS", "aqua")
    c["equation_of_state"]["z_eos"] = z_eos
    # H-He EOS: cd (Chabrier&Debras 2021, default) or cms (Chabrier+2019 + HG23).
    hhe_eos = os.environ.get("HHE_EOS", "cd")
    c["equation_of_state"]["hhe_eos"] = hhe_eos
    # rotation=True couples a spherically-averaged centrifugal term + ToF figure.
    # omega = (2pi/period)*min(C_MoI*M*R_p^2 / I_structure, 1), so a single
    # init=True call seeds omega off the (cold) initial guess and over-inflates;
    # the figure<->structure<->omega loop must be ITERATED to self-consistency.
    c["hydrostatic_equilibrium"]["rotation"] = "True"
    c["hydrostatic_equilibrium"]["tof_calc"] = "True"
    c["hydrostatic_equilibrium"]["tof_order"] = "7"
    c["hydrostatic_equilibrium"]["period"] = "35730"
    c["hydrostatic_equilibrium"]["p_atm"] = "1.0"

    from initial import initialize_grid, mixtures_eos, mantle_eos  # noqa: E402
    from hydrostatic import hydrostatic_equilibrium                # noqa: E402

    N = int(c["general"]["N"])
    M_planet = const.M_jup

    # ---- 2. mass grid only ----
    (_r0, _p0, m_b, dm, _S0, _Y0, _Z0, f_rock, kcore, kcore_fe) = initialize_grid(
        N, M_planet,
        float(c["initial"]["S_ini"]), float(c["initial"]["Y_ini"]),
        float(c["initial"]["Z_ini"]), 0.0)
    kcore = N if int(kcore) < 0 else int(kcore)
    kcore_fe = N if int(kcore_fe) < 0 else int(kcore_fe)
    f_rock = np.zeros(N)

    # ---- MH24 profiles vs enclosed-mass fraction (ascending for np.interp) ----
    L = mh24_io.load_layers(mid)
    mf = L["CMass"]                         # 1 at surface -> ~0 center
    o = np.argsort(mf)
    def imh(col_or_arr):
        arr = L[col_or_arr] if isinstance(col_or_arr, str) else col_or_arr
        return np.interp(mf_cell, mf[o], arr[o])

    m_cell = 0.5 * (m_b[:-1] + m_b[1:])     # cell-center mass (g), descending
    mf_cell = m_cell / M_planet
    mf_b = m_b / M_planet

    Y = imh("Y"); Z = imh("Z")
    P_cell = imh("P") * 1e10                # GPa -> dyn/cm^2
    T_cell = imh("T")
    r_b_guess = np.interp(mf_b, mf[o], (L["rS"] * const.rj_eq)[o])  # volumetric radius

    # ---- entropy: envelope from H-He-Z EOS; core from mantle EOS ----
    S = np.zeros(N)
    env = slice(0, kcore)
    S[env] = mixtures_eos.get_s_pt(np.log10(P_cell[env]), np.log10(T_cell[env]),
                                   Y[env], Z[env], 0.0) * const.erg_to_kbbar
    if kcore < N:
        # compact core: invert mantle T(S,P) for S that reproduces MH24 core T
        from scipy.optimize import brentq
        Pc_GPa = P_cell[kcore:] * mantle_eos.dyn_to_GPa
        Tc = T_cell[kcore:]
        S_core = np.empty(N - kcore)
        for i, (Pg, Tg) in enumerate(zip(Pc_GPa, Tc)):
            f = lambda s: float(mantle_eos.get_t_sp(np.array([s]), np.array([Pg]))[0]) - Tg
            try:
                S_core[i] = brentq(f, 1e-3, 5.0, xtol=1e-6)
            except Exception:
                S_core[i] = float(c["core"]["mantle_entropy"])
        S[kcore:] = S_core
        Y[kcore:] = 0.0; Z[kcore:] = 0.0

    # ---- HSE solve, ITERATED to self-consistency (rotation<->figure<->omega) ----
    log_r, log_p, logt = np.log(r_b_guess), np.log(P_cell), np.log10(T_cell)
    omega_prev, init = 0.0, True
    R_eq_prev = np.inf
    hist = []
    for it in range(12):
        out = hydrostatic_equilibrium(
            S, log_r, log_p, m_b, Y, Z, f_rock, kcore, kcore_fe,
            omega_prev, 0.0, logt_old=logt, init=init)
        r_b, p, rho, temp, S, g, omega, j2n, shapes, niter = out
        R_eq = j2n[5]
        hist.append(dict(it=it, omega=float(omega), R_eq_km=R_eq / 1e5,
                         J2=float(j2n[0])))
        if abs(R_eq - R_eq_prev) / R_eq < 1e-6 and it >= 2:
            break
        R_eq_prev = R_eq
        log_r, log_p, logt = np.log(r_b), np.log(p), np.log10(temp)
        omega_prev, init = omega, False
    j2, j4, j6, j8, obl, R_eq, R_pol, MoI = j2n

    H = mh24_io.load_harmonics(mid)
    cms = {n: H[n]["int"] * 1e6 for n in (2, 4, 6, 8)}
    res = dict(
        mid=mid, label=meta["label"], mcore_Me=mcore_Me, mantle_comp=mantle_comp,
        z_eos=z_eos, hhe_eos=hhe_eos,
        niter=int(niter), n_rot_iter=len(hist), rot_hist=hist,
        R_eq_km=R_eq / 1e5, R_pol_km=R_pol / 1e5, oblat=obl,
        tof7={2: j2, 4: j4, 6: j6, 8: j8}, cms=cms,
        # structure comparison vs MH24 (interp to same mass grid)
        rho_or=rho.tolist(), T_or=temp.tolist(),
        rho_mh=imh("rho").tolist(), T_mh=T_cell.tolist(),
        mf_cell=mf_cell.tolist(), kcore=kcore,
    )
    suffix = "" if (z_eos == "aqua" and hhe_eos == "cd") else f"_{hhe_eos}_{z_eos}"
    out_path = os.path.join(HERE, "results", f"hse_{mid}{suffix}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(res, fh)
    # concise stdout summary
    print(f"{mid} {meta['label']:<26} rot_iters={len(hist):>2} "
          f"R_eq={R_eq/1e5:9.2f}km (target 71492) obl={obl:.5f}")
    for n in (2, 4, 6, 8):
        t = res["tof7"][n]; cv = cms[n]
        print(f"    J{n}: ORCHARD={t:14.4f}  CMS={cv:14.4f}  "
              f"rel={ (t-cv)/cv:+.3e}")
    print(f"    saved -> {out_path}")


if __name__ == "__main__":
    main()
