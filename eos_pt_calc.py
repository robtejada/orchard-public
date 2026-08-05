"""
Compute H-He-Z equation-of-state tables on a (P, T) grid.

This script builds 4-D look-up tables of entropy S(P,T,Y',Z),
density log10(rho)(P,T,Y',Z), and internal energy log10(u)(P,T,Y',Z)
by evaluating the volume-addition-law mixing functions in
``eos.eos_class.mixtures`` at every grid point.

The four independent axes are:

    logpvals  -- log10(P / dyn cm-2), default 6.0 to 14.0 in steps of 0.05
    logtvals  -- log10(T / K),        default 2.1 to 4.30 in steps of 0.05
    yvals     -- Y' = Y/(1-Z)  (He mass fraction in the H-He sub-mixture),
                 default 0.05 to 0.95 in steps of 0.05
    zvals     -- Z  (total heavy-element mass fraction),
                 default 0.0 to 1.0 in steps of 0.01

The Z component can be one of several EOS models:

    mixture       -- water + rock + iron  (zmix_eos1, zmix_eos2, zmix_eos3)
    ice_mixture   -- CH4 + NH3 + H2O ices, mixed with the CNO number ratio
                     given by --cno_ratio (e.g. '111' = 1:1:1, '417' = 4:1:7)
    aqua / ppv2 / iron2 / ...  -- single-component Z

For ``ice_mixture``, the --cno_ratio flag sets the C:N:O number ratio.
This ratio is converted to mass fractions via ``number_to_mass_fraction()``
and then to the (zm, za) parameterisation expected by ``ice_eos``:

    za = Z_NH3                          (ammonia fraction of total Z)
    zm = Z_CH4 / (1 - Z_NH3)           (methane fraction of non-ammonia Z)
    water fraction = (1-zm)*(1-za) = Z_H2O   (remainder)

These are held constant across all Z values; the ``zvals`` axis controls
how much total Z is mixed into the H-He background.

Usage
-----
    python eos_pt_calc.py --hhe_eos cd --z_eos ice_mixture --cno_ratio 111
    python eos_pt_calc.py --hhe_eos cd --z_eos mixture --zmix_eos1 aqua --zmix_eos2 mgsio3_l
    python eos_pt_calc.py --hhe_eos cd --z_eos aqua --smooth True

Command-line arguments
----------------------
    --hhe_eos    : H-He EOS model ('cd' or 'cms').  Default: 'cd'.
    --z_eos      : Z EOS model ('mixture', 'ice_mixture', 'aqua', 'ppv2', ...).
                   Default: 'mixture'.
    --zmix_eos1  : first component in the Z mixture.   Default: 'aqua'.
    --zmix_eos2  : second component in the Z mixture.  Default: 'mgsio3_l'.
    --zmix_eos3  : third component in the Z mixture.   Default: 'iron'.
    --f_ppv      : rock mass fraction within Z (for 'mixture' or 'ice_rock').
                   Default: 0.0.
    --f_fe       : iron mass fraction within Z.  Default: 0.0.
    --hg         : include Howard & Guillot (2023a) non-ideal H-He corrections
                   (only for cms).  Default: False.
    --y_prime    : treat the Y axis as Y' = Y/(1-Z).  Default: True.
    --smooth     : apply Gaussian smoothing to the output tables.
                   Default: False.
    --cno_ratio  : C:N:O number ratio as 3 concatenated digits (e.g. '111',
                   '417').  Only used when z_eos = 'ice_mixture'.
                   Default: '417' (solar-like 4:1:7 C:N:O ratio).
    --custom_range : when True, interactively prompt for min, max, and step
                   of each grid axis (logP, logT, Y', Z) instead of using
                   the hard-coded defaults.  Default: False.

Output
------
A single compressed NumPy archive (.npz) saved to

    eos/<hhe_eos>/<filename>_pt.npz

where <filename> encodes the EOS choices.  Examples:

    eos/cd/cd_1.0_0.0_aqua_mgsio3_l_pt.npz     (mixture, pure water+rock)
    eos/cd/cd_ice_mixture_cno111_aqua_pt.npz     (ice_mixture, 1:1:1)
    eos/cd/cd_1.0_0.0_aqua_aqua_pt.npz          (single-component aqua)

The archive contains the following arrays:

    logpvals   -- 1-D, shape (Np,)
    logtvals   -- 1-D, shape (Nt,)
    yvals      -- 1-D, shape (Ny,)
    zvals      -- 1-D, shape (Nz,)
    s_pt       -- 4-D, shape (Np, Nt, Ny, Nz),  entropy        [erg g-1 K-1]
    logrho_pt  -- 4-D, shape (Np, Nt, Ny, Nz),  log10(density) [log10 g cm-3]
    logu_pt    -- 4-D, shape (Np, Nt, Ny, Nz),  log10(energy)  [log10 erg g-1]
"""

import argparse
from tqdm import tqdm
import numpy as np
from scipy.ndimage import gaussian_filter1d, gaussian_filter
import os

from eos import eos_class

# molar masses in g mol-1  (CODATA 2018 / CRC Handbook)
MOLAR_MASS = {
    "CH4": 16.04,        # methane
    "NH3": 17.03052,     # ammonia
    "H2O": 18.01528,     # water
}

def number_to_mass_fraction(n_CH4, n_NH3, n_H2O,
                            mw=MOLAR_MASS, return_as="array"):
    """
    Convert an arbitrary CH4 : NH3 : H2O number ratio to mass fractions.

    Parameters
    ----------
    n_CH4, n_NH3, n_H2O : int | float
        The number of molecules (or moles) in the mixture.  Only the ratios matter,
        so 4:1:2 is identical to 40:10:20.
    mw : mapping, optional
        Molecular-weight lookup table in g mol-1.  Override if you need custom values.
    return_as : {"dict", "array"}, optional
        - "dict"  → `{"H2O": x_H2O, "CH4": x_CH4, "NH3": x_NH3}`
        - "array" → NumPy array in the order (x_H2O, x_CH4, x_NH3)

    Returns
    -------
    dict | np.ndarray
        Mass fractions that sum to unity.
    """

    # masses for each component
    m_CH4 = n_CH4 * mw["CH4"]
    m_NH3 = n_NH3 * mw["NH3"]
    m_H2O = n_H2O * mw["H2O"]

    total_mass = m_CH4 + m_NH3 + m_H2O
    if total_mass == 0:
        raise ValueError("At least one of the input numbers must be non-zero.")

    Z_CH4 = m_CH4 / total_mass
    Z_NH3 = m_NH3 / total_mass
    Z_H2O = m_H2O / total_mass

    if return_as == "array":
        return np.array([Z_CH4, Z_NH3, Z_H2O])
    return {"CH4": Z_CH4, "NH3": Z_NH3, "H2O": Z_H2O}

def str2bool(v):
    """
    Convert a string to a boolean:
      'true', '1', 't', 'yes', 'y'  => True
      'false', '0', 'f', 'no', 'n'  => False
    """
    if isinstance(v, bool):
        return v
    v_lower = v.lower()
    if v_lower in ('true', '1', 't', 'yes', 'y'):
        return True
    elif v_lower in ('false', '0', 'f', 'no', 'n'):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected for --hg or --y_prime.")

def prompt_grid_ranges():
    """
    Interactively prompt the user for custom grid ranges.

    For each of the four axes (log10 P, log10 T, Y', Z), the user is asked
    for the minimum, maximum, and step size.  The resulting arrays are built
    with ``np.arange(min, max + step/2, step)`` so that the endpoint is
    always included (guarding against floating-point round-off).

    Returns
    -------
    logpvals, logtvals, yvals, zvals : np.ndarray
        The four 1-D grid arrays.
    """
    print("\n--- Custom grid range setup ---")
    print("For each axis, enter: min, max, step")
    print("(The endpoint is always included.)")
    print()
    print("Reference points for log10(P / dyn cm-2):")
    print("    6  = 1 bar,  10 = 10 kbar,  12 = 1 Mbar,  14 = 100 Mbar")
    print("Reference points for log10(T / K):")
    print("    2  = 100 K,  3 = 1000 K,  4 = 10,000 K")
    print()

    def _ask(label, unit_hint, default_min, default_max, default_step):
        """Prompt for one axis and return an np.arange array."""
        prompt = (f"  {label} ({unit_hint})  [default: {default_min}, {default_max}, {default_step}]\n"
                  f"    min, max, step (or press Enter for defaults): ")
        ans = input(prompt).strip()
        if ans == "":
            lo, hi, step = default_min, default_max, default_step
        else:
            parts = [s.strip() for s in ans.replace(",", " ").split()]
            if len(parts) != 3:
                raise ValueError(f"Expected 3 numbers for {label}, got {len(parts)}.")
            lo, hi, step = float(parts[0]), float(parts[1]), float(parts[2])
        arr = np.arange(lo, hi + step / 2, step)
        print(f"    -> {label}: {arr[0]:.4g} to {arr[-1]:.4g}, "
              f"{len(arr)} points, step {step}")
        return arr

    logpvals = _ask("log10(P)", "log10 dyn/cm2; 6=1bar, 12=1Mbar", 6.0, 14.0, 0.05)
    logtvals = _ask("log10(T)", "log10 K; 3=1000K, 4=10000K",       2.1,  4.30, 0.05)
    yvals    = _ask("Y'",       "He mass fraction in H-He, 0 to 1", 0.05, 0.95, 0.05)
    zvals    = _ask("Z",        "total metal mass fraction, 0 to 1", 0.0,  1.0,  0.01)

    total_pts = len(logpvals) * len(logtvals) * len(yvals) * len(zvals)
    print(f"\nTotal grid points: {total_pts:,}\n")

    return logpvals, logtvals, yvals, zvals

def main():
    """Parse command-line arguments, build the (P,T) EOS grid, and save to disk."""
    parser = argparse.ArgumentParser(
        description="Compute H-He-Z EOS tables on a (P, T, Y', Z) grid using the volume addition law."
    )
    parser.add_argument(
        "--hhe_eos",
        type=str,
        default="cd",
        help="Choice of H-He EOS (default: cd)."
    )
    parser.add_argument(
        "--z_eos",
        type=str,
        default="mixture",
        help="Choice of Z EOS (default: mixture)."
    )
    parser.add_argument(
        "--zmix_eos1",
        type=str,
        default="aqua",
        help="Choice of Z1 EOS (default: aqua)."
    )
    parser.add_argument(
        "--zmix_eos2",
        type=str,
        default="mgsio3_l",
        help="Choice of Z2 EOS (default: mgsio3_l)."
    )

    parser.add_argument(
        "--zmix_eos3",
        type=str,
        default="iron",
        help="Choice of Z3 EOS (default: iron)."
    )
    parser.add_argument(
        "--f_ppv",
        type=float,
        default=0.0,
        help="Choice of rock fraction (default: 0.0)."
    )
    parser.add_argument(
        "--f_fe",
        type=float,
        default=0.0,
        help="Choice of iron fraction (default: 0.0)."
    )
    parser.add_argument(
        "--hg",
        type=str2bool,
        default=False,
        help="Use hg? (default: True; pass 'False' to disable. Only for cms H-He EOS)."
    )
    parser.add_argument(
        "--y_prime",
        type=str2bool,
        default=True,
        help="Use y_prime? (default: True; pass 'False' to disable)."
    )

    parser.add_argument(
        "--smooth",
        type=str2bool,
        default=False,
        help="Whether to apply gaussian smoothing to EOS tables."
    )

    parser.add_argument(
        "--cno_ratio",
        type=str,
        default="417",
        help="C:N:O number ratio as concatenated digits, e.g. '111' for 1:1:1, '417' for 4:1:7. Only used when z_eos='ice_mixture'."
    )

    parser.add_argument(
        "--custom_range",
        type=str2bool,
        default=False,
        help="Prompt for custom grid ranges (default: False). When True, the user is "
             "asked interactively for the min, max, and step of each axis."
    )

    args = parser.parse_args()

    if args.z_eos == 'mixture':
        mix = eos_class.mixtures(
            hhe_eos=args.hhe_eos,
            z_eos=args.z_eos,
            zmix_eos1=args.zmix_eos1,
            zmix_eos2=args.zmix_eos2,
            zmix_eos3=args.zmix_eos3,
            # f_ppv=args.f_ppv,
            # f_fe=args.f_fe,
            hg=args.hg,
            y_prime=args.y_prime,
            new_z_mix=True
        )

    else:
        mix = eos_class.mixtures(
            hhe_eos=args.hhe_eos,
            z_eos=args.z_eos,
            zmix_eos1=args.zmix_eos1,
            zmix_eos2=args.zmix_eos2,
            zmix_eos3=args.zmix_eos3,
            hg=args.hg,
            y_prime=args.y_prime,
            new_z_mix=True
        )

        # mix = eos_class.mixtures(hhe_eos='cd', z_eos='mixture', f_ppv=0.0, hg=False, y_prime=True, new_z_mix=True)
    print('Calculating PT table for: hhe={}, z_eos={}, z_eos1 = {}, f_ppv = {}'.format(args.hhe_eos, args.z_eos, args.zmix_eos1, args.f_ppv))

    if args.custom_range:
        logpvals, logtvals, yvals, zvals = prompt_grid_ranges()
    else:
        logpvals = np.arange(6, 14.05, 0.05) # between 1 bar and 100 Mbar in steps of 0.05 cgs dex
        if args.z_eos == 'ice_mixture' or args.z_eos == 'ice_rock':
            logtvals = np.arange(2.1, 4.35, 0.05) # between 125 K and 22,400 K in steps of 0.05 dex
        else:
            logtvals = np.arange(2.1, 5.0, 0.05) # between 125 K and 100,000 K in steps of 0.05 dex
        yvals = np.arange(0.05, 1.0, 0.05) # between 0.05 and 0.95 in steps of 0.05 (Y' = He mass fraction in H-He sub-mixture)
        zvals = np.arange(0, 1.01, 0.01) # between 0 and 1 in steps of 0.01 (Z = total metal mass fraction)

    zr_arr = np.full_like(zvals, args.f_ppv) # for rock mixtures
    zfe_arr = np.zeros(len(zvals))

    if args.z_eos == 'ice_mixture':
        # Parse CNO ratio digits (e.g. '111' -> 1:1:1, '417' -> 4:1:7)
        cno = args.cno_ratio
        if len(cno) != 3:
            raise ValueError("--cno_ratio must be exactly 3 digits, e.g. '111' or '417'.")
        n_C, n_N, n_O = int(cno[0]), int(cno[1]), int(cno[2])
        mass_fracs = number_to_mass_fraction(n_C, n_N, n_O)  # [Z_CH4, Z_NH3, Z_H2O]
        Z_CH4, Z_NH3, Z_H2O = mass_fracs[0], mass_fracs[1], mass_fracs[2]
        # ice_eos expects _zm = methane fraction of non-ammonia Z, _za = ammonia fraction of total Z
        za_val = Z_NH3
        zm_val = Z_CH4 / (1.0 - Z_NH3) if Z_NH3 < 1.0 else 0.0
        zm_arr = np.full_like(zvals, zm_val)
        za_arr = np.full_like(zvals, za_val)
        print(f'Ice mixture CNO ratio: {n_C}:{n_N}:{n_O}')
        print(f'  Mass fractions -> CH4: {Z_CH4:.4f}, NH3: {Z_NH3:.4f}, H2O: {Z_H2O:.4f}')
        print(f'  ice_eos params -> zm: {zm_val:.4f}, za: {za_val:.4f}')
    else:
        zm_arr = np.zeros(len(zvals))
        za_arr = np.zeros(len(zvals))

    if args.smooth:
        print('Applying gaussian smoothing to EOS tables...')

    s_res = []
    rho_res = []
    u_res = []

    for lgp in tqdm(logpvals):
        logp_arr = np.full_like(zvals, lgp)
        s_res_p = []
        rho_res_p = []
        u_res_p = []
        for lgt in logtvals:
            logt_arr = np.full_like(zvals, lgt)
            s_res_t = []
            rho_res_t = []
            u_res_t = []
            for y in yvals:
                y_arr = np.full_like(zvals, y)

                s = mix.get_s_pt_val(logp_arr, logt_arr, y_arr, zvals, _zm=zm_arr, _za=za_arr, _zr=zr_arr, _zfe=zfe_arr)
                logrho = mix.get_logrho_pt_val(logp_arr, logt_arr, y_arr, zvals, _zm=zm_arr, _za=za_arr, _zr=zr_arr, _zfe=zfe_arr)
                logu = np.log10(mix.get_u_pt_val(logp_arr, logt_arr, y_arr, zvals, _zm=zm_arr, _za=za_arr, _zr=zr_arr, _zfe=zfe_arr))

                if args.smooth:
                    s_noglitch = mix.return_noglitch(zvals, s)
                    logrho_noglitch = mix.return_noglitch(zvals, logrho)
                    logu_noglitch = mix.return_noglitch(zvals, logu)

                    # s_smooth = gaussian_filter1d(s, sigma=2.5, mode='reflect')
                    # logrho_smooth = gaussian_filter1d(logrho, sigma=2.5, mode='reflect')
                    # logu_smooth = gaussian_filter1d(logu, sigma=2.5, mode='reflect')

                    s_res_t.append(s_noglitch)
                    rho_res_t.append(logrho_noglitch)
                    u_res_t.append(logu_noglitch)

                else:
                    s_res_t.append(s)
                    rho_res_t.append(logrho)
                    u_res_t.append(logu)

            s_res_p.append(s_res_t)
            rho_res_p.append(rho_res_t)
            u_res_p.append(u_res_t)

        s_res.append(s_res_p)
        rho_res.append(rho_res_p)
        u_res.append(u_res_p)

    output_dir = f"eos/{args.hhe_eos}"
    os.makedirs(output_dir, exist_ok=True)

    if args.smooth:
        s_array = gaussian_filter(np.array(s_res), sigma=(0, 0, 2.5, 2.5), mode='reflect')
        rho_array = gaussian_filter(np.array(rho_res), sigma=(0, 0, 2.5, 2.5), mode='reflect')
        u_array = gaussian_filter(np.array(u_res), sigma=(0, 0, 2.5, 2.5), mode='reflect')

    else:

        s_array = np.array(s_res)
        rho_array = np.array(rho_res)
        u_array = np.array(u_res)


    # --- Build output filename ---
    if args.z_eos == 'mixture':
        f_rock = np.round(args.f_ppv, 2)
        f_ice = np.round(1.0 - f_rock, 2)
        filename = f"{output_dir}/{args.hhe_eos}_{f_ice}_{f_rock}_{args.zmix_eos1}_{args.zmix_eos2}_pt.npz"
    elif args.z_eos == 'ice_mixture':
        filename = f"{output_dir}/{args.hhe_eos}_{args.z_eos}_cno{args.cno_ratio}_{args.zmix_eos1}_pt.npz"
    else:

        f_rock = np.round(args.f_ppv, 2)
        f_ice = np.round(1.0 - f_rock, 2)

        if args.smooth:
            filename = f"{output_dir}/{args.hhe_eos}_{args.z_eos}_smooth_pt.npz"
        else:
            filename = f"{output_dir}/{args.hhe_eos}_{f_ice}_{f_rock}_{args.z_eos}_{args.zmix_eos1}_pt.npz"

    # Tag with "_custom" so custom-range tables never overwrite default ones
    if args.custom_range:
        filename = filename.replace("_pt.npz", "_custom_pt.npz")

    np.savez_compressed(filename,
                        logpvals = logpvals,
                        logtvals = logtvals,
                        yvals = yvals,
                        zvals = zvals,
                        s_pt = s_array,
                        logrho_pt = rho_array,
                        logu_pt = u_array)

    print('FINISHED FROCK={}'.format(args.f_ppv))

    print(f"Data saved successfully to {filename}.")

if __name__ == "__main__":
    main()