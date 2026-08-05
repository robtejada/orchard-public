#!/usr/bin/env python3
import argparse
import configparser
from pathlib import Path
import sys
import numpy as np

# --- user inputs ---
# TEMPLATE_FILE = Path("parameters_subnepunes_initcondexp/subnep_10mearth_9.7mantle_3.2core_0.5zout_6.0Sout.ini")
TEMPLATE_FILE = Path("parameters_superearths_Fecomb/superearth_1.0mearth_menv0.01_mcore0.333_Fecomb.ini")
# DST_FOLDER = Path("parameters_subneptunes_production_nolat_novisc_norad")
DST_FOLDER = Path("parameters_superearths_7.7Sout_1Zsol_Fecomb")

# values to sweep
#S_INI_VALS = [5.5, 6.0, 6.25, 6.5, 6.75, 7.0]          # k_B/baryon
S_INI_VALS = [7.7]
S_INI_CORE = [0.5, 0.52, 0.54, 0.56, 0.58, 0.6]          # k_B/baryon
MASS_TOT_VALS = [1.0, 3.0, 5.0, 7.0, 10.0]           # -> [initial] M_factor
Z_INI_VALS = [0.017]                   # -> [initial] Z_ini
TEQ_VALS   = [55, 400, 800]               # -> [boundary_condition] T_eq

# core mass models to sweep
# CORE_FRAC = core+mantle mass / total mass
# IRON_FRAC = iron core mass / core mass
CORE_FRAC_LIST = [0.5, 0.75, 0.95, 0.995]
IRON_FRAC_LIST = [0.1, 0.33, 0.5, 0.75]
 
# transport toggles to sweep (set to [True] to force on, [False] to force off, or [True, False] to iterate)
# LATENT_HEAT_LIST = [True]           # -> [transport] latent_heat_effect
# VISC_CORE_CONV_LIST = [True]        # -> [transport] viscous_core_convection
# RADIOACTIVE_LIST = [True]           # -> [transport] radioactive
# --------------------


def clean_destination(dst: Path):
    """Delete all .ini files under dst."""
    for ini in dst.glob("*.ini"):
        try:
            ini.unlink()
        except Exception as e:
            print(f"Failed to delete {ini}: {e}", file=sys.stderr)


def _round_safe(x: float, ndigits: int = 6) -> float:
    """Round to kill float noise in names and files."""
    return round(float(x), ndigits)


def _fmt(x: float) -> str:
    """
    Compact float string for filenames and updated values.
    10.0 -> '10'
    3.25 -> '3.25'
    1e-4 -> '0.0001'
    """
    return f"{x:g}"


def load_config(path: Path) -> configparser.ConfigParser:
    """Read template .ini with case preserved."""
    cfg = configparser.ConfigParser(interpolation=None)
    cfg.optionxform = str  # preserve case of keys like M_factor and T_eq
    with path.open("r") as fp:
        cfg.read_file(fp)
    return cfg


def write_config(cfg: configparser.ConfigParser, path: Path) -> None:
    with path.open("w") as fp:
        cfg.write(fp)


def _suffix(flag: bool, yes: str, no: str) -> str:
    """Return filename suffix based on boolean flag."""
    return yes if flag else no


def build_filename(
    m_tot: float,
    m_core: float,
    m_core_fe: float,
    z_ini: float,
    s_ini: float,
    s_core: float,
    teq: float,
    # lat: bool,
    # visc: bool,
    # rad: bool,
) -> str:
    """
    superearth_{mass_tot}mearth_{mass_core}mantle_{mass_ironcore}core_{metallicity}Zout_{S_ini}Sout_{Teq}Teq[_lat|_nolat][_visc|_novisc][_rad|_norad]_Fecomb.ini
    """
    return (
        "superearth_"
        + f"{_fmt(_round_safe(m_tot))}mearth_"
        + f"{_fmt(_round_safe(m_core))}mantle_"
        + f"{_fmt(_round_safe(m_core_fe))}core_"
        + f"{_fmt(_round_safe(z_ini))}Zout_"
        + f"{_fmt(_round_safe(s_ini))}Sout_"
        + f"{_fmt(_round_safe(s_core))}Score_"
        + f"{_fmt(_round_safe(teq))}Teq"
        # + _suffix(lat, "_lat", "_nolat")
        # + _suffix(visc, "_visc", "_novisc")
        # + _suffix(rad, "_rad", "_norad")
        + "_Fecomb.ini"
    )


def generate_all():
    """
    Loops over:
      Z_INI_VALS, S_INI_VALS, MASS_TOT_VALS, CORE_FRAC_LIST, IRON_FRAC_LIST, TEQ_VALS,
      LATENT_HEAT_LIST, VISC_CORE_CONV_LIST, RADIOACTIVE_LIST

    For each combo:
      [initial]:  M_factor, S_ini, Z_ini
      [core]:     mass_core = core_frac * M_factor
                  mass_core_fe = iron_frac * mass_core
      [boundary_condition]: T_eq
      [transport]: latent_heat_effect, viscous_core_convection, radioactive

    Writes output .ini to DST_FOLDER with filename encoding all parameters.
    """
    new_files = []

    for z_ini in Z_INI_VALS:
        for s_ini in S_INI_VALS:
            for s_core in S_INI_CORE:
                for m_tot in MASS_TOT_VALS:
                    for core_frac in CORE_FRAC_LIST:
                        for iron_frac in IRON_FRAC_LIST:
                            for teq in TEQ_VALS:
                                # for lat in LATENT_HEAT_LIST:
                                #     for visc in VISC_CORE_CONV_LIST:
                                #         for rad in RADIOACTIVE_LIST:

                                m_core = _round_safe(core_frac * m_tot)
                                m_core_fe = _round_safe(iron_frac * m_core)

                                cfg = load_config(TEMPLATE_FILE)

                                # update [initial]
                                cfg["initial"]["M_Mearth"] = _fmt(_round_safe(m_tot))
                                cfg["initial"]["S_ini"] = _fmt(_round_safe(s_ini))
                                cfg["initial"]["Z_ini"] = _fmt(_round_safe(z_ini))

                                # update [core]
                                cfg["core"]["mass_core"] = _fmt(m_core)
                                cfg["core"]["mass_core_fe"] = _fmt(m_core_fe)
                                cfg["core"]["mantle_entropy"] = _fmt(s_core)

                                # update [boundary_condition]
                                cfg["boundary_condition"]["T_eq"] = _fmt(_round_safe(teq))

                                # update [transport]
                                # cfg["transport"]["latent_heat_effect"] = "True" if lat else "False"
                                # cfg["core"]["viscous_core_convection"] = "True" if visc else "False"
                                # cfg["transport"]["radioactive"] = "True" if rad else "False"

                                # filename
                                fname = build_filename(
                                    m_tot=m_tot,
                                    m_core=core_frac,               # use computed masses in name
                                    m_core_fe=iron_frac,
                                    z_ini=z_ini,
                                    s_ini = s_ini,
                                    s_core=s_core,
                                    teq=teq,
                                    # lat=lat,
                                    # visc=visc,
                                    # rad=rad,
                                )
                                outpath = DST_FOLDER / fname
                                write_config(cfg, outpath)
                                new_files.append(outpath)

    print(f"Generated {len(new_files)} parameter files in {DST_FOLDER}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate sub-Neptune APPLE .ini grids over Z_ini, S_ini, total mass, core fraction, iron fraction, T_eq, and transport toggles"
    )
    parser.add_argument(
        "-c", "--clean",
        action="store_true",
        help="delete existing .ini files in destination before regenerating"
    )
    parser.add_argument(
        "-n", "--count",
        action="store_true",
        help="count the number of files that would be generated without writing"
    )
    args = parser.parse_args()

    # ensure destination exists
    DST_FOLDER.mkdir(parents=True, exist_ok=True)

    # optional clean
    if args.clean:
        print(f"Cleaning old .ini files in {DST_FOLDER}...")
        clean_destination(DST_FOLDER)
        print(f"Destination folder {DST_FOLDER} cleaned.")

    # optional count-only
    if args.count:
        total = (
            len(Z_INI_VALS)
            * len(S_INI_VALS)
            * len(S_INI_CORE)
            * len(MASS_TOT_VALS)
            * len(CORE_FRAC_LIST)
            * len(IRON_FRAC_LIST)
            * len(TEQ_VALS)
            # * len(LATENT_HEAT_LIST)
            # * len(VISC_CORE_CONV_LIST)
            # * len(RADIOACTIVE_LIST)
        )
        print(f"{total} files will be saved in {DST_FOLDER}.")
        sys.exit(0)

    # generate all combos
    generate_all()


if __name__ == "__main__":
    main()