#!/usr/bin/env python3
"""
Standalone plotting script for ORCHARD evolution models.

Generates still plots, age comparison plots, or movie frames directly
from the command line using saved model data -- no need to re-run the
evolution or open a Jupyter notebook.

Usage examples
--------------

  # Single still plot at the last saved timestep (default):
  python plots.py --config parameters/parameter_user.ini

  # Still plot at a specific age (in Gyr):
  python plots.py --config parameters/parameter_user.ini --age 4.56

  # Compare two models side-by-side:
  python plots.py --config parameters/param_A.ini parameters/param_B.ini \
                   --labels "Model A" "Model B"

  # Age comparison plot (multiple ages, color-coded):
  python plots.py --config parameters/parameter_user.ini \
                   --mode age --ages 1.0 4.56 10.0

  # Generate a movie and compile to mp4:
  python plots.py --config parameters/parameter_user.ini --mode movie

  # Movie with custom time spacing and 6 panels:
  python plots.py --config parameters/parameter_user.ini \
                   --mode movie --dt_gyr 0.05 --panel_num 6
"""

import plot_class
import numpy as np
import argparse
import configparser
import os
import subprocess


def str2bool(v):
    """Convert a string to a boolean."""
    if isinstance(v, bool):
        return v
    v_lower = v.lower()
    if v_lower in ('true', '1', 't', 'yes', 'y'):
        return True
    elif v_lower in ('false', '0', 'f', 'no', 'n'):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def main():
    parser = argparse.ArgumentParser(
        description='Plot evolution data from ORCHARD model parameter files.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ---- required / positional ----
    parser.add_argument(
        '--config', type=str, nargs='+',
        default=['parameters_default.ini'],
        help='Configuration file(s) (.ini) or model folder path(s).',
    )

    # ---- plot mode ----
    parser.add_argument(
        '--mode', type=str, default='still',
        choices=['still', 'age', 'movie'],
        help='Plot mode: "still" (single snapshot), "age" (multi-age comparison), '
             'or "movie" (frames + mp4).  Default: still.',
    )

    # ---- labels ----
    parser.add_argument(
        '--labels', type=str, nargs='+',
        help='Legend labels (one per config path). Auto-generated if omitted.',
    )

    # ---- shared layout options ----
    parser.add_argument(
        '--panel_num', type=int, default=6, choices=[4, 6, 9],
        help='Number of panels (4, 6, or 9).  Default: 6.',
    )
    parser.add_argument(
        '--leg_size', type=int, default=12,
        help='Legend font size.  Default: 12.',
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help='Output filename (still/age) or folder name (movie). '
             'Defaults: "still_plot" / "age_plot" / "<model>_movie".',
    )

    # ---- init_plot constructor flags ----
    parser.add_argument('--plot_Js', type=str2bool, default=False,
                        help='Plot equatorial radius, J2, J4.')
    parser.add_argument('--plot_rock_phases', type=str2bool, default=False,
                        help='Overlay rock phase / melt-fraction curves.')
    parser.add_argument('--zm_plot', type=str2bool, default=False,
                        help='Plot Z miscibility overlay.')
    parser.add_argument('--ym_plot', type=str2bool, default=False,
                        help='Plot Y miscibility overlay.')

    # ---- still_plot / age_plot shared options ----
    parser.add_argument('--plot_water_misc', action='store_true',
                        help='Overlay H2-water miscibility curve.')
    parser.add_argument('--plot_he_rain', action='store_true',
                        help='Overlay helium rain (immiscibility) curve.')
    parser.add_argument('--rad_profiles', action='store_true',
                        help='Use radius (instead of mass) as x-axis for profiles.')
    parser.add_argument('--rad_temp_plot', action='store_true',
                        help='Overlay normalised radius on the temperature panel.')
    # ---- still_plot specific ----
    parser.add_argument(
        '--age', type=float, default=-1,
        help='Age (Gyr) for the still plot snapshot.  -1 = last saved step.',
    )
    parser.add_argument(
        '--save', action='store_true', default=True,
        help='Save the plot to disk (default: True).',
    )

    # ---- age_plot specific ----
    parser.add_argument(
        '--ages', type=float, nargs='+',
        help='List of ages (Gyr) for the age comparison plot.',
    )
    parser.add_argument(
        '--cmap', type=str, default='magma_r',
        help='Colormap for age comparison.  Default: magma_r.',
    )

    # ---- movie specific ----
    parser.add_argument(
        '--dt_gyr', type=float, default=None,
        help='Time spacing (Gyr) between movie frames.  '
             'If omitted, uses every saved snapshot.',
    )
    parser.add_argument(
        '--tmin_gyr', type=float, default=None,
        help='Start time (Gyr) for movie frames.',
    )
    parser.add_argument(
        '--tmax_gyr', type=float, default=None,
        help='End time (Gyr) for movie frames.',
    )
    parser.add_argument(
        '--no_mp4', action='store_true',
        help='Skip ffmpeg mp4 compilation after generating frames.',
    )

    args = parser.parse_args()

    # ---- derive labels ----
    model_paths = args.config
    if args.labels:
        if len(args.labels) != len(model_paths):
            raise ValueError(
                f"Number of labels ({len(args.labels)}) must match "
                f"number of config paths ({len(model_paths)})."
            )
        labels = args.labels
    else:
        # Auto-generate from config filenames
        labels = []
        for p in model_paths:
            base = os.path.basename(p)
            name = base.replace('.ini', '')
            labels.append(name)

    # ---- derive output name ----
    if args.output:
        out_name = args.output
    else:
        if args.mode == 'still':
            out_name = 'still_plot'
        elif args.mode == 'age':
            out_name = 'age_plot'
        else:
            # movie: use first config's basename
            base = os.path.basename(model_paths[0]).replace('.ini', '')
            out_name = base + '_movie'

    # ---- auto-enable rock phases for sub-Neptunes / super-Earths ----
    rock_phases = args.plot_rock_phases
    if not rock_phases:
        for p in model_paths:
            cfg = configparser.ConfigParser()
            cfg.read(p)
            planet = cfg.get('boundary_condition', 'planet', fallback='')
            if planet in ('Sub_Neptune', 'Super_Earth'):
                rock_phases = True
                break

    # ---- initialise plot_class ----
    plotter = plot_class.init_plot(
        paths=model_paths,
        plot_Js=args.plot_Js,
        plot_rock_phases=rock_phases,
        zm_plot=args.zm_plot,
        ym_plot=args.ym_plot,
    )

    # ---- dispatch ----
    if args.mode == 'still':
        plotter.still_plot(
            age=args.age,
            output_file=out_name,
            labels=labels,
            leg_size=args.leg_size,
            save=args.save,
            plot_water_misc=args.plot_water_misc,
            rad_profiles=args.rad_profiles,
            plot_he_rain=args.plot_he_rain,

            panel_num=args.panel_num,
            rad_temp_plot=args.rad_temp_plot,
        )

    elif args.mode == 'age':
        if not args.ages:
            raise ValueError(
                "--ages is required for age mode.  "
                "Example: --ages 1.0 4.56 10.0"
            )
        plotter.age_plot(
            ages=args.ages,
            output_file=out_name,
            labels=labels,
            leg_size=args.leg_size,
            save=args.save,
            plot_water_misc=args.plot_water_misc,
            rad_profiles=args.rad_profiles,
            plot_he_rain=args.plot_he_rain,

            panel_num=args.panel_num,
            rad_temp_plot=args.rad_temp_plot,
            cmap=args.cmap,
        )

    elif args.mode == 'movie':
        import matplotlib
        matplotlib.use('Agg')

        plotter.movie_evol_plots_by_age(
            output_file=out_name,
            labels=labels,
            leg_size=args.leg_size,
            panel_num=args.panel_num,
            dt_gyr=args.dt_gyr,
            tmin_gyr=args.tmin_gyr,
            tmax_gyr=args.tmax_gyr,

            plot_water_misc=args.plot_water_misc,
            rad_profiles=args.rad_profiles,
            plot_he_rain=args.plot_he_rain,
            rad_temp_plot=args.rad_temp_plot,
        )

        movie_outdir = os.path.join('plots', out_name)
        print(f'Movie frames saved to {movie_outdir}/')

        # Compile frames into mp4
        if not args.no_mp4:
            mp4_path = os.path.join(movie_outdir, f'{out_name}.mp4')
            ffmpeg_cmd = (
                f"ffmpeg -y -framerate 30 -pattern_type glob "
                f"-i '{movie_outdir}/*.png' "
                f"-vf \"scale=trunc(iw/2)*2:trunc(ih/2)*2\" "
                f"-c:v libx264 -pix_fmt yuv420p {mp4_path}"
            )
            print('Generating mp4...')
            ret = subprocess.run(ffmpeg_cmd, shell=True,
                                 capture_output=True, text=True)
            if ret.returncode == 0:
                print(f'Movie saved to {mp4_path}')
            else:
                print(f'ffmpeg failed (returncode {ret.returncode}): '
                      f'{ret.stderr.strip()}')


if __name__ == '__main__':
    main()
