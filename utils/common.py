from utils import const
import numpy as np
import os
import sys
import argparse
import configparser
import logging
from types import SimpleNamespace
import pdb # so that all other files get it by default

BASE_CONFIG = 'parameters_default.ini'
ZERO = 1e-20 # underflow

# mode calculations params: run for all timesteps ages[start::stride]
# (YS) shared in calculation and plotting code, so putting here
MODE_START = 1
MODE_STRIDE = 10

# ---------------------------------------------------------------------------
# Library mode: skip CLI parsing, log file creation, and folder setup so this
# module can be imported in notebooks / other scripts purely for its utility
# functions. Triggered by either:
#   - setting ORCHARD_LIBRARY_MODE=1 in the environment before import, OR
#   - automatic detection of a Jupyter / IPython kernel.
# ---------------------------------------------------------------------------
def _resolve_core_mass_fractions(config):
    """If [core].use_mass_fractions is True, derive mass_core and mass_core_fe
    from mantle_fraction / iron_core_fraction and M_factor, overwriting the
    absolute values in-place so all downstream consumers see the resolved mass.

    Definitions:
        mass_core    = mantle_fraction    * M_factor           (mantle + iron core)
        mass_core_fe = iron_core_fraction * mass_core          (iron core)
    """
    if not config.has_section('core'):
        return None
    if not config['core'].getboolean('use_mass_fractions', fallback=False):
        return None
    M_factor_Me = float(config['initial'].get(
        'M_Mearth', config['initial'].get('M_factor', '317.907')))
    mantle_frac = float(config['core'].get('mantle_fraction', '0.0'))
    iron_frac = float(config['core'].get('iron_core_fraction', '0.0'))
    if not (0.0 <= mantle_frac <= 1.0):
        raise ValueError(
            f'[core] mantle_fraction must be in [0, 1]; got {mantle_frac}')
    if not (0.0 <= iron_frac <= 1.0):
        raise ValueError(
            f'[core] iron_core_fraction must be in [0, 1]; got {iron_frac}')
    mass_core_Me = mantle_frac * M_factor_Me
    mass_core_fe_Me = iron_frac * mass_core_Me
    config['core']['mass_core'] = f'{mass_core_Me:.10g}'
    config['core']['mass_core_fe'] = f'{mass_core_fe_Me:.10g}'
    return (mass_core_Me, mass_core_fe_Me, mantle_frac, iron_frac, M_factor_Me)


def _is_library_mode():
    if os.environ.get('ORCHARD_LIBRARY_MODE'):
        return True
    if 'ipykernel' in sys.modules:
        return True
    return False

LIBRARY_MODE = _is_library_mode()

if LIBRARY_MODE:
    # Minimal, side-effect-free initialization. Provides the same names that
    # downstream `from utils.common import *` consumers expect.
    args = SimpleNamespace(
        config=BASE_CONFIG,
        data_folder='models',
        output_folder='plots',
        log_folder='logs',
        modes_folder='modes',
        clean=False,
        plot_end=False,
        plot_active=False,
    )
    config = configparser.ConfigParser()
    if os.path.exists(BASE_CONFIG):
        config.read(BASE_CONFIG)
    _resolve_core_mass_fractions(config)
    config_fldr = ''
    config_basename = BASE_CONFIG
    model_fldr = config_basename.replace('.ini', '')

    logger = logging.getLogger('APPLE')
    while len(logger.handlers) > 0:
        logger.removeHandler(logger.handlers[-1])
    logger.addHandler(logging.NullHandler())
    loggerpath = None

    modes_folder = '%s/%s/' % (args.modes_folder, model_fldr)
    modes_basefn = 'freqs.pkl'
    modes_fn = '%s/%s/%s' % (args.modes_folder, model_fldr, modes_basefn)

else:
    # command line args parser
    parser = argparse.ArgumentParser(description='Run planetary evolution model.')
    parser.add_argument('--config',
                        type=str,
                        required=False,
                        help='Path to the primary configuration file'
                        '[other options will be read from parameters/parameters_default.ini')
    parser.add_argument("--data_folder",
                        type=str,
                        default="models",
                        help="Path to the parent folder of all data files")
    parser.add_argument("--output_folder",
                        type=str,
                        default="plots",
                        help="Path to the parent folder of all plots")
    parser.add_argument("--log_folder",
                        type=str,
                        default="logs",
                        help="Path to the folder with all logs")
    parser.add_argument("--modes_folder",
                        type=str,
                        default="modes",
                        help="Path to the parent folder to store seismology data")
    parser.add_argument("-c", "--clean",
                        help="Remove existing folders/files",
                        action="store_true")
    parser.add_argument("-plot_end", "--plot_end",
                        help="Generate movie frames from saved data after evolution completes",
                        action="store_true")
    parser.add_argument("-plot_active", "--plot_active",
                        help="Generate still_plot frames during the evolution at each save step",
                        action="store_true")
    args = parser.parse_args()
    config = configparser.ConfigParser()
    # config just reads in an empty dict, rather than error-ing, if the config file
    # isn't found
    config.read(BASE_CONFIG)
    if args.config is not None:
        if not os.path.exists(args.config):
            raise ValueError('Config file "%s" not found' % args.config)
        # print('Reading parameters from', args.config)
        config.read(args.config)
    else:
        # if no config specified, run default params alone
        args.config = BASE_CONFIG

    _resolved_core_fracs = _resolve_core_mass_fractions(config)

    # parse out config file into fldr + basename
    config_parts = args.config.split('/')
    config_fldr = '/'.join(config_parts[ :-1])
    config_basename = config_parts[-1]
    model_fldr = config_basename.replace('.ini', '')
    for f in [config_fldr, args.data_folder, args.output_folder, args.log_folder]:
        if f:
            os.makedirs(f, exist_ok=True)

    # Create a logger for both file & stdout
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
    )
    logger_basename = config_basename.replace('.ini', '.log')
    loggerpath = '%s/%s' % (args.log_folder, logger_basename)
    logfilehandler = logging.FileHandler(loggerpath, mode='w')
    logfilehandler.setLevel(logging.INFO)
    logger = logging.getLogger('APPLE')
    # in apple_cli, logger may already have handlers attached, clear them
    # naive for loop doesn't seem to work idk why
    while len(logger.handlers) > 0:
        logger.removeHandler(logger.handlers[-1])
    logger.addHandler(logfilehandler)

    if _resolved_core_fracs is not None:
        _mc_Me, _mcfe_Me, _mfrac, _ifrac, _Mf = _resolved_core_fracs
        logger.info(
            f'[core] use_mass_fractions=True: resolved mass_core={_mc_Me:.5f} Mearth, '
            f'mass_core_fe={_mcfe_Me:.5f} Mearth '
            f'(from mantle_fraction={_mfrac}, iron_core_fraction={_ifrac}, '
            f'M_factor={_Mf} Mearth)'
        )

    # modes folder
    modes_folder = '%s/%s/' % (args.modes_folder, model_fldr)
    modes_basefn = 'freqs.pkl'
    modes_fn = '%s/%s/%s' % (args.modes_folder, model_fldr, modes_basefn)



#### COMMON UTILITY FUNCTIONS ####
def z_to_metallicity_factor(z, z_solar=0.017):
    """
    Convert mass fraction Z to enrichment factor relative to solar (Z/X).
    Works with scalars or numpy arrays.
    """
    return (z / (1.0 - z)) / (z_solar / (1.0 - z_solar))

def metallicity_factor_to_z(metallicity_factor, z_solar=0.017):
    """
    Convert enrichment factor relative to solar (Z/X) to mass fraction Z.
    Works with scalars or numpy arrays.
    """
    return (metallicity_factor * z_solar) / (1.0 + metallicity_factor * z_solar)

def mean_molecular_weight_from_X(X_H, X_He, X_water,
                                 mu_H2=2.0, mu_He=4.0, mu_H2O=18.0):
    """
    Mean molecular weight μ for a mixture of H2, He, and H2O.

    Parameters
    ----------
    X_H : float or array
        Mass fraction of hydrogen (assumed to be in H2).
    X_He : float or array
        Mass fraction of helium.
    X_water : float or array
        Mass fraction of water (the 'metal' component).

    mu_H2, mu_He, mu_H2O : floats, optional
        Molecular weights (amu) of H2, He, and H2O.

    Returns
    -------
    mu_mix : float or array
        Mean molecular weight in units of proton mass (amu).
    """
    X_H = np.asarray(X_H, dtype=float)
    X_He = np.asarray(X_He, dtype=float)
    X_water = np.asarray(X_water, dtype=float)

    # sanity: they should sum to ~1
    # (but we don't enforce in case you want to play games with them)
    inv_mu = (X_H / mu_H2) + (X_He / mu_He) + (X_water / mu_H2O)
    return 1.0 / inv_mu

def mean_molecular_weight_HHe_plus_water(Z_water,
                                         X_HHe=None,
                                         Y_over_XplusY=0.27,
                                         mu_H2=2.0, mu_He=4.0, mu_H2O=18.0):
    """
    Mean molecular weight μ for a mixture where the 'metal' is pure water
    and the rest is an H/He mixture.

    Parameters
    ----------
    Z_water : float or array
        Mass fraction of water (the metal component).
    X_HHe : float or array, optional
        Combined H+He mass fraction. If None, assumed to be 1 - Z_water.
    Y_over_XplusY : float, optional
        Helium mass fraction within the H+He mixture:
            Y / (X + Y)
        Default ~0.27 (roughly solar).

    mu_H2, mu_He, mu_H2O : floats, optional
        Molecular weights (amu) of H2, He, and H2O.

    Returns
    -------
    mu_mix : float or array
        Mean molecular weight in units of proton mass (amu).
    """
    Z_water = np.asarray(Z_water, dtype=float)

    if X_HHe is None:
        X_HHe = 1.0 - Z_water
    else:
        X_HHe = np.asarray(X_HHe, dtype=float)

    # split H+He mixture by mass
    X_He = Y_over_XplusY * X_HHe
    X_H  = (1.0 - Y_over_XplusY) * X_HHe

    return mean_molecular_weight_from_X(X_H, X_He, Z_water,
                                        mu_H2=mu_H2, mu_He=mu_He, mu_H2O=mu_H2O)

def get_rtransit_isothermal_guillot2010(R_ref, g, T_irr, mu=2.3, gamma=None):
    """
    Compute the transit radius offset Δz in the isothermal limit of
    Guillot (2010) Eq. (60):

        Δz = H * ln[ gamma * sqrt( 2π R_ref / H ) ]

    where H is the pressure scale height evaluated at R_ref.

    Parameters
    ----------
    R_ref : `~astropy.units.Quantity`
        Reference radius r at which g and T_irr are evaluated
        (e.g. 1-bar or photospheric radius), with length units.
    g : `~astropy.units.Quantity`
        Local gravitational acceleration at R_ref (e.g. m/s^2).
    T_irr : `~astropy.units.Quantity`
        Irradiation temperature used to characterize the upper-atmosphere
        temperature (typically T_irr or T_eq), in Kelvin.
    mu : float, optional
        Mean molecular weight in proton masses (dimensionless).
        Default is 2.3 for a H/He-dominated atmosphere.
    gamma : float, optional
        Opacity ratio gamma = kappa_v / kappa_th. If None, uses the
        Tang et al. scaling:
            gamma = 0.6 * sqrt( T_irr / 2000 K ).

    Returns
    -------
    delta_z : `~astropy.units.Quantity`
        Transit radius offset Δz (same length units as R_ref).
        The transit radius would be R_tr = R_ref + delta_z.
    """

    # Scale height H = k_B T / (mu m_p g)
    H = (const.kb * T_irr / (mu * const.mh * g))

    # Default gamma from Tang et al. (2024): gamma = 0.6 * sqrt(T_irr / 2000 K)
    if gamma is None:
        gamma = 0.6 * np.sqrt((T_irr / (2000.0)))

    # Argument of the log must be dimensionless
    R_over_H = (R_ref / H)
    log_arg = gamma * np.sqrt(2.0 * np.pi * R_over_H)

    if np.any(log_arg <= 0):
        raise ValueError("log argument <= 0 in delta_z_guillot_isothermal; "
                         "check gamma, R_ref, H.")

    delta_z = H * np.log(log_arg)

    return R_ref + delta_z

def assert_all_not_nan(*args):
    for i, var in enumerate(args):
        arr = np.asarray(var)
        bad = ~np.isfinite(arr)
        if not np.any(bad):
            continue

        # Scalar path: np.where on 0d arrays raises ValueError.
        if arr.ndim == 0:
            raise AssertionError(
                f"Variable {i} has non-finite scalar value: {arr.item()}"
            )

        idxs = np.where(bad)[0]
        raise AssertionError(
            f"Variable {i} has non-finite values at indices {idxs[:10]} "
            f"(showing up to 10). Example bad value: {arr[idxs[0]]}"
        )

def scifmt(num):
    ''' returns string corresponding to num in LaTeX scientific notation '''
    if num == 0:
        return '0'
    exp = np.floor(np.log10(np.abs(num)))
    mant = num / 10**exp
    return r'$%.3f \times 10^{%d}$' % (mant, exp)
def get_arith_mean(x):
    return (x[1: ] + x[ :-1]) / 2
def get_geom_mean(x):
    return np.sqrt(x[1: ] * x[ :-1])
def get_harmonic_mean(x):
    return 2 * (x[1:] * x[:-1])/(x[1:] + x[:-1])
def centers_to_edge_arith(x):
    ''' edge to center is easy, how to do center to edge? extrapolate at limits
    of array, means everywhere else '''
    x_padded = np.concatenate(([2 * x[0] - x[1]], x, [2 * x[-1] - x[-2]]))
    return get_arith_mean(x_padded)
def centers_to_edge_geom(x):
    x_padded = np.concatenate(([x[0]**2/x[1]], x, [x[-1]**2/x[-2]]))
    return get_geom_mean(x_padded)
def get_number_fracs(y, m_h=2, m_he=4):
    # vector-compatible expressions
    f_h = (1 - y) / m_h
    f_he = y / m_he
    f_tot = f_h + f_he
    return f_h / f_tot, f_he / f_tot

# --------------- backward-compatible [diffusion] / [transport] helpers ----- #
import warnings as _warnings

_SENTINEL = object()

def get_diffusion_param(key, default=_SENTINEL, cast=float):
    """Read *key* from [diffusion], falling back to [transport] for backward
    compatibility with older .ini files."""
    for section in ('diffusion', 'transport'):
        if config.has_option(section, key):
            if section == 'transport':
                _warnings.warn(
                    f"Parameter '{key}' found in [transport]; "
                    f"please move it to [diffusion]. "
                    f"[transport] fallback will be removed in a future release.",
                    DeprecationWarning, stacklevel=2,
                )
            raw = config[section][key]
            return cast(raw) if cast is not None else raw
    if default is not _SENTINEL:
        return default
    raise KeyError(f"Parameter '{key}' not found in [diffusion] or [transport]")

def getbool_diffusion_param(key, fallback=_SENTINEL):
    """Boolean variant of get_diffusion_param using getboolean()."""
    for section in ('diffusion', 'transport'):
        if config.has_option(section, key):
            if section == 'transport':
                _warnings.warn(
                    f"Parameter '{key}' found in [transport]; "
                    f"please move it to [diffusion]. "
                    f"[transport] fallback will be removed in a future release.",
                    DeprecationWarning, stacklevel=2,
                )
            return config[section].getboolean(key)
    if fallback is not _SENTINEL:
        return fallback
    raise KeyError(f"Boolean parameter '{key}' not found in [diffusion] or [transport]")

def mh_mass_relation(Mplanet, Mcore=14.7, fz=0.09):
    "Mass-metallicity relation (Chachan et al. 2025)'"
    Mz = Mcore + fz * (Mplanet - Mcore)
    return Mz
