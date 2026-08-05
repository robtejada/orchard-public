import os, pickle, lzma
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from utils import const
import os
from astropy.constants import k_B, m_e, m_p, m_n, M_jup, M_earth, R_jup, R_earth
from astropy import units as u
from astropy.constants import u as amu
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import AutoMinorLocator
kb = k_B.to('erg/K')
erg_to_kbbar = (u.erg/u.Kelvin/u.gram).to(k_B/amu)
kbbar_to_erg = 1/erg_to_kbbar
from scipy.interpolate import RegularGridInterpolator as RGI
from matplotlib.lines import Line2D
from misc.misc import get_misc_p
from scipy.interpolate import interp1d
from initial import initialize_grid
from utils.const import G

from utils.common import *
import shutil
from utils import seis

folder = '%s/%s/' % (args.data_folder, model_fldr)

R_rho = get_diffusion_param('R_rho')
alpha = float(config['diffusion']['alpha'])
M_factor = float(config['initial'].get('M_Mearth', config['initial'].get('M_factor', '317.907')))
misc_curve = str(config['diffusion']['misc_curve'])
deltaT = float(config['diffusion']['deltaT'])
mc = float(config['core']['mass_core'])

M_jup = 1.89914e30
R_jup = 7.15e9
M_e = 5.9722e27
a = 7.57e-15
c = 3e10
msun = 1.98e33
MODES_FN = 'freqs.pkl'


N = int(config['general']['N'])
M_factor = float(config['initial'].get('M_Mearth', config['initial'].get('M_factor', '317.907')))
M_planet = float(config['initial'].get('M_planet_unit_cgs', config['initial'].get('M_planet', '5.972167867791379e27')))*M_factor
S_ini = float(config['initial']['S_ini'])
Y_ini = float(config['initial']['Y_ini'])
Z_ini = float(config['initial']['Z_ini'])
f_rock_ini = float(config['initial']['f_rock_ini'])

(   r_b_old,
    p_old,
    m_b,
    dm,
    S,
    Y,
    Z,
    f_rock,
    kcore,
    kcore_fe
    ) = initialize_grid(N, M_planet, S_ini, Y_ini, Z_ini, f_rock_ini)

def calc_modes(modes_folder, stride=1):
    case = 'Saturn'

    modes_all = []
    m_metals_all = []
    j2_all = []

    modes_fn = os.path.join(modes_folder, MODES_FN)
    if os.path.exists(modes_fn):
        logger.info('Modes file "%s" exists, not recomputing' % modes_fn)
        return

    config_sim = configparser.ConfigParser()
    config_sim.read(folder + config_basename)
    r_b = np.genfromtxt(folder+"rad_boundary_profiles.txt")
    m_b = np.array(np.genfromtxt(folder + "/mgrid.txt"))
    press = np.genfromtxt(folder+"press_profiles.txt")
    rho = np.genfromtxt(folder+"rho_profiles.txt")
    temp = np.genfromtxt(folder+"temp_profiles.txt")
    S = np.genfromtxt(folder+"Sprofiles.txt")
    data_age = np.genfromtxt(folder+"evolution_data.txt",unpack=True,usecols=0)
    data_tint = np.genfromtxt(folder+"evolution_data.txt",unpack=True,usecols=1)
    data_teff = np.genfromtxt(folder+"evolution_data.txt",unpack=True,usecols=2)
    brunt = np.genfromtxt(folder+"brunt.txt")

    Y = np.genfromtxt(folder+"Yprofiles.txt")
    Z = np.genfromtxt(folder+"Zprofiles.txt")

    # get modes using last appended data
    M0 = float(config['initial'].get('M_planet_unit_cgs', config['initial'].get('M_planet', '5.972167867791379e27')))
    factor = float(config['initial'].get('M_Mearth', config['initial'].get('M_factor', '317.907')))
    Mp = M0 * factor
    # my code doesn't deal with zero mass in center
    modes_ret = seis.get_all_modes(
        r_b, S, rho, press, m_b, Y, Z, brunt**2,
            modes_folder, 'modes', stride=stride)
    modes_lst = [r[0] for r in modes_ret]
    m_metals = [r[1][0] for r in modes_ret]
    j2_s = [r[1][1] for r in modes_ret]

    with lzma.open(modes_fn, 'wb') as f:
        pickle.dump((modes_lst, m_metals, j2_s), f)

if __name__ == "__main__":
    os.makedirs(modes_folder, exist_ok=True)
    calc_modes(modes_folder, stride=MODE_STRIDE)
