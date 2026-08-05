import os, pickle, lzma
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
# plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=20)
plt.rc('lines', lw=2.5)
plt.rc('xtick', direction='in', top=True, bottom=True)
plt.rc('ytick', direction='in', left=True, right=True)
plt.rc('xtick.minor', visible=True)
plt.rc('ytick.minor', visible=True)
plt.rc('figure', figsize=(8.0, 8.0), dpi=300)
plt.rc('savefig', bbox='tight')
from utils.common import *
from utils.const import G

MHZ = 1e6

if __name__ == '__main__':
    folder = '%s/%s/' % (args.data_folder, model_fldr)
    output_folder = '%s/%s/' % (args.output_folder, model_fldr)
    os.makedirs(output_folder, exist_ok=True)
    f_size = 14

    if not os.path.exists(modes_fn):
        logger.info('Modes have not been calculated, skipping')
        exit(0)
    else:
        with lzma.open(modes_fn, 'rb') as f:
            (modes_lst, m_metals, j2_s) = pickle.load(f)
    age = np.genfromtxt(folder+"evolution_data.txt",unpack=True,usecols=0)
    r_b = np.genfromtxt(folder+"rad_boundary_profiles.txt")
    m_b = np.genfromtxt(folder + "/mgrid.txt")
    age_Gyr = age[MODE_START::MODE_STRIDE] * const.s_to_Myr/1e3
    i = len(age_Gyr)

    def get_modes(modes_all_slice, ln_pairs):
        ret = []
        for l, n in ln_pairs:
            freqs = []
            for modes_curr in modes_all_slice:
                cands = []
                for lmode, nmode, mode in modes_curr:
                    if lmode == l and nmode == n:
                        cands.append(mode)
                if len(cands) > 0:
                    freqs.append(min(cands))
                else:
                    freqs.append(-1)
            ret.append(np.array(freqs))
        return ret
    ln_pairs2 = [(2, 0), (2, -1), (2, -2), (2, -3)]
    ln_pairs3 = [(3, 0), (3, -1), (3, -2)]
    modes_by_ln2 = get_modes(modes_lst[ :i], ln_pairs2)
    modes_by_ln3 = get_modes(modes_lst[ :i], ln_pairs3)
    ls_arr = ['-', '--', ':', '-.']
    lw = 1.5
    for (l, n), freqs, ls in zip(ln_pairs2, modes_by_ln2, ls_arr):
        idx = freqs > 0
        plt.plot(age_Gyr[idx], freqs[idx],
                 label='(%d, %d)' % (l, n),
                 color='b', ls=ls, lw=lw)
    for (l, n), freqs, ls in zip(ln_pairs3, modes_by_ln3, ls_arr):
        idx = freqs > 0
        plt.plot(age_Gyr[idx], freqs[idx],
                 label='(%d, %d)' % (l, n),
                 color='r', ls=ls, lw=lw)
    plt.legend(ncol=2)

    M = m_b[0]
    R = r_b[MODE_START::MODE_STRIDE, 0]
    wdyn = np.sqrt(G * M / (4 * np.pi**2 * R**3)) * MHZ
    ylim = plt.ylim()
    plt.plot(age_Gyr, 0.05 * wdyn, 'k--', lw=0.7)
    plt.plot(age_Gyr, 30 * wdyn, 'k--', lw=0.7)
    plt.ylim(0, ylim[1])

    plt.tight_layout()
    plt.xlabel(r'Age [Gyr]')
    plt.ylabel(r'Freq ($\mu$Hz)')
    plt.xlim(0, int(config['general']['final_age']))
    if float(config['initial'].get('M_Mearth', config['initial'].get('M_factor', '317.907'))) < 0.5:
        kwargs = dict(lw=1.0, alpha=0.5)
        plt.ylim(top=145)
        age = 4.56
        plt.plot(age, 86.83, 'bx', **kwargs)
        plt.plot(age, 66.99, 'bx', **kwargs)
        plt.plot(age, 61.1, 'bx', **kwargs)
        plt.plot(age, 61.7, 'bx', **kwargs)
        plt.plot(age, 87.9, 'rx', **kwargs)
        plt.plot(age, 88.3, 'rx', **kwargs)
        plt.plot(age, 88.5, 'rx', **kwargs)
    plt.savefig(f"%s/modes"%(output_folder), facecolor='white',bbox_inches='tight')
