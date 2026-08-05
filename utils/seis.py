import matplotlib.pyplot as plt

import os, pickle, lzma
import numpy as np

from collections import defaultdict
from multiprocessing import Pool
from scipy.integrate import solve_ivp, quad
from scipy.interpolate import interp1d
import scipy.optimize as opt

from eos import mixtures_eos as mix_eos
import os, pickle, lzma
from utils.common import *

ZERO = 1e-8
MSUN = 1.989e33
RSUN = 6.9599e10
MJUP = 1.898e30
RJUP = 6.991e9
MSAT = 5.683e29
RSAT = 5.823e9
P0JUP = 3.506e13
T0JUP = 1e4
Ws_sq_today_SAT = 0.138504 # SAT
Ws_sq_today_JUP = 0.083496 # JUP
MEARTH = 5.972e27
from utils.const import G
MHZ = 1e6
# needs to be interior to the innermost convectively stable region, else the
# outwards integration will diverge
XMID = 0.3
ONE_BAR = 1e6

PBOUNDS = (10, 13.6)
TBOUNDS = (4.2, 5)
SBOUNDS = (6, 9.5)
METHOD = 'DOP853'

TOL = 1e-9 # for structure
TOL2 = 1e-8 # for shooting

M_DEFAULT = 10

Z_EOS = 'ideal'

l2_freqs = [69.92, 70.58, 75.81, 95.64]
l3_freqs = [94.55, 95.00, 95.16]

hhe_eos = str(config['equation_of_state']['hhe_eos'])
hg = config['equation_of_state'].getboolean('hg')
z_eos = str(config['equation_of_state']['z_eos'])

N = int(config['general']['N'])
M_factor = float(config['initial'].get('M_Mearth', config['initial'].get('M_factor', '317.907')))
M_planet = float(config['initial'].get('M_planet_unit_cgs', config['initial'].get('M_planet', '5.972167867791379e27')))*M_factor
S_ini = float(config['initial']['S_ini'])
Y_ini = float(config['initial']['Y_ini'])
Z_ini = float(config['initial']['Z_ini'])
f_rock_ini = float(config['initial']['f_rock_ini'])

from initial import initialize_grid
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

def get_csq(r, S, rho, P, Mr, yarr, zarr, eos, m=M_DEFAULT):
    logP = np.log10(P)
    g1 = eos.get_gamma1(S, logP, yarr, zarr, hhe_eos=hhe_eos, z_eos=z_eos, hg=hg)
    csq_inv = rho / (g1 * P)
    return csq_inv

def get_interps_for_structure(r, S, rho, P, Mr, yarr, zarr, N2,
                              eos, m=M_DEFAULT):
    x = r / r.max() * (1 - ZERO)
    g = G * Mr / r**2
    csq_inv = get_csq(r, S, rho, P, Mr, yarr, zarr, eos, m=M_DEFAULT)
    Vg = g * r * csq_inv
    U = 4 * np.pi * rho * r**3 / Mr
    c1 = (r / r[-1])**3 / (Mr / Mr[-1])
    As = N2 * r / g

    Vg_x = interp1d(x, Vg)
    c1_x = interp1d(x, c1)
    As_x = interp1d(x, As)
    U_x = interp1d(x, U)

    return Vg_x, c1_x, As_x, U_x

def dydx(x, y, wsq, l, Vg_x, U_x, c1_x, As_x):
    # this should never happen, really...
    x = max(min(x, 1 - ZERO), 0)
    y1, y2, y3, y4 = y
    Vg = Vg_x(x)
    U = U_x(x)
    c1 = c1_x(x)
    As = As_x(x)

    return np.array([
        y1 * (Vg - 3) + (l * (l + 1) / (c1 * wsq) - Vg) * y2 + Vg * y3,
        (c1 * wsq - As) * y1 + (As - U + 1) * y2 - As * y3,
        (1 - U) * y3 + y4,
        U * As * y1 + U * Vg * y2 + (l * (l + 1) - U * Vg) * y3 - U * y4,
    ]) / x

def wrons(wsq, l, Vg_x, U_x, c1_x, As_x, xmid=XMID,
          xin=ZERO, xout=1-ZERO,
          rtol=TOL2, method=METHOD, m=M_DEFAULT, **kwargs):
    '''
    we need two linearly independent solutions; yc1, yc2, ys1, ys2
    '''
    xmid = max(xin, xmid)
    f1 = 1
    f2 = -1
    g1 = 1
    g2 = -1
    yc_0_base = xin**(l - 2)
    yc1_0 = [
        yc_0_base,
        c1_x(xin) * wsq / l * yc_0_base,
        f1 * yc_0_base,
        f1 * l * yc_0_base,
    ]
    yc2_0 = [
        yc_0_base,
        c1_x(xin) * wsq / l * yc_0_base,
        f2 * yc_0_base,
        f2 * l * yc_0_base,
    ]
    ys1_0 = [
        1,
        1 + g1,
        g1,
        -g1 * (l + 1),
    ]
    ys2_0 = [
        1,
        1 + g2,
        g2,
        -g2 * (l + 1),
    ]
    kwargs = dict(rtol=rtol, method=method, **kwargs)
    args = [wsq, l, Vg_x, U_x, c1_x, As_x]
    retc1 = solve_ivp(dydx, (xin, xmid), yc1_0, args=args, **kwargs)
    retc2 = solve_ivp(dydx, (xin, xmid), yc2_0, args=args, **kwargs)
    rets1 = solve_ivp(dydx, (xout, xmid), ys1_0, args=args, **kwargs)
    rets2 = solve_ivp(dydx, (xout, xmid), ys2_0, args=args, **kwargs)

    wronskian = np.array([
        retc1.y[ :,-1],
        retc2.y[ :,-1],
        -rets1.y[ :,-1],
        -rets2.y[ :,-1]])
    det = np.linalg.det(wronskian)
    # logger.info('%.15f' % wsq, det)
    return det

def wrons_nokw(wsq, l, Vg_x, U_x, c1_x, As_x, xmid, xin, xout, rtol, method, m):
    return wrons(wsq, l, Vg_x, U_x, c1_x, As_x,
                 xmid=xmid, xin=xin, xout=xout, rtol=rtol, method=method, m=m)
def opt_func(yl, yr, l, Vg_x, U_x, c1_x, As_x,
             xmid=XMID, xin=ZERO, xout=1 - ZERO,
             rtol=TOL2, method=METHOD, m=M_DEFAULT):
    my_opt = lambda wsq: wrons_nokw(wsq, l, Vg_x, U_x, c1_x, As_x,
                                      xmid, xin, xout, rtol, method, m)
    try:
        return opt.brenth(my_opt, yl, yr, xtol=1e-4)
    except ValueError:
        logger.info('ERROR: f(a) f(b) do not have opposite signs', yl, yr)
        raise

def get_y(wsq, l, Vg_x, U_x, c1_x, As_x, xmid=XMID, xin=ZERO, xout=1 - ZERO,
          npts=2000, **kwargs):
    '''
    we need two linearly independent solutions; yc1, yc2, ys1, ys2
    '''
    xmid = max(xin, xmid)
    f1 = 1e-3
    f2 = -1e-3
    g1 = 1e-3
    g2 = -1e-3
    yc_0_base = 1e3 # xin**(l - 2)
    yc1_0 = [
        yc_0_base,
        c1_x(xin) * wsq / l * yc_0_base,
        f1 * yc_0_base,
        f1 * l * yc_0_base,
    ]
    yc2_0 = [
        yc_0_base,
        c1_x(xin) * wsq / l * yc_0_base,
        f2 * yc_0_base,
        l * yc_0_base,
    ]
    ys1_0 = [
        1,
        1 + g1,
        g1,
        -g1 * (l + 1),
    ]
    ys2_0 = [
        1,
        1 + g2,
        g2,
        -g2 * (l + 1),
    ]
    args = [wsq, l, Vg_x, U_x, c1_x, As_x]
    retc1 = solve_ivp(dydx, (xin, xmid), yc1_0, args=args, dense_output=True, **kwargs)
    retc2 = solve_ivp(dydx, (xin, xmid), yc2_0, args=args, dense_output=True,**kwargs)
    rets1 = solve_ivp(dydx, (xout, xmid), ys1_0, args=args, dense_output=True,**kwargs)
    rets2 = solve_ivp(dydx, (xout, xmid), ys2_0, args=args, dense_output=True,**kwargs)
    wronskian = np.array([
        retc1.y[ :,-1],
        retc2.y[ :,-1],
        -rets1.y[ :,-1],
        -rets2.y[ :,-1]]).T
    eigs, eigvs = np.linalg.eig(wronskian)
    z_idx = np.argmin(np.abs(eigs))
    mults = np.real(eigvs[ :, z_idx])

    _c_t = np.sort(np.concatenate((retc1.t, retc2.t)))
    c_t = np.linspace(_c_t.min(), _c_t.max(), npts // 2)
    y_c1 = retc1.sol(c_t)
    y_c2 = retc2.sol(c_t)
    _s_t = np.sort(np.concatenate((rets1.t, rets2.t)))
    s_t = np.linspace(_s_t.min(), _s_t.max(), npts // 2)
    y_s1 = rets1.sol(s_t)
    y_s2 = rets2.sol(s_t)

    y_crit = np.concatenate((
        mults[0] * y_c1 + mults[1] * y_c2,
        mults[2] * y_s1 + mults[3] * y_s2), axis=1)
    y_crit *= np.sign(y_crit[0, -1])
    x_crit = np.concatenate((c_t, s_t))
    return x_crit, y_crit

def get_J2(r, rho, Mr, Ws_sq_today):
    '''
    returns J_2 / (Omega_s / Omega_dyn)^2, using Fuller 2014 App a.

    NB: we solve as an IVP first, then normalize epsilon(r=1)
    '''
    rho_interp = interp1d(r, rho)
    Mr_interp = interp1d(r, Mr)
    # Ws_sq = W^2 / (GM/R^3) at constant spin AM yields that
    # Ws_sq \propto 1 / R
    # TODO more elegantly please...
    if abs(Mr[-1] - MJUP) < 0.1:
        Ws_sq = Ws_sq_today * RJUP / r[-1]
    else:
        Ws_sq = Ws_sq_today * RSAT / r[-1]

    # F14 A.2
    def deps_dr(r, y):
        eps, deps = y
        # g = G * Mr / r**2
        # G * rho / g = rho * r**2 / Mr
        ret_dr = (
            deps,
            6 / r**2 * eps - (8 * np.pi * rho_interp(r) * r**2 / Mr_interp(r))
                * (deps + eps / r)
        )
        return ret_dr
    ret = solve_ivp(deps_dr, (r.min(), r.max()), (1, 0),
                    method='DOP853', rtol=1e-9, dense_output=True)
    norm = ret.y[0, -1] # divide eps, deps by this number

    # F14 A.3
    def q_int(r_):
        eps, deps = ret.sol(r_) / norm
        return 4 * np.pi * rho_interp(r_) * r_**4 * (5 * eps + r_ * deps)
    Qbase = (5 * Mr[-1] * r[-1]**2)
    tol= 1e-4 * Qbase
    # scipy.integrate.quadrature was removed in SciPy 1.14; quad is the
    # supported replacement (same (value, error) return convention).
    Q, Qerr = quad(q_int, r.min(), r.max(), epsrel=1e-4, epsabs=tol)
    Q, Qerr = Q / Qbase, Qerr / Qbase
    j2 = Q / (3 * (1 - Q)) * Ws_sq
    return j2

def sweep(x, M, R, Vg_x, c1_x, As_x, U_x,
          l=2, wsq_arr=np.geomspace(0.05, 30, 401), nthreads=4,
          xmid=XMID, xin=ZERO, xout=1 - ZERO,
          tol=TOL2, method=METHOD, fn='2polytrope', m=M_DEFAULT):
    dwsq = np.mean(np.diff(wsq_arr))

    # plot W(wsq)
    pkl_fn = '%s.pkl' % fn
    if not os.path.exists(pkl_fn):
        logger.info('Running %s' % pkl_fn)
        args = [
            (wsq, l, Vg_x, U_x, c1_x, As_x, xmid, xin, xout, tol, method, m)
            for wsq in wsq_arr
        ]
        if nthreads > 1:
            with Pool(nthreads) as p:
                w_arr = p.starmap(wrons_nokw, args)
        else:
            w_arr = [wrons_nokw(*a) for a in args]
        w_arr = np.array(w_arr)

        sign_changes = np.where(w_arr[1: ] / w_arr[ :-1] < 0)[0]
        cands_left = wsq_arr[sign_changes]
        cands_right = wsq_arr[sign_changes + 1]
        logger.info('%d sign changes detected' % len(cands_left))

        x_crit_lst = []
        y_crit_lst = []
        args = [
            (yl, yr, l, Vg_x, U_x, c1_x, As_x, xmid, xin, xout, tol, method, m)
            for yl, yr in zip(cands_left, cands_right)
        ]
        if nthreads > 1:
            with Pool(nthreads) as p:
                wsq_crits = p.starmap(opt_func, args)
        else:
            wsq_crits = [opt_func(*a) for a in args]
        for wsq_crit in wsq_crits:
            x_crit, y_crit = get_y(wsq_crit, l, Vg_x, U_x, c1_x, As_x,
                                   xin=xin, xout=xout, rtol=tol, method=method)
            x_crit_lst.append(x_crit)
            y_crit_lst.append(y_crit)
        with lzma.open(pkl_fn, 'wb') as f:
            pickle.dump((w_arr, cands_left, cands_right, wsq_crits, x_crit_lst, y_crit_lst), f)
    else:
        with lzma.open(pkl_fn, 'rb') as f:
            logger.info('Loading %s' % pkl_fn)
            (w_arr, cands_left, cands_right, wsq_crits, x_crit_lst, y_crit_lst) = pickle.load(f)

    wsq_crits = np.array(wsq_crits)
    nu_mhzs = np.sqrt(
        G * M / (4 * np.pi**2 * R**3) * wsq_crits) * MHZ

    # try to plot the modes closest to n=0
    absnmin = np.inf
    absnmin_idx = -1
    n_arr = []
    for idx, (x_crit, y_crit, nu) in enumerate(
            zip(x_crit_lst, y_crit_lst, nu_mhzs)):
        # get the mode order (ignore innermost points)
        offset = 3
        y1_zero_idxs = np.where(
            y_crit[0, offset + 1: ] / y_crit[0, offset:-1] < 0)[0]
        n = (
            0 if y_crit[0, offset] * y_crit[1, offset] > 0
            else 1
        )
        for z_idx in y1_zero_idxs:
            z_idx += offset
            dy1 = y_crit[0, z_idx + 1] - y_crit[0, z_idx]
            y2 = np.mean(y_crit[1, z_idx:z_idx + 2])
            n -= int(np.sign(dy1 * y2))
        n_arr.append(n)
        print('\t\tl, n, nu, w/wdyn', l, n, nu, np.sqrt(wsq_crits[idx]))
        if abs(n) < absnmin:
            absnmin = abs(n)
            absnmin_idx = idx

    return absnmin_idx, n_arr, nu_mhzs

def get_modes_for_profile(
        r, S, rho, P, Mr, yarr, zarr, N2, fldr, fn, eos=mix_eos,
        lmin=2, lmax=3, nmin=3, nmax=3,
        nthreads=int(os.getenv('NTHREADS', '8')),
):
    # flip order of all vars
    r = r[::-1]
    S = S[::-1]
    rho = rho[::-1]
    P = P[::-1]
    Mr = Mr[::-1]
    yarr = yarr[::-1]
    zarr = zarr[::-1]
    N2 = N2[::-1]
    sweep_kws = dict(nthreads=nthreads, method='DOP853')
    Vg, c1, As, U = get_interps_for_structure(
        r, S, rho, P, Mr, yarr, zarr, N2, eos)
    if abs(Mr[-1] - MJUP) < 0.1:
        Ws_sq_today = Ws_sq_today_JUP
    else:
        Ws_sq_today = Ws_sq_today_SAT
    j2_val = get_J2(r, rho, Mr, Ws_sq_today)
    m_metals = 0

    for rho_curr, rin, rout, z in zip(rho, r[ :-1], r[1: ], zarr):
        m_metals += (
            4 * np.pi / 3 * (rout**3 - rin**3) * z
            * rho_curr)
    logger.info('Metal Frac of %s: %.3f M_earth' % (fldr, m_metals / MEARTH))

    modes = []
    xin = r.min() / r.max()
    # xin = r[len(r) - kcore + 1] / r.max()
    for l in range(lmin, lmax + 1):
        ret_sweep = sweep(
            r / r.max(), Mr.max(), r.max(), Vg, c1, As, U, l=l,
            xin=xin,
            fn=os.path.join(fldr, '%s_l%d' % (fn, l)), **sweep_kws)
        cent, n_arr, nu_mhzs = ret_sweep
        for n, nu in list(zip(n_arr, nu_mhzs))[max(cent - nmin, 0):cent + nmax + 1]:
            modes.append((l, n, nu))
    return modes, (m_metals / MEARTH, j2_val)

def get_all_modes(r_b_arr, S_arr, rho_arr, P_arr, M_b, yarr_arr, zarr_arr,
                  brunt_arr, folder, base_fn, stride=1, **kwargs):
    to_ret = []
    for idx, (r_b, S, rho, P, yarr, zarr, N2_i) in list(
        enumerate(zip(r_b_arr, S_arr, rho_arr, P_arr, yarr_arr, zarr_arr,
                      brunt_arr))
    )[MODE_START::stride]:
        # upconvert N2, whatever. Drop innermost boundary
        N2 = np.concatenate((N2_i, [0]))
        to_ret.append(get_modes_for_profile(
            r_b[ :-1], S, rho, P, M_b[ :-1], yarr, zarr, N2, folder, '%s_%d' % (base_fn, idx)))
    return to_ret
