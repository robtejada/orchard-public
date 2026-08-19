#
# Calculate gravitational moments based on Theory of Figures-4th order approach of Nettelmann 2017
# Code adapted from Chris Mankovich (2017)
#
# Modified by Ankan Sur
# Date: September 21 2024
#
# saves .txt file with the columns: J2, J4, J6, J8, oblateness, MoI, Req, Rpol
# =======================================================================================

import numpy as np
from numba import njit
from utils.const import G
try:
    from scipy.integrate import trapz, cumtrapz
except ImportError:
    from scipy.integrate import trapezoid as trapz
    from scipy.integrate import cumulative_trapezoid as cumtrapz
from scipy.interpolate import splrep, splev, interp1d
# from scipy.interpolate import interp1d
from scipy.special import legendre, roots_legendre
from scipy.integrate import quad
import time
import os
import pdb
import glob
from utils import const
import argparse
import signal
import sys
from tqdm import tqdm
from utils.common import *

R_jup_eq = 7.1492e9
R_sat_eq = 6.0268e9
R_ura_eq = 2.5559e9
R_nep_eq = 2.4764e9
M_jup = 1.89914e30
msun = 1.988e33
rjup_mean = 69911e5
G = const.G

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
pp0 = np.poly1d(legendre(0))
pp2 = np.poly1d(legendre(2))
pp4 = np.poly1d(legendre(4))
pp6 = np.poly1d(legendre(6))
pp8 = np.poly1d(legendre(8))

def set_req_rpol(l, s0, s2, s4, s6, s8):
    '''
    calculate equatorial and polar radius vectors from the figure functions s_2n and legendre polynomials P_2n.
    see N17 eq. (B.1) or ZT78 eq. (27.1).
    also calculates q from m and new r_eq[-1].
    '''

    #equator: mu = cos(pi/2) = 0
    r_eq = l * (1. + s0 * pp0(0.) \
                                + s2 * pp2(0.) \
                                + s4 * pp4(0.) \
                                + s6 * pp6(0.) \
                                + s8 * pp8(0.))

    # pole: mu = cos(0) = 1
    r_pol = l * (1. + s0 * pp0(1.) \
                                + s2 * pp2(1.) \
                                + s4 * pp4(1.) \
                                + s6 * pp6(1.) \
                                + s8 * pp8(1.))

    return r_eq, r_pol


@njit(cache=True)
def set_f2n_f2np(s2, s4, s6, s8):
    '''
    N17 eqs. (B.16) and (B.17).

    @njit-compiled: ~11x faster than the equivalent numpy expression because
    Numba fuses the nine output arrays into a single pass and avoids
    per-statement temporary allocations. Numerical output agrees with the
    pure-numpy reference to <=2.3e-16 relative error (float64 machine eps).
    '''

    f2 = 3. / 5 * s2 + 12. / 35 * s2 ** 2 + 6. / 175 * s2 ** 3 \
                + 24. / 35 * s2 * s4 + 40. / 231 * s4 ** 2 \
                + 216. / 385 * s2 ** 2 * s4 - 184. / 1925 * s2 ** 4

    f4 = 1. / 3 * s4 + 18. / 35 * s2 ** 2 + 40. / 77 * s2 * s4 \
                + 36. / 77 * s2 ** 3 + 90. / 143 * s2 * s6 \
                + 162. / 1001 * s4 ** 2 + 6943. / 5005 * s2 ** 2 * s4 \
                + 486. / 5005 * s2 ** 4

    f6 = 3. / 13 * s6 + 120. / 143 * s2 * s4 + 72. / 143 * s2 ** 3 \
                + 336. / 715 * s2 * s6 + 80. / 429 * s4 ** 2 \
                + 216. / 143 * s2 ** 2 * s4 + 432. / 715 * s2 ** 4

    f8 = 3. / 17 * s8 + 168. / 221 * s2 * s6 + 2450. / 7293 * s4 ** 2 \
                + 3780. / 2431 * s2 ** 2 * s4 + 1296. / 2431 * s2 ** 4

    f0p = 3. / 2 - 3. / 10 * s2 ** 2 - 2. / 35 * s2 ** 3 - 1. / 6 * s4 ** 2 \
                    - 6. / 35 * s2 ** 2 * s4 + 3. / 50 * s2 ** 4

    f2p = 3. / 5 * s2 - 3. / 35 * s2 ** 2 - 6. / 35 * s2 * s4 \
                + 36. / 175 * s2 ** 3 - 10. / 231 * s4 ** 2 - 17. / 275 * s2 ** 4 \
                + 36. / 385 * s2 ** 2 * s4

    f4p = 1. / 3 * s4 - 9. / 35 * s2 ** 2 - 20. / 77 * s2 * s4 \
                - 45. / 143 * s2 * s6 - 81. / 1001 * s4 ** 2 + 72. / 385 * s2 ** 3\
                + 4579/5005 * s2 **2 * s4 - 12798/25025 * s2 **4
                # f4p has an s_2**3 in Z+T. NN says it shouldn't be there (Oct 4 2017).

    f6p = 3. / 13 * s6 - 75. / 143 * s2 * s4 + 270. / 1001 * s2 ** 3 \
                - 50. / 429 * s4 ** 2 + 810. / 1001 * s2 ** 2 * s4 - 54. / 143 * s2 ** 4 \
                - 42. / 143 * s2 * s6

    f8p = 3. / 17 * s8 - 588. / 1105 * s2 * s6 - 1715. / 7293 * s4 ** 2 \
                + 2352. / 2431 * s2 ** 2 * s4 - 4536. / 12155 * s2 ** 4


    return f2, f4, f6, f8, f0p, f2p, f4p, f6p, f8p


def _cumtrapz0(y, x):
    """Exact inline of scipy.integrate.cumulative_trapezoid(y, x=x,
    initial=0.): res = cumsum(diff(x) * (y[1:] + y[:-1]) / 2.0) with a
    prepended 0.  Same operations in the same order (the inner sum is a
    single commutative FP add), so the result is BIT-IDENTICAL to the
    scipy call -- this exists purely to bypass scipy's per-call
    validation/dispatch overhead (~760k calls per evolution run)."""
    res = np.cumsum(np.diff(x) * (y[1:] + y[:-1]) / 2.0)
    out = np.empty(res.size + 1, dtype=res.dtype)
    out[0] = 0.0
    out[1:] = res
    return out


def set_ss2n_ss2np(l, rho, m,  f2, f4, f6, f8, f0p, f2p, f4p, f6p, f8p):
    '''
    N17 eq. (B.9).
    '''
    if l[0] == 0.:
        l[0] = l[1] / 2

    z = l / l[-1]
    mtot = m[-1]
    rhobar = 3. * mtot / 4 / np.pi / l[-1] ** 3

    ss2_integral = _cumtrapz0(z ** (2. + 3) * f2 / rhobar, rho)
    ss4_integral = _cumtrapz0(z ** (4. + 3) * f4 / rhobar, rho)
    ss6_integral = _cumtrapz0(z ** (6. + 3) * f6 / rhobar, rho)
    ss8_integral = _cumtrapz0(z ** (8. + 3) * f8 / rhobar, rho)

    ss0p_integral = _cumtrapz0(z ** (2. - 0) * f0p / rhobar, rho)
    ss2p_integral = _cumtrapz0(z ** (2. - 2) * f2p / rhobar, rho)
    ss4p_integral = _cumtrapz0(z ** (2. - 4) * f4p / rhobar, rho)
    ss6p_integral = _cumtrapz0(z ** (2. - 6) * f6p / rhobar, rho)
    ss8p_integral = _cumtrapz0(z ** (2. - 8) * f8p / rhobar, rho)

    # int_z^1 = int_0^1 - int_0^z
    ss0p_integral = ss0p_integral[-1] - ss0p_integral
    ss2p_integral = ss2p_integral[-1] - ss2p_integral
    ss4p_integral = ss4p_integral[-1] - ss4p_integral
    ss6p_integral = ss6p_integral[-1] - ss6p_integral
    ss8p_integral = ss8p_integral[-1] - ss8p_integral

    ss0 = m / mtot / z ** 3 # (B.8)

    ss2 = rho / rhobar * f2 - 1. / z ** (2. + 3) * ss2_integral
    ss4 = rho / rhobar * f4 - 1. / z ** (4. + 3) * ss4_integral
    ss6 = rho / rhobar * f6 - 1. / z ** (6. + 3) * ss6_integral
    ss8 = rho / rhobar * f8 - 1. / z ** (8. + 3) * ss8_integral

    ss0p = -1. * rho / rhobar * f0p + 1. / z ** (2. - 0) \
                    * (rho[-1] / rhobar * f0p[-1] - ss0p_integral)

    ss2p = -1. * rho / rhobar * f2p + 1. / z ** (2. - 2) \
                * (rho[-1] / rhobar * f2p[-1] - ss2p_integral)

    ss4p = -1. * rho / rhobar * f4p + 1. / z ** (2. - 4) \
                * (rho[-1] / rhobar * f4p[-1] - ss4p_integral)

    ss6p = -1. * rho / rhobar * f6p + 1. / z ** (2. - 6) \
                * (rho[-1] / rhobar * f6p[-1] - ss6p_integral)

    ss8p = -1. * rho / rhobar * f8p + 1. / z ** (2. - 8) \
                * (rho[-1] / rhobar * f8p[-1] - ss8p_integral)


    return ss0, ss2, ss4, ss6, ss8, ss0p, ss2p, ss4p, ss6p, ss8p

@njit(cache=True)
def aa2n(small, s2n, ss0, ss2, ss2p, ss4, ss4p, ss6, ss6p, ss8, ss8p):
    """B.12-15

    @njit-compiled: ~14x faster than the equivalent pure-numpy version for
    the same reason set_f2n_f2np is (loop fusion, fewer allocations).
    Output agrees with numpy to <=1.3e-17 relative error.
    """
    s2 = s2n[0]; s4 = s2n[1]; s6 = s2n[2]; s8 = s2n[3]


    s2new = 1/ss0 *(( 2. / 7 * s2 ** 2 + 4. / 7 * s2 * s4 - 29. / 35 * s2 ** 3 + 100. / 693 * s4 ** 2 \
            + 454. / 1155 * s2 ** 4 - 36. / 77 * s2 ** 2 * s4) * ss0 \
            + (1. - 6. / 7 * s2 - 6. / 7 * s4 + 111. / 35 * s2 ** 2 - 1242. / 385 * s2 ** 3 + 144. / 77 * s2 * s4) * ss2 \
            + (-10. / 7 * s2 - 500. / 693 * s4 + 180. / 77 * s2 ** 2) * ss4 \
            + (1. + 4. / 7 * s2 + 1. / 35 * s2 ** 2 + 4. / 7 * s4 - 16. / 105 * s2 ** 3 + 24. / 77 * s2 * s4) * ss2p \
            + (8. / 7 * s2 + 72. / 77 * s2 ** 2 + 400. / 693 * s4) * ss4p \
            + small / 3 * (-1. + 10. / 7 * s2 + 9. / 35 * s2 ** 2 - 4. / 7 * s4 + 20./ 77 * s2 * s4 - 26. / 105 * s2 ** 3))

    s4new = 1/ss0 *((18. / 35 * s2 ** 2 - 108. / 385 * s2 ** 3 + 40. / 77 * s2 * s4 + 90. / 143 * s2 * s6 + 162. / 1001 * s4 ** 2 \
            + 16902. / 25025 * s2 ** 4 - 7369. / 5005 * s2 ** 2 * s4) * ss0 \
            + (-54. / 35 * s2 - 60. / 77 * s4 + 648. / 385 * s2 ** 2 \
            - 135. / 143 * s6 + 21468. / 5005 * s2 * s4 - 122688. / 25025 * s2 ** 3) * ss2 \
            + (1. - 100. / 77 * s2 - 810. / 1001 * s4 + 6368. / 1001 * s2 ** 2) * ss4 \
            - 315. / 143 * s2 * ss6 \
            + (36. / 35 * s2 + 108. / 385 * s2 ** 2 + 40. / 77 * s4 + 3578. / 5005 * s2 * s4 \
            - 36. / 175 * s2 ** 3 + 90. / 143 * s6) * ss2p \
            + (1. + 80. / 77 * s2 + 1346. / 1001 * s2 ** 2 + 648. / 1001 * s4) * ss4p \
            + 270. / 143 * s2 * ss6p \
            + small / 3 * (-36. / 35 * s2 + 114. / 77 * s4 + 18. / 77 * s2 ** 2 \
            - 978. / 5005 * s2 * s4 + 36. / 175 * s2 ** 3 - 90. / 143 * s6))

    s6new = 1/ss0 *((10. / 11 * s2 * s4 - 18. / 77 * s2 ** 3 + 28. / 55 * s2 * s6 + 72. / 385 * s2 ** 4 + 20. / 99 * s4 ** 2 \
            - 54. / 77 * s2 ** 2 * s4) * ss0 \
            + (-15. / 11 * s4 + 108. / 77 * s2 ** 2 - 42. / 55 * s6 - 144. / 77 * s2 ** 3 + 216. / 77 * s2 * s4) * ss2 \
            + (-25. / 11 * s2 - 100. / 99 * s4 + 270. / 77 * s2 ** 2) * ss4 \
            + (1. - 98. / 55 * s2) * ss6 \
            + (10. / 11 * s4 + 18. / 77 * s2 ** 2 + 36. / 77 * s2 * s4 + 28. / 55 * s6) * ss2p \
            + (20. / 11 * s2 + 108. / 77 * s2 ** 2 + 80. / 99 * s4) * ss4p \
            + (1. + 84. / 55 * s2) * ss6p \
            + small / 3 * (-10. / 11 * s4 - 18. / 77 * s2 ** 2 + 34. / 77 * s2 * s4 + 82. / 55 * s6))

    s8new = 1/ss0 *((56. / 65 * s2 * s6 + 72. / 715 * s2 ** 4 + 490. / 1287 * s4 ** 2 - 84. / 143 * s2 ** 2 * s4) * ss0 \
            + (-84. / 65 * s6 - 144. / 143 * s2 ** 3 + 336. / 143 * s2 * s4) * ss2 \
            + (-2450. / 1287 * s4 + 420. / 143 * s2 ** 2) * ss4 \
            - 196. / 65 * s2 * ss6 \
            + ss8 \
            + (56. / 65 * s6 + 56. / 143 * s2 * s4) * ss2p \
            + (1960. / 1287 * s4 + 168. / 143 * s2 ** 2) * ss4p \
            + 168. / 65 * s2 * ss6p \
            + ss8p \
            + small / 3 * (-56. / 65 * s6 - 56. / 143 * s2 * s4))


    # Build the (4, nz) stack explicitly — numba prefers np.empty + row
    # assignment over np.array([...]) of a Python list.
    out = np.empty((4, s2.shape[0]))
    out[0] = s2new
    out[1] = s4new
    out[2] = s6new
    out[3] = s8new
    return out


# Module-level Gauss-Legendre nodes and weights for new_MoI's latitude
# integration. 32 nodes is exact for polynomials of degree <= 63; the MoI
# integrand is degree 5*8 + 2 = 42 in mu, so this is machine-precision
# accurate with headroom. Computed once at import.
_MOI_GL_NODES, _MOI_GL_WEIGHTS = roots_legendre(32)


def new_MoI(s_values, rho_values, s0_values, s2_values, s4_values,
            s6_values, s8_values):
    """
    Axial moment of inertia of the converged ToF figure.

        I = 2*pi int ds rho(s) int dmu r^4 (1 - mu^2) (dr/ds)|_mu
          = 2*pi int dmu (1 - mu^2) int rho d( r(s, mu)^5 / 5 )

    with r(s, mu) = s * (1 + s0 + s2*P2 + s4*P4 + s6*P6 + s8*P8) and
    mu = cos(theta). The level-surface Jacobian (dr/ds)|_mu of the volume
    element is carried exactly by evaluating the radial integral as a
    Stieltjes integral in x = r^5/5 (the antiderivative of r^4 dr) per
    Gauss-Legendre latitude node, so no finite-difference derivatives of
    the figure functions are needed. In the constant-density limit the
    radial integral telescopes to the boundary-only value, recovering the
    homogeneous-body invariant I/(M R_eq^2) = 2/5 to ToF truncation error.
    Matches the Nettelmann (2021) by-parts MoI form to <~1e-7 relative.
    """
    mu = _MOI_GL_NODES              # shape (M,)
    w  = _MOI_GL_WEIGHTS            # shape (M,)
    # Precomputed Legendre polynomials at the GL nodes (same for every zone):
    P2 = 0.5 * (3 * mu**2 - 1)
    P4 = (1.0 / 8.0) * (35 * mu**4 - 30 * mu**2 + 3)
    P6 = (1.0 / 16.0) * (231 * mu**6 - 315 * mu**4 + 105 * mu**2 - 5)
    P8 = (1.0 / 128.0) * (6435 * mu**8 - 12012 * mu**6 + 6930 * mu**4
                          - 1260 * mu**2 + 35)
    sin2 = 1.0 - mu**2              # = sin^2(theta)
    # Broadcast over zones (Z) and latitude nodes (M):
    f = (s0_values[:, None]
         + s2_values[:, None] * P2[None, :]
         + s4_values[:, None] * P4[None, :]
         + s6_values[:, None] * P6[None, :]
         + s8_values[:, None] * P8[None, :])
    r_zm = s_values[:, None] * (1.0 + f)          # (Z, M)
    x = r_zm**5 / 5.0
    y = np.broadcast_to(rho_values[:, None], x.shape)
    # Radial Stieltjes integral int rho d(r^5/5) at each latitude node:
    I_mu = np.trapezoid(y, x=x, axis=0)           # shape (M,)
    # Gauss-Legendre sum over mu:
    return 2.0 * np.pi * np.sum(w * sin2 * I_mu)


def get_moments(l, rho, m, omega, initial_guess=None):

    tolerance = 1e-12
    max_iterations = 1000
    omega0 = np.sqrt(G * m[-1] / l[-1] ** 3)
    small = (omega / omega0) ** 2
    nz = len(rho)
    rm = l[-1]

    def jacobi_iteration(guess, tolerance, max_iterations):
        s2, s4, s6, s8 = guess
        prev_s2, prev_s4, prev_s6, prev_s8 = guess

        for iteration in range(max_iterations):

            f2, f4, f6, f8, f0p, f2p, f4p, f6p, f8p = set_f2n_f2np(prev_s2, prev_s4, prev_s6, prev_s8)

            ss0, ss2, ss4, ss6, ss8, ss0p, ss2p, ss4p, ss6p, ss8p = set_ss2n_ss2np(l, rho, m,  f2, f4, f6, f8, f0p, f2p, f4p, f6p, f8p)

            s2n = np.array([prev_s2, prev_s4, prev_s6, prev_s8])

            s2_new, s4_new, s6_new, s8_new = aa2n(small, s2n, ss0, ss2, ss2p, ss4, ss4p, ss6, ss6p, ss8, ss8p)

            # Max-abs relative error across the four figure functions. The
            # previous `.all()`-based expression returned booleans (0/1), so
            # `error` was stuck at 0.5 every iteration and the loop always ran
            # to `max_iterations` regardless of actual convergence. With a
            # proper norm + tight tolerance (1e-12), warm-started calls
            # typically converge in tens of iterations instead of 1000, and
            # the converged s-functions agree with the previous pseudo-1000-
            # iter answer to <= tolerance.
            max_diff = max(
                np.max(np.abs(s2_new - prev_s2)),
                np.max(np.abs(s4_new - prev_s4)),
                np.max(np.abs(s6_new - prev_s6)),
                np.max(np.abs(s8_new - prev_s8)),
            )
            max_val = max(
                np.max(np.abs(prev_s2)),
                np.max(np.abs(prev_s4)),
                np.max(np.abs(prev_s6)),
                np.max(np.abs(prev_s8)),
                1e-300,  # floor to avoid 0/0 if all prev arrays are exact zero
            )
            error = max_diff / max_val
            if error < tolerance:
                break

            #Update the previous values for the next iteration
            prev_s2, prev_s4, prev_s6, prev_s8 = s2_new, s4_new, s6_new, s8_new

        return s2_new, s4_new, s6_new, s8_new, ss2, ss4, ss6, ss8

    #Initial guess for s2, s4, s6, s8
    if initial_guess is None:
        err = small
        guess = [np.zeros(nz) + err,
                 np.zeros(nz) + err,
                 np.zeros(nz) + err,
                 np.zeros(nz) + err]
    else:
        guess = initial_guess
    #err = small
    #initial_guess = [np.zeros(nz)+err, np.zeros(nz)+err, np.zeros(nz)+err, np.zeros(nz)+err]

    #Jacobi iteration
    solution = jacobi_iteration(guess, tolerance, max_iterations)

    s2 = solution[0]
    s4 = solution[1]
    s6 = solution[2]
    s8 = solution[3]
    ss2 = solution[4]
    ss4 = solution[5]
    ss6 = solution[6]
    ss8 = solution[7]
    s0 = - 1. / 5 * s2 ** 2 \
            - 2. / 105 * s2 ** 3 \
            - 1. / 9 * s4 ** 2 \
            - 2. / 35 * s2 ** 2 * s4

    r_eq, r_pol = set_req_rpol(l, s0, s2, s4, s6, s8)
    oblateness = (r_eq[-1]-r_pol[-1])/r_eq[-1]

    j2 = - 1. * (rm / r_eq[-1]) ** 2. * ss2[-1]
    j4 = - 1. * (rm / r_eq[-1]) ** 4. * ss4[-1]
    j6 = - 1. * (rm / r_eq[-1]) ** 6. * ss6[-1]
    j8 = - 1. * (rm / r_eq[-1]) ** 8. * ss8[-1]
    new_I = new_MoI(l, rho, s0, s2, s4, s6, s8)

    j2n = np.array([j2*1e6, j4*1e6, j6*1e6, j8*1e6, oblateness, r_eq[-1], r_pol[-1], new_I])
    return j2n, (s0, s2, s4, s6, s8)

def save_results(folder,  J2_list, J4_list, J6_list, J8_list, f_list, I_list, I2_list, Req, Rpol, Rvol):
    np.savetxt(f"{folder}/Js_tof2.txt", np.c_[ J2_list, J4_list, J6_list, J8_list, f_list, I_list, I2_list, Req, Rpol, Rvol])

def handle_exit(signal, frame):
    print("Interrupted! Saving data...")
    save_results(folder,  J2_list, J4_list, J6_list, J8_list, f_list, I_list, I2_list, Req, Rpol, Rvol)
    sys.exit(0)

def get_MOI(r, rho):
    I = np.abs(8*np.pi/3*np.trapezoid(r**4*rho,x=r))
    return I


if __name__ == "__main__":
    folder = '%s/%s' % (args.data_folder, model_fldr)

    planet = config['boundary_condition']['planet'].lower()
    mgrid = np.genfromtxt(f"{folder}/mgrid.txt")[::-1]
    age = np.genfromtxt(f"{folder}/evolution_data.txt",usecols=0)
    m = (mgrid[:-1]+mgrid[1:])/2.0
    length = len(np.genfromtxt(f"{folder}/rad_profiles.txt"))
    Idata = np.genfromtxt(f"{folder}/evolution_data.txt",usecols=7) 
    omega = np.genfromtxt(f"{folder}/evolution_data.txt",usecols=6) # rotational frequency in rad/s
    age_456 = np.where(age>4.56*const.Gyr_to_s)[0][0] # present age in seconds
    C_MoI = float(config['hydrostatic_equilibrium']['C_MoI']) # moment of inertia factor
    period = float(config['hydrostatic_equilibrium']['period']) # rotation period in seconds

    mp = float(config['initial'].get('M_planet_unit_cgs', config['initial'].get('M_planet', '5.972167867791379e27'))) * float(config['initial'].get('M_Mearth', config['initial'].get('M_factor', '317.907')))

    # if planet=="jupiter":
    #     omega_0 = 2*np.pi/(period)
    #     mp = mjup
    #     I_0 = C_MoI*mjup*R_jup**2
    #     #I_0 = Idata[age_456]

    # else:
        #omega_0 = 2*np.pi/(10*3600+33*60+34)

    rp =  R_jup_eq if planet=="jupiter" else R_sat_eq if planet=="saturn" else R_ura_eq if planet=="uranus" else R_nep_eq
    omega_0 = 2 * np.pi / period
    I_0 = C_MoI*mp*rp**2
    ## list to store values: f is oblateness
    J2_list, J4_list, J6_list, J8_list, f_list, I_list, I2_list, Req, Rpol, Rvol = [], [], [], [], [], [], [], [], [], []

    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)

    try:
        prev_guess = None
        for idx in tqdm(range(length)):
            radius = np.genfromtxt(f"{folder}/rad_profiles.txt")[idx][::-1]
            rho = np.genfromtxt(f"{folder}/rho_profiles.txt")[idx][::-1]
            (j2,j4,j6,j8,obl,eq_rad,pol_rad,new_I),(s0,s2,s4,s6,s8) =  get_moments(radius, rho, m, omega[idx],initial_guess=prev_guess)
            prev_guess = [s2, s4, s6, s8]
            J2_list.append(j2)
            J4_list.append(j4)
            J6_list.append(j6)
            J8_list.append(j8)
            f_list.append(obl)
            I_list.append(Idata[idx])
            I2_list.append(new_I)
            Req.append(eq_rad)
            Rpol.append(pol_rad)
            Rvol.append(eq_rad**(2/3) * pol_rad**(1/3))

            # saving as it calculates
            save_results(folder, J2_list, J4_list, J6_list, J8_list, f_list, I_list, I2_list, Req, Rpol, Rvol)

    except Exception as e:
        #print(f"An error occurred: {e}")
        save_results(folder, J2_list, J4_list, J6_list, J8_list, f_list, I_list, I2_list, Req, Rpol, Rvol)
        raise
        sys.exit(1)

    # Final save after the loop completes
    print('Saving final data ...')
    save_results(folder, J2_list, J4_list, J6_list, J8_list, f_list, I_list, I2_list, Req, Rpol, Rvol)