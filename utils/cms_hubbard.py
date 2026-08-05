#!/usr/bin/env python3
"""
Clean-room Concentric Maclaurin Spheroid (CMS) gravity solver — ORCHARD Phase 1.

Implemented from the published equations of Hubbard (2013, ApJ 768, 43) and
Militzer & Hubbard (2024, Icarus 411, 115955), NOT from the GPL nmovshov/CMS-planet
code (used only to cross-check equation forms). Computes the gravitational
harmonics J_2n of a rotating fluid planet represented as N nested constant-density
oblate spheroids, by iterating each spheroid's equipotential shape.

Conventions (Hubbard 2013):
  - Spheroids indexed i = 0 (outermost) ... N-1 (innermost).
  - lambda_i = a_i / a_0  : equatorial radius normalized to the outer one (lambda_0 = 1, descending).
  - zeta_i(mu) = r_i(mu)/a_i : shape, zeta_i(0)=1 at the equator (mu=cos(colat)=0), <1 at the pole.
  - delta_i = rho_i - rho_{i-1} : density jump (rho_{-1}=0), positive going inward.
  - q = omega^2 a_0^3 / (G M) : rotation parameter.
  - mu in [0,1] (even symmetry); integrals use Gauss-Legendre on [0,1].

Validation oracle: a single homogeneous Maclaurin spheroid has, exactly,
  zeta(mu) = 1/sqrt(1 + e^2 mu^2/(1-e^2)),  J_2n = (-1)^(n+1) 3/((2n+1)(2n+3)) e^(2n),
so J_2 = e^2/5, J_4 = -3/35 e^4, ...

Run self-tests:
    python utils/cms_hubbard.py
"""
import numpy as np
from scipy.special import eval_legendre

try:
    from numba import njit, prange
    _HAVE_NUMBA = True
except Exception:                       # pragma: no cover - numba optional
    _HAVE_NUMBA = False

    def njit(*args, **kwargs):          # no-op fallback decorator
        if len(args) == 1 and callable(args[0]):
            return args[0]

        def _wrap(f):
            return f
        return _wrap
    prange = range


# ----------------------------------------------------------------------------- quadrature / Legendre
def _gl01(L):
    """Gauss-Legendre nodes/weights on [0, 1]."""
    x, w = np.polynomial.legendre.leggauss(L)
    return 0.5 * (x + 1.0), 0.5 * w


def _legendre_even(mu, kmax):
    """Return array P[k, :] = P_{2k}(mu) for k = 0..kmax."""
    return np.array([eval_legendre(2 * k, mu) for k in range(kmax + 1)])


# ----------------------------------------------------------------------------- J_n from shapes (MH24 Eq 11-12)
def _Jn_from_shapes(lam, delta, zeta, mu, wt, nmax):
    """Conventional gravity harmonics J_0, J_2, ..., J_{2*nmax} from the shapes.

    J_n = (1/M) sum_i delta_i * Jhat_{i,n},
    Jhat_{i,n} = -(2pi/(n+3)) lam_i^(n+3) int_{-1}^{1} P_n(mu) zeta_i^(n+3) dmu
               = -(4pi/(n+3)) lam_i^(n+3) int_0^1 P_n zeta_i^(n+3) dmu   (even),
    M = sum_i delta_i (4pi/3) lam_i^3 int_0^1 zeta_i^3 dmu.
    The 4pi cancels in the ratio, so we drop it.
    """
    N = len(lam)
    P = _legendre_even(mu, nmax)              # (nmax+1, L)
    # mass (drop 4pi): sum_i delta_i lam_i^3 (1/3) int zeta^3
    Mtot = np.sum(delta * lam ** 3 * (1.0 / 3.0) * np.sum(wt * zeta ** 3, axis=1))
    Js = np.zeros(nmax + 1)
    for k in range(nmax + 1):
        n = 2 * k
        integ = np.sum(wt * P[k] * zeta ** (n + 3), axis=1)        # per layer
        Jhat = -(1.0 / (n + 3)) * lam ** (n + 3) * integ           # drop 4pi
        Js[k] = np.sum(delta * Jhat) / Mtot
    return Js                                                      # Js[0]=J0(=-1), Js[1]=J2, ...


# ----------------------------------------------------------------------------- numba multipole integrals (prange over layers)
@njit(cache=True, parallel=True)
def _multipoles_njit(lam, delta, zeta, wt, Pk_mu, kmax):
    N, L = zeta.shape
    K = kmax + 1
    dl3 = np.empty(N)
    D = 0.0
    for i in range(N):
        dl3[i] = delta[i] * lam[i] ** 3
        s3 = 0.0
        for a in range(L):
            s3 += wt[a] * zeta[i, a] ** 3
        D += dl3[i] * s3
    Jt = np.zeros((N, K))
    Jtp = np.zeros((N, K))
    for i in prange(N):
        for k in range(K):
            tk = 2 * k
            si = 0.0
            se = 0.0
            for a in range(L):
                P = Pk_mu[k, a]
                z = zeta[i, a]
                w = wt[a]
                si += w * P * z ** (tk + 3)
                if k == 0:
                    se += w * P * z * z
                elif k == 1:
                    se += w * P * np.log(z)
                else:
                    se += w * P * z ** (2 - tk)
            Jt[i, k] = -(3.0 / (tk + 3)) * dl3[i] * si / D
            if k == 0:
                Jtp[i, k] = -(3.0 / 2.0) * dl3[i] * se / D
            elif k == 1:
                Jtp[i, k] = -3.0 * dl3[i] * se / D
            else:
                Jtp[i, k] = -(3.0 / (2 - tk)) * dl3[i] * se / D
    Jpp = 0.5 * dl3 / D
    return Jt, Jtp, Jpp, D


# ----------------------------------------------------------------------------- multipole integrals (Hubbard Eq 40-43)
def _multipoles(lam, delta, zeta, mu, wt, kmax):
    """Interior J~_{i,2k}, exterior J~'_{i,2k}, and J~''_i, normalized by D.

    D = sum_j delta_j lam_j^3 int_0^1 zeta_j^3 dmu
    Interior (Eq 40):       J~_{i,2k}  = -(3/(2k+3)) delta_i lam_i^3 int P_2k zeta^(2k+3) / D
    Exterior (Eq 41-43):    J~'_{i,2k} = -(3/(2-2k)) delta_i lam_i^3 int P_2k zeta^(2-2k) / D   (k>=2)
                            J~'_{i,2}  = -3        delta_i lam_i^3 int P_2  ln(zeta)       / D   (k=1)
                            J~'_{i,0}  = -(3/2)    delta_i lam_i^3 int P_0  zeta^2         / D   (k=0)
    Quadratic term:         J~''_i     =  (1/2)    delta_i lam_i^3                          / D
    """
    P = _legendre_even(mu, kmax)             # (kmax+1, L)
    if _HAVE_NUMBA:
        return _multipoles_njit(lam, delta, zeta, wt, P, kmax)
    dl3 = delta * lam ** 3                    # (N,)
    D = np.sum(dl3 * np.sum(wt * zeta ** 3, axis=1))
    Jt = np.zeros((len(lam), kmax + 1))       # interior  J~_{i,2k}
    Jtp = np.zeros((len(lam), kmax + 1))      # exterior  J~'_{i,2k}
    for k in range(kmax + 1):
        n = 2 * k
        # interior
        Jt[:, k] = -(3.0 / (n + 3)) * dl3 * np.sum(wt * P[k] * zeta ** (n + 3), axis=1) / D
        # exterior
        if k == 0:
            Jtp[:, k] = -(3.0 / 2.0) * dl3 * np.sum(wt * P[k] * zeta ** 2, axis=1) / D
        elif k == 1:
            Jtp[:, k] = -3.0 * dl3 * np.sum(wt * P[k] * np.log(zeta), axis=1) / D
        else:
            Jtp[:, k] = -(3.0 / (2 - n)) * dl3 * np.sum(wt * P[k] * zeta ** (2 - n), axis=1) / D
    Jpp = 0.5 * dl3 / D                        # J~''_i
    return Jt, Jtp, Jpp, D


# ----------------------------------------------------------------------------- per-surface coefficient sums (precomputed once per iteration)
def _coeffs(lam, Jt, Jtp, Jpp, kmax):
    """Precompute, for every surface j and order k, the layer-summed coefficients
    in the equipotential equation (so the per-(j,mu) root-find is O(kmax), not O(N)):
      A[j,k]  = sum_{i>=j} J~_{i,2k} (lam_i/lam_j)^{2k}          (interior multipoles)
      B[j,k]  = sum_{i<j}  J~'_{i,2k} (lam_j/lam_i)^{2k+1}        (exterior multipoles)
      C[j]    = sum_{i<j}  J~''_i (lam_j/lam_i)^3                 (quadratic term)
      A0[k]   = sum_i J~_{i,2k} lam_i^{2k}                        (outer surface, Eq 50)
    Reverse/forward cumulative sums give all j at once.
    """
    N = len(lam)
    A = np.zeros((N, kmax + 1))
    B = np.zeros((N, kmax + 1))
    A0 = np.zeros(kmax + 1)
    lam2k = lam[:, None] ** (2 * np.arange(kmax + 1))[None, :]      # (N,kmax+1): lam_i^{2k}
    lam2k1 = lam[:, None] ** (2 * np.arange(kmax + 1) + 1)[None, :]  # lam_i^{2k+1}
    for k in range(kmax + 1):
        # interior: sum_{i>=j} Jt_i lam_i^{2k} / lam_j^{2k}
        rev = np.cumsum((Jt[:, k] * lam2k[:, k])[::-1])[::-1]        # sum_{i>=j}
        A[:, k] = rev / lam ** (2 * k)
        A0[k] = np.sum(Jt[:, k] * lam ** (2 * k))
        # exterior: sum_{i<j} Jtp_i / lam_i^{2k+1} * lam_j^{2k+1}
        pref = Jtp[:, k] / lam2k1[:, k]
        cum = np.concatenate([[0.0], np.cumsum(pref)[:-1]])         # sum_{i<j} (exclusive)
        B[:, k] = cum * lam ** (2 * k + 1)
    cpp = np.concatenate([[0.0], np.cumsum(Jpp / lam ** 3)[:-1]])    # sum_{i<j} Jpp_i/lam_i^3
    C = cpp * lam ** 3
    return A, B, A0, C


# ----------------------------------------------------------------------------- numba-JIT shape solvers (per-element loops, prange over layers)
@njit(cache=True, parallel=True)
def _newton_interior_njit(zin, Aj, Bj, Cj, lj3, Pk_mu, Pk0, P2_mu, P2_0,
                          qrot, kmax, niter, tol, zlo, zhi):
    """Eq 51 Newton for interior surfaces (j>0), parallel over layers (prange)."""
    Nm1, L = zin.shape
    K = kmax + 1
    out = zin.copy()
    for j in prange(Nm1):
        s = 0.0
        for k in range(K):
            s += (Aj[j, k] + Bj[j, k]) * Pk0[k]
        rhs_j = -(s + Cj[j]) + (1.0 / 3.0) * qrot * lj3[j] * (1.0 - P2_0)
        for a in range(L):
            zc = (1.0 / 3.0) * qrot * lj3[j] * (1.0 - P2_mu[a]) - Cj[j]
            zz = zin[j, a]
            for _ in range(niter):
                F = zc * zz * zz - rhs_j
                Fp = 2.0 * zc * zz
                for k in range(K):
                    P = Pk_mu[k, a]
                    AP = Aj[j, k] * P
                    BP = Bj[j, k] * P
                    tk = 2 * k
                    F += -AP * zz ** (-(tk + 1)) - BP * zz ** tk
                    Fp += AP * (tk + 1) * zz ** (-(tk + 2)) - BP * tk * zz ** (tk - 1)
                step = F / Fp
                zz -= step
                if zz < zlo:
                    zz = zlo
                elif zz > zhi:
                    zz = zhi
                if abs(step) < tol:
                    break
            out[j, a] = zz
    return out


@njit(cache=True)
def _newton_outer_njit(zin, A0, Pk_mu, Pk0, P2_mu, P2_0, qrot, kmax,
                       niter, tol, zlo, zhi):
    """Eq 50 Newton for the outer surface (j=0), over latitudes."""
    L = zin.shape[0]
    K = kmax + 1
    out = zin.copy()
    rhs0 = 1.0 + (1.0 / 3.0) * qrot * (1.0 - P2_0)
    for k in range(1, K):
        rhs0 -= A0[k] * Pk0[k]
    for a in range(L):
        zz = zin[a]
        for _ in range(niter):
            F = 1.0 / zz + (1.0 / 3.0) * qrot * zz * zz * (1.0 - P2_mu[a]) - rhs0
            Fp = -1.0 / (zz * zz) + (2.0 / 3.0) * qrot * zz * (1.0 - P2_mu[a])
            for k in range(1, K):
                c = A0[k] * Pk_mu[k, a]
                tk = 2 * k
                F += -c * zz ** (-(tk + 1))
                Fp += c * (tk + 1) * zz ** (-(tk + 2))
            step = F / Fp
            zz -= step
            if zz < zlo:
                zz = zlo
            elif zz > zhi:
                zz = zhi
            if abs(step) < tol:
                break
        out[a] = zz
    return out


# ----------------------------------------------------------------------------- vectorized shape solve (replaces per-(j,mu) brentq)
def _solve_shapes(zeta, A, B, A0, C, lam, qrot, Pk_mu, Pk0, P2_mu, P2_0, kmax,
                  newton_iters=60, tol=1e-13):
    # Fast path: numba-JIT per-element Newton (parallel over layers).
    if _HAVE_NUMBA:
        N = zeta.shape[0]
        out = zeta.copy()
        out[0] = _newton_outer_njit(zeta[0].copy(), A0, Pk_mu, Pk0, P2_mu, P2_0,
                                    qrot, kmax, newton_iters, tol, 0.20, 1.0000001)
        if N > 1:
            out[1:] = _newton_interior_njit(
                zeta[1:].copy(), A[1:], B[1:], C[1:], (lam[1:] ** 3).copy(),
                Pk_mu, Pk0, P2_mu, P2_0, qrot, kmax, newton_iters, tol, 0.20, 1.0000001)
        return out
    """Solve the equipotential shape equations for ALL surfaces and latitudes at
    once by a vectorized Newton iteration (analytic derivative), replacing the
    O(N*L) scipy.brentq root-finds. The equation is smooth and monotonic in zeta
    near the root, so Newton (clipped to [0.2, 1.05]) converges in a few passes.
    Eq 50 for the outer surface (j=0), Eq 51 for interior surfaces (j>0)."""
    N, L = zeta.shape
    out = zeta.copy()

    # --- outer surface j=0 (Eq 50), vectorized over mu ---
    z = zeta[0].copy()
    rhs0 = 1.0 - np.sum(A0[1:] * Pk0[1:]) + (1.0 / 3.0) * qrot * (1.0 - P2_0)
    c0 = A0[1:, None] * Pk_mu[1:]                # (kmax, L): A0[k] P_2k(mu), k>=1
    kk0 = (2 * np.arange(1, kmax + 1))[:, None]   # 2k for k>=1
    for _ in range(newton_iters):
        zpow = z ** (-(kk0 + 1))                  # (kmax, L): z^{-(2k+1)}
        F = 1.0 / z + (1.0 / 3.0) * qrot * z ** 2 * (1.0 - P2_mu) - rhs0 - np.sum(c0 * zpow, axis=0)
        Fp = -1.0 / z ** 2 + (2.0 / 3.0) * qrot * z * (1.0 - P2_mu) \
            + np.sum(c0 * (kk0 + 1) * z ** (-(kk0 + 2)), axis=0)
        step = F / Fp
        z = np.clip(z - step, 0.20, 1.0000001)
        if np.max(np.abs(step)) < tol:
            break
    out[0] = z

    # --- interior surfaces j>0 (Eq 51), vectorized over (j, mu) ---
    if N > 1:
        z = zeta[1:].copy()                       # (N-1, L)
        Aj, Bj, Cj = A[1:], B[1:], C[1:]
        lj3 = lam[1:] ** 3
        rhs = -(Aj @ Pk0 + Bj @ Pk0 + Cj) + (1.0 / 3.0) * qrot * lj3 * (1.0 - P2_0)
        zc = (1.0 / 3.0) * qrot * lj3[:, None] * (1.0 - P2_mu[None, :]) - Cj[:, None]  # (N-1,L)
        rhs_c = rhs[:, None]
        AP = Aj.T[:, :, None] * Pk_mu[:, None, :]  # (kmax+1, N-1, L): A[j,k] P_2k(mu)
        BP = Bj.T[:, :, None] * Pk_mu[:, None, :]
        twok = 2 * np.arange(kmax + 1)
        for _ in range(newton_iters):
            F = zc * z ** 2 - rhs_c
            Fp = 2.0 * zc * z
            for k in range(kmax + 1):
                F += -AP[k] * z ** (-(twok[k] + 1)) - BP[k] * z ** twok[k]
                Fp += AP[k] * (twok[k] + 1) * z ** (-(twok[k] + 2)) \
                    - BP[k] * twok[k] * z ** (twok[k] - 1)
            step = F / Fp
            z = np.clip(z - step, 0.20, 1.0000001)
            if np.max(np.abs(step)) < tol:
                break
        out[1:] = z
    return out


# ----------------------------------------------------------------------------- main CMS driver
def cms(lam, rho, qrot, kmax=10, L=48, tol=1e-10, maxiter=300, verbose=False,
        initial_guess=None):
    """Solve the CMS shapes and return gravity harmonics (Hubbard 2013 Eq 40-52).

    lam  : equatorial radii normalized to outer (lam[0]=1, descending), shape (N,)
    rho  : densities (any units; only ratios matter), shape (N,), surface->center
    qrot : omega^2 a_0^3 / (G M)
    initial_guess : (N,L) zeta from a previous call to WARM-START the Jacobi
        iteration (e.g. across evolution timesteps); None -> cold spherical start.
    Returns Js (J0,J2,...,J_{2*kmax}), converged zeta shapes (N,L), diagnostics.
    """
    lam = np.asarray(lam, float)
    rho = np.asarray(rho, float)
    N = len(lam)
    delta = rho.copy()
    delta[1:] = rho[1:] - rho[:-1]            # delta_i = rho_i - rho_{i-1}, delta_0 = rho_0
    mu, wt = _gl01(L)
    Pk_mu = _legendre_even(mu, kmax)          # (kmax+1, L)
    P2_mu = Pk_mu[1]                           # P_2(mu) at the nodes
    Pk0 = _legendre_even(np.array([0.0]), kmax)[:, 0]   # P_2k(mu=0), equator reference
    P2_0 = Pk0[1]                              # P_2(0) = -1/2
    if initial_guess is not None and np.shape(initial_guess) == (N, L):
        zeta = np.array(initial_guess, float)   # warm start
    else:
        zeta = np.ones((N, L))                  # cold (spherical) start

    for it in range(maxiter):
        Jt, Jtp, Jpp, D = _multipoles(lam, delta, zeta, mu, wt, kmax)
        A, B, A0, C = _coeffs(lam, Jt, Jtp, Jpp, kmax)
        zeta_new = _solve_shapes(zeta, A, B, A0, C, lam, qrot,
                                 Pk_mu, Pk0, P2_mu, P2_0, kmax)
        err = np.max(np.abs(zeta_new - zeta))
        zeta = zeta_new
        if verbose:
            print(f"  cms iter {it:3d}: max|dzeta| = {err:.3e}")
        if err < tol and it >= 1:
            break
    Js = _Jn_from_shapes(lam, delta, zeta, mu, wt, kmax)
    return Js, zeta, dict(iters=it + 1, err=err, mu=mu, wt=wt)


# ----------------------------------------------------------------------------- ORCHARD-style wrapper
def get_moments_cms(lam_eq, rho, m_total, omega, G=6.6743e-8, kmax=10, L=48,
                    tol=1e-10, maxiter=300, verbose=False, initial_guess=None):
    """CMS gravity harmonics with an ORCHARD-style interface.

    lam_eq  : EQUATORIAL radii (cm), surface->center (descending), shape (N,)
    rho     : densities (g/cm^3), surface->center
    m_total : total planet mass (g), used for qrot = omega^2 a0^3/(G M)
    omega   : rotation rate (rad/s)
    initial_guess : (N,L) zeta to warm-start (reuse the returned zeta next call).
    Returns j2n = [J2, J4, J6, J8, oblat, R_eq, R_pol, I_MoI] (J's x1e6, like TOF7), zeta.
    """
    lam_eq = np.asarray(lam_eq, float)
    rho = np.asarray(rho, float)
    a0 = lam_eq[0]                              # outer equatorial radius
    qrot = omega ** 2 * a0 ** 3 / (G * m_total)
    lam = lam_eq / a0                           # lam[0] = 1
    Js, zeta, info = cms(lam, rho, qrot, kmax=kmax, L=L, tol=tol,
                         maxiter=maxiter, verbose=verbose, initial_guess=initial_guess)
    mu, wt = info["mu"], info["wt"]
    # outer-surface polar value: ledge node closest to mu=1 (Gauss node ~0.999)
    zeta0_pole = zeta[0, -1]
    R_eq, R_pol = a0, a0 * zeta0_pole
    oblat = 1.0 - zeta0_pole
    # polar moment of inertia C = sum_i drho_i (4pi/5) a0^5 lam_i^5 int_0^1 (1-mu^2) zeta_i^5 dmu
    delta = rho.copy(); delta[1:] = rho[1:] - rho[:-1]
    C = np.sum(delta * (4 * np.pi / 5.0) * a0 ** 5 * lam ** 5
               * np.sum(wt * (1 - mu ** 2) * zeta ** 5, axis=1))
    j2n = np.array([Js[1] * 1e6, Js[2] * 1e6, Js[3] * 1e6, Js[4] * 1e6,
                    oblat, R_eq, R_pol, C])
    return j2n, zeta, info


# ----------------------------------------------------------------------------- ORCHARD _call_tof-compatible wrapper (mean radii in)
def get_moments_cms_from_mean(r_mean, rho, m_total, omega, G=6.6743e-8, kmax=10,
                              L=48, tol=1e-10, maxiter=300, lam_iters=6,
                              lam_tol=1e-7, initial_guess=None):
    """CMS harmonics from MEAN (volumetric) radii, matching TOF7's input convention
    so it can drop into hydrostatic.py:_call_tof.

    CMS is parameterized by EQUATORIAL radii lambda, but the HSE structure gives
    volumetric radii rS = lambda*(int_0^1 zeta^3 dmu)^{1/3} < lambda. We therefore
    iterate lambda so the spheroids' volumetric radii match the input r_mean
    (mass-conserving): lambda <- lambda * r_mean / rS. Converges in a few passes.

    r_mean : volumetric radii (cm), surface->center (descending), shape (N,)
    Returns j2n (TOF7-compatible 8-element) and the converged zeta (warm-start).
    """
    r_mean = np.asarray(r_mean, float)
    lam_eq = r_mean.copy()                         # initial guess: lambda = r_mean
    zeta = initial_guess
    j2n = info = None
    for _ in range(lam_iters):
        j2n, zeta, info = get_moments_cms(lam_eq, rho, m_total, omega, G=G,
                                          kmax=kmax, L=L, tol=tol, maxiter=maxiter,
                                          initial_guess=zeta)
        wt = info["wt"]
        vfac = np.sum(wt * zeta ** 3, axis=1) ** (1.0 / 3.0)   # (int_0^1 zeta^3)^{1/3}
        rS = lam_eq * vfac
        lam_new = lam_eq * (r_mean / rS)
        rel = np.max(np.abs(lam_new - lam_eq) / lam_eq)
        lam_eq = lam_new
        if rel < lam_tol:
            break
    return j2n, zeta, info


# ============================================================================= self-tests
def _maclaurin_q_of_e(e):
    """Exact Maclaurin rotation parameter q = omega^2 a0^3/(GM) for eccentricity e."""
    # omega^2/(pi G rho) = 2 sqrt(1-e^2)/e^3 (3-2e^2) arcsin(e) - 6(1-e^2)/e^2   (Chandrasekhar)
    s = np.sqrt(1 - e ** 2)
    w2_piGrho = (2 * s / e ** 3) * (3 - 2 * e ** 2) * np.arcsin(e) - 6 * (1 - e ** 2) / e ** 2
    # M = (4/3) pi rho a^2 c = (4/3) pi rho a^3 sqrt(1-e^2); q = omega^2 a^3/(GM)
    #   = omega^2/( (4/3) pi G rho sqrt(1-e^2) ) = (3/4)/sqrt(1-e^2) * omega^2/(pi G rho)
    return (3.0 / 4.0) / s * w2_piGrho


def _test_Jn_integral():
    print("=" * 70)
    print("Test 1: J_n integral on exact Maclaurin ellipsoid (no shape solver)")
    print("=" * 70)
    mu, wt = _gl01(48)
    ok = True
    for e in (0.1, 0.3, 0.5):
        zeta = (1.0 / np.sqrt(1 + e ** 2 * mu ** 2 / (1 - e ** 2)))[None, :]
        Js = _Jn_from_shapes(np.array([1.0]), np.array([1.0]), zeta, mu, wt, 4)
        analytic = {1: e ** 2 / 5, 2: -3 / 35 * e ** 4, 3: 9 / 315 * e ** 6}
        print(f"  e={e}: J0={Js[0]:.6f}(want -1)  "
              f"J2={Js[1]:.6e} (want {analytic[1]:.6e}, rel {Js[1]/analytic[1]-1:+.1e})  "
              f"J4={Js[2]:.6e} (want {analytic[2]:.6e})")
        if abs(Js[1] / analytic[1] - 1) > 1e-6 or abs(Js[0] + 1) > 1e-9:
            ok = False
    print(f"  => {'PASS' if ok else 'FAIL'}")
    return ok


def _test_single_spheroid():
    print("\n" + "=" * 70)
    print("Test 2: single-layer shape solver vs analytic Maclaurin spheroid")
    print("=" * 70)
    ok = True
    for e in (0.1, 0.3):
        q = _maclaurin_q_of_e(e)
        Js, zeta, info = cms(np.array([1.0]), np.array([1.0]), q, kmax=10, L=48)
        # analytic shape and J2
        mu = info["mu"]
        zeta_exact = 1.0 / np.sqrt(1 + e ** 2 * mu ** 2 / (1 - e ** 2))
        dz = np.max(np.abs(zeta[0] - zeta_exact))
        J2_an = e ** 2 / 5
        print(f"  e={e:.2f} q={q:.5f}: iters={info['iters']} max|dzeta|={dz:.2e}  "
              f"J2={Js[1]:.6e} (want {J2_an:.6e}, rel {Js[1]/J2_an-1:+.2e})")
        if dz > 1e-4 or abs(Js[1] / J2_an - 1) > 1e-3:
            ok = False
    print(f"  => {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    t1 = _test_Jn_integral()
    t2 = _test_single_spheroid()
    print("\n" + "=" * 70)
    print(f"Summary: Jn-integral {'PASS' if t1 else 'FAIL'}, "
          f"single-spheroid {'PASS' if t2 else 'FAIL'}")
