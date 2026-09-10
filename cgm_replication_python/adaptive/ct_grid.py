"""Grid-based HJB solver: the curse-of-dimensionality baseline.

Controls from the HJB first-order conditions

    c     = (v_w)^(-1/gamma)
    alpha = -(v_w/(w v_ww)) * (mu/sigma^2)

and the value function solves the steady-state (or finite-horizon) linear PDE
implied by the current policy.  A log-spaced wealth grid is used because
v ~ w^(1-gamma) is extremely steep near the borrowing constraint.
"""
from __future__ import annotations
import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import ct_model as M


def loggrid(wmin, wmax, n):
    return np.exp(np.linspace(np.log(wmin), np.log(wmax), n))


def derivs(x, v):
    """First and second derivatives on an arbitrary (increasing) grid."""
    hm = np.diff(x)                     # h[i] = x[i+1]-x[i], length n-1
    n = len(x)
    vw = np.empty(n); vww = np.empty(n)
    # interior: three-point non-uniform stencil
    hb = x[1:-1] - x[:-2]               # backward spacing
    hf = x[2:] - x[1:-1]                # forward spacing
    vw[1:-1] = (-hb / (hf * (hb + hf))) * v[2:] \
             + ((hb - hf) / (hb * hf)) * v[1:-1] \
             + (hf / (hb * (hb + hf))) * v[:-2]
    vww[1:-1] = (2.0 / (hb * (hb + hf))) * v[2:] \
              - (2.0 / (hb * hf)) * v[1:-1] \
              + (2.0 / (hf * (hb + hf))) * v[:-2]
    vw[0] = (v[1] - v[0]) / hm[0]
    vww[0] = (v[2] - 2 * v[1] + v[0]) / hm[0] ** 2
    vw[-1] = (v[-1] - v[-2]) / hm[-1]
    vww[-1] = (v[-1] - 2 * v[-2] + v[-3]) / hm[-1] ** 2
    return vw, vww


def operator(f, q, x):
    """Sparse matrix for  f v_x + q v_xx  on a non-uniform grid (upwind advection)."""
    n = len(x)
    hb = x[1:-1] - x[:-2]                       # (n-2,) spacing below interior rows
    hf = x[2:] - x[1:-1]                        # (n-2,) spacing above interior rows
    hbar = 0.5 * (hb + hf)
    fp = np.maximum(f, 0.0); fm = np.minimum(f, 0.0)
    a_sub = np.empty(n); a_sup = np.empty(n)
    a_sub[1:n - 1] = -fm[1:n - 1] / hb
    a_sup[1:n - 1] = fp[1:n - 1] / hf
    d_sub = np.zeros(n); d_sup = np.zeros(n)
    d_sub[1:n - 1] = 2.0 * q[1:n - 1] / (hf * hbar)
    d_sup[1:n - 1] = 2.0 * q[1:n - 1] / (hb * hbar)
    diag = -(a_sub + a_sup + d_sub + d_sup)
    subd = np.concatenate([[0.0], a_sub[1:n - 1] - d_sub[1:n - 1]])
    supd = np.concatenate([a_sup[1:n - 1] + d_sup[1:n - 1], [0.0]])
    A = sp.diags([subd, diag, supd], [-1, 0, 1], shape=(n, n), format="lil")
    A[0, :] = 0.0                                # boundary row handled by the caller
    hL = x[-1] - x[-2]
    A[n - 1, n - 2] = -q[n - 1] / hL ** 2
    A[n - 1, n - 1] = q[n - 1] / hL ** 2
    return sp.csr_matrix(A)


def solve_merton_1d(N=400, L=40.0, wmin=1e-6, maxit=60, tol=1e-13, verbose=False):
    """Infinite-horizon no-income Merton problem with a no-borrowing constraint."""
    gamma = M.GAMMA; rho = M.RHO; r = M.R; mu = M.MU; sigma = M.SIGMA
    x = loggrid(wmin, L, N)
    kstar = M.MERTON_K; astar = M.MERTON_ALPHA
    v = -(x ** (1 - gamma)) / (gamma - 1) * kstar ** (-gamma)
    t0 = time.perf_counter()
    for it in range(maxit):
        vw, vww = derivs(x, v)
        vww = np.minimum(vww, -1e-300)
        c = np.minimum(np.maximum(np.where(vw > 0, vw ** (-1.0 / gamma), x), 1e-300), x)
        al = np.clip(-(vw / (x * vww)) * (mu / sigma ** 2), 0.0, 1.0)
        f = (r + al * mu) * x - c
        q = 0.5 * (al * sigma * x) ** 2
        u = c ** (1 - gamma) / (1 - gamma)
        A = operator(f, q, x)
        L = (A - rho * sp.identity(N)).tolil()
        L[0, :] = 0.0; L[0, 0] = 1.0
        rhs = u.copy(); rhs[0] = v[0]
        vv = spla.spsolve(sp.csr_matrix(L), rhs)
        change = np.max(np.abs(vv - v)) / max(1.0, np.max(np.abs(vv)))
        v = vv
        if verbose:
            print("   it %3d rel change %.3e" % (it, change))
        if change < tol:
            break
    vw, vww = derivs(x, v); vww = np.minimum(vww, -1e-300)
    c = np.minimum(np.maximum(np.where(vw > 0, vw ** (-1.0 / gamma), x), 1e-300), x)
    al = np.clip(-(vw / (x * vww)) * (mu / sigma ** 2), 0.0, 1.0)
    return dict(x=x, v=v, c=c, alpha=al, iterations=it + 1,
                seconds=time.perf_counter() - t0, grid_points=N)


def grid_points(n_states, n_per_dim):
    return int(n_per_dim) ** int(n_states)


def grid_cost_table(dims=(1, 2, 3, 5, 10, 20, 24), n_per_dim=40, bytes_per_point=8.0,
                    seconds_per_point=1e-7):
    rows = []
    for d in dims:
        n = grid_points(d, n_per_dim)
        rows.append(dict(states=d, points_per_dim=n_per_dim, total_points=n,
                         memory_GB=n * bytes_per_point / 1e9,
                         est_seconds=n * seconds_per_point))
    return rows