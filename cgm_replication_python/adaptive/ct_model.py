"""Continuous-time reformulation of the Cocco-Gomes-Maenhout (2005) life-cycle problem.

State vector (normalised by permanent labour income P_t):
    s = (w, y_1, ..., y_{d-1})
where
    w      financial wealth / permanent income
    y_k    extra risky "background" state variables (asset returns, extra income
           factors, health, family size, ...).  They are included so that the
           SAME code can be run with d = 1, 3, 5, ... states, which is what makes
           the curse of dimensionality measurable.

Economics (continuous-time limit of the CGM economy):
    dW  = [ (r + alpha*mu) W + Y - C ] dt + alpha*sigma W dB
    dP/P = g(a) dt + sigma_P dB_P
    dY_k = -kappa_k Y_k dt + s_k dB_k
    death: hazard nu(a)

Because the problem is homogeneous of degree one in (W, P), the value function
satisfies V(W,P,y) = P^(1-gamma) v(w,y), and the HJB equation for v is

    0 = sup_{c,alpha} { c^(1-gamma)/(1-gamma) - (rho + nu) v
        + v_w [ (r + alpha*mu - g) w + y_1 - c ]
        + sum_k v_{y_k} ( -kappa_k y_k )
        + 0.5 v_ww alpha^2 sigma^2 w^2
        + 0.5 sum_k v_{y_k y_k} s_k^2
        + alpha sigma w sum_k v_{w y_k} s_k rho_k
        + sum_{k<l} v_{y_k y_l} s_k s_l rho_kl }

For the infinitely-lived, no-labour-income Merton case (d = 1, y = 0, nu = 0)
the closed-form solution is
    alpha* = mu/(gamma sigma^2),
    c*/w   = ( rho + (gamma-1)(r + mu^2/(2 gamma sigma^2)) ) / gamma .
This is used to validate the solvers in 1-D.
"""
from __future__ import annotations
import math
import numpy as np

# ----------------------------------------------------------------- CGM calibration
GAMMA      = 10.0                 # relative risk aversion (CGM Table 4)
BETA       = 0.96                 # annual discount factor
RHO        = -math.log(BETA)      # continuous discount rate  0.040822
RF         = 1.02                 # gross riskless return
MU_ANN     = 0.04                 # equity premium (CGM text, section 2.2)
SIGMA_ANN  = 0.157                # sd of stock returns
R          = math.log(RF)         # continuous riskless rate 0.019803
MU         = MU_ANN               # equity premium is a drift
SIGMA      = SIGMA_ANN            # diffusion of the risky asset
SIGMA_P    = math.sqrt(0.0106)    # permanent income innovation sd
TRANS_VAR  = 0.0738               # transitory income variance
REPL       = 0.68212              # retirement replacement ratio (high-school group)
RETIRE     = 65.0                 # last working age
TERMINAL   = 100.0

# analytic Merton benchmark
MERTON_ALPHA = MU / (GAMMA * SIGMA ** 2)
MERTON_K     = (RHO + (GAMMA - 1.0) * (R + 0.5 * MU ** 2 / (GAMMA * SIGMA ** 2))) / GAMMA


def income_growth(age):
    """f'(age) for the CGM high-school income polynomial, in continuous time."""
    b = (0.5304, 0.16818, -0.00323371, 0.000019704)
    a = np.asarray(age, dtype=float)
    return b[1] + 2.0 * b[2] * a + 3.0 * b[3] * a * a


# CGM / NCHS conditional survival probabilities, ages 20-99 (from the authors' template)
SURV = np.array([0.99845,0.99839,0.99833,0.9983,0.99827,0.99826,0.99824,0.9982,0.99813,0.99804,
0.99795,0.99785,0.99776,0.99766,0.99755,0.99743,0.9973,0.99718,0.99707,0.99696,
0.99685,0.99672,0.99656,0.99635,0.9961,0.99579,0.99543,0.99504,0.99463,0.9942,
0.9937,0.99311,0.99245,0.99172,0.99091,0.99005,0.98911,0.98803,0.9868,0.98545,
0.98409,0.9827,0.98123,0.97961,0.97786,0.97603,0.97414,0.97207,0.9697,0.96699,
0.96393,0.96055,0.9569,0.9531,0.94921,0.94508,0.94057,0.9357,0.93031,0.92424,
0.91717,0.90922,0.90089,0.89282,0.88503,0.87622,0.86576,0.8544,0.8423,0.82942,
0.8154,0.80002,0.78404,0.76842,0.75382,0.73996,0.72464,0.71057,0.6961,0.6809])

def hazard(age):
    """Mortality hazard implied by the CGM survival table (ages 20-100)."""
    a = np.asarray(age, dtype=float)
    idx = np.clip((a - 20.0).astype(int), 0, len(SURV) - 1)
    p = np.where(a >= TERMINAL, 0.0, SURV[idx])
    p = np.clip(p, 1e-8, 1.0 - 1e-12)
    return -np.log(p)


def drift_diffusion(s, c, alpha, params):
    """Continuous-time state dynamics.

    s      (n, d)    first column is w, remaining columns are the y_k
    c      (n,)      consumption
    alpha  (n,)      equity share
    returns (drift (n,d), diffusion (n,d,m))  with m = 1 + d - 1 = d independent
    Brownian shocks: shock 0 drives the stock, shock k drives state k+1.
    """
    d = s.shape[1]
    w = s[:, 0]
    g = income_growth(params["age"])
    y = s[:, 1:].sum(axis=1) if d > 1 else np.zeros_like(w)
    drift = np.empty_like(s)
    drift[:, 0] = (R + alpha * MU - g) * w + y - c
    for k in range(1, d):
        drift[:, k] = -params["kappa"][k - 1] * s[:, k]
    diff = np.zeros((s.shape[0], d, d))
    diff[:, 0, 0] = alpha * SIGMA * w
    for k in range(1, d):
        diff[:, k, k] = params["sk"][k - 1]
    return drift, diff


def ito_generator(v, s, drift, diff, dV=None, d2V=None):
    """L v = v_s . f + 0.5 tr(g g' v_ss).  Derivatives may be supplied."""
    n, d = s.shape
    if dV is None:
        dV = np.stack([np.gradient(v, s[:, k], axis=0) for k in range(d)], axis=1)
    if d2V is None:
        d2V = np.zeros((n, d, d))
        for k in range(d):
            d2V[:, k, k] = np.gradient(dV[:, k], s[:, k], axis=0)
    quad = 0.5 * np.einsum("nik,nik->n", np.einsum("nij,njk->nik", diff, np.swapaxes(diff, 1, 2)), d2V)
    return (dV * drift).sum(axis=1) + quad
