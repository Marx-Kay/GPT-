"""Neural-network solution of the full CGM life-cycle model by Bellman residual
minimisation (deep-Galerkin / deep-BSDE style).

Value-function parameterisation (exactly consistent with CRRA homogeneity)
-------------------------------------------------------------------------
For gamma > 1 the value function V_t(x) is negative and increasing in normalised
cash-on-hand x.  Write

        - V_t(x) = A_t(x) * x^(1-gamma),      A_t(x) > 0
        log A_t(x) = NN( log(1+x), age )

At the terminal age the agent consumes everything, so V = x^(1-gamma)/(1-gamma)
and therefore A = 1/(gamma-1) exactly; the network output is replaced by that
constant at the terminal age, which imposes the terminal condition exactly.

Differentiating,

        v_w  = -( A' x^(1-gamma) + (1-gamma) A x^(-gamma) )
        v_ww = -( A'' x^(1-gamma) + 2(1-gamma) A' x^(-gamma)
                  + (1-gamma)(-gamma) A x^(-gamma-1) )

and the first-order conditions give the controls

        c     = (v_w)^(-1/gamma)                 (<= x holds by construction)
        alpha = -(v_w / (x v_ww)) * mu / sigma^2 (clamped to [0, 1]).

Loss = Bellman residual normalised by its own local magnitude

        R = [ u(c) + beta p_t E( G^(1-gamma) v_{t+1}(x') ) - v_t(x) ] / |terms|
"""
from __future__ import annotations
import json, math, os, sys, time
import numpy as np
import torch
from torch import nn

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "discrete"))
from cgm_model import SURV, F  # noqa: E402

torch.set_default_dtype(torch.float64)

GAMMA = 10.0; BETA = 0.96; RF = 1.02; MU = 0.04; SE = 0.157
S2E = 0.0738; S2U = 0.0106; REPL = 0.68212
A20, AK, AT = 20, 65, 100
AGES = np.arange(A20, AT + 1)
T = len(AGES)
SURVX = np.concatenate([SURV[:T - 1], [0.0]])
Q = 1.0 - GAMMA                                  # = -9
EXPO = 1.0 - GAMMA                               # = -9
LOG_A_TERMINAL = math.log(1.0 / abs(EXPO))       # A = 1/(gamma-1) at terminal age


def gauss_hermite(n):
    x, w = np.polynomial.hermite_e.hermegauss(n)
    return x, w / np.sqrt(2 * np.pi)


def shock_grid(ny=3, nr=3):
    zy, wy = gauss_hermite(ny); zr, wr = gauss_hermite(nr)
    ZY, ZR = np.meshgrid(zy, zr, indexing="ij")
    return ZY.ravel(), ZR.ravel(), np.outer(wy, wr).ravel(), ZY.size


def transition(age, ZY):
    if age >= AK:
        return np.ones_like(ZY), REPL * np.ones_like(ZY)
    G = (F(min(age + 1, AK)) / F(min(age, AK))) * np.exp(np.sqrt(S2U) * ZY)
    return G, np.exp(np.sqrt(S2E) * ZY)


class ValueNet(nn.Module):
    """V_t(x) = -A_t(x) x^(1-gamma),  log A_t(x) = NN(log(1+x), age)."""

    def __init__(self, width=64, depth=3):
        super().__init__()
        layers, d = [], 2
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.Tanh()]
            d = width
        layers += [nn.Linear(d, 1)]
        self.body = nn.Sequential(*layers)
        with torch.no_grad():
            self.body[-1].weight.mul_(0.1)
            self.body[-1].bias.fill_(0.0)

    def logA(self, x, age_s):
        raw = self.body(torch.stack([torch.log1p(x), age_s], dim=-1)).squeeze(-1)
        term = (age_s > 1.0 - 1e-9).to(raw.dtype)
        # keep 'raw' in the graph even when term == 1 so autograd never detaches
        return raw * (1.0 - term) + LOG_A_TERMINAL * term

    def value(self, x, age_s):
        return -torch.exp(self.logA(x, age_s)) * x.pow(EXPO)


def controls(net, x, age_s, create_graph=True):
    """c and alpha from the first-order conditions."""
    xr = x.clone().requires_grad_(True)
    A = torch.exp(net.logA(xr, age_s))
    dA = torch.autograd.grad(A.sum(), xr, create_graph=True)[0]
    d2A = torch.autograd.grad(dA.sum(), xr, create_graph=create_graph, retain_graph=create_graph)[0]
    V = -A * xr.pow(EXPO)
    vw = -(dA * xr.pow(EXPO) + EXPO * A * xr.pow(EXPO - 1.0))
    vww = -(d2A * xr.pow(EXPO)
            + 2.0 * EXPO * dA * xr.pow(EXPO - 1.0)
            + EXPO * (EXPO - 1.0) * A * xr.pow(EXPO - 2.0))
    c = torch.clamp(vw.pow(-1.0 / GAMMA), min=1e-12)
    c = torch.minimum(c, xr)
    denom = torch.clamp(xr * vww, max=-1e-12)
    alpha = torch.clamp(-(vw / denom) * (MU / SE ** 2), 0.0, 1.0)
    return V, c, alpha


def bellman_target(net, x, age_s, t_idx, ZY, ZRt, Wt, NQ, batch, create_graph=True):
    V, c, alpha = controls(net, x, age_s, create_graph=create_graph)
    s = x - c
    G = torch.empty(batch, NQ); Yq = torch.empty(batch, NQ)
    for k, t in enumerate(t_idx.numpy()):
        g, y = transition(int(AGES[t]), ZY)
        G[k] = torch.tensor(g); Yq[k] = torch.tensor(y)
    rp = RF + alpha[:, None] * (ZRt - RF)[None, :]
    xp = s[:, None] * rp / G + Yq
    Vn = net.value(xp.reshape(-1), age_s[:, None].expand_as(xp).reshape(-1)).reshape(xp.shape)
    surv = torch.tensor(SURVX[t_idx.numpy()])[:, None]
    cont = (BETA * surv * G.pow(Q) * Vn * Wt[None, :]).sum(dim=1)
    return V, c, alpha, cont


@torch.no_grad()
def evaluate(net, x, age_s, t_idx, ZY, ZRt, Wt_np, NQ):
    return bellman_target(net, x, age_s, t_idx, ZY, ZRt, Wt_np, NQ, x.numel(),
                          create_graph=False)


def _normalised_residual(u, cont, V):
    denom = torch.clamp(torch.maximum(torch.maximum(u.abs(), cont.abs()), V.abs()), min=1e-300)
    return (u + cont - V) / denom


def train(iterations=3000, batch=128, width=64, depth=3, lr=2e-3, seed=2026,
          verbose=True, xmin=0.3, xmax=30.0, ny=3, nr=3):
    torch.manual_seed(seed)
    net = ValueNet(width, depth)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iterations, eta_min=lr / 50)
    ZY, ZR, Wt_np, NQ = shock_grid(ny, nr)
    ZRt = torch.tensor(ZR); Wt = torch.tensor(Wt_np)
    g = torch.Generator().manual_seed(seed + 999)
    hist = []; t0 = time.perf_counter()
    for it in range(iterations):
        ti = torch.randint(0, T - 1, (batch,), generator=g)
        x = xmin + (xmax - xmin) * torch.rand(batch, generator=g).pow(2.0)
        age_s = (ti.double() / (T - 1)) * 2.0 - 1.0
        V, c, alpha, cont = bellman_target(net, x, age_s, ti, ZY, ZRt, Wt, NQ, batch)
        u = c.pow(Q) / Q
        res = _normalised_residual(u, cont, V)
        loss = (res ** 2).mean()
        if not torch.isfinite(loss):
            if verbose:
                print("   it %5d  non-finite loss, stopping" % it)
            break
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step(); sched.step()
        if it % 200 == 0 or it == iterations - 1:
            hist.append((it, float(loss.sqrt())))
            if verbose:
                print("   it %5d  Bellman residual rms %.4e  (%.0fs)"
                      % (it, float(loss.sqrt()), time.perf_counter() - t0))
    return net, hist, time.perf_counter() - t0


def holdout_residual(net, n=4000, seed=7, xmin=0.3, xmax=30.0, ny=3, nr=3):
    ZY, ZR, Wt_np, NQ = shock_grid(ny, nr)
    ZRt = torch.tensor(ZR); Wt = torch.tensor(Wt_np)
    g = torch.Generator().manual_seed(seed)
    ti = torch.randint(0, T - 1, (n,), generator=g)
    x = xmin + (xmax - xmin) * torch.rand(n, generator=g).pow(2.0)
    age_s = (ti.double() / (T - 1)) * 2.0 - 1.0
    V, c, alpha, cont = evaluate(net, x, age_s, ti, ZY, ZRt, Wt, NQ)
    res = _normalised_residual(c.pow(Q) / Q, cont, V)
    return float((res ** 2).mean().sqrt()), c.numpy(), alpha.numpy(), x.numpy(), ti.numpy()


def extract_policy(net, na=400, amax=80.0):
    xs = torch.tensor(np.concatenate([[5e-3], np.linspace(1e-2, amax, na - 1)]))
    Xg, Cp, Ap = [], [], []
    for t in range(T):
        if t == T - 1:
            Xg.append(np.concatenate([[0.0], xs.numpy()]))
            Cp.append(np.concatenate([[0.0], xs.numpy()]))
            Ap.append(np.zeros(na)); continue
        age_s = torch.full_like(xs, (t / (T - 1)) * 2.0 - 1.0)
        V, c, al = controls(net, xs, age_s)
        Xg.append(np.concatenate([[0.0], (xs + c).numpy()]))
        Cp.append(np.concatenate([[0.0], c.numpy()]))
        Ap.append(np.concatenate([[0.0], al.numpy()]))
    return dict(x=Xg, c=Cp, alpha=Ap, ages=AGES, p_surv=SURVX,
                params=dict(gamma=GAMMA, beta=BETA, Rf=RF, mu=MU, se=SE, s2e=S2E,
                            s2u=S2U, repl=REPL, aK=AK, a20=A20, aT=AT))


if __name__ == "__main__":
    net, hist, secs = train(iterations=3000, batch=128, verbose=True)
    out = os.path.join(_HERE, "..", "results"); os.makedirs(out, exist_ok=True)
    torch.save(net.state_dict(), os.path.join(out, "nn_lifecycle_vf.pt"))
    hr, c, a, x, ti = holdout_residual(net)
    json.dump(dict(iterations=3000, seconds=secs, history=hist, holdout_residual_rms=hr),
              open(os.path.join(out, "nn_lifecycle_vf_training.json"), "w"), indent=2)
    print("trained in %.0f s; final train rms %.3e; holdout rms %.3e" % (secs, hist[-1][1], hr))
