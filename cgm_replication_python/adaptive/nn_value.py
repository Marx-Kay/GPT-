"""Neural-network solution of the full CGM life-cycle model by Bellman residual
minimisation (deep-Galerkin style).

Value-function parameterisation (exactly consistent with CRRA homogeneity)
-------------------------------------------------------------------------
For gamma > 1 the value function V_t(x) is negative and increasing in normalised
cash-on-hand x.  Write

        - V_t(x) = A_t(x) * x^(1-gamma),      A_t(x) > 0
        log A_t(x) = NN( log(1+x), age )

At the terminal age the agent consumes everything, so V = x^(1-gamma)/(1-gamma)
and A = 1/(gamma-1) exactly: the network output is replaced by that constant
there, imposing the terminal condition exactly.

The controls follow from the first-order conditions

        c     = (v_w)^(-1/gamma)                  (c <= x holds by construction)
        alpha = -(v_w/(x v_ww)) * mu / sigma^2    (clamped to [0, 1])

Loss = Bellman residual, normalised by its own local magnitude.

NOTE: all evaluation helpers run inside their own torch.enable_grad() block,
because they need autograd to differentiate the value function; do not wrap them
in torch.no_grad().
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
Q = 1.0 - GAMMA
EXPO = 1.0 - GAMMA
LOG_A_TERMINAL = math.log(1.0 / abs(EXPO))


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
        return raw * (1.0 - term) + LOG_A_TERMINAL * term

    def value(self, x, age_s):
        return -torch.exp(self.logA(x, age_s)) * x.pow(EXPO)


def state_terms(net, x, age_s, need_graph):
    """V, consumption and equity share at (x, age) from the first-order conditions."""
    with torch.enable_grad():
        xr = x.detach().clone().requires_grad_(True)
        lA = net.logA(xr, age_s)
        d1 = torch.autograd.grad(lA.sum(), xr, create_graph=True)[0]
        d2 = torch.autograd.grad(d1.sum(), xr, create_graph=True)[0]
        if not need_graph:
            lA, d1, d2 = lA.detach(), d1.detach(), d2.detach()
    A = torch.exp(lA); A1 = A * d1; A2 = A * (d2 + d1 ** 2)
    xc = torch.clamp(x, min=1e-12)
    V = -A * xc.pow(EXPO)
    vw = -(A1 * xc.pow(EXPO) + EXPO * A * xc.pow(EXPO - 1.0))
    vww = -(A2 * xc.pow(EXPO) + 2.0 * EXPO * A1 * xc.pow(EXPO - 1.0)
            + EXPO * (EXPO - 1.0) * A * xc.pow(EXPO - 2.0))
    c = torch.minimum(torch.clamp(vw.pow(-1.0 / GAMMA), min=1e-12), xc)
    alpha = torch.clamp(-(vw / torch.clamp(xc * vww, max=-1e-12)) * (MU / SE ** 2), 0.0, 1.0)
    return V, c, alpha


def _GY(t_idx, ZY, NQ, batch):
    G = np.empty((batch, NQ)); Yq = np.empty((batch, NQ))
    for k, t in enumerate(t_idx):
        g, y = transition(int(AGES[t]), ZY)
        G[k] = g; Yq[k] = y
    return torch.tensor(G), torch.tensor(Yq)


def normalised_residual(u, cont, V):
    denom = torch.clamp(torch.maximum(torch.maximum(u.abs(), cont.abs()), V.abs()), min=1e-300)
    return (u + cont - V) / denom


def continuation(net, x, c, alpha, age_s, G, Yq, surv, Wt, ZRt):
    rp = RF + alpha[:, None] * (ZRt - RF)[None, :]
    xp = (x - c)[:, None] * rp / G + Yq
    with torch.no_grad():
        Vn = net.value(xp.reshape(-1), age_s[:, None].expand_as(xp).reshape(-1)).reshape(xp.shape)
    return (BETA * surv[:, None] * G.pow(Q) * Vn * Wt[None, :]).sum(dim=1)


def train(iterations=4000, batch=128, width=64, depth=3, lr=2e-3, seed=2026,
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
        ti = torch.randint(0, T - 1, (batch,), generator=g).numpy()
        x = xmin + (xmax - xmin) * torch.rand(batch, generator=g).pow(2.0)
        age_s = torch.tensor((ti / (T - 1)) * 2.0 - 1.0)
        V, c, alpha = state_terms(net, x, age_s, need_graph=True)
        G, Yq = _GY(ti, ZY, NQ, batch)
        surv = torch.tensor(SURVX[ti])
        cont = continuation(net, x, c, alpha, age_s, G, Yq, surv, Wt, ZRt)
        res = normalised_residual(c.pow(Q) / Q, cont, V)
        loss = (res ** 2).mean()
        if not torch.isfinite(loss):
            if verbose: print("   it %5d non-finite; stopping" % it)
            break
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step(); sched.step()
        if it % 250 == 0 or it == iterations - 1:
            hist.append((it, float(loss.sqrt())))
            if verbose:
                print("   it %5d  Bellman residual rms %.4e  (%.0fs)"
                      % (it, float(loss.sqrt()), time.perf_counter() - t0))
    return net, hist, time.perf_counter() - t0


def holdout_residual(net, n=4000, seed=7, xmin=0.3, xmax=30.0, ny=3, nr=3):
    ZY, ZR, Wt_np, NQ = shock_grid(ny, nr)
    ZRt = torch.tensor(ZR); Wt = torch.tensor(Wt_np)
    g = torch.Generator().manual_seed(seed)
    ti = torch.randint(0, T - 1, (n,), generator=g).numpy()
    x = xmin + (xmax - xmin) * torch.rand(n, generator=g).pow(2.0)
    age_s = torch.tensor((ti / (T - 1)) * 2.0 - 1.0)
    V, c, alpha = state_terms(net, x, age_s, need_graph=False)
    G, Yq = _GY(ti, ZY, NQ, n)
    surv = torch.tensor(SURVX[ti])
    cont = continuation(net, x, c, alpha, age_s, G, Yq, surv, Wt, ZRt)
    res = normalised_residual(c.pow(Q) / Q, cont, V)
    return float((res ** 2).mean().sqrt()), c.numpy(), alpha.numpy(), x.numpy(), ti


def extract_policy(net, na=400, amax=80.0):
    xs = torch.tensor(np.concatenate([[5e-3], np.linspace(1e-2, amax, na - 1)]))
    Xg, Cp, Ap = [], [], []
    for t in range(T):
        if t == T - 1:
            Xg.append(np.concatenate([[0.0], xs.numpy()]))
            Cp.append(np.concatenate([[0.0], xs.numpy()]))
            Ap.append(np.zeros(na)); continue
        age_s = torch.full_like(xs, (t / (T - 1)) * 2.0 - 1.0)
        V, c, al = state_terms(net, xs, age_s, need_graph=False)
        Xg.append(np.concatenate([[0.0], (xs + c).numpy()]))
        Cp.append(np.concatenate([[0.0], c.numpy()]))
        Ap.append(np.concatenate([[0.0], al.numpy()]))
    return dict(x=Xg, c=Cp, alpha=Ap, ages=AGES, p_surv=SURVX,
                params=dict(gamma=GAMMA, beta=BETA, Rf=RF, mu=MU, se=SE, s2e=S2E,
                            s2u=S2U, repl=REPL, aK=AK, a20=A20, aT=AT))


if __name__ == "__main__":
    net, hist, secs = train(iterations=4000, batch=128, verbose=True)
    out = os.path.join(_HERE, "..", "results"); os.makedirs(out, exist_ok=True)
    torch.save(net.state_dict(), os.path.join(out, "nn_lifecycle_vf.pt"))
    hr, c, a, x, ti = holdout_residual(net)
    json.dump(dict(iterations=4000, seconds=secs, history=hist, holdout_residual_rms=hr),
              open(os.path.join(out, "nn_lifecycle_vf_training.json"), "w"), indent=2)
    print("trained in %.0f s; final train rms %.3e; holdout rms %.3e" % (secs, hist[-1][1], hr))
