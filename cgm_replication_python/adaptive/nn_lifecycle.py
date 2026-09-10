"""Neural-network solution of the FULL CGM (2005) life-cycle model.

Next step after the replication: solve the same discrete-time CGM problem with a
neural network instead of a grid, so the method can later be extended to many
more state variables.

Problem (identical to the grid/EGM solution in ../discrete/cgm_model.py):

    v_t(x) = max_{0<=c<=x, 0<=alpha<=1}  u(c) + beta p_t E[ G^{1-gamma} v_{t+1}(x') ]
    x'     = s R^p / G + y',      s = x - c,   R^p = Rf + alpha (R - Rf)
    utility u(c) = c^{1-gamma}/(1-gamma),  terminal age 100 consumes everything.

The controls are recovered with the SAME two-step logic as the grid solver, so
the two solutions are directly comparable:

  1. alpha is chosen by maximising the continuation value
         E[ v_{t+1}(x') ]  ~  E[ c_{t+1}(x')^{1-gamma} ]
     on a grid over [0, 1].
  2. consumption then comes from the Euler equation
         c^{-gamma} = beta p_t E[ G^{-gamma} c_{t+1}(x')^{-gamma} R^p ],
     a scalar fixed point in c, solved by bisection.

The network supplies only the consumption rule, through the parameterisation

    u'(c) = x^{-gamma} * phi,   phi = exp(-softplus(NN) / gamma) in (0, 1]
    =>  c = x * phi^{1/gamma} <= x

so the no-borrowing constraint c <= x holds by construction and phi = 1
reproduces the constrained rule c = x.
"""
from __future__ import annotations
import json, math, os, sys, time
import numpy as np
import torch
from torch import nn

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "discrete"))

from cgm_model import SURV, F, solve as solve_grid, simulate as simulate_grid  # noqa: E402

torch.set_default_dtype(torch.float64)

GAMMA = 10.0; BETA = 0.96; RF = 1.02; MU = 0.04; SE = 0.157
S2E = 0.0738; S2U = 0.0106; REPL = 0.68212
A20, AK, AT = 20, 65, 100
AGES = np.arange(A20, AT + 1)
SURVX = np.concatenate([SURV[:len(AGES) - 1], [0.0]])


def gauss_hermite(n):
    x, w = np.polynomial.hermite_e.hermegauss(n)
    return x, w / np.sqrt(2 * np.pi)


def shock_grid(ny=3, nr=3):
    zy, wy = gauss_hermite(ny); zr, wr = gauss_hermite(nr)
    ZY, ZR = np.meshgrid(zy, zr, indexing="ij")
    W = np.outer(wy, wr).ravel()
    return ZY.ravel(), ZR.ravel(), W, ZY.size


class PhiNet(nn.Module):
    """Multiplier phi(x, age) in (0, 1] with c = x * phi**(1/gamma)."""

    def __init__(self, width=96, depth=3):
        super().__init__()
        layers, d = [], 2
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.Tanh()]
            d = width
        layers += [nn.Linear(d, 1)]
        self.body = nn.Sequential(*layers)
        with torch.no_grad():
            self.body[-1].weight.mul_(0.05)
            self.body[-1].bias.fill_(0.0)

    def forward(self, x, age_scaled):
        raw = self.body(torch.stack([torch.log1p(x), age_scaled], dim=-1)).squeeze(-1)
        return torch.exp(-torch.nn.functional.softplus(raw) / GAMMA)


def transition(age, ZY, ZR):
    """Next-period normalised income y' and permanent-income growth G."""
    if age >= AK:
        return np.ones_like(ZY), REPL * np.ones_like(ZY)
    G = (F(min(age + 1, AK)) / F(min(age, AK))) * np.exp(np.sqrt(S2U) * ZY)
    Yq = np.exp(np.sqrt(S2E) * ZY)
    return G, Yq


def euler_c(x, alpha, Gt, Yqt, au, net, surv_t, ZRt):
    """Consumption from the Euler equation by bisection on log c."""
    lo = torch.full_like(x, -50.0); hi = torch.log(x)
    for _ in range(45):
        mid = 0.5 * (lo + hi); c = mid.exp(); s = x - c
        rp = RF + alpha[:, None] * (ZRt - RF)[None, :]
        xp = s[:, None] * rp / Gt + Yqt
        phi2 = net(xp.reshape(-1), au[:, None].expand_as(xp).reshape(-1)).reshape(xp.shape)
        cn2 = xp * phi2.pow(1.0 / GAMMA)
        rhs = (BETA * surv_t[:, None] * cn2.pow(-GAMMA) * rp).sum(dim=1)
        too_small = c.pow(-GAMMA) > rhs     # marginal utility too high => c below optimum
        lo = torch.where(too_small, mid, lo); hi = torch.where(too_small, hi, mid)
    return (0.5 * (lo + hi)).exp()


def train(iterations=4000, batch=192, width=96, depth=3, lr=3e-3, seed=2026,
          nalpha=41, verbose=True, xmax=40.0, ny=3, nr=3):
    torch.manual_seed(seed)
    net = PhiNet(width, depth)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iterations, eta_min=lr / 200)
    ZY, ZR, Wq, NQ = shock_grid(ny, nr)
    ZYt = torch.tensor(ZY); ZRt = torch.tensor(ZR); Wt = torch.tensor(Wq)
    ag = torch.tensor(np.linspace(0.0, 1.0, nalpha))
    rp_all = RF + ag[:, None] * (ZRt - RF)[None, :]
    hist = []; t0 = time.perf_counter()
    for it in range(iterations):
        ti = torch.randint(0, len(AGES) - 1, (batch,))
        x = 1e-3 + (xmax - 1e-3) * torch.rand(batch).pow(2.0)
        age_np = AGES[ti.numpy()]
        G = np.empty((batch, NQ)); Yq = np.empty((batch, NQ))
        for k, a in enumerate(age_np):
            G[k], Yq[k] = transition(int(a), ZY, ZR)
        Gt = torch.tensor(G); Yqt = torch.tensor(Yq)
        au = (ti.double() / (len(AGES) - 1)) * 2.0 - 1.0
        surv_t = torch.tensor(SURVX[ti.numpy()])
        phi = net(x, au)
        # step 1: alpha by maximising the continuation value
        xp_all = x[:, None, None] * rp_all[None, :, :] / Gt[:, None, :] + Yqt[:, None, :]
        aue = au[:, None, None].expand_as(xp_all)
        phin = net(xp_all.reshape(-1), aue.reshape(-1)).reshape(xp_all.shape)
        cn = xp_all * phin.pow(1.0 / GAMMA)
        cont = (cn.pow(1 - GAMMA) * Wt[None, None, :]).sum(dim=2)
        alpha = ag[cont.argmax(dim=1)]
        # step 2: consumption from the Euler equation
        c = euler_c(x, alpha, Gt, Yqt, au, net, surv_t, ZRt)
        phi_euler = (c / x).pow(GAMMA)
        loss = ((torch.log(phi) - torch.log(phi_euler)) ** 2).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
        opt.step(); sched.step()
        if it % 200 == 0:
            hist.append((it, float(loss.sqrt())))
            if verbose:
                print("   it %5d  log-phi residual rms %.3e  (%.0fs)"
                      % (it, float(loss.sqrt()), time.perf_counter() - t0))
    return net, hist, time.perf_counter() - t0


@torch.no_grad()
def extract_policy(net, na=400, amax=80.0, nalpha=201, ny=5, nr=5):
    """Policy arrays in the same format the grid EGM solver returns."""
    xs = torch.tensor(np.concatenate([[1e-4], np.linspace(1e-4, amax, na - 1)]))
    ZY, ZR, Wq, NQ = shock_grid(ny, nr)
    ZYt = torch.tensor(ZY); ZRt = torch.tensor(ZR); Wt = torch.tensor(Wq)
    ag = np.linspace(0.0, 1.0, nalpha)
    Xg, Cp, Ap = [], [], []
    for t, age in enumerate(AGES):
        if t == len(AGES) - 1:
            Xg.append(np.concatenate([[0.0], xs.numpy()]))
            Cp.append(np.concatenate([[0.0], xs.numpy()]))
            Ap.append(np.zeros(na)); continue
        au = torch.full_like(xs, (t / (len(AGES) - 1)) * 2.0 - 1.0)
        G, Yq = transition(int(age), ZY, ZR)
        Gt = torch.tensor(G); Yqt = torch.tensor(Yq)
        best = np.zeros(xs.numel()); bestv = np.full(xs.numel(), -np.inf)
        for al in ag:
            rp = RF + al * (ZRt - RF)
            xp = xs[:, None] * rp[None, :] / Gt[None, :] + Yqt[None, :]
            phin = net(xp.reshape(-1), au[:, None].expand_as(xp).reshape(-1)).reshape(xp.shape)
            cn = xp * phin.pow(1.0 / GAMMA)
            val = (cn.pow(1 - GAMMA) * Wt[None, :]).sum(dim=1).numpy()
            imp = val > bestv
            bestv = np.where(imp, val, bestv); best = np.where(imp, al, best)
        alpha_t = torch.tensor(best)
        surv_t = torch.full_like(xs, SURVX[t])
        c = euler_c(xs, alpha_t, Gt, Yqt, au, net, surv_t, ZRt)
        Xg.append(np.concatenate([[0.0], (xs + c).numpy()]))
        Cp.append(np.concatenate([[0.0], c.numpy()]))
        Ap.append(np.concatenate([[0.0], best]))
    return dict(x=Xg, c=Cp, alpha=Ap, ages=AGES, p_surv=SURVX,
                params=dict(gamma=GAMMA, beta=BETA, Rf=RF, mu=MU, se=SE, s2e=S2E,
                            s2u=S2U, repl=REPL, aK=AK, a20=A20, aT=AT))


if __name__ == "__main__":
    net, hist, secs = train(iterations=4000, batch=192, verbose=True)
    out = os.path.join(_HERE, "..", "results"); os.makedirs(out, exist_ok=True)
    torch.save(net.state_dict(), os.path.join(out, "nn_lifecycle.pt"))
    json.dump(dict(iterations=4000, seconds=secs, history=hist),
              open(os.path.join(out, "nn_lifecycle_training.json"), "w"), indent=2)
    print("trained in %.0f s; final log-phi residual rms %.3e" % (secs, hist[-1][1]))
