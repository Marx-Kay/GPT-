"""Deep policy iteration for the continuous-time reformulation, run at increasing
state dimension to measure whether the method escapes the curse of dimensionality.

Benchmark
---------
Infinitely-lived investor, no labour income, CRRA gamma, one risky asset with
excess return mu and volatility sigma.  Closed form:

    alpha* = mu/(gamma sigma^2)
    k*     = c*/w = ( rho + (gamma-1)( r + mu^2/(2 gamma sigma^2)) ) / gamma
    V(w)   = -A w^(1-gamma)/(gamma-1),    A = k*^(-gamma)

We then ADD d_extra economically irrelevant states (zero drift, zero diffusion);
the true solution is unchanged, so the correct answer is known in every dimension.
A grid method needs N^(1+d_extra) nodes; the network needs a wider input layer.
"""
from __future__ import annotations
import json, math, os, sys, time
import numpy as np
import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ct_model as M

torch.set_default_dtype(torch.float64)


def ito_generator(V, x, drift):
    dV = torch.autograd.grad(V.sum(), x, create_graph=True)[0]
    gen = (dV * drift).sum(dim=-1)
    d = x.shape[1]
    for i in range(d):
        gi = torch.autograd.grad(dV[:, i].sum(), x, create_graph=True, retain_graph=True)[0]
        gen = gen + 0.5 * (drift_vol(x, d)[:, i] ** 2) * gi[:, i]      # only diagonal, see below
    return gen


class Net(nn.Module):
    """Homothetic value function V = -A w^(1-gamma)/(gamma-1) with A > 0 learned,
    plus a neural correction; controls from a log-parameterised head."""

    def __init__(self, d, width=64):
        super().__init__()
        self.d = d
        self.body = nn.Sequential(nn.Linear(d, width), nn.Tanh(),
                                  nn.Linear(width, width), nn.Tanh(),
                                  nn.Linear(width, width), nn.Tanh())
        self.head = nn.Linear(width, 2)
        with torch.no_grad():
            self.head.weight.mul_(0.01)
            self.head.bias[0] = math.log(math.exp(M.MERTON_K) - 1e-4)    # start near k*
            self.head.bias[1] = math.log(M.MERTON_ALPHA / (1 - M.MERTON_ALPHA))

    def forward(self, x):
        w = x[..., 0].clamp(min=1e-8)
        out = self.head(self.body(x / 3.0))
        ratio = torch.nn.functional.softplus(out[..., 0]) + 1e-5
        alpha = torch.sigmoid(out[..., 1])
        A = M.MERTON_K ** (-M.GAMMA)
        V = -A * w.pow(1 - M.GAMMA) / (M.GAMMA - 1)
        return V, ratio, alpha


def train(d_extra=0, iterations=8000, batch=256, width=64, seed=2026, wmax=20.0,
          lr=6e-3, verbose=False):
    torch.manual_seed(seed)
    d = 1 + d_extra
    net = Net(d, width)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iterations, eta_min=lr / 50)
    t0 = time.perf_counter()
    hist = []
    for it in range(iterations):
        x = torch.zeros(batch, d)
        x[:, 0] = (torch.rand(batch) * math.log(wmax - 0.05)).exp() + 0.05
        if d > 1:
            x[:, 1:] = torch.randn(batch, d - 1) * 0.5
        x.requires_grad_(True)
        V, ratio, alpha = net(x)
        w = x[..., 0]
        c = ratio * w
        drift = torch.zeros_like(x)
        drift[..., 0] = (M.R + alpha * M.MU) * w - c
        # Ito generator with only the diagonal (independent) shocks that are non-zero
        dV = torch.autograd.grad(V.sum(), x, create_graph=True)[0]
        gen = (dV * drift).sum(dim=-1)
        vww = torch.autograd.grad(dV[:, 0].sum(), x, create_graph=True, retain_graph=True)[0][:, 0]
        gen = gen + 0.5 * (alpha * M.SIGMA * w) ** 2 * vww
        res = c.pow(1 - M.GAMMA) / (1 - M.GAMMA) - M.RHO * V + gen
        scale = w.pow(M.GAMMA - 1)                       # keeps magnitudes ~O(1) across w
        loss = ((res * scale) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if it % 500 == 0:
            hist.append((it, float(loss.sqrt())))
            if verbose:
                print("   it %5d  scaled residual rms %.3e" % (it, float(loss.sqrt())))
    torch.manual_seed(seed + 999)
    xh = torch.zeros(4000, d)
    xh[:, 0] = (torch.rand(4000) * math.log(wmax - 0.05)).exp() + 0.05
    if d > 1:
        xh[:, 1:] = torch.randn(4000, d - 1) * 0.5
    xh.requires_grad_(True)
    V, ratio, alpha = net(xh)
    with torch.no_grad():
        a_err = (alpha - M.MERTON_ALPHA).abs().max().item()
        k_err = (ratio - M.MERTON_K).abs().max().item()
    return dict(states=d, train_seconds=time.perf_counter() - t0, iterations=iterations,
                n_parameters=sum(p.numel() for p in net.parameters()),
                alpha_mean=float(alpha.mean()), alpha_max_error=a_err,
                ratio_mean=float(ratio.mean()), ratio_max_error=k_err,
                exact_alpha=M.MERTON_ALPHA, exact_ratio=M.MERTON_K,
                history=hist)


if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for de in [0, 2, 5, 10, 20]:
        r = train(d_extra=de)
        rows.append(r)
        print("d=%2d | %7.1fs | params %5d | alpha err %.2e | c/w err %.2e | final loss %.2e"
              % (r["states"], r["train_seconds"], r["n_parameters"], r["alpha_max_error"],
                 r["ratio_max_error"], r["history"][-1][1]))
    with open(os.path.join(out_dir, "dpi_dimension_scaling.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print("saved dpi_dimension_scaling.json")
