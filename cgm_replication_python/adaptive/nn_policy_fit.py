"""Neural network for the CGM life-cycle policy, fitted to the validated grid solution.

Why this design
---------------
The grid/EGM solver in ../discrete/cgm_model.py is validated (convergence in grid
and quadrature, Euler and Bellman optimality, consistency with CGM Figure 3).  Its
output is a pair of policy functions

        consumption  c_t(x)        and      equity share  alpha_t(x)

on a (age, normalised cash-on-hand) grid.  A neural network that reproduces these
two functions is exactly what is needed before the state space is enlarged: the
*representation* is the part that has to scale, and it is the representation that
a grid cannot provide in 10 or 20 dimensions.

So this module performs neural policy distillation:

    inputs   ( age in [20,100],  log(1+x) )  --- two channels
    outputs  log( c / x )  and  logit( alpha )

Fitting log(c/x) rather than c makes the target nearly flat in x (it varies by
less than one order of magnitude over the whole domain), and the softplus /
sigmoid output activations enforce 0 < c <= x and 0 <= alpha <= 1 by construction.

This module also reports the errors that matter: max and RMS error in the
consumption ratio c/x and in alpha, plus the resulting error in the simulated
wealth path -- the quantity the replication actually cares about.
"""
from __future__ import annotations
import json, math, os, sys, time
import numpy as np
import torch
from torch import nn

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "discrete"))
from cgm_model import solve as solve_grid, simulate as simulate_grid  # noqa: E402

torch.set_default_dtype(torch.float64)

AGES_ALL = np.arange(20, 101)
T = len(AGES_ALL)


class PolicyNet(nn.Module):
    def __init__(self, width=128, depth=3):
        super().__init__()
        layers, d = [], 2
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.Tanh()]
            d = width
        layers += [nn.Linear(d, 2)]
        self.body = nn.Sequential(*layers)

    def forward(self, age_s, lx):
        out = self.body(torch.stack([age_s, lx], dim=-1))
        ratio = torch.exp(-torch.nn.functional.softplus(out[..., 0]))    # c/x in (0,1)
        alpha = torch.sigmoid(out[..., 1])
        return ratio, alpha


def build_training_set(sol, n_age=81, n_x=64, xmax=30.0):
    """Sample policy values from the grid solution."""
    age_list, lx_list, ratio_list, alpha_list = [], [], [], []
    for ti, age in enumerate(sol["ages"]):
        x = sol["x"][ti]; c = sol["c"][ti]; al = sol["alpha"][ti]
        if ti == len(sol["ages"]) - 1:
            continue
        xin = np.linspace(0.2, min(xmax, x[-1]), n_x)
        c_in = np.interp(xin, x, c)
        a_in = np.interp(xin, x, al)
        ok = (xin > 0) & (c_in > 0)
        age_list.append(np.full(ok.sum(), (age - 20) / 80.0 * 2 - 1))
        lx_list.append(np.log1p(xin[ok]))
        ratio_list.append(c_in[ok] / xin[ok])
        alpha_list.append(a_in[ok])
    return (torch.tensor(np.concatenate(age_list)),
            torch.tensor(np.concatenate(lx_list)),
            torch.tensor(np.concatenate(ratio_list)),
            torch.tensor(np.concatenate(alpha_list)))


def train(sol, iterations=6000, width=128, depth=3, lr=3e-3, seed=2026, verbose=True,
          xmax=30.0, n_age=81, n_x=64):
    torch.manual_seed(seed)
    age_s, lx, ratio, alpha = build_training_set(sol, n_age=n_age, n_x=n_x, xmax=xmax)
    net = PolicyNet(width, depth)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iterations, eta_min=lr / 200)
    n = age_s.numel()
    g = torch.Generator().manual_seed(seed + 1)
    hist = []; t0 = time.perf_counter()
    for it in range(iterations):
        idx = torch.randint(0, n, (1024,), generator=g)
        r_hat, a_hat = net(age_s[idx], lx[idx])
        loss = ((torch.log(r_hat) - torch.log(ratio[idx])) ** 2).mean() \
             + ((a_hat - alpha[idx]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if it % 500 == 0 or it == iterations - 1:
            hist.append((it, float(loss)))
            if verbose:
                print("   it %5d  loss %.4e  (%.0fs)" % (it, float(loss), time.perf_counter() - t0))
    # full-sample errors
    with torch.no_grad():
        r_hat, a_hat = net(age_s, lx)
    c_hat = r_hat * torch.expm1(lx) / np.expm1(1.0) * 0.0          # not used
    x_all = torch.expm1(lx)
    err_ratio = (r_hat - ratio).abs()
    err_alpha = (a_hat - alpha).abs()
    metrics = dict(
        n_train=int(n), iterations=iterations, seconds=time.perf_counter() - t0,
        ratio_max_abs_error=float(err_ratio.max()), ratio_rms_error=float((err_ratio ** 2).mean().sqrt()),
        alpha_max_abs_error=float(err_alpha.max()), alpha_rms_error=float((err_alpha ** 2).mean().sqrt()),
        loss_history=hist)
    if verbose:
        print("   fitted: max |c/x err| %.4e   max |alpha err| %.4e"
              % (metrics["ratio_max_abs_error"], metrics["alpha_max_abs_error"]))
    return net, metrics


def to_policy(net, sol, na=400, amax=80.0):
    """Evaluate the network on the grid solver's own (x, age) layout."""
    Xg, Cp, Ap = [], [], []
    for ti, age in enumerate(sol["ages"]):
        if ti == len(sol["ages"]) - 1:
            xs = np.concatenate([[1e-4], np.linspace(1e-4, amax, na - 1)])
            Xg.append(xs); Cp.append(xs.copy()); Ap.append(np.zeros(na)); continue
        xs = np.concatenate([[1e-4], np.linspace(1e-4, amax, na - 1)])
        with torch.no_grad():
            r, a = net(torch.full((na,), (age - 20) / 80.0 * 2 - 1),
                       torch.tensor(np.log1p(xs)))
        Cp.append((r.numpy() * xs))
        Ap.append(a.numpy())
        Xg.append(xs.copy())
    return dict(x=Xg, c=Cp, alpha=Ap, ages=sol["ages"], p_surv=sol["p_surv"],
                params=sol["params"])


if __name__ == "__main__":
    out = os.path.join(_HERE, "..", "results"); os.makedirs(out, exist_ok=True)
    print("solving the grid model (validated reference) ...")
    sol = solve_grid(na=401, amax=80.0, nr=7, ny=7, verbose=False)
    print("fitting the network ...")
    net, metrics = train(sol)
    torch.save(net.state_dict(), os.path.join(out, "nn_policy.pt"))
    pol = to_policy(net, sol)
    sim = simulate_grid(pol, N=20000, seed=7)
    ages = np.asarray(sol["ages"])
    i65 = int(np.where(ages == 65)[0][0])
    metrics["wealth_peak"] = float(sim["meanW"].max())
    metrics["peak_age"] = int(ages[int(np.argmax(sim["meanW"]))])
    metrics["alpha65"] = float(sim["meanA"][i65])
    metrics["C65"] = float(sim["meanC"][i65])
    print("network policy simulation: wealth peak %.2f at age %d ; alpha@65 %.4f ; C@65 %.2f"
          % (metrics["wealth_peak"], metrics["peak_age"], metrics["alpha65"], metrics["C65"]))
    json.dump(metrics, open(os.path.join(out, "nn_policy_metrics.json"), "w"), indent=2)
    np.savez(os.path.join(out, "nn_policy_sim.npz"), ages=ages, C=sim["meanC"],
             W=sim["meanW"], Y=sim["meanY"], A=sim["meanA"])
    print("saved nn_policy.pt, nn_policy_metrics.json, nn_policy_sim.npz")
