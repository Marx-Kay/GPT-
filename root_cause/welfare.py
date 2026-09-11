"""CGM (2005) Table 6 replication: consumption-equivalent cost of fixed portfolio rules.

Appendix C: for each rule the consumption/savings problem is re-solved optimally subject
to the exogenous portfolio weight, then lifetime expected utility from age 20 with zero
initial financial wealth is converted into an equivalent constant consumption stream.
Reported loss = 1 - EC^rule / EC^optimal, in per cent.
"""
import numpy as np, ref_cgm as R

AGES = R.AGES
NT = len(AGES)
MERTON_NI = 0.04 / (10.0 * 0.157 ** 2)          # mu/(gamma sigma^2) = 0.16228


def p_survival():
    return np.concatenate([R.SURV[:NT - 1], [0.0]])


# ---------------------------------------------------------------- policy with exogenous alpha
def solve_rule(rule, s2u=0.0106, nq=7, na=401, amax=200.0):
    beta, gamma = .96, 10.
    Rf, prem, sig = 1.02, .04, .157
    s2e, repl = .0738, .68212
    zi, wi = R.gh(nq)
    U3, E3, R3 = np.meshgrid(zi, zi, zi, indexing="ij")
    wq = (wi[:, None, None] * wi[None, :, None] * wi[None, None, :]).ravel()
    uu, ee, rr = U3.ravel(), E3.ravel(), R3.ravel()
    Rr = Rf + prem + sig * rr
    Nu, Ne = np.exp(np.sqrt(s2u) * uu), np.exp(np.sqrt(s2e) * ee)
    ps = p_survival()
    a = np.concatenate([[0.0], np.geomspace(1e-7, amax, na - 1)])
    N = a.size
    X = np.zeros((NT, N + 1)); C = np.zeros((NT, N + 1))
    X[-1] = np.concatenate([[0.0], a]); C[-1] = X[-1]
    for i in range(NT - 2, -1, -1):
        age, child = AGES[i], AGES[i + 1]
        if child >= 66:
            G = np.ones_like(Rr); y = repl * np.ones_like(Rr)
        else:
            G = R.F(child) / R.F(age) * Nu; y = Ne
        al = min(1.0, max(0.0, float(rule(age))))
        rp = Rf + al * (Rr - Rf)
        xp = a[:, None] * rp[None, :] / G[None, :] + y[None, :]
        cn = np.interp(xp, X[i + 1], C[i + 1])
        exp_ = ((G * cn) ** (-gamma) * rp[None, :] * wq[None, :]).sum(1)
        ct = (beta * ps[i] * exp_) ** (-1.0 / gamma)
        xt = a + ct
        X[i] = np.concatenate([[0.0], xt]); C[i] = np.concatenate([[0.0], ct])
    return dict(X=X, C=C, ps=ps, s2u=s2u)


# ---------------------------------------------------------------- simulation / utility
def _sim_utility(X, C, AL, rule, s2u=0.0106, nsim=40000, seed=777, pdv=None):
    """pdv: optional array PDV_t/P_t used by the 'no income risk' rule."""
    beta, gamma = .96, 10.
    Rf, prem, sig = 1.02, .04, .157
    s2e, repl = .0738, .68212
    ps = p_survival()
    rng = np.random.default_rng(seed); half = nsim // 2
    z = rng.standard_normal((NT, half, 3)); z = np.concatenate([z, -z], 1)
    nsim = 2 * half
    Rr = Rf + prem + sig * z[:, :, 2]
    W = np.zeros(nsim); P = np.full(nsim, R.F(20.0))
    util = 0.0; disc = 1.0
    for i, age in enumerate(AGES):
        if i > 0:
            if age <= 65:
                P = P * R.F(age) / R.F(age - 1) * np.exp(np.sqrt(s2u) * z[i, :, 0])
        Y = P * np.exp(np.sqrt(s2e) * z[i, :, 1]) if age <= 65 else repl * P
        cash = W + Y
        xn = cash / P
        if i == NT - 1:
            Cn = cash
        else:
            Cn = np.minimum(cash, np.interp(xn, X[i], C[i]) * P)
        util += disc * float(np.mean(Cn ** (1 - gamma) / (1 - gamma)))
        if i < NT - 1:
            if rule is None:
                al = np.clip(np.interp(xn, X[i], AL[i]), 0, 1)
            elif isinstance(rule, str) and rule == "no_income_risk":
                Wn = W / P                                     # normalised financial wealth
                al = np.clip(MERTON_NI * (Wn + pdv[i]) / np.maximum(Wn, 1e-6), 0, 1)
            else:
                al = np.clip(float(rule(age)), 0, 1) * np.ones(nsim)
            W = (cash - Cn) * (Rf + al * (Rr[i] - Rf))
        disc *= beta * ps[i]
    S = 0.0; disc = 1.0
    for i in range(NT):
        S += disc; disc *= beta * ps[i]
    return (util * (1 - gamma) / S) ** (1.0 / (1 - gamma))


def pdv_ratio(s2u=0.0106):
    """PDV of future labour income / current permanent income, by age (mortality-adjusted)."""
    ps = p_survival(); Rf = 1.02; s2e = .0738; repl = .68212
    out = np.zeros(NT)
    for i, age in enumerate(AGES):
        s = 0.0; cum = 1.0
        for j in range(i, NT - 1):
            cum *= ps[j] / Rf
            if AGES[j + 1] <= 65:
                s += cum * R.F(AGES[j] + 1) / R.F(AGES[j]) * np.exp(0.5 * s2u + 0.5 * s2e)
            else:
                base = R.F(65) / R.F(age) * np.exp(0.5 * s2u * max(0, 65 - age)) if age <= 65 else 1.0
                s += cum * repl * base
        out[i] = s
    return out


def table6(s2u=0.0106, nsim=40000):
    pdv = pdv_ratio(s2u)
    opt = R.solve(nq=7, na=401, s2u=s2u)
    EC_opt = _sim_utility(opt["X"], opt["C"], opt["AL"], None, s2u=s2u, nsim=nsim)
    rules = {
        "100-Age": lambda age: (100.0 - age) / 100.0,
        "No income": lambda age: MERTON_NI,
        "Zero": lambda age: 0.0,
        "Approx.": lambda age: min(1.0, max(0.5, 2.0 - 0.025 * age)),
    }
    out = {}
    for name, r in rules.items():
        s = solve_rule(r, s2u=s2u)
        out[name] = 100 * (1 - _sim_utility(s["X"], s["C"], None, r, s2u=s2u, nsim=nsim) / EC_opt)
    s = solve_rule(lambda age: MERTON_NI, s2u=s2u)
    out["No income risk"] = 100 * (1 - _sim_utility(s["X"], s["C"], None, "no_income_risk",
                                                   s2u=s2u, nsim=nsim, pdv=pdv) / EC_opt)
    return out


if __name__ == "__main__":
    print("CGM Table 6, benchmark row (loss, % of lifetime consumption)")
    for lbl, d in [("literature value (paper)", None), ("DP sigma_u^2 = 0.0106 (literal)", 0.0106),
                   ("DP sigma_u^2 = 0.0", 0.0)]:
        if d is None:
            print(f"  {lbl:32s} 100-Age= 0.637  No income= 1.531  No income risk= 0.152  Zero= 2.108  Approx.= 0.084")
            continue
        o = table6(s2u=d)
        print(f"  {lbl:32s} " + "  ".join(f"{k}={v:6.3f}" for k, v in o.items()))
