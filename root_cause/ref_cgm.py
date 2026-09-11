"""
Reference implementation of CGM (2005) benchmark (high-school group, annual, ages 20-100).

Purpose: an implementation written directly from the paper, with every modelling
choice exposed as an explicit switch, so that the numerical consequences of each
choice can be isolated (ablation study).

Model (paper equations 1-10):
    log Y_it = f(t) + v_it + eps_it          (working age, t <= 65)
    v_it     = v_i,t-1 + u_it                (random walk)
    Y_it     = lambda * P_iK  for t > K      (retirement, K = 65)
    P_it     = exp(f(t) + v_it)              (permanent income)
    X_it     = W_it + Y_it                   (cash-on-hand)
    W_i,t+1  = R_p,i,t+1 * (X_it - C_it)
    R_p      = alpha*R + (1-alpha)*Rf
    0 <= alpha <= 1 ,  C <= X  (no borrowing), no bequests

Normalised state x = X / P  (P frozen after retirement):
    v_t(x) = max_{c,alpha}  u(c) + delta*p_t*E[ G_{t+1}^{1-gamma} v_{t+1}(x') ]
    x'     = a * R_p(alpha) / G_{t+1} + y_{t+1},      a = x - c
    G_{t+1} = P_{t+1}/P_t ,  y_{t+1} = Y_{t+1}/P_{t+1}

Euler / envelope:
    c_t^{-gamma} = delta*p_t * E[ (G_{t+1} c_{t+1})^{-gamma} R_p ]
    portfolio FOC: E[ (G_{t+1} c_{t+1})^{-gamma} (R - Rf) ] = 0
"""
import numpy as np

# ----------------------------------------------------------------------------- parameters
IC = np.array([0.530339, 0.16818, -0.00323371, 0.000019704])  # Table 1 const 2.700381 + Table 2 -2.170042


def F(age):
    a = np.asarray(age, float)
    return np.exp(IC[0] + IC[1] * a + IC[2] * a ** 2 + IC[3] * a ** 3)


# conditional survival probabilities survprob(i) = P(survive from age 19+i to age 20+i)
SURV = np.array([
    0.99845, 0.99839, 0.99833, 0.99830, 0.99827, 0.99826, 0.99824, 0.99820, 0.99813, 0.99804,
    0.99795, 0.99785, 0.99776, 0.99766, 0.99755, 0.99743, 0.99730, 0.99718, 0.99707, 0.99696,
    0.99685, 0.99672, 0.99656, 0.99635, 0.99610, 0.99579, 0.99543, 0.99504, 0.99463, 0.99420,
    0.99370, 0.99311, 0.99245, 0.99172, 0.99091, 0.99005, 0.98911, 0.98803, 0.98680, 0.98545,
    0.98409, 0.98270, 0.98123, 0.97961, 0.97786, 0.97603, 0.97414, 0.97207, 0.96970, 0.96699,
    0.96393, 0.96055, 0.95690, 0.95310, 0.94921, 0.94508, 0.94057, 0.93570, 0.93031, 0.92424,
    0.91717, 0.90922, 0.90089, 0.89282, 0.88503, 0.87622, 0.86576, 0.85440, 0.84230, 0.82942,
    0.81540, 0.80002, 0.78404, 0.76842, 0.75382, 0.73996, 0.72464, 0.71057, 0.69610, 0.68090])


def gh(n):
    """Gauss-Hermite nodes/weights for N(0,1)."""
    x, w = np.polynomial.hermite_e.hermegauss(n)
    return x, w / np.sqrt(2 * np.pi)


AGES = np.arange(20, 101)          # 81 ages, 20 ... 100
T = len(AGES)
K = 65                             # last working year; retirement income from age 66


# ----------------------------------------------------------------------------- solver
def solve(beta=.96, gamma=10., Rf=1.02, prem=.04, sig=.157, s2e=.0738, s2u=.0106,
          repl=.68212, nq=9, na=601, amax=200.0,
          euler_g=True,       # include the G^{-gamma} permanent-income term in the Euler eq.
          indep_shocks=True,  # permanent / transitory / return shocks drawn as independent nodes
          cons_cap="x",       # "x": c <= x  (paper) ; "a": c <= a (spurious extra cap c <= x/2)
          alpha_foc="envelope",   # "envelope": E[(Gc')^{-g}(R-Rf)] ; "oneperiod": Var of U(c')
          trans_mean1=False,  # if True rescale exp(eps) to mean 1
          nu_forget=False,    # permanent shock hits income but does NOT persist in the
                              # state used for discounting (the "normalise v_it to one" reading)
          verbose=False):
    n_node = nq ** 3
    zi, wi = gh(nq)
    # 3-D tensor: axis order (u, eps, ret)
    U3, E3, R3 = np.meshgrid(zi, zi, zi, indexing="ij")
    W3 = wi[:, None, None] * wi[None, :, None] * wi[None, None, :]
    u = U3.ravel(); e = E3.ravel(); r = R3.ravel(); wq = W3.ravel()
    R = Rf + prem + sig * r
    assert (R > 0).all()

    ss_u = np.sqrt(s2u); ss_e = np.sqrt(s2e)
    Nu = np.exp(ss_u * u)                       # permanent growth shock
    if trans_mean1:
        Ne = np.exp(ss_e * e - 0.5 * s2e)
    else:
        Ne = np.exp(ss_e * e)                   # transitory level shock (paper eq. 2)
    if not indep_shocks:
        # both income shocks driven by one common node  (the DeepSeek-Python defect)
        u = u.copy(); Nu = np.exp(ss_u * u); Ne = np.exp(ss_e * u)

    p_surv = np.concatenate([SURV[:T - 1], [0.0]])   # p_surv[i] = P(survive age 20+i -> 21+i)

    a = np.concatenate([[0.0], np.geomspace(1e-7, amax, na - 1)])
    N = a.size
    X = np.zeros((T, N + 1)); C = np.zeros((T, N + 1)); AL = np.zeros((T, N + 1))
    # terminal: age 100, consume everything
    X[T - 1] = np.concatenate([[0.0], a]); C[T - 1] = np.concatenate([[0.0], a])
    AL[T - 1] = 0.0

    for i in range(T - 2, -1, -1):
        age = AGES[i]
        child = AGES[i + 1]
        if child >= K + 1:                      # child is retired: P frozen -> G = 1, y = lambda
            G = np.ones(n_node); y = repl * np.ones(n_node)
        elif nu_forget:
            # shock arrives in income but the permanent level used for scaling does not
            # carry it: G deterministic, and the permanent shock acts like an extra
            # transitory shock in the normalised state.
            G = np.full(n_node, F(child) / F(age)); y = Ne * Nu
        else:                                   # child is working
            G = F(child) / F(age) * Nu; y = Ne
        xnext = X[i + 1]; cnext = C[i + 1]

        # ---- portfolio choice for every savings level a (independent of c)
        lo = np.zeros(N); hi = np.ones(N)
        for _ in range(60):
            al = 0.5 * (lo + hi)
            rp = Rf + al[:, None] * (R - Rf)[None, :]
            xp = a[:, None] * rp / G[None, :] + y[None, :]
            cn = np.interp(xp, xnext, cnext)
            if alpha_foc == "envelope":
                foc = ((G * cn) ** (-gamma) * (R - Rf)[None, :] * wq[None, :]).sum(1)
            else:                               # one-period-utility rule (DeepSeek-Python defect)
                xx = xp[:, :, None] if False else None
                foc = ((cn ** (1 - gamma)) * (R - Rf)[None, :] * wq[None, :]).sum(1)
            lo = np.where(foc > 0, al, lo); hi = np.where(foc <= 0, al, hi)
        al = 0.5 * (lo + hi)
        al = np.clip(al, 0.0, 1.0)
        rp = Rf + al[:, None] * (R - Rf)[None, :]
        xp = a[:, None] * rp / G[None, :] + y[None, :]
        cn = np.interp(xp, xnext, cnext)
        marg = (G * cn) ** (-gamma) if euler_g else cn ** (-gamma)
        expected = (marg * rp * wq[None, :]).sum(1)
        ct = (beta * p_surv[i] * expected) ** (-1.0 / gamma)
        if cons_cap == "a":                     # spurious cap c <= a  <=>  c <= x/2
            ct = np.minimum(ct, a)
        xt = a + ct
        assert np.all(np.diff(xt) > 0), "non-monotone endogenous grid at age %d" % age
        X[i] = np.concatenate([[0.0], xt])
        C[i] = np.concatenate([[0.0], ct])
        AL[i] = np.concatenate([[al[0]], al])
        if verbose:
            print("  age %3d  c(a=5)=%.4f  alpha(a=5)=%.4f" % (age, ct[np.argmin(abs(a - 5))],
                                                               al[np.argmin(abs(a - 5))]))
    return dict(X=X, C=C, AL=AL, p_surv=p_surv, a=a,
                params=dict(beta=beta, gamma=gamma, Rf=Rf, prem=prem, sig=sig, s2e=s2e, s2u=s2u,
                            repl=repl, nq=nq, na=na, amax=amax))


# ----------------------------------------------------------------------------- simulator
def simulate(sol, nsim=20000, seed=20260909, ret_kind="persistent", init_draw=False,
             ret_rule=None, mean1=False, s2u=None):
    """ret_kind: 'persistent' (paper) or 'iid' (shock loaded on the level).
    ret_rule: None -> optimal alpha from the policy function;
              a scalar or callable(age) -> exogenous portfolio rule (portfolio fixed).
    s2u: override the permanent-shock variance used in the *simulated* income process
         (used to test solve/simulate consistency)."""
    beta = sol["params"]["beta"]; gamma = sol["params"]["gamma"]
    Rf, prem, sig = sol["params"]["Rf"], sol["params"]["prem"], sol["params"]["sig"]
    s2e = sol["params"]["s2e"]; repl = sol["params"]["repl"]
    s2u = sol["params"]["s2u"] if s2u is None else s2u
    X, C, AL = sol["X"], sol["C"], sol["AL"]
    rng = np.random.default_rng(seed)
    half = nsim // 2
    z = rng.standard_normal((T, half, 3))
    z = np.concatenate([z, -z], axis=1)          # antithetic pairs
    nsim = 2 * half
    R = Rf + prem + sig * z[:, :, 2]

    W = np.zeros((T, nsim)); P = np.zeros((T, nsim)); Y = np.zeros((T, nsim))
    Cc = np.zeros((T, nsim)); Ac = np.zeros((T, nsim)); S = np.zeros((T, nsim))
    for i in range(T):
        age = AGES[i]
        if i == 0:
            v0 = np.sqrt(s2u) * z[0, :, 0] if init_draw else 0.0
            P[i] = F(age) * np.exp(v0)
        elif age <= K:
            P[i] = P[i - 1] * F(age) / F(age - 1) * np.exp(np.sqrt(s2u) * z[i, :, 0])
        else:
            P[i] = P[i - 1]
        if age <= K:
            Y[i] = P[i] * np.exp(np.sqrt(s2e) * z[i, :, 1] - (0.5 * s2e if mean1 else 0.0))
            if ret_kind == "iid":
                Y[i] = Y[i] * np.exp(np.sqrt(s2u) * z[i, :, 0])
        else:
            Y[i] = repl * P[i]
        cash = W[i] + Y[i]
        xn = cash / P[i]
        if i == T - 1:
            Cc[i] = cash; Ac[i] = 0.0
        else:
            Cc[i] = np.minimum(cash, np.interp(xn, X[i], C[i]) * P[i])
            if ret_rule is None:
                Ac[i] = np.clip(np.interp(xn, X[i], AL[i]), 0, 1)
            else:
                a_r = ret_rule(age) if callable(ret_rule) else ret_rule
                Ac[i] = np.clip(a_r, 0, 1)
        S[i] = cash - Cc[i]
        if i < T - 1:
            rp = Rf + Ac[i] * (R[i] - Rf)
            assert (rp > 0).all()
            W[i + 1] = S[i] * rp
    return dict(age=AGES, W=W, Y=Y, C=Cc, A=Ac, P=P, S=S,
                mW=W.mean(1), mY=Y.mean(1), mC=Cc.mean(1), mA=Ac.mean(1))


# ----------------------------------------------------------------------------- paper targets
PAPER = dict(wealth_peak=221.476, wealth_peak_age=65, cons65=35.553, inc65=32.359, alpha65=0.4992)


def report(sim, tag=""):
    i65 = np.where(sim["age"] == 65)[0][0]
    ipk = sim["mW"].argmax()
    out = dict(tag=tag,
               wealth_peak=round(float(sim["mW"].max()), 2),
               peak_age=int(sim["age"][ipk]),
               wealth65=round(float(sim["mW"][i65]), 2),
               cons65=round(float(sim["mC"][i65]), 2),
               inc65=round(float(sim["mY"][i65]), 2),
               alpha65=round(float(sim["mA"][i65]), 4),
               alpha20=round(float(sim["mA"][0]), 4),
               cons_inc_ratio_25=round(float(sim["mC"][np.where(sim["age"] == 25)[0][0]] /
                                            sim["mY"][np.where(sim["age"] == 25)[0][0]]), 3))
    return out
