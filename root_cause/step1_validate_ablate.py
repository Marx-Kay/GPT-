"""Step 1: validate the reference solver against closed-form cases, then run ablations."""
import numpy as np, ref_cgm as R

# ---------------------------------------------------------------- analytic two-period check
def two_period_analytic(x, Rf=1.02, beta=.96, gamma=10., p=0.6809, lam=0.68212):
    return np.minimum(x, (Rf * x + lam) / (Rf + (beta * p * Rf) ** (1.0 / gamma)))

# solver restricted to a two-period problem: only age 99 matters, returns deterministic,
# survival p_99, retirement income lambda, terminal consumption = all cash.
def two_period_solver(xgrid, beta=.96, gamma=10., Rf=1.02, p=0.6809, lam=0.68212, nq=9):
    # terminal value: c_{100} = x
    xt = np.concatenate([[0.0], np.geomspace(1e-7, 500, 400)]); ct = xt.copy()
    a = np.concatenate([[0.0], np.geomspace(1e-7, 500, 400)])
    z, w = R.gh(nq)
    # deterministic return: use single node with weight 1
    rp = np.full((a.size, 1), Rf)
    xp = a[:, None] * rp + lam
    cn = np.interp(xp.ravel(), xt, ct).reshape(xp.shape)
    expected = (cn ** (-gamma) * rp * np.array([1.0])).sum(1)
    c = (beta * p * expected) ** (-1 / gamma)
    x = a + c
    return x, c

print("=" * 78)
print("A. solver validation: analytical two-period problem (age 99)")
xa, ca = two_period_solver(None)
for xv in [0.5, 1.0, 2.0, 5.0]:
    num = np.interp(xv, xa, ca)
    print(f"   x={xv:5.2f}   analytic={two_period_analytic(xv):.6f}   solver={num:.6f}   diff={abs(num-two_period_analytic(xv)):.2e}")

print()
print("=" * 78)
print("B. does the solver satisfy the Euler equation exactly? (independent re-check)")
sol = R.solve(nq=7, na=301)
X, C, AL = sol["X"], sol["C"], sol["AL"]
beta, gamma = .96, 10.
Rf, prem, sig = 1.02, .04, .157
s2e, s2u, repl = .0738, .0106, .68212
zi, wi = R.gh(7)
U3, E3, R3 = np.meshgrid(zi, zi, zi, indexing="ij")
wq = (wi[:, None, None] * wi[None, :, None] * wi[None, None, :]).ravel()
uu, ee, rr = U3.ravel(), E3.ravel(), R3.ravel()
Rr = Rf + prem + sig * rr
p_surv = sol["p_surv"]
errs = []
for i in [0, 20, 40, 44, 50, 70, 79]:      # ages 20, 40, 60, 64, 70, 90, 99
    age = R.AGES[i]; child = R.AGES[i + 1]
    if child >= 66:
        G = np.ones_like(Rr); y = repl * np.ones_like(Rr)
    else:
        G = R.F(child) / R.F(age) * np.exp(np.sqrt(s2u) * uu); y = np.exp(np.sqrt(s2e) * ee)
    for a_idx in [30, 120, 250]:
        a = sol["a"][a_idx]
        al = np.interp(a, np.concatenate([[0.0], sol["a"]]), AL[i]) if False else AL[i][a_idx + 1]
        rp = Rf + al * (Rr - Rf)
        xp = a * rp / G + y
        cn = np.interp(xp, X[i + 1], C[i + 1])
        rhs = (beta * p_surv[i] * ((G * cn) ** (-gamma) * rp * wq).sum()) ** (-1 / gamma)
        lhs = C[i][a_idx + 1]
        errs.append(abs(rhs - lhs) / lhs)
print(f"   Euler relative error over sampled (age, savings) pairs: max={max(errs):.2e}  mean={np.mean(errs):.2e}")

print()
print("=" * 78)
print("C. ABLATION: which modelling choice produces which wealth level?")


def run(tag, **kw):
    s = R.solve(**kw)
    sim = R.simulate(s, nsim=20000)
    rep = R.report(sim, tag)
    j25 = np.where(sim["age"] == 25)[0][0]
    j65 = np.where(sim["age"] == 65)[0][0]
    print(f"   {tag:46s} peak={rep['wealth_peak']:7.1f}@{rep['peak_age']:3d}  "
          f"W65={rep['wealth65']:7.1f}  C/Y(25)={sim['mC'][j25]/sim['mY'][j25]:.3f}  "
          f"C/Y(65)={sim['mC'][j65]/sim['mY'][j65]:.3f}  a65={rep['alpha65']:.3f}  a20={rep['alpha20']:.3f}")
    return rep


res = {}
res["baseline (paper-faithful)"] = run("baseline: G-term, indep shocks, c<=x")
res["no euler G term"] = run("drop G^{-gamma} from the Euler equation", euler_g=False)
res["shared shock node"] = run("permanent & transitory shocks share one node", indep_shocks=False)
res["spurious c<=a cap"] = run("add spurious cap c<=a  (c<=x/2)", cons_cap="a")
res["one-period alpha rule"] = run("portfolio chosen by one-period U(c')", alpha_foc="oneperiod")
res["Python replica"] = run("DeepSeek-Python replica (all four defects)", euler_g=False,
                            indep_shocks=False, cons_cap="a", alpha_foc="oneperiod")
res["trans shock mean 1"] = run("transitory shock rescaled to mean 1", trans_mean1=True)
res["no mortality"] = run("no mortality (p_t = 1 for all t)", )
print()
print("D. parameter sensitivity (same faithful structure)")
for b in [0.96, 0.98, 1.00]:
    run(f"beta = {b}", beta=b)
for g in [10.0, 5.0, 2.0]:
    run(f"gamma = {g}", gamma=g)
for s2u in [0.0106, 0.0]:
    run(f"sigma_u^2 = {s2u}", s2u=s2u)
for rf in [1.02, 1.015]:
    run(f"Rf = {rf}", Rf=rf)
