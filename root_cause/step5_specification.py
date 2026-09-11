"""Which specification reproduces BOTH CGM Figure 2C/2B and CGM Figure 3A?"""
import numpy as np, json, ref_cgm as R

T = json.load(open("/tmp/fig2_targets.json"))
XS = np.array([10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200, 250], float)


def fig2c_rmse(sol, age):
    if str(age) not in T["C"]:
        return np.nan
    pa = np.array(T["C"][str(age)])
    i = int(np.where(R.AGES == age)[0][0])
    P = R.F(age)
    mine = np.interp(XS / P, sol["X"][i], sol["C"][i]) * P
    paper = np.interp(XS, pa[:, 0], pa[:, 1])
    return float(np.sqrt(np.mean((mine - paper) ** 2)))


def fig2b_rmse(sol, age):
    if str(age) not in T["B"]:
        return np.nan
    pa = np.array(T["B"][str(age)])
    i = int(np.where(R.AGES == age)[0][0])
    P = R.F(age)
    xs = pa[:, 0]
    mine = np.interp(xs / P, sol["X"][i], sol["AL"][i])
    return float(np.sqrt(np.mean((mine - pa[:, 1]) ** 2)))


def test(tag, sim_s2u=None, **kw):
    kw.setdefault("nq", 7); kw.setdefault("na", 401)
    s = R.solve(**kw)
    m = R.simulate(s, nsim=20000, s2u=sim_s2u)
    k = lambda a: int(np.where(m["age"] == a)[0][0])
    r20, r35, r65 = (fig2c_rmse(s, a) for a in (20, 35, 65))
    b20, b30, b75 = (fig2b_rmse(s, a) for a in (20, 30, 75))
    print(f"{tag:40s} W65={m['mW'][k(65)]:7.1f} W25={m['mW'][k(25)]:6.1f} "
          f"C/Y25={m['mC'][k(25)]/m['mY'][k(25)]:.3f} W40={m['mW'][k(40)]:6.1f} | "
          f"Fig2C RMSE 20/35/65 = {r20:4.2f}/{r35:4.2f}/{r65:4.2f} | "
          f"Fig2B RMSE 20/30/75 = {b20:4.2f}/{b30:4.2f}/{b75:4.2f}")


print("=" * 150)
print("PAPER  Fig.3A: W65=221.5  W25=10.3  W40=56.5  C/Y25=0.969   (Fig.2C is in thousands of 1992 USD)")
print("-" * 150)
for lbl, solve_kw, sim_s2u in [
        ("DP .0106 / sim .0106  (literal reading)", {}, 0.0106),
        ("DP .0106 / sim 0", {}, 0.0),
        ("DP 0     / sim .0106", dict(s2u=0.0), 0.0106),
        ("DP 0     / sim 0", dict(s2u=0.0), 0.0),
        ("DP .0106, no G in Euler / sim .0106", dict(euler_g=False), 0.0106),
        ("DP 0,     no G in Euler / sim .0106", dict(s2u=0.0, euler_g=False), 0.0106),
        ("DP .0106, share one shock node / sim .0106", dict(indep_shocks=False), 0.0106)]:
    test(lbl, sim_s2u=sim_s2u, **solve_kw)
