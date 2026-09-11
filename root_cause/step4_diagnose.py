"""Cross-sectional diagnosis: why does the simulation drift away from the paper's Figure 3A
even though the policy functions (Figure 2C/2B) match?"""
import numpy as np, ref_cgm as R

sol = R.solve(nq=7, na=401)
sim = R.simulate(sol, nsim=40000)
ages = sim["age"]
j = lambda a: int(np.where(ages == a)[0][0])

print("=" * 120)
print("1. full cross-sectional means (my model, paper-faithful)")
print("age  meanY   meanW   meanC   meanS   meanC/X  meanC/Y  mean_alpha  mean_x=X/P")
for a in [20, 21, 22, 25, 30, 35, 40, 45, 50, 55, 60, 65, 66, 70, 80, 90, 100]:
    i = j(a); P = sim["P"][i]
    mY, mW, mC = sim["mY"][i], sim["mW"][i], sim["mC"][i]
    x = (sim["W"][i] + sim["Y"][i]) / P
    print(f"{a:4d} {mY:7.2f} {mW:7.2f} {mC:7.2f} {mW+mY-mC:7.2f} {mC/(mW+mY):8.3f} "
          f"{mC/mY:8.3f} {sim['mA'][i]:11.3f} {x.mean():10.3f}")

print()
print("=" * 120)
print("2. age-20 detail: distribution of normalised cash x = exp(eps) and the model's own policy")
i = j(20)
rng = np.random.default_rng(1)
eps = rng.standard_normal(200000) * np.sqrt(R.s2e if False else 0.0738)
x = np.exp(eps)
cn = np.interp(x, sol["X"][i], sol["C"][i])
print("   E[x]=%.4f  E[c/x]=%.4f  share c/x=1 (constrained)=%.3f" %
      (x.mean(), (cn / x).mean(), (np.abs(cn / x - 1) < 1e-6).mean()))
xs = np.array([0.3, 0.5, 0.7, 0.9, 1.0, 1.2, 1.5, 2.0, 3.0, 4.0])
print("   x_norm   :" + "".join("%8.2f" % v for v in xs))
print("   c/x      :" + "".join("%8.3f" % v for v in np.interp(xs, sol["X"][i], sol["C"][i]) / xs))
print("   X_plot   :" + "".join("%8.1f" % v for v in xs * R.F(20)))
print("   (paper Fig.2C at age 20, C/x): 1.00 0.91 0.78 0.69 0.61 0.51 0.44 0.33 0.27 0.20")
print("   (paper Fig.2C age 20, X=10,15,20,25,30,40,50,75,100,150,200,250 -> C=10.0,13.7,15.6,17.3,18.3,20.4,21.8,24.8,27.1,29.8,32.0,33.8)")

print()
print("=" * 120)
print("3. same diagnostics under alternative specifications")
for tag, kw in [("baseline", {}), ("sigma_u2=0", dict(s2u=0.0)),
                ("no mortality", None), ("no transitory risk", dict(s2e=1e-8)),
                ("no return risk", dict(sig=1e-8))]:
    if kw is None:
        continue
    s = R.solve(nq=7, na=301, **kw)
    m = R.simulate(s, nsim=20000)
    k = lambda a: int(np.where(m["age"] == a)[0][0])
    print(f"   {tag:22s} W65={m['mW'][k(65)]:7.1f}  W25={m['mW'][k(25)]:6.1f}  "
          f"C/Y(20)={m['mC'][k(20)]/m['mY'][k(20)]:.3f}  C/Y(25)={m['mC'][k(25)]/m['mY'][k(25)]:.3f}  "
          f"C/Y(40)={m['mC'][k(40)]/m['mY'][k(40)]:.3f}  a65={m['mA'][k(65)]:.3f}")
print("   paper Figure 3A            W65=  221.5  W25=  10.3  C/Y(20)=  n/a  C/Y(25)=0.969  C/Y(40)=0.940  a65=0.499")
