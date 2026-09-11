"""Compare the model's policy functions (in levels, thousands of 1992 USD) with the
policy functions digitised from CGM (2005) Figure 2."""
import numpy as np, json, ref_cgm as R

T = json.load(open("/tmp/fig2_targets.json"))
XS = [10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200, 250]


def pol(sol, age, x_thousand, P_level):
    X, C = sol["X"], sol["C"]
    i = int(np.where(R.AGES == age)[0][0])
    xn = np.asarray(x_thousand) / P_level
    return np.interp(xn, X[i], C[i]) * P_level


def show(tag, sol, ages, key, Pof, fmt="%8.1f"):
    print("-" * 112)
    print(f"{tag}   (permanent income conditioned at {key})")
    for age in ages:
        paper = [(x, np.interp(x, [p[0] for p in T["C"][str(age)]], [p[1] for p in T["C"][str(age)]]))
                 if str(age) in T["C"] else None for x in XS]
        if paper[0] is None:
            continue
        Pl = Pof(age)
        mine = pol(sol, age, XS, Pl)
        print(f"  age {age:3d}")
        print("    cash-on-hand :" + "".join("%8d" % x for x in XS))
        print("    paper Fig 2C :" + "".join(fmt % v for _, v in paper))
        print("    my model     :" + "".join(fmt % v for v in mine))
        print("    paper C/x    :" + "".join("%8.2f" % (v / x) for (x, v) in paper))
        print("    my    C/x    :" + "".join("%8.2f" % (v / x) for x, v in zip(XS, mine)))


for lbl, kw in [("A. baseline (paper-parameter, paper-faithful)", {}),
                ("B. sigma_u^2 = 0  (no permanent income risk)", dict(s2u=0.0))]:
    sol = R.solve(nq=7, na=401, **kw)
    print("=" * 112)
    print(lbl)
    show("consumption policy in LEVELS", sol, [20, 35, 65],
         "C", lambda a: R.F(a), fmt="%8.1f")
    show("consumption policy in LEVELS", sol, [20, 35, 65],
         "C", lambda a: R.F(a) * (1.0 + 0.0), fmt="%8.1f")

# portfolio share comparison
sol = R.solve(nq=7, na=401)
print("=" * 112)
print("portfolio share alpha(x,t)  --  paper Fig. 2B vs model")
for age in [20, 30, 75]:
    if str(age) not in T["B"]:
        continue
    pa = T["B"][str(age)]
    xs = [p[0] for p in pa]; ys = [p[1] for p in pa]
    i = int(np.where(R.AGES == age)[0][0])
    Pl = R.F(age)
    mine = [float(np.interp(x / Pl, sol["X"][i], sol["AL"][i])) for x in XS]
    print(f"  age {age:3d}   cash-on-hand :" + "".join("%8d" % x for x in XS))
    print("            paper Fig 2B :" + "".join("%8.2f" % np.interp(x, xs, ys) for x in XS))
    print("            my model     :" + "".join("%8.2f" % v for v in mine))
