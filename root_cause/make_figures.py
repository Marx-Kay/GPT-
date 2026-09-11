"""Produce the figures and the numeric record used in the final report."""
import numpy as np, json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import ref_cgm as R

plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": .3,
                     "figure.dpi": 160, "savefig.bbox": "tight"})
OUT = "/Users/zwkai/Desktop/复旦新开始/root_cause"

paper = {}
import csv
with open("/Users/zwkai/Desktop/复旦新开始/GPT_repo_sync/cgm_replication/paper_targets/"
          "figure3_annual_targets.csv") as f:
    for row in csv.DictReader(f):
        paper[int(row["age"])] = {k: (float(v) if v not in ("", "NaN") else np.nan)
                                  for k, v in row.items() if k != "age"}

fig2 = json.load(open("/tmp/fig2_targets.json"))

SPECS = [
    ("literal  $\\sigma_u^2=0.0106$", dict(), 0.0106),
    ("$\\sigma_u^2=0$ in DP", dict(s2u=0.0), 0.0106),
    ("$\\sigma_u^2=5\\times10^{-4}$ in DP", dict(s2u=0.0005), 0.0106),
]
rec = {}
curves = {}
for tag, kw, simsu in SPECS:
    s = R.solve(nq=7, na=401, **kw)
    m = R.simulate(s, nsim=40000, s2u=simsu)
    curves[tag] = m
    k = lambda a: int(np.where(m["age"] == a)[0][0])
    rec[tag] = dict(solve=kw, sim_s2u=simsu, W65=float(m["mW"][k(65)]),
                    W_peak=float(m["mW"].max()), W25=float(m["mW"][k(25)]),
                    W40=float(m["mW"][k(40)]), C65=float(m["mC"][k(65)]),
                    a65=float(m["mA"][k(65)]), a20=float(m["mA"][k(20)]),
                    CY25=float(m["mC"][k(25)] / m["mY"][k(25)]),
                    CY65=float(m["mC"][k(65)] / m["mY"][k(65)]))
    np.savetxt(f"{OUT}/sim_{tag.replace(' ','_').replace('$','').replace('\\\\','')}.csv",
               np.c_[m["age"], m["mY"], m["mW"], m["mC"], m["mA"]], delimiter=",",
               header="age,income,wealth,consumption,alpha", comments="")
rec["paper"] = dict(W65=221.476, W25=10.328, W40=56.497, C65=35.553, a65=0.4992,
                    a20=0.9629, CY25=0.969, CY65=1.099)
json.dump(rec, open(f"{OUT}/root_cause_results.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps(rec, indent=1, ensure_ascii=False))

ageP = np.array(sorted(paper))
gv = lambda key: np.array([paper[a].get(key, np.nan) for a in ageP])

# ------------------------------------------------------------------ figure 1: wealth path
fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.2))
ax = axes[0]
ax.plot(ageP, gv("wealth"), "o-", ms=2.6, lw=1.3, color="#c0392b", label="CGM (2005) Fig. 3A")
for tag, m in curves.items():
    ax.plot(m["age"], m["mW"], lw=1.4, label=tag)
ax.set_xlabel("age"); ax.set_ylabel("mean financial wealth\n(thousands of 1992 USD)")
ax.set_title("Figure 3A — wealth"); ax.legend(fontsize=6.4, loc="upper left"); ax.set_xlim(20, 100)
ax = axes[1]
ax.plot(ageP, gv("income"), "o-", ms=2.6, lw=1.3, color="#c0392b", label="paper")
ax.plot(curves[list(curves)[0]]["age"], curves[list(curves)[0]]["mY"], lw=1.4, color="#2c3e50",
        label="model")
ax.set_xlabel("age"); ax.set_ylabel("mean labour income"); ax.set_title("Figure 3A — income")
ax.legend(fontsize=7); ax.set_xlim(20, 100)
ax = axes[2]
ax.plot(ageP, gv("alpha_mean"), "o-", ms=2.6, lw=1.3, color="#c0392b", label="paper")
for tag, m in curves.items():
    ax.plot(m["age"], m["mA"], lw=1.4, label=tag)
ax.set_xlabel("age"); ax.set_ylabel("mean stock share $\\alpha$")
ax.set_title("Figure 3C — portfolio share"); ax.legend(fontsize=6.4); ax.set_xlim(20, 100)
fig.savefig(f"{OUT}/fig_repl_fig3.pdf"); plt.close(fig)

# ------------------------------------------------------------------ figure 2: policy functions
fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.4))
solv = {tag: R.solve(nq=7, na=401, **kw) for tag, kw, _ in SPECS}
XS = np.array([10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200, 250])
for ax, age in zip(axes, [20, 35, 65]):
    pa = np.array(fig2["C"][str(age)])
    ax.plot(pa[:, 0], pa[:, 1], "o", ms=3.2, color="#c0392b", label="CGM (2005) Fig. 2C", zorder=5)
    for tag, s in solv.items():
        i = int(np.where(R.AGES == age)[0][0]); P = R.F(age)
        ax.plot(XS, np.interp(XS / P, s["X"][i], s["C"][i]) * P, lw=1.5, label=tag)
    ax.set_xlabel("cash-on-hand"); ax.set_title(f"consumption policy, age {age}")
    ax.set_xlim(0, 260)
axes[0].set_ylabel("consumption\n(thousands of 1992 USD)")
axes[0].legend(fontsize=6.4)
fig.savefig(f"{OUT}/fig_repl_fig2c.pdf"); plt.close(fig)

# ------------------------------------------------------------------ figure 3: ablation bars
abl = json.load(open("/tmp/ablation.json")) if False else None
print("figures written")
