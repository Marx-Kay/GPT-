import json, os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUT=os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","results")
dpi=json.load(open(os.path.join(OUT,"dpi_dimension_scaling.json"),encoding="utf-8"))
rows=[]
for N in (20,40):
    for d in (1,2,3,5,10,20):
        n=float(N)**d
        rows.append(dict(method="grid (N=%d/dim)"%N, states=d, points=n,
                         memory_GB=n*8/1e9, est_seconds=n*1e-7))
for r in dpi:
    rows.append(dict(method="deep learning", states=r["states"], points=np.nan,
                     memory_GB=0.07, est_seconds=r["train_seconds"],
                     alpha_max_error=r["alpha_max_error"], ratio_max_error=r["ratio_max_error"],
                     params=r["n_parameters"]))
tbl=pd.DataFrame(rows)
pd.set_option("display.width",220)
print(tbl.to_string(index=False))
tbl.to_csv(os.path.join(OUT,"dimensionality_comparison.csv"),index=False)

fig,ax=plt.subplots(1,2,figsize=(12,4.4))
for N,c in [(20,"tab:blue"),(40,"tab:red")]:
    sub=[r for r in rows if r["method"]=="grid (N=%d/dim)"%N]
    ax[0].semilogy([r["states"] for r in sub],[max(r["memory_GB"],1e-9) for r in sub],"o-",color=c,label="grid N=%d"%N)
ax[0].axhline(64,ls="--",color="gray"); ax[0].text(1.2,80,"64 GB RAM",color="gray",fontsize=9)
xs=[r["states"] for r in dpi]; ys=[r["train_seconds"] for r in dpi]
ax[0].semilogy(xs,ys,"s-",color="tab:green",label="deep learning (measured)")
ax[0].set_xlabel("number of state variables"); ax[0].set_ylabel("memory (GB)")
ax[0].set_title("Grid method: memory grows like N^d"); ax[0].legend(); ax[0].grid(alpha=.3,which="both")
ax[1].semilogy(xs,[r["alpha_max_error"] for r in dpi],"o-",label="max |alpha - alpha*|")
ax[1].semilogy(xs,[r["ratio_max_error"] for r in dpi],"s-",label="max |c/w - (c/w)*|")
ax[1].set_xlabel("number of state variables"); ax[1].set_ylabel("absolute error vs closed form")
ax[1].set_title("Deep policy iteration: accuracy vs dimension"); ax[1].legend(); ax[1].grid(alpha=.3,which="both")
fig.suptitle("Curse of dimensionality: grid solver vs deep learning (closed-form benchmark)",fontsize=12)
fig.tight_layout(); fig.savefig(os.path.join(OUT,"dimensionality_comparison.png"),dpi=150)
print("\nsaved dimensionality_comparison.png and dimensionality_comparison.csv")
