import numpy as np, pandas as pd, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0,"/Users/zwkai/Desktop/复旦新开始/cgm_python")
from cgm_model import solve, simulate
ROOT="/Users/zwkai/Desktop/复旦新开始"; OUT=os.path.join(ROOT,"cgm_python","results")
T=pd.read_csv(os.path.join(ROOT,"GPT_repo_sync","cgm_replication","paper_targets","figure3_annual_targets.csv")).set_index("age")
sol=solve(na=401,amax=80.0,nr=7,ny=7,verbose=False)
sim=simulate(sol,N=50000,seed=2026)
mine=pd.DataFrame(dict(age=sol["ages"],consumption=sim["meanC"],wealth=sim["meanW"],
                       income=sim["meanY"],alpha_mean=sim["meanA"])).set_index("age")
mc=pd.read_csv(os.path.join(ROOT,"GPT_repo_sync","cgm_replication","matlab_final","cgm_seed_20260909.csv"),header=None)
# the repo CSV is [age, meanC, meanW, meanY, meanSavings, meanAlpha, seW, seAlpha]
mc.columns=["age","C","W","Y","S","alpha_mean","seW","seA"]
repo=pd.DataFrame(dict(age=mc.age,consumption=mc.C,wealth=mc.W,income=mc.Y,
                       alpha_mean=mc.alpha_mean)).set_index("age")
def rmse(a,b):
    m=(~np.isnan(a))&(~np.isnan(b)); return float(np.sqrt(np.mean((np.asarray(a)[m]-np.asarray(b)[m])**2)))
rows=[]
for nm,d,ispaper in [("paper (digitised Fig 3)",T,True),("python re-implementation (this work)",mine,False),("previous Matlab solver (repo)",repo,False)]:
    r={"series":nm}
    for col,lab in [("wealth","W"),("consumption","C"),("income","Y"),("alpha_mean","a")]:
        r[lab+"@65"]=round(float(d.loc[65,col]),4); r[lab+"peak"]=round(float(d[col].max()),4)
    if not ispaper:
        for col,lab in [("wealth","W"),("consumption","C"),("income","Y"),("alpha_mean","a")]:
            r[lab+"_RMSE_vs_paper"]=round(rmse(d[col],T[col]),4)
    rows.append(r)
tbl=pd.DataFrame(rows); pd.set_option("display.width",260)
print(tbl.to_string(index=False))
tbl.to_csv(os.path.join(OUT,"figure3_comparison_table.csv"),index=False)
fig,ax=plt.subplots(1,3,figsize=(15,4.2))
for col,t,axi in [("wealth","Wealth (USD '000)",0),("consumption","Consumption (USD '000)",1),("alpha_mean","Mean equity share",2)]:
    ax[axi].plot(T.index,T[col],"k-",lw=2,label="paper (digitised)")
    ax[axi].plot(mine.index,mine[col],"b--",lw=2,label="python re-implementation")
    ax[axi].plot(repo.index,repo[col],"r:",lw=2,label="previous Matlab solver")
    ax[axi].set_title(t); ax[axi].set_xlabel("age"); ax[axi].grid(alpha=.3)
    if axi==0: ax[axi].legend()
ax[1].plot(T.index,T.income,"g-.",lw=1.5,label="income (paper)"); ax[1].legend()
fig.suptitle("CGM (2005) Figure 3: paper vs two independent solvers",fontsize=12)
fig.tight_layout(); fig.savefig(os.path.join(OUT,"figure3_replication_comparison.png"),dpi=150)
print("saved figure3_replication_comparison.png")
