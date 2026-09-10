import numpy as np, pandas as pd, os, sys
sys.path.insert(0,"/Users/zwkai/Desktop/复旦新开始/cgm_python")
from cgm_model import solve, simulate
ROOT="/Users/zwkai/Desktop/复旦新开始"; OUT=os.path.join(ROOT,"cgm_python","results")
T=pd.read_csv(os.path.join(ROOT,"GPT_repo_sync","cgm_replication","paper_targets","figure3_annual_targets.csv")).set_index("age")
def rmse(a,b):
    m=(~np.isnan(a))&(~np.isnan(b)); return (float(np.sqrt(np.mean((a[m]-b[m])**2))),int(m.sum())) if m.sum()>2 else (np.nan,0)
cands={}
mc=pd.read_csv(os.path.join(ROOT,"GPT_repo_sync","cgm_replication","matlab_final","cgm_seed_20260909.csv"),header=None)
mc.columns=["age","C","W","Y","alpha_mean","x","y","z"]
cands["A. previous Matlab solver (repo)"]=pd.DataFrame(dict(age=mc.age,C=mc.C,W=mc.W,Y=mc.Y,a=mc.alpha_mean))
for nm,fn in [("B. python EGM, timing A","sim_timing_A.csv"),("C. python EGM, timing B","sim_timing_B.csv")]:
    d=pd.read_csv(os.path.join(OUT,fn)); cands[nm]=pd.DataFrame(dict(age=d.age,C=d.C,W=d.W,Y=d.Y,a=d.a))
p=os.path.join(ROOT,"GPT_repo_sync","cgm_replication","experiments","results","two_by_two_annual.csv")
if os.path.exists(p):
    d=pd.read_csv(p)
    hc=d[(d.policy_model=="historical_iid")|(d.policy_model=="historical")] if "historical" in set(d.policy_model) else d[d.policy_model==d.policy_model.unique()[0]]
    for pm in d.policy_model.unique():
        for sm in d.simulation_model.unique():
            sub=d[(d.policy_model==pm)&(d.simulation_model==sm)]
            cands["D. %s policy / %s sim"%(pm,sm)]=pd.DataFrame(dict(age=sub.age,C=sub.consumption,W=sub.wealth,Y=sub.income,a=sub.alpha))
rows=[]
for nm,d in cands.items():
    m=d.groupby("age").mean(numeric_only=True)
    ipk=int(np.argmax(d.W.values))
    r={"replication":nm,"W_peak":round(float(d.W.values[ipk]),1),"peak_age":int(d.age.values[ipk])}
    for col,key in [("wealth","W"),("consumption","C"),("income","Y"),("alpha_mean","a")]:
        if key not in m.columns or col not in T.columns: continue
        v,n=rmse(m.reindex(T.index)[key].values,T[col].values); r[key+"_rmse"]=round(v,4) if v==v else None
        r[key+"_at65"]=round(float(m.loc[65,key]),3) if 65 in m.index else None
    rows.append(r)
out=pd.DataFrame(rows); pd.set_option("display.width",260)
print(out.to_string(index=False)); out.to_csv(os.path.join(OUT,"replication_scorecard.csv"),index=False)
print()
print("PAPER (digitised Figure 3): W65=%.1f C65=%.1f Y65=%.1f a65=%.4f"%(T.loc[65,"wealth"],T.loc[65,"consumption"],T.loc[65,"income"],T.loc[65,"alpha_mean"]))
# final validated solution
sol=solve(na=401,amax=80.0,nr=7,ny=7,verbose=False)
sim=simulate(sol,N=50000,seed=2026)
df=pd.DataFrame(dict(age=sol["ages"],consumption=sim["meanC"],wealth=sim["meanW"],income=sim["meanY"],alpha_mean=sim["meanA"]))
df.to_csv(os.path.join(OUT,"python_cgm_baseline.csv"),index=False)
np.savez(os.path.join(OUT,"python_cgm_policy.npz"),
   **{"x_%d"%i:sol["x"][i] for i in range(len(sol["ages"]))},
   **{"c_%d"%i:sol["c"][i] for i in range(len(sol["ages"]))},
   **{"a_%d"%i:sol["alpha"][i] for i in range(len(sol["ages"]))})
print("saved python_cgm_baseline.csv : peak %.1f @ %d"%(df.wealth.max(),df.age[int(np.argmax(df.wealth.values))]))
