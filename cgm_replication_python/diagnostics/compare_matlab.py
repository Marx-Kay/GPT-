import numpy as np, os, sys, pandas as pd
sys.path.insert(0,"/Users/zwkai/Desktop/复旦新开始/cgm_python")
from cgm_model import solve, F, SURV
OUT="/Users/zwkai/Desktop/复旦新开始/cgm_python/results"
sol=solve(verbose=False); np.savez(os.path.join(OUT,"sol_baseline.npz"),
    **{k:v for k,v in sol.items() if k!="params"})
x,c,al,ages=sol["x"],sol["c"],sol["alpha"],sol["ages"]
mc=pd.read_csv("/Users/zwkai/Desktop/复旦新开始/GPT_repo_sync/cgm_replication/matlab_final/cgm_seed_20260909.csv",header=None)
mc.columns=["age","C","W","Y","S","alpha","Wse","Ase"]
print("age | x_n | MATLAB c_n  python c_n | MATLAB alpha  python alpha")
rows=[]
for a in [20,25,30,35,40,45,50,55,60,65,70,80,90]:
    i=int(np.where(ages==a)[0][0])
    sub=mc[mc.age==a]
    # permanent income scale implied by the model: P_t = F_t before 65, F_65 from 65 on
    Pn = F(min(a,65))
    xn=(sub.W.values+sub.Y.values)/Pn
    cn=sub.C.values/Pn; an=sub.alpha.values
    med=np.median(xn); idx=np.argsort(abs(xn-med))[:5]
    pyc=np.interp(xn[idx],x[i],c[i]); pya=np.interp(xn[idx],x[i],al[i])
    print("%3d | %.3f | %.4f  %.4f | %.3f  %.3f"%(a,med,cn[idx].mean(),pyc.mean(),an[idx].mean(),pya.mean()))
    rows.append((a,med,cn[idx].mean(),pyc.mean(),an[idx].mean(),pya.mean()))
pd.DataFrame(rows,columns=["age","x_n","matlab_cn","python_cn","matlab_a","python_a"]).to_csv(os.path.join(OUT,"policy_compare_matlab.csv"),index=False)
print()
print("MATLAB sim mean consumption ratio C/X by age:")
for a in [25,40,55,65,80,95]:
    sub=mc[mc.age==a]; print("  %3d  C/X=%.3f  C/Y=%.3f  W/Y=%.2f"%(a,(sub.C/(sub.W+sub.Y)).mean(),(sub.C/sub.Y).mean(),(sub.W/sub.Y).mean()))
