import numpy as np, sys, os, scipy.io as sio, pandas as pd
sys.path.insert(0,"/Users/zwkai/Desktop/复旦新开始/cgm_python")
from cgm_model import solve, simulate, F
OUT="/Users/zwkai/Desktop/复旦新开始/cgm_python/results"
M=sio.loadmat("/tmp/cgmwork/matlab_policy.mat"); ages=M["ages"].ravel()
xM=[np.asarray(M["x"][i,0]).ravel() for i in range(len(ages))]
cM=[np.asarray(M["c"][i,0]).ravel() for i in range(len(ages))]
aM=[np.asarray(M["alpha"][i,0]).ravel() for i in range(len(ages))]
rows=[]
for nalpha in [61,201,801]:
    for (nr,ny) in [(3,3),(7,7)]:
        sol=solve(na=401,amax=80.0,nr=nr,ny=ny,nalpha=nalpha,verbose=False)
        sim=simulate(sol,N=20000,seed=7)
        i65=int(np.where(sol["ages"]==65)[0][0])
        err=[]
        for a in [20,30,40,50,60,65]:
            i=int(np.where(ages==a)[0][0])
            xs=np.linspace(0.7,15,40)
            cm=np.interp(xs,xM[i],cM[i]); cp=np.interp(xs,sol["x"][i],sol["c"][i])
            am=np.interp(xs,xM[i],aM[i]); ap=np.interp(xs,sol["x"][i],sol["alpha"][i])
            err.append((np.abs(cm-cp).max(),np.abs(am-ap).max()))
        err=np.array(err)
        rows.append(dict(nalpha=nalpha,nr=nr,ny=ny,
            wealth_peak=float(sim["meanW"].max()),
            C65=float(sim["meanC"][i65]), alpha65=float(sim["meanA"][i65]),
            max_c_policy_err=float(err[:,0].max()), max_alpha_policy_err=float(err[:,1].max())))
t=pd.DataFrame(rows); pd.set_option("display.width",200)
print(t.to_string(index=False))
print()
print("MATLAB reference: wealth peak 433.4 (repo sim) ; policy at x_n=4/8/13 at age 40:")
i=int(np.where(ages==40)[0][0])
print("  c:",[round(float(np.interp(v,xM[i],cM[i])),4) for v in [4,8,13]])
print("  a:",[round(float(np.interp(v,xM[i],aM[i])),4) for v in [4,8,13]])
