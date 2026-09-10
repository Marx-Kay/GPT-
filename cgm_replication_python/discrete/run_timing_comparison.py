import numpy as np, os, sys, pandas as pd
sys.path.insert(0,"/Users/zwkai/Desktop/复旦新开始/cgm_python")
from cgm_model import solve, simulate
OUT="/Users/zwkai/Desktop/复旦新开始/cgm_python/results"
rows=[]
for timing in ["A","B"]:
    sol=solve(timing=timing,verbose=False)
    sim=simulate(sol,N=20000,seed=7)
    df=pd.DataFrame(dict(age=sim["ages"],C=sim["meanC"],W=sim["meanW"],Y=sim["meanY"],a=sim["meanA"]))
    pk=df.W.idxmax()
    print("timing %s: wealth peak %.2f at age %d | alpha@65 %.4f | C@65 %.2f | C/Y@65 %.3f"%(
        timing, df.W.max(), df.age[pk], df.a[df.age==65].values[0], df.C[df.age==65].values[0],
        df.C[df.age==65].values[0]/df.Y[df.age==65].values[0]))
    print(df.iloc[[0,10,20,30,40,45,46,55,70,80]].to_string(index=False))
    df.to_csv(os.path.join(OUT,"sim_timing_%s.csv"%timing),index=False)
