import numpy as np, scipy.io as sio, sys, os, pandas as pd
sys.path.insert(0,"/Users/zwkai/Desktop/复旦新开始/cgm_python")
from cgm_model import solve, F
OUT="/Users/zwkai/Desktop/复旦新开始/cgm_python/results"
M=sio.loadmat("/tmp/cgmwork/matlab_policy.mat")
ages=M["ages"].ravel()
xM=[np.asarray(M["x"][i,0]).ravel() for i in range(len(ages))]
cM=[np.asarray(M["c"][i,0]).ravel() for i in range(len(ages))]
aM=[np.asarray(M["alpha"][i,0]).ravel() for i in range(len(ages))]
print("MATLAB policy loaded: ages %d, grid pts %d, quadrature %s, rf %s premium %s sigma %s"%(
  len(ages), len(xM[0]), M["quadrature"].ravel(), M["rf"].ravel(), M["premium"].ravel(), M["sigma"].ravel()))
base=solve(na=401,amax=80.0,nr=7,ny=7,verbose=False)
xP=base["x"]; cP=base["c"]; aP=base["alpha"]
print()
print("  age |    x_n |  MATLAB c_n  |  python c_n  | MATLAB a | python a")
for a in [20,30,40,50,65]:
    i=int(np.where(ages==a)[0][0])
    for xn in [0.5,1.0,2.0,4.0,8.0,13.0,18.0]:
        cm=np.interp(xn,xM[i],cM[i]); cp=np.interp(xn,xP[i],cP[i])
        am=np.interp(xn,xM[i],aM[i]); ap=np.interp(xn,xP[i],aP[i])
        print("%5d | %6.2f | %11.4f | %12.4f | %8.3f | %8.3f"%(a,xn,cm,cp,am,ap))
print()
print("MATLAB c_n range at age 40: [%.4f, %.4f]; x_n range [%.2f, %.2f]"%(
  cM[20].min(),cM[20].max(),xM[20].min(),xM[20].max()))
print("MATLAB alpha values at age 40: min %.3f max %.3f ; unique count %d"%(
  aM[20].min(),aM[20].max(),len(np.unique(np.round(aM[20],3)))))
