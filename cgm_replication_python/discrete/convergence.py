import numpy as np, sys, os
sys.path.insert(0,"/Users/zwkai/Desktop/复旦新开始/cgm_python")
import cgm_model as M
from cgm_model import F, simulate
OUT="/Users/zwkai/Desktop/复旦新开始/cgm_python/results"
rows=[]
for (nr,ny,na,nq) in [(3,3,201,400),(5,5,201,400),(7,7,301,400),(9,9,301,400)]:
    sol=M.solve(na=na,amax=80.0,nr=nr,ny=ny,verbose=False)
    sim=simulate(sol,N=20000,seed=7)
    age=M.ages if False else sol["ages"]
    pk=int(np.argmax(sim["meanW"]))
    i65=int(np.where(age==65)[0][0])
    print("nr=%d ny=%d na=%d : wealth peak %8.2f @%3d | alpha65 %.4f | C65 %6.2f | C/Y65 %.3f"%(
        nr,ny,na,sim["meanW"].max(),age[pk],sim["meanA"][i65],sim["meanC"][i65],
        sim["meanC"][i65]/sim["meanY"][i65]))
    rows.append((nr,ny,na,sim["meanW"].max(),age[pk],sim["meanA"][i65],sim["meanC"][i65]))
import pandas as pd
pd.DataFrame(rows,columns=["nr","ny","na","wealth_peak","peak_age","alpha65","C65"]).to_csv(
   os.path.join(OUT,"convergence_grid_quadrature.csv"),index=False)
