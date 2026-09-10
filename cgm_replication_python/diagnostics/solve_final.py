import numpy as np, sys, os, time, pandas as pd
sys.path.insert(0,"/Users/zwkai/Desktop/复旦新开始/cgm_python")
from cgm_model import solve as old_solve, F, simulate
OUT="/Users/zwkai/Desktop/复旦新开始/cgm_python/results"
b0=old_solve(verbose=False); ages=b0["ages"]; p=b0["params"]; psurv=b0["p_surv"]
gamma=p["gamma"]; beta=p["beta"]; Rf=p["Rf"]; mu=p["mu"]; se=p["se"]
s2e=p["s2e"]; s2u=p["s2u"]; repl=p["repl"]; aK=p["aK"]; T=len(ages); qq=1.0-gamma

nyq=5; nrq=5
def herm(n):
    x,w=np.polynomial.hermite_e.hermegauss(n); return x,w/np.sqrt(2*np.pi)
zy,wy=herm(nyq); zr,wr=herm(nrq)
Gf=np.repeat(np.exp(np.sqrt(s2u)*zy),nrq); Yf=np.repeat(np.exp(np.sqrt(s2e)*zy),nrq)
Rf_=np.tile(Rf+mu+se*zr,nyq); Wq=np.repeat(wy,nrq)*np.tile(wr,nyq); NQ=nyq*nrq

def solve_final(nalpha=161,na=400,amax=150.0,alpha_lo=0.0):
    """For each alpha: consumption from the Euler equation (bisection on log c).
       Select alpha by comparing ln(-V) (correct monotone transform of a negative value fn)."""
    a=np.concatenate([[0.0],np.logspace(-8,np.log10(amax),na-1)])
    lV=[None]*T; Xg=[None]*T; Cp=[None]*T; Ap=[None]*T
    lV[T-1]=np.where(a>0,(1.0-gamma)*np.log(np.maximum(a,1e-300))-np.log(gamma-1.0),np.inf)
    Xg[T-1]=a.copy(); Cp[T-1]=a.copy(); Ap[T-1]=np.zeros_like(a)
    ag=np.linspace(alpha_lo,1.0,nalpha)
    for t in range(T-2,-1,-1):
        age=ages[t]
        if age>=aK: G=np.ones(NQ); Yq=repl*np.ones(NQ)
        else: G=F(age+1)/F(age)*Gf; Yq=Yf
        xN=Xg[t+1]; lVN=lV[t+1]
        bV=np.full(len(a),np.inf); bC=np.zeros(len(a)); bA=np.zeros(len(a))
        for al in ag:
            rp=Rf+al*(Rf_-Rf)
            lo=np.full(len(a),-60.0); hi=np.log(np.maximum(a,1e-300))-1e-12
            for it in range(70):
                mid=0.5*(lo+hi); c=np.exp(mid); s=a-c
                xp=s[:,None]*rp[None,:]/G[None,:]+Yq[None,:]
                LV=np.interp(xp,xN,lVN,left=np.inf,right=lVN[-1])
                lhs=c**(-gamma)
                rhs=beta*psurv[t]*((np.exp(-gamma*LV))*rp[None,:]*Wq[None,:]).sum(axis=1)
                more=lhs>rhs
                hi=np.where(more,mid,hi); lo=np.where(more,lo,mid)
            c=np.exp(0.5*(lo+hi)); s=a-c
            xp=s[:,None]*rp[None,:]/G[None,:]+Yq[None,:]
            LV=np.interp(xp,xN,lVN,left=np.inf,right=lVN[-1])
            LM=qq*LV                       # = ln|V'|  (positive, decreasing in V')
            M=LM.max(axis=1)
            EVq=np.exp(M)*(np.exp(LM-M[:,None])*Wq[None,:]).sum(axis=1)   # E[|V'|^qq] = -E[V']
            tot=np.where(s>1e-14,np.exp(qq*np.log(np.maximum(s,1e-300)))+beta*psurv[t]*EVq,np.inf)
            imp=tot<bV
            bV=np.where(imp,tot,bV); bC=np.where(imp,c,bC); bA=np.where(imp,al,bA)
        lV[t]=np.log(bV); Xg[t]=a+bC; Cp[t]=bC.copy(); Ap[t]=bA.copy()
        Xg[t][0]=0.0; Cp[t][0]=0.0; Ap[t][0]=0.0
    return dict(ages=ages,x=Xg,c=Cp,alpha=Ap,p_surv=psurv,params=p)

t0=time.time(); sol=solve_final(nalpha=81,na=300); print("solve_final %.0f s"%(time.time()-t0))
np.savez("/tmp/cgmwork/sol_final.npz",**{"x_%d"%i:sol["x"][i] for i in range(T)},
   **{"c_%d"%i:sol["c"][i] for i in range(T)},**{"a_%d"%i:sol["alpha"][i] for i in range(T)})
print()
print(" c_n at x_n=0.5 / 1.0 / 2.0 / 6.0")
for a_ in [20,25,30,40,50,65,80]:
    i=int(np.where(ages==a_)[0][0])
    print("  age %3d : %8.4f %8.4f %8.4f %8.4f"%(a_,np.interp(0.5,sol["x"][i],sol["c"][i]),
       np.interp(1.0,sol["x"][i],sol["c"][i]),np.interp(2.0,sol["x"][i],sol["c"][i]),
       np.interp(6.0,sol["x"][i],sol["c"][i])))
sim=simulate(sol,N=20000,seed=7)
print()
print("FINAL: wealth peak %.2f at age %d ; alpha@65 %.4f ; C@65 %.2f ; C/Y@65 %.3f"%(
  sim["meanW"].max(),ages[int(np.argmax(sim["meanW"]))],sim["meanA"][ages==65][0],
  sim["meanC"][ages==65][0],sim["meanC"][ages==65][0]/sim["meanY"][ages==65][0]))
pd.DataFrame(dict(age=ages,C=sim["meanC"],W=sim["meanW"],Y=sim["meanY"],A=sim["meanA"])).to_csv(
  os.path.join(OUT,"sim_final.csv"),index=False)
