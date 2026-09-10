import numpy as np, os, sys, time, pandas as pd
sys.path.insert(0,"/Users/zwkai/Desktop/复旦新开始/cgm_python")
from cgm_model import solve, F, simulate
OUT="/Users/zwkai/Desktop/复旦新开始/cgm_python/results"
base=solve(verbose=False); ages=base["ages"]; p=base["params"]; psurv=base["p_surv"]
gamma=p["gamma"]; beta=p["beta"]; Rf=p["Rf"]; mu=p["mu"]; se=p["se"]
s2e=p["s2e"]; s2u=p["s2u"]; repl=p["repl"]; aK=p["aK"]; T=len(ages); qq=1.0-gamma
def herm(n):
    x,w=np.polynomial.hermite_e.hermegauss(n); return x,w/np.sqrt(2*np.pi)
NY=7; NR=7
zy,wy=herm(NY); zr,wr=herm(NR)
Rf_=(Rf+mu+se*zr)                                   # (NR,)
Gf=np.repeat(np.exp(np.sqrt(s2u)*zy),NR)            # (NY*NR,)
Yf=np.repeat(np.exp(np.sqrt(s2e)*zy),NR)
Rflat=np.tile(Rf_,NY); Wq=np.repeat(wy,NR)*np.tile(wr,NY)
NQ=NY*NR
def linterp(xp,X,L):
    return np.interp(xp,X,L)
def solve_hybrid(nalpha=101,na=400,amax=150.0):
    a=np.concatenate([[0.0],np.logspace(-9,np.log10(amax),na-1)])
    logV=[None]*T; Xg=[None]*T; Cp=[None]*T; Ap=[None]*T
    logV[T-1]=np.where(a>0,(1.0-gamma)*np.log(np.maximum(a,1e-300)),np.inf)
    Xg[T-1]=a.copy(); Cp[T-1]=a.copy(); Ap[T-1]=np.zeros_like(a)
    ag=np.linspace(0,1,nalpha)
    for t in range(T-2,-1,-1):
        age=ages[t]
        if age>=aK: G=np.ones(NQ); Yq=repl*np.ones(NQ)
        else: G=F(age+1)/F(age)*Gf; Yq=Yf
        xN=Xg[t+1]; lVN=logV[t+1]
        bV=np.full(len(a),np.inf); bC=np.zeros(len(a)); bA=np.zeros(len(a))
        for al in ag:
            rp=Rf+al*(Rflat-Rf)                       # (NQ,)
            lo=np.full(len(a),-40.0); hi=np.log(np.maximum(a,1e-300))
            for it in range(50):
                mid=0.5*(lo+hi); c=np.exp(mid); s=np.maximum(a-c,0.0)
                xp=s[:,None]*rp[None,:]/G[None,:]+Yq[None,:]
                lhs=(c**(-gamma))[:,None]*np.ones((1,NQ))
                rhs=beta*psurv[t]*((np.exp(linterp(xp,xN,lVN))**(-gamma))*rp[None,:]*Wq[None,:]).sum(axis=1,keepdims=True)
                more=lhs>rhs
                hi=np.where(more[:,0],mid,hi); lo=np.where(more[:,0],lo,mid)
            c=np.exp(0.5*(lo+hi)); s=np.maximum(a-c,0.0)
            xp=s[:,None]*rp[None,:]/G[None,:]+Yq[None,:]
            LM=qq*linterp(xp,xN,lVN)
            M=LM.max(axis=1)
            EVq=np.exp(M)*(np.exp(LM-M[:,None])*Wq[None,:]).sum(axis=1)
            tot=np.where(s>0,np.exp(qq*np.log(np.maximum(s,1e-300)))+beta*psurv[t]*EVq,np.inf)
            imp=tot<bV
            bV=np.where(imp,tot,bV); bC=np.where(imp,c,bC); bA=np.where(imp,al,bA)
        logV[t]=np.log(bV); Xg[t]=a+bC; Cp[t]=bC.copy(); Ap[t]=bA.copy()
        Xg[t][0]=0.0; Cp[t][0]=0.0; Ap[t][0]=0.0
    return dict(ages=ages,x=Xg,c=Cp,alpha=Ap,p_surv=psurv,params=p)
t0=time.time(); sol=solve_hybrid(nalpha=61,na=300); print("hybrid %.0f s"%(time.time()-t0))
sim=simulate(sol,N=20000,seed=7)
df=pd.DataFrame(dict(age=ages,C=sim["meanC"],W=sim["meanW"],Y=sim["meanY"],A=sim["meanA"]))
print("HYBRID: wealth peak %.2f at age %d ; alpha@65 %.4f ; C@65 %.2f ; C/Y@65 %.3f"%(
   df.W.max(),df.age[df.W.idxmax()],df.A[df.age==65].values[0],df.C[df.age==65].values[0],
   df.C[df.age==65].values[0]/df.Y[df.age==65].values[0]))
print(df.iloc[[0,5,10,20,30,40,45,46,55,70,80]].to_string(index=False))
df.to_csv(os.path.join(OUT,"sim_hybrid.csv"),index=False)
np.savez(os.path.join(OUT,"sol_hybrid.npz"),
   **{"x_%d"%i:sol["x"][i] for i in range(T)},**{"c_%d"%i:sol["c"][i] for i in range(T)},
   **{"a_%d"%i:sol["alpha"][i] for i in range(T)})
