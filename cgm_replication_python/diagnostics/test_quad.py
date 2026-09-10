import numpy as np, sys, os, pandas as pd
sys.path.insert(0,"/Users/zwkai/Desktop/复旦新开始/cgm_python")
from cgm_model import solve, simulate, F
OUT="/Users/zwkai/Desktop/复旦新开始/cgm_python/results"
base=solve(verbose=False); ages=base["ages"]; p=base["params"]; psurv=base["p_surv"]
gamma=p["gamma"]; beta=p["beta"]; Rf=p["Rf"]; mu=p["mu"]; se=p["se"]
s2u=p["s2u"]; repl=p["repl"]; aK=p["aK"]; T=len(ages); qq=1.0-gamma

def egm_quad(nq,nalpha=201,na=500,amax=150.0,alpha_grid=True):
    """EGM with an nq-node Gauss-Hermite rule and coupled (return, permanent) shocks, as in
       the Matlab template.  alpha chosen by maximising the continuation value."""
    if nq==9: v=np.sqrt(np.arange(1,9.0)); J=np.diag(v,1)+np.diag(v,-1)
    else:
        v=np.sqrt(np.arange(1,float(nq))); J=np.diag(v,1)+np.diag(v,-1)
    D,Q=np.linalg.eigh(J); ix=np.argsort(D); z=Q[0,:][ix]; w=z**2
    Rv=Rf+mu+se*z                                  # (nq,)
    a=np.concatenate([[0.0],np.logspace(-8,np.log10(amax),na-1)])
    Xg=[None]*T; Cp=[None]*T; Ap=[None]*T
    Xg[T-1]=a.copy(); Cp[T-1]=a.copy(); Ap[T-1]=np.zeros_like(a)
    lV=[None]*T; lV[T-1]=np.where(a>0,(1.0-gamma)*np.log(np.maximum(a,1e-300)),np.inf)
    ag=np.linspace(0,1,nalpha)
    for t in range(T-2,-1,-1):
        age=ages[t]
        if age>=aK: G=np.ones(nq); Yq=repl*np.ones(nq)
        else:
            G=F(age+1)/F(age)*np.exp(np.sqrt(s2u)*z)
            Yq=np.exp(np.sqrt(2*s2u)*z)          # template: Y shares the SAME z as R and G
        xN=Xg[t+1]; lVN=lV[t+1]
        bV=np.full(len(a),np.inf); bC=np.zeros(len(a)); bA=np.zeros(len(a))
        for al in ag:
            rp=Rf+al*(Rv-Rf)
            lo=np.full(len(a),-40.0); hi=np.log(np.maximum(a,1e-300))
            for it in range(60):
                mid=0.5*(lo+hi); c=np.exp(mid); s=np.maximum(a-c,0.0)
                xp=s[:,None]*rp[None,:]/G[None,:]+Yq[None,:]
                LV=np.interp(xp,xN,lVN)
                lhs=c**(-gamma)
                rhs=beta*psurv[t]*((np.exp(LV)**(-gamma))*rp[None,:]*w[None,:]).sum(axis=1)
                more=lhs>rhs
                hi=np.where(more,mid,hi); lo=np.where(more,lo,mid)
            c=np.exp(0.5*(lo+hi)); s=np.maximum(a-c,0.0)
            xp=s[:,None]*rp[None,:]/G[None,:]+Yq[None,:]
            LV=np.interp(xp,xN,lVN)
            LM=qq*LV; M=LM.max(axis=1)
            EVq=np.exp(M)*(np.exp(LM-M[:,None])*w[None,:]).sum(axis=1)
            tot=np.where(s>0,np.exp(qq*np.log(np.maximum(s,1e-300)))+beta*psurv[t]*EVq,np.inf)
            imp=tot<bV
            bV=np.where(imp,tot,bV); bC=np.where(imp,c,bC); bA=np.where(imp,al,bA)
        lV[t]=np.log(bV); Xg[t]=a+bC; Cp[t]=bC.copy(); Ap[t]=bA.copy()
        Xg[t][0]=0.0; Cp[t][0]=0.0; Ap[t][0]=0.0
    return dict(ages=ages,x=Xg,c=Cp,alpha=Ap,p_surv=psurv,params=p)

for nq in [3,5,7,9]:
    sol=egm_quad(nq,nalpha=81,na=300)
    sim=simulate(sol,N=20000,seed=7)
    mw,ma,mc=sim["meanW"],sim["meanA"],sim["meanC"]
    print("quadrature nq=%d (%d coup. nodes): wealth peak %8.2f at age %2d | alpha@65 %.4f | C@65 %6.2f"%(
        nq,nq,mw.max(),ages[int(np.argmax(mw))],ma[ages==65][0],mc[ages==65][0]))
