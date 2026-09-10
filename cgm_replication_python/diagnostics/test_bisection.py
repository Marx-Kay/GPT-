import numpy as np, sys, os, pandas as pd
sys.path.insert(0,"/Users/zwkai/Desktop/复旦新开始/cgm_python")
from cgm_model import solve, F, simulate, SURV
OUT="/Users/zwkai/Desktop/复旦新开始/cgm_python/results"

base=solve(verbose=False)
ages=base["ages"]; p=base["params"]; psurv=base["p_surv"]
gamma=p["gamma"]; beta=p["beta"]; Rf=p["Rf"]; mu=p["mu"]; se=p["se"]
s2e=p["s2e"]; s2u=p["s2u"]; repl=p["repl"]; aK=p["aK"]; T=len(ages)

def egm(bisect_sign="matlab", timing="A"):
    v=np.sqrt(np.arange(1,9.0)); J=np.diag(v,1)+np.diag(v,-1)
    D,Q=np.linalg.eigh(J); ix=np.argsort(D); z=Q[0,:][ix]; q=z**2
    zr,zp,zt=np.meshgrid(z,z,z,indexing='ij'); qr,qp,qt=np.meshgrid(q,q,q,indexing='ij')
    w=(qr*qp*qt).ravel(); Rv=(Rf+mu+se*zr.ravel())[None,:]
    a=np.concatenate([[0.0],np.logspace(-5,np.log10(200.0),800)])
    xx=[None]*T; cc=[None]*T; aa=[None]*T
    xx[T-1]=np.array([0.0,1e6]); cc[T-1]=xx[T-1].copy(); aa[T-1]=np.array([0.0,0.0])
    nonmono=0
    for t in range(T-2,-1,-1):
        age=ages[t]
        if age>=aK:
            G=np.ones((1,Rv.shape[1])); Y=repl*np.ones((1,Rv.shape[1]))
        else:
            G=(F(age+1)/F(age)*np.exp(np.sqrt(s2u)*zp.ravel()))[None,:]
            Y=np.exp(np.sqrt(s2e)*zt.ravel())[None,:]
        lo=np.zeros(len(a)); hi=np.ones(len(a))
        for k in range(28):
            al_=(lo+hi)/2.0
            rp=Rf+al_[:,None]*(Rv-Rf)
            xp=(a[:,None]*rp/G+Y) if timing=="A" else ((a[:,None]*rp+Y)/G)
            cn=np.interp(xp,xx[t+1],cc[t+1]); m=(G*cn)**(-gamma)
            foc=(m*(Rv-Rf))@w
            if bisect_sign=="matlab": up=foc>0
            else:                     up=foc<0
            lo[up]=al_[up]; hi[~up]=al_[~up]
        al_=(lo+hi)/2.0
        rp=Rf+al_[:,None]*(Rv-Rf)
        xp=(a[:,None]*rp/G+Y) if timing=="A" else ((a[:,None]*rp+Y)/G)
        cn=np.interp(xp,xx[t+1],cc[t+1])
        expd=((G*cn)**(-gamma)*rp)@w
        ct=(beta*psurv[t]*expd)**(-1.0/gamma)
        xt=a+ct
        if np.any(np.diff(xt)<=0): nonmono+=1
        xx[t]=np.concatenate([[0.0],xt]); cc[t]=np.concatenate([[0.0],ct])
        aa[t]=np.concatenate([[al_[0]],al_])
    return xx,cc,aa,nonmono

for sign in ["matlab","corrected"]:
    xx,cc,aa,nm=egm(bisect_sign=sign)
    sol=dict(ages=ages,x=xx,c=cc,alpha=aa,p_surv=psurv,params=p)
    sim=simulate(sol,N=20000,seed=7)
    print("=== bisection sign: %s (non-monotone ages: %d) ==="%(sign,nm))
    print("   c_n at x_n=1.0 : age25 %.4f  age40 %.4f  age65 %.4f"%(
        np.interp(1.0,xx[5],cc[5]),np.interp(1.0,xx[20],cc[20]),np.interp(1.0,xx[45],cc[45])))
    print("   c_n at x_n=6.0 : age25 %.4f  age40 %.4f  age65 %.4f"%(
        np.interp(6.0,xx[5],cc[5]),np.interp(6.0,xx[20],cc[20]),np.interp(6.0,xx[45],cc[45])))
    print("   sim wealth peak %.2f at age %d ; alpha@65 %.4f ; C@65 %.2f"%(
        sim["meanW"].max(),ages[int(np.argmax(sim["meanW"]))],sim["meanA"][ages==65][0],sim["meanC"][ages==65][0]))
    print("   age25 c=%.3f W=%.3f | age45 c=%.3f W=%.3f | age65 c=%.3f W=%.3f"%(
        sim["meanC"][5],sim["meanW"][5],sim["meanC"][25],sim["meanW"][25],sim["meanC"][45],sim["meanW"][45]))
