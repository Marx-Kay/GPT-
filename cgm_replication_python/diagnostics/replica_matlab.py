import numpy as np, sys, os, pandas as pd
sys.path.insert(0,"/Users/zwkai/Desktop/复旦新开始/cgm_python")
from cgm_model import F, simulate, solve
OUT="/Users/zwkai/Desktop/复旦新开始/cgm_python/results"
b0=solve(verbose=False); ages=b0["ages"]; p=b0["params"]; psurv=b0["p_surv"]
gamma=p["gamma"]; beta=p["beta"]; Rf=p["Rf"]; mu=p["mu"]; se=p["se"]
s2e=p["s2e"]; s2u=p["s2u"]; repl=p["repl"]; aK=p["aK"]; T=len(ages)

def herm3():
    v=np.sqrt(np.arange(1,9.0)); J=np.diag(v,1)+np.diag(v,-1)
    D,Q=np.linalg.eigh(J); ix=np.argsort(D); z=Q[0,:][ix]; return z,z**2

def matlab_egm(flag="orig", nq=9, na=801, amax=200.0, nb=28):
    """Faithful replica of cgm_solve.m. flag='orig' uses the code's own comparison."""
    z,q=herm3()
    zr,zp,zt=np.meshgrid(z,z,z,indexing='ij')
    qr,qp,qt=np.meshgrid(q,q,q,indexing='ij')
    w=(qr*qp*qt).ravel(); Rv=(Rf+mu+se*zr.ravel())[None,:]
    a=np.concatenate([[0.0],np.logspace(-5,np.log10(amax),na-1)])
    Xg=[None]*T; Cp=[None]*T; Ap=[None]*T
    Xg[T-1]=np.array([0.0,1e6]); Cp[T-1]=Xg[T-1].copy(); Ap[T-1]=np.array([0.0,0.0])
    zpf=zp.ravel(); ztf=zt.ravel()
    for t in range(T-2,-1,-1):
        age=ages[t]
        if age>=aK: G=np.ones(Rv.shape[1]); Yq=repl*np.ones(Rv.shape[1])
        else:
            G=F(age+1)/F(age)*np.exp(np.sqrt(s2u)*zpf); Yq=np.exp(np.sqrt(s2e)*ztf)
        lo=np.zeros(len(a)); hi=np.ones(len(a))
        for k in range(nb):
            al=(lo+hi)/2.0; rp=Rf+al[:,None]*(Rv-Rf)
            xp=a[:,None]*rp/G[None,:]+Yq[None,:]
            cn=np.interp(xp,Xg[t+1],Cp[t+1]); m=(G[None,:]*cn)**(-gamma)
            foc=(m*(Rv-Rf))@w
            if flag=="orig": up=foc>0
            else:            up=foc<0
            lo=np.where(up,al,lo); hi=np.where(up,hi,al)
        al=(lo+hi)/2.0; rp=Rf+al[:,None]*(Rv-Rf)
        xp=a[:,None]*rp/G[None,:]+Yq[None,:]
        cn=np.interp(xp,Xg[t+1],Cp[t+1])
        expd=((G[None,:]*cn)**(-gamma)*rp)@w
        ct=(beta*psurv[t]*expd)**(-1.0/gamma)
        xt=a+ct
        Xg[t]=np.concatenate([[0.0],xt]); Cp[t]=np.concatenate([[0.0],ct])
        Ap[t]=np.concatenate([[al[0]],al])
    return dict(ages=ages,x=Xg,c=Cp,alpha=Ap,p_surv=psurv,params=p)

for flag in ["orig","flipped"]:
    s=matlab_egm(flag=flag)
    sim=simulate(s,N=20000,seed=7)
    age=ages; i65=int(np.where(age==65)[0][0])
    print("replica[%s]: wealth peak %8.2f @ %3d | alpha65 %.4f | C65 %6.2f | W65 %7.2f"%(
        flag,sim["meanW"].max(),age[int(np.argmax(sim["meanW"]))],sim["meanA"][i65],
        sim["meanC"][i65],sim["meanW"][i65]))
    print("   alpha path:",np.round([sim["meanA"][int(np.where(age==x)[0][0])] for x in [20,30,40,50,60,65,70,80,90]],3))
    pd.DataFrame(dict(age=age,C=sim["meanC"],W=sim["meanW"],Y=sim["meanY"],A=sim["meanA"])).to_csv(
        os.path.join(OUT,"sim_replica_%s.csv"%flag),index=False)
