import numpy as np, sys, os, time
sys.path.insert(0,"/Users/zwkai/Desktop/复旦新开始/cgm_python")
from cgm_model import solve, F
base=solve(verbose=False); ages=base["ages"]; p=base["params"]; psurv=base["p_surv"]
gamma=p["gamma"]; beta=p["beta"]; Rf=p["Rf"]; mu=p["mu"]; se=p["se"]
s2e=p["s2e"]; s2u=p["s2u"]; repl=p["repl"]; aK=p["aK"]; T=len(ages); qq=1.0-gamma   # qq<0
def herm(n):
    x,w=np.polynomial.hermite_e.hermegauss(n); return x,w/np.sqrt(2*np.pi)
nyq=5; nrq=5
zy,wy=herm(nyq); zr,wr=herm(nrq)
R=Rf+mu+se*zr
Gf=np.repeat(np.exp(np.sqrt(s2u)*zy),nrq); Yf=np.repeat(np.exp(np.sqrt(s2e)*zy),nrq)
Rf_=np.tile(R,nyq); Wq=np.repeat(wy,nrq)*np.tile(wr,nyq); NQ=nyq*nrq
SURVX=np.concatenate([psurv,[0.0]])
NX=201; XMAX=120.0
xg=np.concatenate([[0.0],np.linspace(1e-4,XMAX,NX-1)])
LOG0=-500.0

def vfi(nc=121,nalpha=81):
    """All values in logs of |V|.  logV is stored as ln(-V) since qq<0 -> V<0."""
    logV=np.full((T,NX),-np.inf)
    with np.errstate(divide='ignore'):
        logV[T-1]=np.where(xg>0,qq*np.log(np.maximum(xg,1e-300)),-np.inf)   # V=-x^9/9 -> ln(-V)=9 ln x -ln9 ; sign convention
    logV[T-1]=np.where(xg>0,(1.0-gamma)*np.log(np.maximum(xg,1e-300))-np.log(gamma-1.0),-np.inf)
    cf=np.linspace(0.0,1.0,nc)
    ag=np.linspace(0.0,1.0,nalpha)
    rp=Rf+ag[:,None]*(Rf_-Rf)
    cbest=np.zeros((T,NX)); abest=np.zeros((T,NX))
    for t in range(T-2,-1,-1):
        age=ages[t]
        if age>=aK: G=np.ones(NQ); Yq=repl*np.ones(NQ)
        else: G=F(age+1)/F(age)*Gf; Yq=Yf
        SAV=xg[:,None,None]*(1.0-cf[None,:,None])            # (nx,nc,1)
        xp=SAV[:,:,:,None]*rp[None,:,:]/G[None,None,None,:]+Yq[None,None,None,:]
        # log V_{t+1}(xp): same sign (negative) => use log|V| and keep qq sign
        LV=np.interp(xp,xg,logV[t+1],left=-np.inf,right=np.nan)
        LV=np.where(xp>XMAX,logV[t+1][-1]+(1.0-gamma)*(np.log(xp)-np.log(XMAX)),LV)
        LV=np.where((xp>0)&(xp<xg[1]),logV[t+1][1]+(1.0-gamma)*(np.log(xp)-np.log(xg[1])),LV)
        LV=np.where(xp<=0,-np.inf,LV)
        # E[V^qq] with V<0 and qq<0  =>  V^qq = exp(qq*logV)  (positive, increasing in V)
        M=np.nanmax(qq*LV,axis=3,keepdims=True)
        M=np.where(np.isfinite(M),M,0.0)
        MS=np.broadcast_to(M,(NX,nc,nalpha,1))
        EVq=np.exp(M[...,0])*(np.exp(qq*LV-MS)*Wq[None,None,None,:]).sum(axis=3)
        EVq=np.where(np.isfinite(M[...,0]),EVq,np.inf)
        # objective: Z = (x-c)^qq + beta*p*EVq   (both positive, want MAX)
        Zc=np.where(SAV[:,:,0]>0,np.exp(qq*np.log(np.maximum(SAV[:,:,0],1e-300))),0.0)
        tot=Zc[:,:,None]+beta*SURVX[t]*EVq
        tot=np.nan_to_num(tot,nan=-np.inf,posinf=np.inf)
        j=np.argmax(tot,axis=2)
        zb=np.take_along_axis(tot,j[:,:,None],axis=2)[:,:,0]
        i=np.argmax(zb,axis=1)
        Zbest=zb[np.arange(NX),i]
        with np.errstate(divide='ignore'):
            logV[t]=np.where(Zbest>0,np.log(Zbest)/qq,-np.inf)
        cbest[t]=xg*cf[i]; abest[t]=ag[j[np.arange(NX),i]]
    return logV,cbest,abest
t0=time.time(); logV,cv,av=vfi(); print("VFI %.1fs"%(time.time()-t0))
print("logV increasing in x? ",all(np.all(np.diff(logV[t][1:])>=-1e-6) for t in range(T)))
for a_ in [25,40,55,65,80]:
    i=int(np.where(ages==a_)[0][0])
    for xn in [1.0,2.0,6.0,12.0]:
        k=int(np.argmin(abs(xg-xn)))
        cp=np.interp(xg[k],base["x"][i],base["c"][i]); ap=np.interp(xg[k],base["x"][i],base["alpha"][i])
        print("age %3d x=%6.2f | VFI c=%7.4f a=%5.3f | EGM c=%7.4f a=%5.3f"%(a_,xg[k],cv[i,k],av[i,k],cp,ap))
def sim_vfi(cv,av,N=20000,seed=7):
    rng=np.random.default_rng(seed)
    Y=np.zeros((T,N)); Wv=np.zeros((T,N)); C=np.zeros((T,N)); A=np.zeros((T,N)); Pn=np.zeros((T,N))
    e_eta=rng.standard_normal((T,N)); e_eps=rng.standard_normal((T,N)); e_u=rng.standard_normal((T,N))
    Rt=Rf+mu+se*e_eta
    for t in range(T):
        age=ages[t]
        if t==0: Pn[t]=F(age)
        elif ages[t-1]<aK: Pn[t]=Pn[t-1]*F(age)/F(age-1)*np.exp(np.sqrt(s2u)*e_u[t])
        else: Pn[t]=Pn[t-1]
        Y[t]=Pn[t]*np.exp(np.sqrt(s2e)*e_eps[t]) if age<=aK else repl*Pn[t]
        cash=Wv[t]+Y[t]; xn=cash/Pn[t]
        if age==ages[-1]: C[t]=cash
        else:
            C[t]=np.minimum(cash,np.interp(xn,xg,cv[t])*Pn[t]); A[t]=np.clip(np.interp(xn,xg,av[t]),0,1)
        if t<T-1: Wv[t+1]=(cash-C[t])*(Rf+A[t]*(Rt[t]-Rf))
    return C.mean(1),Wv.mean(1),Y.mean(1),A.mean(1)
mc,mw,my,ma=sim_vfi(cv,av)
print()
print("VFI-policy: wealth peak %.2f at age %d ; alpha@65 %.4f ; C@65 %.2f ; C/Y@65 %.3f"%(mw.max(),ages[int(np.argmax(mw))],ma[ages==65][0],mc[ages==65][0],mc[ages==65][0]/my[ages==65][0]))
print("age25 c=%.2f W=%.2f | age40 c=%.2f W=%.2f | age50 c=%.2f W=%.2f | age65 c=%.2f W=%.2f"%(mc[5],mw[5],mc[20],mw[20],mc[30],mw[30],mc[45],mw[45]))
np.savez("/tmp/cgmwork/vfi.npz",logV=logV,c=cv,a=av,xg=xg,ages=ages)
