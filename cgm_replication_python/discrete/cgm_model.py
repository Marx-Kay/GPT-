import numpy as np, os, json, time
from numpy.polynomial.hermite_e import hermegauss

ROOT="/Users/zwkai/Desktop/复旦新开始"; OUT=os.path.join(ROOT,"cgm_python","results"); os.makedirs(OUT,exist_ok=True)

SURV=np.array([0.99845,0.99839,0.99833,0.9983,0.99827,0.99826,0.99824,0.9982,0.99813,0.99804,
0.99795,0.99785,0.99776,0.99766,0.99755,0.99743,0.9973,0.99718,0.99707,0.99696,
0.99685,0.99672,0.99656,0.99635,0.9961,0.99579,0.99543,0.99504,0.99463,0.9942,
0.9937,0.99311,0.99245,0.99172,0.99091,0.99005,0.98911,0.98803,0.9868,0.98545,
0.98409,0.9827,0.98123,0.97961,0.97786,0.97603,0.97414,0.97207,0.9697,0.96699,
0.96393,0.96055,0.9569,0.9531,0.94921,0.94508,0.94057,0.9357,0.93031,0.92424,
0.91717,0.90922,0.90089,0.89282,0.88503,0.87622,0.86576,0.8544,0.8423,0.82942,
0.8154,0.80002,0.78404,0.76842,0.75382,0.73996,0.72464,0.71057,0.6961,0.6809])

IC=np.array([0.5304,0.16818,-0.00323371,0.000019704])
def F(age):
    a=np.asarray(age,float); return np.exp(IC[0]+IC[1]*a+IC[2]*a**2+IC[3]*a**3)
def nodes(n):
    x,w=hermegauss(n); return x,w/np.sqrt(2*np.pi)

def solve(beta=0.96,gamma=10.0,Rf=1.02,mu=0.04,se=0.157,s2e=0.0738,s2u=0.0106,repl=0.68212,
          a20=20,aK=65,aT=100,na=401,amax=60.0,nr=7,ny=7,alpha_max=1.0,timing="A",
         nalpha=61,verbose=True):
    ages=np.arange(a20,aT+1); T=len(ages)
    p_surv=np.concatenate([SURV[:T-1],[0.0]])
    zr,wr=nodes(nr); zy,wy=nodes(ny)
    R=Rf+mu+se*zr; W=(wy[:,None]*wr[None,:]); gpow=1.0-gamma
    a=np.concatenate([[0.0],np.linspace(1e-4,amax,na-1)])
    x=np.zeros((T,na)); c=np.zeros((T,na)); al=np.zeros((T,na))
    x[T-1]=a.copy(); c[T-1]=a.copy()
    for t in range(T-2,-1,-1):
        age=ages[t]
        if age>=aK:
            G=np.ones((ny,nr)); Y=repl*np.ones((ny,nr))
        else:
            gdet=F(age+1)/F(age)
            G=gdet*np.exp(np.sqrt(s2u)*zy)[:,None]*np.ones((1,nr))
            Y=np.exp(np.sqrt(s2e)*zy)[:,None]*np.ones((1,nr))
        cprev=c[t+1]; xprev=x[t+1]
        def xprime(s,alpha):
            rp=Rf+alpha*(R-Rf)
            s3=s[:,None,None]; rp3=rp[None,None,:]; G3=G[None,:,:]; Y3=Y[None,:,:]
            return (s3*rp3/G3+Y3) if timing=="A" else ((s3*rp3+Y3)/G3)
        alphas=np.linspace(0.0,alpha_max,nalpha)
        best=np.zeros(na); bestval=np.full(na,-np.inf)
        for alpha in alphas:
            xp=xprime(a,alpha); cn=np.interp(xp,xprev,cprev)
            vv=(cn**gpow*W[None,:,:]).sum(axis=(1,2))/(1.0-gamma)
            imp=vv>bestval; bestval[imp]=vv[imp]; best[imp]=alpha
        rp3=(Rf+best[:,None,None]*(R-Rf)[None,None,:])          # (na,ny,nr)
        if timing=="A":
            xp=a[:,None,None]*rp3/G[None,:,:]+Y[None,:,:]
        else:
            xp=(a[:,None,None]*rp3+Y[None,:,:])/G[None,:,:]
        cn=np.interp(xp,xprev,cprev)
        mu_c=(cn**(-gamma)*rp3*W[None,:,:]).sum(axis=(1,2))
        ct=np.where(a>0,(beta*p_surv[t]*mu_c)**(-1.0/gamma),0.0)
        ct=np.minimum(ct,a)
        x[t]=a+ct; c[t]=ct; al[t]=best
        x[t][0]=0.0; c[t][0]=0.0; al[t][0]=0.0
    return dict(ages=ages,x=x,c=c,alpha=al,p_surv=p_surv,
                params=dict(beta=beta,gamma=gamma,Rf=Rf,mu=mu,se=se,s2e=s2e,s2u=s2u,repl=repl,
                            a20=a20,aK=aK,aT=aT,na=na,amax=amax,nr=nr,ny=ny,timing=timing))

def simulate(sol,N=20000,seed=1,s2e=None,s2u=None,repl=None,beta=None):
    ages=sol["ages"];x=sol["x"];c=sol["c"];al=sol["alpha"];p=sol["params"]
    s2e=p["s2e"] if s2e is None else s2e; s2u=p["s2u"] if s2u is None else s2u
    repl=p["repl"] if repl is None else repl
    Rf,mu,se=p["Rf"],p["mu"],p["se"]; aK=p["aK"]
    rng=np.random.default_rng(seed); T=len(ages)
    Y=np.zeros((T,N)); Wv=np.zeros((T,N)); C=np.zeros((T,N)); A=np.zeros((T,N)); Pn=np.zeros((T,N))
    e_eta=rng.standard_normal((T,N)); e_eps=rng.standard_normal((T,N)); e_u=rng.standard_normal((T,N))
    Rt=Rf+mu+se*e_eta
    for t in range(T):
        age=ages[t]
        if t==0: Pn[t]=F(age)
        else:
            if ages[t-1]<aK: Pn[t]=Pn[t-1]*F(age)/F(age-1)*np.exp(np.sqrt(s2u)*e_u[t])
            else: Pn[t]=Pn[t-1]
        if age<aK: Y[t]=Pn[t]*np.exp(np.sqrt(s2e)*e_eps[t])
        elif age==aK: Y[t]=Pn[t]*np.exp(np.sqrt(s2e)*e_eps[t])
        else: Y[t]=repl*Pn[aK-1+ (0 if False else 0)] if False else repl*Pn[t]
        cash=Wv[t]+Y[t]; xn=cash/Pn[t]
        if age==ages[-1]:
            C[t]=cash; A[t]=0.0
        else:
            C[t]=np.minimum(cash,np.interp(xn,x[t],c[t])*Pn[t]); A[t]=np.clip(np.interp(xn,x[t],al[t]),0,1)
        Sv=cash-C[t]
        if t<T-1: Wv[t+1]=Sv*(Rf+A[t]*(Rt[t]-Rf))
    return dict(ages=ages,meanC=C.mean(1),meanW=Wv.mean(1),meanY=Y.mean(1),meanA=A.mean(1))

if __name__=="__main__":
    t0=time.time(); sol=solve(); print("solve secs %.1f"%(time.time()-t0))
    sim=simulate(sol)
    import pandas as pd
    df=pd.DataFrame(dict(age=sim["ages"],consumption=sim["meanC"],wealth=sim["meanW"],
                         income=sim["meanY"],alpha=sim["meanA"]))
    df.to_csv(os.path.join(OUT,"baseline_sim.csv"),index=False)
    print(df.iloc[[0,5,10,20,30,40,44,45,46,50,60,70,80]].to_string(index=False))
    print("wealth peak %.2f at age %d"%(df.wealth.max(),df.age[df.wealth.idxmax()]))
    print("alpha at 65: %.4f"%df.alpha[df.age==65].values[0])
