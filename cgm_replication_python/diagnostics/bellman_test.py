import numpy as np, os, sys, pandas as pd
sys.path.insert(0,"/Users/zwkai/Desktop/复旦新开始/cgm_python")
from cgm_model import solve, F
from numpy.polynomial.hermite_e import hermegauss

OUT="/Users/zwkai/Desktop/复旦新开始/cgm_python/results"
sol=solve(verbose=False)
x,c,al,ages,p=sol["x"],sol["c"],sol["alpha"],sol["ages"],sol["params"]
gamma=p["gamma"]; beta=p["beta"]; Rf=p["Rf"]; mu=p["mu"]; se=p["se"]
s2e=p["s2e"]; s2u=p["s2u"]; repl=p["repl"]; aK=p["aK"]
T=len(ages); psurv=sol["p_surv"]
zr,wr=hermegauss(9); wr=wr/np.sqrt(2*np.pi); zy,wy=hermegauss(9); wy=wy/np.sqrt(2*np.pi)
R=Rf+mu+se*zr

def continuation(t,xx):
    """value of continuing from normalized cash xx at age index t (i.e. V_t(x))"""
    return np.interp(xx,x[t],c[t])   # placeholder replaced below

def V_next(tp,xx):
    """normalized value function V at age index tp, evaluated by its own policy's Bellman fixpoint proxy:
       we use the policy's continuation directly (u(c)/(1-gamma) + ...) is circular, so instead we
       evaluate the value implied by the policy. We do it recursively once, which is enough to test
       local optimality of c_t."""
    return None

# Recursive evaluation of the value function implied by a policy (used only for the maturity
# check). To keep it exact we recompute V by backward recursion over the policy grid.
def value_of_policy(xx_grid,x,c,al,ages,p,psurv,order="A"):
    T=len(ages); V=[None]*T
    V[T-1]=xx_grid**(1-gamma)/(1-gamma)
    for t in range(T-2,-1,-1):
        age=ages[t]
        if age>=aK: G=np.ones((len(zy),len(zr))); Y=repl*np.ones((len(zy),len(zr)))
        else:
            G=F(age+1)/F(age)*np.exp(np.sqrt(s2u)*zy)[:,None]*np.ones((1,len(zr)))
            Y=np.exp(np.sqrt(s2e)*zy)[:,None]*np.ones((1,len(zr)))
        Wq=wy[:,None]*wr[None,:]
        cn=np.interp(xx_grid,x[t],c[t]); an=np.interp(xx_grid,x[t],al[t])
        s=np.maximum(xx_grid-cn,0.0)
        rp=Rf+an[:,None,None]*(R-Rf)[None,None,:]
        if order=="A": xp=s[:,None,None]*rp/G[None,:,:]+Y[None,:,:]
        else:          xp=(s[:,None,None]*rp+Y[None,:,:])/G[None,:,:]
        Vn=np.interp(xp,xx_grid,V[t+1])
        cont=(Vn*Wq[None,:,:]).sum(axis=(1,2))
        V[t]=cn**(1-gamma)/(1-gamma)+beta*psurv[t]*cont
    return V

g=np.linspace(1e-4,60.0,4001)
V=value_of_policy(g,x,c,al,ages,p,psurv,order="A")
print("Bellman-residual test of the PYTHON policy (should be ~0 if it is a fixpoint)")
print(" age   x_n   c_policy   c_argmax   V(policy)   V(max)    diff")
for a in [25,40,55,65,80]:
    t=int(np.where(ages==a)[0][0])
    for xn in [2.0,6.0,12.0,18.0]:
        if xn>x[t][-1]: continue
        cpol=np.interp(xn,x[t],c[t]); apol=np.interp(xn,x[t],al[t])
        age=a
        if age>=aK: G=np.ones((len(zy),len(zr))); Y=repl*np.ones((len(zy),len(zr)))
        else:
            G=F(age+1)/F(age)*np.exp(np.sqrt(s2u)*zy)[:,None]*np.ones((1,len(zr)))
            Y=np.exp(np.sqrt(s2e)*zy)[:,None]*np.ones((1,len(zr)))
        Wq=wy[:,None]*wr[None,:]
        cs=np.linspace(1e-6,xn,400)
        best=(-1e18,None,None)
        for cc in cs:
            s=xn-cc
            for aa in np.linspace(0,1,101):
                rp=Rf+aa*(R-Rf)
                xp=(s*rp/G+Y) if True else None
                vv=np.interp(xp,g,V[t+1])
                cont=(vv*Wq).sum()
                val=cc**(1-gamma)/(1-gamma)+beta*psurv[t]*cont
                if val>best[0]: best=(val,cc,aa)
        vpol=cpol**(1-gamma)/(1-gamma)
        s=xn-cpol; rp=Rf+apol*(R-Rf); xp=s*rp/G+Y
        vpol=vpol+beta*psurv[t]*(np.interp(xp,g,V[t+1])*Wq).sum()
        print("%4d %6.2f  %8.4f  %8.4f  %10.5f %10.5f  %9.2e"%(age,xn,cpol,best[1],vpol,best[0],best[0]-vpol))
