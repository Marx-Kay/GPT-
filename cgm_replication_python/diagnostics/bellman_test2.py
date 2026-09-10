import numpy as np, os, sys, pandas as pd
sys.path.insert(0,"/Users/zwkai/Desktop/复旦新开始/cgm_python")
from cgm_model import solve, F
from numpy.polynomial.hermite_e import hermegauss
OUT="/Users/zwkai/Desktop/复旦新开始/cgm_python/results"

sol=solve(verbose=False); x,c,al,ages,p=sol["x"],sol["c"],sol["alpha"],sol["ages"],sol["params"]
psurv=sol["p_surv"]
gamma=p["gamma"]; beta=p["beta"]; Rf=p["Rf"]; mu=p["mu"]; se=p["se"]
s2e=p["s2e"]; s2u=p["s2u"]; repl=p["repl"]; aK=p["aK"]; T=len(ages)

# rebuild MATLAB policy from its own EGM, replicated exactly in python
def matlab_egm():
    v=np.sqrt(np.arange(1,9.0))
    J=np.diag(v,1)+np.diag(v,-1)
    D,Q=np.linalg.eigh(J); ix=np.argsort(D)
    z=Q[0,:][ix]; q=z**2
    zr,zp,zt=np.meshgrid(z,z,z,indexing='ij'); qr,qp,qt=np.meshgrid(q,q,q,indexing='ij')
    w=(qr*qp*qt).ravel(); Rv=(Rf+mu+se*zr.ravel())[None,:]
    a=np.concatenate([[0.0],np.logspace(-5,np.log10(200.0),800)])
    xx=[None]*T; cc=[None]*T; aa=[None]*T
    xx[T-1]=np.array([0.0,1e6]); cc[T-1]=xx[T-1].copy(); aa[T-1]=np.array([0.0,0.0])
    for t in range(T-2,-1,-1):
        age=ages[t]
        if age>=aK:
            G=np.ones((1,Rv.shape[1])); Y=repl*np.ones((1,Rv.shape[1]))
        else:
            G=(F(age+1)/F(age)*np.exp(np.sqrt(s2u)*zp.ravel()))[None,:]
            Y=np.exp(np.sqrt(s2e)*zt.ravel())[None,:]
        xpn=xx[t+1]; cpn=cc[t+1]
        lo=np.zeros(len(a)); hi=np.ones(len(a))
        for k in range(28):
            al_=(lo+hi)/2.0
            rp=Rf+al_[:,None]*(Rv-Rf)
            xp=a[:,None]*rp/G+Y
            cn=np.interp(xp,xpn,cpn); m=(G*cn)**(-gamma)
            foc=(m*(Rv-Rf))@w
            up=foc>0; dn=~up
            lo[up]=al_[up]; hi[dn]=al_[dn]
        al_=(lo+hi)/2.0
        rp=Rf+al_[:,None]*(Rv-Rf)
        xp=a[:,None]*rp/G+Y
        cn=np.interp(xp,xpn,cpn)
        expd=((G*cn)**(-gamma)*rp)@w
        ct=(beta*psurv[t]*expd)**(-1.0/gamma)
        xt=a+ct
        xx[t]=np.concatenate([[0.0],xt]); cc[t]=np.concatenate([[0.0],ct])
        aa[t]=np.concatenate([[al_[0]],al_])
    return xx,cc,aa
xM,cM,aM=matlab_egm()
np.savez("/tmp/cgmwork/matlab_egm_py.npz",ages=ages,
         **{"x_%d"%i:xM[i] for i in range(T)},**{"c_%d"%i:cM[i] for i in range(T)})

def value_of_policy(g,poly_x,poly_c,poly_a,order="A"):
    T=len(ages); V=[None]*T; V[T-1]=g**(1-gamma)/(1-gamma)
    zr,wr=hermegauss(9); wr=wr/np.sqrt(2*np.pi)
    zy,wy=hermegauss(9); wy=wy/np.sqrt(2*np.pi)
    R=Rf+mu+se*zr; Wq=wy[:,None]*wr[None,:]
    for t in range(T-2,-1,-1):
        age=ages[t]
        if age>=aK: G=np.ones((len(zy),len(zr))); Y=repl*np.ones((len(zy),len(zr)))
        else:
            G=F(age+1)/F(age)*np.exp(np.sqrt(s2u)*zy)[:,None]*np.ones((1,len(zr)))
            Y=np.exp(np.sqrt(s2e)*zy)[:,None]*np.ones((1,len(zr)))
        cn=np.interp(g,poly_x[t],poly_c[t]); an=np.interp(g,poly_x[t],poly_a[t])
        s=np.maximum(g-cn,0.0)
        rp=Rf+an[:,None,None]*(R-Rf)[None,None,:]
        xp=(s[:,None,None]*rp/G[None,:,:]+Y[None,:,:]) if order=="A" else ((s[:,None,None]*rp+Y[None,:,:])/G[None,:,:])
        Vn=np.interp(xp,g,V[t+1])
        V[t]=cn**(1-gamma)/(1-gamma)+beta*psurv[t]*(Vn*Wq[None,:,:]).sum(axis=(1,2))
    return V

g=np.linspace(1e-4,80.0,6001)
def test(name,px,pc,pa,age_list=(25,40,55,65,80),x_list=(2.0,6.0,12.0,18.0)):
    print("\n===== %s ====="%name)
    V=value_of_policy(g,px,pc,pa)
    zr,wr=hermegauss(9); wr=wr/np.sqrt(2*np.pi); zy,wy=hermegauss(9); wy=wy/np.sqrt(2*np.pi)
    Rr=Rf+mu+se*zr; Wq=wy[:,None]*wr[None,:]
    print(" age    x_n   c_pol    a_pol | c_best   a_best |  V_pol      V_best     gain")
    for a_ in age_list:
        t=int(np.where(ages==a_)[0][0])
        if a_>=aK: G=np.ones((len(zy),len(zr))); Yq=repl*np.ones((len(zy),len(zr)))
        else:
            G=F(a_+1)/F(a_)*np.exp(np.sqrt(s2u)*zy)[:,None]*np.ones((1,len(zr)))
            Yq=np.exp(np.sqrt(s2e)*zy)[:,None]*np.ones((1,len(zr)))
        for xn in x_list:
            cpol=np.interp(xn,px[t],pc[t]); apol=np.interp(xn,px[t],pa[t])
            s=xn-cpol; rp=Rf+apol*(Rr-Rf); xp=s*rp/G+Yq
            vpol=cpol**(1-gamma)/(1-gamma)+beta*psurv[t]*(np.interp(xp,g,V[t+1])*Wq).sum()
            best=(-1e30,None,None)
            for aa in np.linspace(0,1,201):
                rp2=Rf+aa*(Rr-Rf)
                # solve c by scalar search on a fine grid then Newton-free refine
                cs=np.linspace(1e-8,xn,3000)
                xp2=(xn-cs)[:,None,None]*rp2[None,None,:]/G[None,:,:]+Yq[None,:,:]
                vv=np.interp(xp2,g,V[t+1])
                cont=(vv*Wq[None,:,:]).sum(axis=(1,2))
                vals=cs**(1-gamma)/(1-gamma)+beta*psurv[t]*cont
                j=int(np.argmax(vals))
                if vals[j]>best[0]: best=(vals[j],cs[j],aa)
            print("%4d %6.2f  %7.4f  %6.3f | %7.4f  %6.3f | %10.5f %10.5f %8.2e"%(
                a_,xn,cpol,apol,best[1],best[2],vpol,best[0],best[0]-vpol))
test("PYTHON policy",x,c,al)
test("MATLAB policy (py replica)",xM,cM,aM)