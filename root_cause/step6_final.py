import numpy as np, json, ref_cgm as R
T=json.load(open("/tmp/fig2_targets.json"))
XS=np.array([10,15,20,25,30,40,50,75,100,150,200,250],float)
def c_rmse(sol,age):
    if str(age) not in T["C"]: return np.nan
    pa=np.array(T["C"][str(age)]); i=int(np.where(R.AGES==age)[0][0]); P=R.F(age)
    mine=np.interp(XS/P,sol["X"][i],sol["C"][i])*P
    return float(np.sqrt(np.mean((mine-np.interp(XS,pa[:,0],pa[:,1]))**2)))
def run(tag,sim_s2u=0.0106,init_draw=False,**kw):
    kw.setdefault("nq",7); kw.setdefault("na",401)
    s=R.solve(**kw); m=R.simulate(s,nsim=40000,s2u=sim_s2u,init_draw=init_draw)
    k=lambda a:int(np.where(m["age"]==a)[0][0])
    print(f"{tag:44s} W65={m['mW'][k(65)]:7.1f} W25={m['mW'][k(25)]:6.1f} W40={m['mW'][k(40)]:6.1f} "
          f"C/Y25={m['mC'][k(25)]/m['mY'][k(25)]:.3f} C65={m['mC'][k(65)]:6.1f} a65={m['mA'][k(65)]:.3f} "
          f"C/Y65={m['mC'][k(65)]/m['mY'][k(65)]:.3f} | Fig2C {c_rmse(s,20):4.2f}/{c_rmse(s,35):4.2f}/{c_rmse(s,65):4.2f}")
print("PAPER                                         W65=  221.5 W25=  10.3 W40=  56.5 C/Y25=0.969 C65=  35.6 a65=0.499 C/Y65=1.099")
print("-"*150)
for t,kw,idw in [("DP 0 / sim .0106",dict(s2u=0.0),False),
                 ("DP 0 / sim .0106 / random initial perm. income",dict(s2u=0.0),True),
                 ("DP .0005 / sim .0106",dict(s2u=0.0005),False),
                 ("DP .001  / sim .0106",dict(s2u=0.001),False),
                 ("DP .002  / sim .0106",dict(s2u=0.002),False),
                 ("DP 'nu_forget' .0106 / sim .0106",dict(nu_forget=True),False),
                 ("DP 'nu_forget' .0106 / sim .0106 / init",dict(nu_forget=True),True)]:
    run(t, init_draw=idw, **kw)
print()
print("grid robustness of the best-fitting DP-0 specification (sim always .0106):")
for nq,na in [(5,301),(7,401),(9,801),(9,1601)]:
    s=R.solve(nq=nq,na=na,s2u=0.0); m=R.simulate(s,nsim=40000)
    k=lambda a:int(np.where(m["age"]==a)[0][0])
    print(f"   nq={nq} na={na:5d}  W65={m['mW'][k(65)]:7.1f}  W25={m['mW'][k(25)]:5.1f}  C/Y25={m['mC'][k(25)]/m['mY'][k(25)]:.3f}  Fig2C RMSE={c_rmse(s,20):.3f}")
