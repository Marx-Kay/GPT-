import numpy as np, welfare as W, ref_cgm as R
print("sigma_u2(DP)   W65    a65   | 100-Age  NoIncome  NoIncRisk   Zero   Approx")
for d in [0.0106,0.005,0.0035,0.002,0.001,0.0005,0.0]:
    o=W.table6(s2u=d,nsim=30000)
    s=R.solve(nq=7,na=301,s2u=d); m=R.simulate(s,nsim=20000,s2u=0.0106)
    k=lambda a:int(np.where(m["age"]==a)[0][0])
    print(f"  {d:7.4f}   {m['mW'][k(65)]:6.1f}  {m['mA'][k(65)]:.3f}  | "
          f"{o['100-Age']:7.3f} {o['No income']:9.3f} {o['No income risk']:10.3f} {o['Zero']:6.3f} {o['Approx.']:8.3f}",flush=True)
print("  paper      221.5  0.499  |   0.637     1.531      0.152   2.108    0.084")
