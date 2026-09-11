import numpy as np, ref_cgm as R
PAPER=dict(W65=221.476, CY25=0.969, CY65=1.099, a65=0.4992, a20=0.963, C65=35.553)
def row(tag,**kw):
    kw.setdefault("nq",7); kw.setdefault("na",301)
    s=R.solve(**kw); sim=R.simulate(s,nsim=20000)
    j=lambda a: np.where(sim["age"]==a)[0][0]
    print(f"{tag:40s} W65={sim['mW'][j(65)]:7.1f}  peak={sim['mW'].max():7.1f}  "
          f"CY25={sim['mC'][j(25)]/sim['mY'][j(25)]:.3f}  CY65={sim['mC'][j(65)]/sim['mY'][j(65)]:.3f}  "
          f"a65={sim['mA'][j(65)]:.3f}  a20={sim['mA'][j(20)]:.3f}  C65={sim['mC'][j(65)]:6.1f}",flush=True)
print("PAPER TARGET                            W65=  221.5  peak=  221.5  CY25=0.969  CY65=1.099  a65=0.499  a20=0.963  C65=  35.6")
print("-"*135)
for tag,kw in [
  ("baseline sigma_u2=.0106",{}),
  ("sigma_u2=0.0",dict(s2u=0.0)),
  ("sigma_u2=0.0005",dict(s2u=0.0005)),
  ("sigma_u2=0.001",dict(s2u=0.001)),
  ("sigma_u2=0.002",dict(s2u=0.002)),
  ("sigma_u2=0.005",dict(s2u=0.005)),
  ("drop euler G-term",dict(euler_g=False)),
  ("drop G-term + sigma_u2=0",dict(euler_g=False,s2u=0.0)),
  ("drop euler G, beta=.98",dict(euler_g=False,beta=0.98)),
  ("drop euler G, beta=1.00",dict(euler_g=False,beta=1.0)),
  ("sigma_u2=0, beta=.98",dict(s2u=0.0,beta=0.98)),
  ("sigma_u2=0, gamma=5",dict(s2u=0.0,gamma=5.0)),
  ("sigma_u2=0, beta=.98,gamma=10",dict(s2u=0.0,beta=0.98)),
]: row(tag,**kw)
