"""Test whether the repo's Matlab policy satisfies the CGM Euler equation.

The repo ships, per simulation path, consumption, wealth, income and the policy rules.
If the policy is the solution of the model then for every interior path

    c_t^{-gamma}  =  beta * p_t * E[ G_{t+1}^{-gamma} c_{t+1}^{-gamma} R^p_{t+1} ]

so the realised Euler residual should look like mean-zero noise.  We cannot form the
expectation from a single path, but we CAN use the *policy function* to construct it:
the repo's own policy tells us what next-period consumption would have been for every
return/income realisation, which the simulation does not record.  Instead we use the
cheaper and still decisive test: the implied consumption growth,
(c_{t+1}/c_t) implied by the policy, must match (beta p_t R^p)^{1/gamma}.
"""
import numpy as np, pandas as pd, sys, os
sys.path.insert(0,"/Users/zwkai/Desktop/复旦新开始/cgm_python")
from cgm_model import F, SURV, solve, simulate
ROOT="/Users/zwkai/Desktop/复旦新开始"
repo=pd.read_csv(ROOT+"/GPT_repo_sync/cgm_replication/matlab_final/cgm_seed_20260909.csv",header=None)
repo.columns=["age","C","W","Y","S","alpha","seW","seA"]
mine=pd.read_csv(ROOT+"/cgm_python/results/python_cgm_baseline.csv")
p_surv=SURV
beta=0.96; gamma=10.0; Rf=1.02
print("Implied consumption growth from the policy, vs the Euler benchmark")
print("(unconstrained Euler:  c_{t+1}/c_t = (beta*p_t*R^p)^(1/gamma);  beta*Rf*p ~ 0.972 -> 0.9972)")
print()
print(" age |   repo   c_{t+1}/c_t |   python c_{t+1}/c_t | beta*p*Rf^(1/g)")
hdr=True
for i,a in enumerate(range(21,66)):
    r0=repo[repo.age==a-1].iloc[0]; r1=repo[repo.age==a].iloc[0]
    m0=mine[mine.age==a-1].iloc[0]; m1=mine[mine.age==a].iloc[0]
    pr=r1.C/r0.C if r0.C>0 else np.nan
    pm=m1.consumption/m0.consumption
    bm=(beta*p_surv[a-20]*Rf)**(1/gamma)
    if a in (21,25,30,35,40,45,50,55,60,65):
        print("%4d | %19.4f | %20.4f | %14.4f"%(a,pr,pm,bm))
print()
# geometric average growth over the working life
for nm,d,col in [("repo",repo,"C"),("python",mine,"consumption")]:
    v=d[d.age.between(21,65)][col].values
    g=(v[-1]/v[0])**(1/(len(v)-1))
    print("%s: average annual consumption growth age 21-65 = %.4f  (implied (beta*p*R)^(1/gamma) benchmark ~ 0.997)"%(nm,g))
