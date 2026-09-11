import numpy as np, welfare as W, json
res={}
for d in [0.0106,0.0005]:
    rows=[]
    for seed in [777,101,2024]:
        o={}
        pdv=W.pdv_ratio(d); opt=W.R.solve(nq=7,na=401,s2u=d)
        import ref_cgm as RR
        # patch seed through module-level by re-implementing the call
        def sim(X,C,AL,rule,s2u=d,seed=seed,pdv=pdv):
            return W._sim_utility(X,C,AL,rule,s2u=s2u,nsim=60000,seed=seed,pdv=pdv)
        ECopt=sim(opt["X"],opt["C"],opt["AL"],None)
        rules={"100-Age":lambda a:(100.0-a)/100.0,"No income":lambda a:W.MERTON_NI,
               "Zero":lambda a:0.0,"Approx.":lambda a:min(1.0,max(0.5,2.0-0.025*a))}
        for nm,r in rules.items():
            s=W.solve_rule(r,s2u=d)
            o[nm]=100*(1-sim(s["X"],s["C"],None,r)/ECopt)
        s=W.solve_rule(lambda a:W.MERTON_NI,s2u=d)
        o["No income risk"]=100*(1-sim(s["X"],s["C"],None,"no_income_risk")/ECopt)
        rows.append(o); print(f"  s2u={d} seed={seed} "+" ".join(f"{k}={v:.3f}" for k,v in o.items()),flush=True)
    res[str(d)]=rows
json.dump(res,open("welfare_seed_robustness.json","w"),indent=1)
