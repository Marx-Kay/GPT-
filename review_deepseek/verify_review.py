"""Read-only audit of supplied code; write evidence only in this directory."""
from pathlib import Path
import importlib.util, json, hashlib, time, csv
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / 'cgm_replication_python/discrete/cgm_model.py'
spec = importlib.util.spec_from_file_location('reviewed_model', source)
model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model)
out = {'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest()}
# Exact two-period retirement problem: deterministic return, positive pension,
# terminal consumption of all cash. The earlier periods do not enter age 99.
sol = model.solve(na=101, nr=3, ny=3, se=0, mu=0, verbose=False)
beta, survival, R, pension, gamma = .96, model.SURV[79], 1.02, .68212, 10
k = (beta * survival * R)**(1/gamma)
out['exact_last_decision_test'] = []
for cash in [.5, 1., 2., 5.]:
    actual = float(np.interp(cash, sol['x'][79], sol['c'][79]))
    exact = min(cash, (R*cash+pension)/(R+k))
    out['exact_last_decision_test'].append(dict(cash=cash, python_consumption=actual,
        exact_consumption=exact, absolute_error=abs(actual-exact)))
t0 = time.perf_counter()
baseline = model.solve(na=401, amax=80., nr=7, ny=7, verbose=False)
sim = model.simulate(baseline, N=50000, seed=2026)
out['current_source_run'] = dict(seconds=time.perf_counter()-t0,
    wealth_peak=float(sim['meanW'].max()), peak_age=int(sim['ages'][sim['meanW'].argmax()]),
    C65=float(sim['meanC'][45]), W65=float(sim['meanW'][45]), alpha65=float(sim['meanA'][45]),
    C20=float(sim['meanC'][0]), Y20=float(sim['meanY'][0]))
with (ROOT/'review_deepseek/current_source_simulation.csv').open('w',encoding='utf-8',newline='') as f:
    writer=csv.writer(f);writer.writerow(['age','consumption','wealth','income','alpha'])
    writer.writerows(zip(sim['ages'],sim['meanC'],sim['meanW'],sim['meanY'],sim['meanA']))
z,w=model.nodes(7)
out['solver_log_income_shock_covariance']=float(np.dot(w,(np.sqrt(.0106)*z)*(np.sqrt(.0738)*z)))
out['paper_log_income_shock_covariance']=0.0
(ROOT/'review_deepseek/evidence.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(out,ensure_ascii=False,indent=2))
