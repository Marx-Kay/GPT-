"""Small DPI integration check with a known analytical solution.
Uses homothetic value parameterization and a neural policy. This is NOT the
CGM lifecycle model and is not evidence of high-dimensional speedup.
"""
from pathlib import Path
import math,json,time
import torch
from torch import nn
from ito import ito_generator

torch.set_default_dtype(torch.float64);torch.set_num_threads(2)
torch.manual_seed(20260909)
gamma=10.;beta=-math.log(.96);r=math.log(1.02);mu=.04;sigma=.157
power=1-gamma
logA=nn.Parameter(torch.tensor(-gamma*math.log(.04)))
policy=nn.Sequential(nn.Linear(1,16),nn.Tanh(),nn.Linear(16,2))
with torch.no_grad():
 policy[-1].weight.zero_();policy[-1].bias[:]=torch.tensor([-3.2,-.5])
critic_opt=torch.optim.Adam([logA],lr=.03)
actor_opt=torch.optim.Adam(policy.parameters(),lr=.006)
def value(x):return -logA.exp()*x[:,0].pow(power)/(gamma-1)
def hjb(x,learn_policy):
 raw=policy(x.log()/3)
 ratio=torch.nn.functional.softplus(raw[:,0])+1e-6
 alpha=raw[:,1].sigmoid()
 if not learn_policy:ratio=ratio.detach();alpha=alpha.detach()
 c=ratio*x[:,0]
 f=((r+alpha*mu)*x[:,0]-c)[:,None]
 g=(alpha*sigma*x[:,0])[:,None,None]
 v=value(x)
 residual=c.pow(power)/power-beta*v+ito_generator(value,x,f,g)
 return residual/(-v),ratio,alpha
start=time.perf_counter();history=[]
for step in range(1800):
 x=(torch.rand(96,1)*math.log(100)).exp()
 logA.requires_grad_(False)
 actor_opt.zero_grad();res,_,_=hjb(x,True);(-res.mean()).backward();actor_opt.step()
 logA.requires_grad_(True)
 critic_opt.zero_grad();res,_,_=hjb(x,False);res.square().mean().backward();critic_opt.step()
 if step%300==0:history.append({'step':step,'residual_rms':res.square().mean().sqrt().item()})
# Uniform-log wealth holdout, not used for training or model selection.
x=torch.linspace(0,math.log(100),301).exp()[:,None]
res,ratio,alpha=hjb(x,False)
a_star=mu/(gamma*sigma**2)
k_star=(beta+(gamma-1)*(r+mu**2/(2*gamma*sigma**2)))/gamma
report={'model':'infinite-horizon no-income Merton; not CGM',
 'iterations':1800,'seconds':time.perf_counter()-start,'seed':20260909,
 'exact_alpha':a_star,'learned_alpha_mean':alpha.mean().item(),
 'alpha_max_abs_error':(alpha-a_star).abs().max().item(),
 'exact_consumption_wealth_ratio':k_star,'learned_ratio_mean':ratio.mean().item(),
 'ratio_max_abs_error':(ratio-k_star).abs().max().item(),
 'value_coefficient_relative_error':abs((logA.exp()*k_star**gamma).item()-1),
 'heldout_scaled_hjb_rms':res.square().mean().sqrt().item(),'history':history}
report['passed']=(report['alpha_max_abs_error']<.015 and report['ratio_max_abs_error']<.002 and report['value_coefficient_relative_error']<.1)
out=Path(__file__).resolve().parent
(out/'merton_dpi_results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
torch.save({'policy':policy.state_dict(),'logA':logA.detach()},out/'merton_dpi.pt')
print(json.dumps(report,indent=2))
if not report['passed']:raise SystemExit('Merton validation failed; do not treat solver as validated.')
