import torch, json
from lifecycle_hjb import Parameters,coefficients,hjb_residual
from ito import ito_generator
from pathlib import Path
torch.set_default_dtype(torch.float64)
s=torch.tensor([[30.,2.,1.5,.2],[70.,2.,1.5,.2]])
c=torch.tensor([.8,.8]);a=torch.tensor([.4,.4]);p=Parameters()
f,g=coefficients(s,c,a,p)
assert g[1,2:,:].abs().max()==0
assert f[1,2:].abs().max()==0
v=lambda x:x[:,0]+x[:,1:].square().sum(-1)
actual=ito_generator(v,s,f,g)
exact=f[:,0]+2*(s[:,1:]*f[:,1:]).sum(-1)+g[:,1:,:].square().sum((-1,-2))
assert torch.allclose(actual,exact,atol=1e-12)
# Wealth zero, consume exactly income: zero wealth drift and diffusion.
boundary=s.clone();boundary[:,1]=0
income=torch.tensor([1.5*torch.exp(torch.tensor(.2)),p.replacement*1.5])
bf,bg=coefficients(boundary,income,a,p)
assert bf[:,1].abs().max()<1e-12 and bg[:,1,:].abs().max()==0
report={'generator_max_abs_error':(actual-exact).abs().max().item(),'retirement_freeze_passed':True,'wealth_boundary_passed':True,'trained_lifecycle_solution':False}
Path(__file__).with_name('lifecycle_checks.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(report)
