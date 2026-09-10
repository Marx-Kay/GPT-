"""Independent analytical checks including mixed partials and policy gradients."""
from pathlib import Path
import json, time
import torch
from ito import ito_generator

torch.set_default_dtype(torch.float64)
torch.set_num_threads(2)
torch.manual_seed(20260909)
checks=[]
for dim,m in [(1,1),(3,2),(10,1),(10,5),(100,1),(100,10),(4,0)]:
    s=torch.randn(16,dim); f=torch.randn_like(s);g=torch.randn(16,dim,m)
    A=torch.randn(dim,dim);A=(A+A.T)/2
    v=lambda x: torch.einsum('bi,ij,bj->b',x,A,x)
    actual=ito_generator(v,s,f,g)
    exact=2*torch.einsum('bi,ij,bj->b',s,A,f)+torch.einsum('bim,ij,bjm->b',g,A,g)
    error=(actual-exact).abs().max().item()
    assert torch.allclose(actual,exact,atol=1e-9,rtol=1e-10)
    checks.append({'states':dim,'shocks':m,'max_abs_error':error})
# State-dependent diffusion must be frozen with respect to epsilon, but retain
# its computational dependence on controls for the policy-improvement gradient.
s=torch.tensor([[1.2,0.7],[2.0,1.3]])
a=torch.tensor(0.4,requires_grad=True)
f=a*s;g=(a*s.square()).unsqueeze(-1)
v=lambda x: x.square().sum(-1)
y=ito_generator(v,s,f,g).sum()
grad=torch.autograd.grad(y,a)[0]
exact=(2*s.square().sum()+2*a*s.pow(4).sum()).item()
assert abs(grad.item()-exact)<1e-10
checks.append({'policy_gradient_error':abs(grad.item()-exact)})
out=Path(__file__).resolve().parent/'ito_checks.json'
out.write_text(json.dumps(checks,indent=2),encoding='utf-8')
print(out.read_text())
