function [out,fine,high]=test_convergence(sol,root)
p=sol.p;n=sol.n;n1=n;n1.assets=n.fine_assets;n2=n;n2.quadrature=n.high_quadrature;
fprintf('Fine grid solve\n');fine=cgm_solve(p,n1);fprintf('Higher quadrature solve\n');high=cgm_solve(p,n2);
out.policy=zeros(2,2);
for j=1:2
 if j==1,s=fine;else,s=high;end
 for t=1:numel(p.ages)-1
  xx=logspace(-1,log10(80),201)';[c,a,ss]=cgm_policy(sol,t,xx);[cc,aa]=cgm_policy(s,t,xx);
  out.policy(j,1)=max(out.policy(j,1),max(abs(cc-c)./max(c,.01)));
  valid=ss>1e-5;out.policy(j,2)=max(out.policy(j,2),max(abs(aa(valid)-a(valid))));
 end
end
out.passed=all(out.policy(:,1)<n.consumption_convergence)&all(out.policy(:,2)<n.share_convergence);
write_json(fullfile(root,'results/validation/convergence_policies.json'),out);
assert(out.passed,'CGM:convergence','Policy convergence threshold failed');
end
