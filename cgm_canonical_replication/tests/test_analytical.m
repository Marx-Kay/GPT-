function out=test_analytical(p,n)
n.assets=301;n.quadrature=5;
% Two-period deterministic pension benchmark, derived directly from FOC.
p2=p;p2.ages=[99;100];p2.survival=p.survival(end);p2.stock_sigma=0;p2.premium=0;
s=cgm_solve(p2,n);xx=[.5;1;2;5];cc=cgm_policy(s,1,xx);
exact=min(xx,(p2.rf*xx+p2.replacement)/(p2.rf+(p2.beta*p2.survival*p2.rf)^(1/p2.gamma)));
out.two_period=max(abs(cc-exact));assert(out.two_period<1e-10);assert(cc(2)>.5);
% No-income finite-horizon discrete-time analytic recursion. Independent fzero.
p2=p;p2.income_enabled=false;
s=cgm_solve(p2,n);q=cgm_shocks(n.quadrature,false);R=p.rf+p.premium+p.stock_sigma*q.z;
f=@(a)sum(q.w.*(p.rf+a*(R-p.rf)).^(-p.gamma).*(R-p.rf));
a=fzero(f,[0,1]);M=sum(q.w.*(p.rf+a*(R-p.rf)).^(1-p.gamma));
kappa=ones(numel(p.ages),1);
for t=numel(p.ages)-1:-1:1,kappa(t)=1/(1+(p.beta*p.survival(t)*M)^(1/p.gamma)/kappa(t+1));end
out.no_income_consumption=0;out.no_income_share=0;
for t=1:numel(p.ages)-1
 out.no_income_consumption=max(out.no_income_consumption,max(abs(s.c{t}(2:end)./s.x{t}(2:end)-kappa(t))));
 out.no_income_share=max(out.no_income_share,max(abs(s.alpha{t}(2:end)-a)));
end
assert(out.no_income_consumption<1e-8&&out.no_income_share<1e-7);
% Deterministic finite horizon with income: nonbinding budget branch and
% independently discounted annuity weights; choose wealth high enough to save.
p2=p;p2.ages=(90:100)';p2.survival=p.survival(71:80);p2.premium=0;p2.stock_sigma=0;
s=cgm_solve(p2,n);T=numel(p2.ages);H=0;factor=1;denom=1;
for j=1:T-1
 H=H+p2.replacement/p2.rf^j;
 factor=factor*(p2.beta*p2.survival(j)*p2.rf)^(1/p2.gamma);
 denom=denom+factor/p2.rf^j;
end
xx=100;exact=(xx+H)/denom;cc=cgm_policy(s,1,xx);
out.deterministic=abs(cc-exact);assert(out.deterministic<1e-8);
% Independent exact Bellman borrowing branch for gamma2 and two dates.
pv=p;pv.ages=[20;21];pv.last_work_age=0;pv.survival=1;
pv.gamma=2;pv.beta=1;pv.rf=1;pv.premium=0;pv.stock_sigma=0;pv.replacement=1;
pv.income_coefficients=zeros(1,4);pv.income_log_level=0;pv.transitory_variance=.09;
sv=cgm_solve(pv,n);ev=cgm_evaluate_policy(sv,3);
z=[-sqrt(3);0;sqrt(3)];weight=[1/6;2/3;1/6];xx=exp(.3*z);
ce=xx./(1+xx);ce(xx>1)=(xx(xx>1)+1)/4;
out.value_boundary=abs(ev.initial_moment-sum(weight./ce));assert(out.value_boundary<1e-8);
out.passed=true;
end
