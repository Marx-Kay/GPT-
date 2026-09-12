function sol=cgm_solve(p,n,rule)
% One solver, common transition. Endogenous portfolio uses the envelope FOC.
if nargin<3,rule="optimal";end
start=tic;T=numel(p.ages);H=cgm_human_capital(p);
a=[0;logspace(log10(n.asset_min),log10(n.asset_max),n.assets-1)'];
if ~p.income_enabled,a=a(2:end);end
x=cell(T,1);c=x;al=x;x{T}=[0;1e8];c{T}=x{T};al{T}=[0;0];
sol=struct('p',p,'n',n,'rule',string(rule),'H',H,'x',{x},'c',{c},'alpha',{al});
for t=T-1:-1:1
 age=p.ages(t);q=cgm_shocks(n.quadrature,age<p.last_work_age&&p.income_enabled);
 [R,~]=cgm_returns(q.zr,0,p); assert(all(R>0));
 if string(rule)=="no_income_risk"
  sol=cgm_rule_dp_step(sol,t,a,q);continue
 end
 lo=zeros(size(a));hi=ones(size(a));
 if string(rule)~="optimal",lo=cgm_welfare_rule(rule,age,a,H(t),p);hi=lo;end
 for j=1:n.bisections*(string(rule)=="optimal")
  alpha=(lo+hi)/2;
  [xp,G]=cgm_transition(age,a,alpha,q.zp,q.ze,q.zr,p);
  cn=cgm_policy(sol,t+1,xp);m=(G.*cn).^(-p.gamma);
  d=(m.*(R-p.rf))*q.weight;
  lo(d>0)=alpha(d>0);hi(d<=0)=alpha(d<=0);
 end
 alpha=(lo+hi)/2;
 [xp,G,~,rp]=cgm_transition(age,a,alpha,q.zp,q.ze,q.zr,p);
 cn=cgm_policy(sol,t+1,xp);
 ct=(p.beta*p.survival(t)*((G.*cn).^(-p.gamma).*rp)*q.weight).^(-1/p.gamma);
 xt=a+ct;assert(all(diff(xt)>0),'Nonmonotone endogenous grid');
 sol.x{t}=[0;xt];sol.c{t}=[0;ct];sol.alpha{t}=[alpha(1);alpha];
end
sol.solve_seconds=toc(start);
end
