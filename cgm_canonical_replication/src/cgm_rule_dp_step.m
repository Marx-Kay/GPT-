function sol=cgm_rule_dp_step(sol,t,a,q)
% State-dependent Eq12 rule need not preserve concavity at the cap kink.
% Globally scan savings and refine the best grid bracket; shared Bellman kernel.
p=sol.p;k=1-p.gamma;T=numel(p.ages);
if t==T-1,sol.rulevalue.ce=cell(T,1);sol.rulevalue.ce{T}=sol.x{T};sol.rulevalue.boundary=zeros(T,1);end
m=p.premium/(p.gamma*p.stock_sigma^2);kink=m*sol.H(t)/(1-m);
a=unique([a;kink]);a=a(a<=sol.n.asset_max);alpha=cgm_welfare_rule(sol.rule,p.ages(t),a,sol.H(t),p);
[xp,G,Y]=cgm_transition(p.ages(t),a,alpha,q.zp,q.ze,q.zr,p);
K=p.beta*p.survival(t)*((G.*cgm_ce_at(sol,sol.rulevalue,t+1,xp)).^k)*q.weight;
h=K.^(1/k);hc=@(s)interp1(a,h,s,'pchip','extrap');
xx=[logspace(-4,log10(sol.n.asset_max+30),sol.n.assets)'];
obj=(max(xx-a',realmin).^k+K').^(1/k);obj(xx<=a')=-Inf;
[~,idx]=max(obj,[],2);lo=a(max(1,idx-1));hi=min(xx*(1-1e-12),a(min(numel(a),idx+1)));
phi=(sqrt(5)-1)/2;
for iter=1:45
 l=hi-phi*(hi-lo);u=lo+phi*(hi-lo);
 vl=((xx-l).^k+hc(l).^k).^(1/k);vu=((xx-u).^k+hc(u).^k).^(1/k);
 left=vl>vu;hi(left)=u(left);lo(~left)=l(~left);
end
s=(lo+hi)/2;c=xx-s;
vs=(c.^k+hc(s).^k).^(1/k);v0=(xx.^k+K(1)).^(1/k);
b=v0>=vs;s(b)=0;c(b)=xx(b);
% Explicit kink candidate protects the piecewise allocation rule.
if kink<max(a)
 sk=min(kink,xx*(1-1e-12));vk=((xx-sk).^k+hc(sk).^k).^(1/k);
 take=vk>max(vs,v0);s(take)=sk(take);c(take)=xx(take)-s(take);
end
alpha=cgm_welfare_rule(sol.rule,p.ages(t),s,sol.H(t),p);
sol.x{t}=[0;xx];sol.c{t}=[0;c];sol.alpha{t}=[alpha(1);alpha];
[xp,G]=cgm_transition(p.ages(t),s,alpha,q.zp,q.ze,q.zr,p);
vv=(c.^k+p.beta*p.survival(t)*((G.*cgm_ce_at(sol,sol.rulevalue,t+1,xp)).^k)*q.weight).^(1/k);
sol.rulevalue.ce{t}=[0;vv];sol.rulevalue.boundary(t)=K(1);
end
