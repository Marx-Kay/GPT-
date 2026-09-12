function out=test_extended(sol,sim,root)
p=sol.p;n=sol.n;
% Conditional normalized transition moments from independent Gaussian draws.
rng(n.seed+17);N=200000;z=randn(3,N);
s=2;a=.4;[xp,G,y,rp]=cgm_transition(40,s,a,z(1,:),z(2,:),z(3,:),p);
g=cgm_income(41,p)/cgm_income(40,p);
expected=s*(p.rf+a*p.premium)/g*exp(p.permanent_variance/2)+exp(p.transitory_variance/2);
out.transition_mean_z=abs(mean(xp)-expected)/(std(xp)/sqrt(N));assert(out.transition_mean_z<4);
% Independent quadrature and simulation must agree on all income moments.
q=cgm_shocks(n.validation_quadrature,true);xx=cgm_transition(40,s,a,q.zp,q.ze,q.zr,p);
out.transition_quadrature=abs(xx*q.weight-expected);assert(out.transition_quadrature<1e-9);
% Normalization and zero-permanent-risk representation checks at solver level.
nn=n;nn.assets=101;nn.quadrature=5;
p0=p;p0.permanent_variance=0;s0=cgm_solve(p0,nn);
p1=p0;p1.income_log_level=p1.income_log_level+log(3);s1=cgm_solve(p1,nn);
out.scale_invariance=0;
for t=1:80
 out.scale_invariance=max(out.scale_invariance,max(abs(s0.c{t}-s1.c{t})));
end
assert(out.scale_invariance<1e-10);
% Independent Euler checks at actual observed wealth quantiles after simulation.
er=[];ke=[];
for age=[20 30 50 65 66 85 99]
 t=age-19;q=cgm_shocks(n.validation_quadrature,age<p.last_work_age);
 x=quantile(sim.x(t,:),[.001 .01 .05 .25 .5 .75 .95 .99 .999])';[c,a,s]=cgm_policy(sol,t,x);
 [xp,G,~,rp]=cgm_transition(age,s,a,q.zp,q.ze,q.zr,p);cn=cgm_policy(sol,t+1,xp);m=(G.*cn).^(-p.gamma);
 ratio=(p.beta*p.survival(t)*(m.*rp)*q.weight).^(-1/p.gamma)./c;
 e=abs(ratio-1);b=s<1e-9;e(b)=max(0,1-ratio(b));er=[er;e];
 R=cgm_returns(q.zr,0,p);f=((m.*(R-p.rf))*q.weight)./(m*q.weight);kk=abs(f);kk(a<1e-6)=max(0,f(a<1e-6));kk(a>1-1e-6)=max(0,-f(a>1-1e-6));kk(b)=0;ke=[ke;kk];
end
out.quantile_euler=[median(er),quantile(er,.95),max(er)];out.quantile_kkt=max(ke);
assert(max(er)<n.euler_tolerance&&max(ke)<n.kkt_tolerance);
out.passed=true;write_json(fullfile(root,'results/validation/extended.json'),out);
end
