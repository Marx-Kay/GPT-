function out=test_primitives(p,n)
assert(p.gamma==10&&p.beta==.96&&p.rf==1.02&&p.premium==.04);
assert(p.bequest==0&&p.stock_sigma==.157&&p.replacement==.68212);
assert(p.last_work_age==65&&isequal(p.ages,(20:100)'));
assert(isequal(p.income_coefficients,[-2.1700,.1682,-.0323/10,.0020/100]));
assert(p.income_correlation==0&&p.return_income_correlation==0);
assert(p.permanent_variance==.0106&&p.transitory_variance==.0738);
assert(all(p.survival>0 & p.survival<1)&&numel(p.survival)==80);
q=cgm_shocks(n.quadrature,true);w=q.weight;
out.shock_covariance=abs((q.zp.*q.ze)*w);
assert(out.shock_covariance<1e-12);assert(abs(q.zp.^2*w-1)<1e-12);
[G,y]=cgm_income_process(65,[-2 0 2],[-1 0 1],p);
out.retirement=max(abs(G-1))+max(abs(y-p.replacement));assert(out.retirement==0);
[G,y]=cgm_income_process(64,0,0,p);assert(abs(G-cgm_income(65,p)/cgm_income(64,p))<1e-14&&y==1);
% Permanent impulse persists through wages and retirement.
base=cgm_income(30,p);hit=base*exp(.2);
for age=30:70
 [g,yy]=cgm_income_process(age,0,0,p);base=base*g;hit=hit*g;
 assert(abs(hit/base-exp(.2))<1e-12);
end
% Normalized transition vs independent dollar budget calculation.
P=17;s=[.1;1;4];alpha=[0;.5;1];zp=[-1 0 1];ze=[1 0 -1];zr=[-.5 0 .5];
[xp,G,y,rp]=cgm_transition(40,s,alpha,zp,ze,zr,p);
Xnext=P*s.*rp+P*G.*y;
out.transition=max(abs(Xnext./(P*G)-xp),[],'all');assert(out.transition<1e-12);
p0=p;p0.permanent_variance=0;
[g0,y0]=cgm_income_process(40,zp,ze,p0);[g1,y1]=cgm_income_process(40,zeros(size(zp)),ze,p0);
out.zero_permanent=max(abs(g0-g1))+max(abs(y0-y1));assert(out.zero_permanent==0);
out.passed=true;
end
