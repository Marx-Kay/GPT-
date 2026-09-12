function e=cgm_evaluate_policy(sol,nq)
start=tic;p=sol.p;T=numel(p.ages);k=1-p.gamma;
e.ce=cell(T,1);e.ce{T}=sol.x{T};e.boundary=zeros(T,1);
for t=T-1:-1:1
 q=cgm_shocks(nq,p.ages(t)<p.last_work_age&&p.income_enabled);
 xx=sol.x{t}(2:end);[c,a,s]=cgm_policy(sol,t,xx);
 [xp,G,Y]=cgm_transition(p.ages(t),s,a,q.zp,q.ze,q.zr,p);
 vn=cgm_ce_at(sol,e,t+1,xp);
 e.boundary(t)=p.beta*p.survival(t)*((G.*cgm_ce_at(sol,e,t+1,Y)).^k)*q.weight;
 e.ce{t}=[0;(c.^k+p.beta*p.survival(t)*((G.*vn).^k)*q.weight).^(1/k)];
end
q=cgm_shocks(nq,false);x0=exp(sqrt(p.transitory_variance)*q.z);
% Initial wealth is zero in baseline; for nonzero diagnostic integrate both initial shocks.
[zp,ze]=ndgrid(q.z,q.z);[wp,we]=ndgrid(q.w,q.w);
[P,y,W]=cgm_initial(zp,ze,p);vv=cgm_ce_at(sol,e,1,W./P+y);
e.initial_moment=sum(wp.*we.*(P.*vv).^k,'all');
e.initial_CE=e.initial_moment^(1/k);e.seconds=toc(start);e.quadrature=nq;
end
