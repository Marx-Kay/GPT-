function out=test_optimality(sol,root)
p=sol.p;n=sol.n;rows=[];errors=[];kerrors=[];
for age=[20 25 30 40 50 60 64 65 66 75 85 95 99]
 t=find(p.ages==age);q=cgm_shocks(n.validation_quadrature,age<p.last_work_age);
 xx=unique([logspace(-1,log10(80),151),sol.x{t}(2)*[.99 1 1.01]])';
 [c,alpha,s]=cgm_policy(sol,t,xx);
 [xp,G,~,rp]=cgm_transition(age,s,alpha,q.zp,q.ze,q.zr,p);
 cn=cgm_policy(sol,t+1,xp);m=(G.*cn).^(-p.gamma);
 rhs=p.beta*p.survival(t)*(m.*rp)*q.weight;ratio=rhs.^(-1/p.gamma)./c;
 er=abs(ratio-1);bound=s<1e-9;er(bound)=max(0,1-ratio(bound));
 R=cgm_returns(q.zr,0,p);f=((m.*(R-p.rf))*q.weight)./(m*q.weight);
 ke=abs(f);ke(alpha<1e-6)=max(0,f(alpha<1e-6));ke(alpha>1-1e-6)=max(0,-f(alpha>1-1e-6));ke(bound)=0;
 errors=[errors;er];kerrors=[kerrors;ke];
 rows=[rows;age,median(er),quantile(er,.95),max(er),max(ke)];
 assert(max(abs(xx-c-s))<1e-12&&min(s)>=0&&min(alpha)>=0&&max(alpha)<=1);
end
out.euler_median=median(errors);out.euler_p95=quantile(errors,.95);out.euler_max=max(errors);out.kkt_max=max(kerrors);
out.passed=out.euler_max<n.euler_tolerance&&out.kkt_max<n.kkt_tolerance;
writetable(array2table(rows,'VariableNames',{'age','EulerMedian','EulerP95','EulerMax','KKTMax'}),fullfile(root,'results/validation/optimality.csv'));
assert(out.passed,'CGM:optimality','Independent Euler/KKT threshold failed');
end
