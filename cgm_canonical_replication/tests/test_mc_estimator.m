function out=test_mc_estimator(sol,root)
% Consume all income each date: lifetime expected utility has analytic lognormal
% moments, independent of any solved value function. Checks likelihood algebra.
p=sol.p;T=numel(p.ages);s=sol;
for t=1:T,s.x{t}=[0;1e8];s.c{t}=s.x{t};s.alpha{t}=[0;0];end
s.rule="optimal";m=cgm_welfare_mc({s},10000,99281);
k=1-p.gamma;expected=0;disc=1;
for t=1:T
 age=p.ages(t);v=p.initial_permanent_variance+max(0,min(age,p.last_work_age)-p.ages(1))*p.permanent_variance;
 F=cgm_income(age,p);
 if age<=p.last_work_age,term=F^k*exp(.5*k^2*(v+p.transitory_variance));else,term=(p.replacement*F)^k*exp(.5*k^2*v);end
 expected=expected+disc*term;
 if t<T,disc=disc*p.beta*p.survival(t);end
end
out.expected=expected;out.estimate=m.moments;out.SE=sqrt(m.moment_covariance);out.z=(m.moments-expected)/out.SE;
out.passed=abs(out.z)<4;assert(out.passed,'Importance-sampling analytic moment test failed');
write_json(fullfile(root,'results/validation/mc_analytic.json'),out);
end
