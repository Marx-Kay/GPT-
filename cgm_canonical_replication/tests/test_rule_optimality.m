function out=test_rule_optimality(solutions,evals,root)
% Independent direct Bellman improvement: scan and fminbnd, not EGM.
rows=[];
for j=2:numel(solutions)
 sol=solutions{j};p=sol.p;k=1-p.gamma;e=evals{j};worst=0;
 for age=[20 40 60 65 80 99]
  t=find(p.ages==age);q=cgm_shocks(sol.n.validation_quadrature,age<p.last_work_age);
  states=[.3 1 3 10 30];
  if sol.rule=="no_income_risk"
   m=p.premium/(p.gamma*p.stock_sigma^2);ss=m*sol.H(t)/(1-m);
   states=[states,ss+[.1 1 3]];
  end
  for x=states
   [c,~,s]=cgm_policy(sol,t,x);
   obj=@(ss)objective(ss,x,sol,e,t,q,k);
   grid=linspace(0,x*(1-1e-10),161);val=obj(grid');[~,ix]=max(val);
   lo=grid(max(1,ix-1));hi=grid(min(numel(grid),ix+1));
   sb=fminbnd(@(ss)-obj(ss),lo,hi,optimset('TolX',1e-9));
   best=max([obj(sb),obj(0),obj(s)]);gain=best/obj(s)-1;
   worst=max(worst,gain);rows=[rows;j,age,x,c,gain];
  end
 end
 out.max_improvement(j)=worst;
end
out.passed=max(out.max_improvement)<3e-4;
writetable(array2table(rows,'VariableNames',{'rule_index','age','cash','consumption','relative_CE_improvement'}),fullfile(root,'results/validation/rule_optimality.csv'));
assert(out.passed,'CGM:ruleOptimality','Fixed-rule consumption fails independent Bellman improvement');
end
function v=objective(s,x,sol,e,t,q,k)
p=sol.p;a=cgm_welfare_rule(sol.rule,p.ages(t),s,sol.H(t),p);
[xp,G]=cgm_transition(p.ages(t),s,a,q.zp,q.ze,q.zr,p);
v=((x-s).^k+p.beta*p.survival(t)*((G.*cgm_ce_at(sol,e,t+1,xp)).^k)*q.weight).^(1/k);
end
