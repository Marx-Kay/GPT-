function out=cgm_sensitivity(sol,root)
% DIAGNOSTIC ONLY. Each experiment uses the SAME canonical kernel.
% These parameters are never written back to the frozen baseline.
p=sol.p;n=sol.n;names=["baseline","initial_permanent_draw","initial_wealth_one_income","historical_full_precision","log_income_scale_plus_10pct"];
rows=[];
for i=1:numel(names)
 s=sol;
 switch i
  case 2,s.p.initial_permanent_variance=p.permanent_variance;
  case 3,s.p.initial_wealth=cgm_income(p.ages(1),p);
  case 4
   pp=p;pp.income_coefficients=p.diagnostic_full_precision_coefficients;s=cgm_solve(pp,n);
  case 5,s.p.income_log_level=p.income_log_level+log(p.diagnostic_income_scale);
 end
 sm=cgm_simulate(s,n.simulation_N,n.seed);[peak,idx]=max(sm.table.wealth);
 rows=[rows;peak,sm.table.age(idx),sm.table.income(46),sm.table.alpha(46)];
end
out=table(names',rows(:,1),rows(:,2),rows(:,3),rows(:,4),'VariableNames',{'diagnostic','wealth_peak','peak_age','income65','alpha65'});
writetable(out,fullfile(root,'results/validation/assumption_sensitivity.csv'));
end
