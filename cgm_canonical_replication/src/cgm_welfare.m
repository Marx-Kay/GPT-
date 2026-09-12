function [out,solutions,evals]=cgm_welfare(sol,root)
p=sol.p;n=sol.n;rules=["optimal","100_age","no_income","no_income_risk","zero","approximation"];
solutions=cell(6,1);evals=cell(6,1);solutions{1}=sol;mom=zeros(6,1);times=zeros(6,1);
for j=1:6
 fprintf('Welfare rule %s\n',rules(j));
 if j>1,solutions{j}=cgm_solve(p,n,rules(j));end
 evals{j}=cgm_evaluate_policy(solutions{j},n.validation_quadrature);mom(j)=evals{j}.initial_moment;times(j)=solutions{j}.solve_seconds;
end
loss=100*((mom(1)./mom).^(1/(1-p.gamma))-1);
mc=cgm_mc_ensemble(solutions,root);
paper=[0;.637;1.531;.152;2.108;.084];delta=mc.loss-loss;
standardized=delta./max(mc.SE,eps);
out.table=table(rules',paper,loss,loss-paper,mc.loss,mc.SE,delta,standardized,times, 'VariableNames',{'rule','paper_loss','Bellman','paper_difference','MonteCarlo','MC_SE','MC_minus_Bellman','standardized_difference','solve_seconds'});
out.mc=mc;out.passed=all(abs(delta)<=n.mc_z_limit*mc.SE+n.welfare_numerical_tolerance);
writetable(out.table,fullfile(root,'results/table6/welfare.csv'));disp(out.table);
write_json(fullfile(root,'results/table6/welfare_validation.json'),mc);
assert(out.passed,'CGM:welfare','Bellman and independent tilted MC disagree');
end
