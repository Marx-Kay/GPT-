function check_welfare_convergence()
base=fileparts(mfilename('fullpath'));addpath(base);
rules=["optimal","100_age","no_income_merton","zero","approximation"];
configs=[801,5;801,9;1601,11];rows=[];
for j=1:size(configs,1)
 p=cgm_parameters();p.n_assets=configs(j,1);p.quadrature=configs(j,2);moments=zeros(5,1);
 for i=1:5
  p.share_rule=rules(i);sol=cgm_solve(p);ev=cgm_evaluate_policy(sol,11);moments(i)=ev.initial_moment;
  if j==3&&i==1,save(fullfile(base,'welfare_validation_solution.mat'),'sol','ev','-v7');end
 end
 losses=100*(1-(moments/moments(1)).^(1/(1-p.gamma)));
 rows=[rows;configs(j,:),losses(2:end)'];fprintf('Welfare convergence %d/%d done.\n',j,size(configs,1));
end
T=array2table(rows,'VariableNames',{'asset_points','solve_nodes','loss_100_age','loss_no_income','loss_zero','loss_approximation'});
writetable(T,fullfile(base,'welfare_convergence.csv'));disp(T);
end
