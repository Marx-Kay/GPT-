function run_all()
% Unique self-contained entry point. Requires base MATLAB with JVM, no toolbox.
root=fileparts(mfilename('fullpath'));addpath(fullfile(root,'src'),fullfile(root,'tests'));
folders={'figure2','figure3','table6','validation'};
for i=1:numel(folders),if ~isfolder(fullfile(root,'results',folders{i})),mkdir(fullfile(root,'results',folders{i}));end;end
start=tic;diary(fullfile(root,'results','run.log'));cleanup=onCleanup(@()diary('off'));
status=struct('completed',false,'stage','initializing');write_json(fullfile(root,'results/validation/run_status.json'),status);
try
 source_start=cgm_provenance(root);p=cgm_parameters();n=numerical_config();
 write_json(fullfile(root,'results/validation/frozen_config.json'),struct('parameters',p,'numerical',n,'matlab',version));
 fprintf('1. Parameters, primitives, analytical tests\n');v=run_tests(p,n);
 fprintf('2. Canonical solve\n');sol=cgm_solve(p,n);
 fprintf('3. Independent Euler/KKT\n');v.optimality=test_optimality(sol,root);
 fprintf('4. Grid and quadrature convergence\n');[v.convergence,fine,high]=test_convergence(sol,root);
 fprintf('5. Lifecycle simulation and consistency\n');sim=cgm_simulate(sol,n.simulation_N,n.seed);v.simulation=test_simulation(sol,sim);v.extended=test_extended(sol,sim,root);v.upper_bound=test_upper_bound(sol,sim,root);
 writetable(sim.table,fullfile(root,'results/figure3/lifecycle.csv'));
 for j=1:2
  if j==1,s=fine;else,s=high;end
  ss=cgm_simulate(s,n.simulation_N,n.seed);v.convergence.wealth(j)=max(abs(ss.table.wealth-sim.table.wealth))/max(sim.table.wealth);
 end
 assert(all(v.convergence.wealth<n.wealth_convergence));
 fprintf('6. Figure 2 and Figure 3 publication comparisons\n');targets=cgm_paper_targets(root);comparison=cgm_compare(sol,sim,targets,root);
 fprintf('7. Table 6, Bellman evaluation and independent welfare MC\n');wt=tic;[w,solutions,evals]=cgm_welfare(sol,root);
 v.rule_optimality=test_rule_optimality(solutions,evals,root);
 v.welfare_convergence=test_welfare_convergence(solutions,evals,fine,high,root);
 v.mc_estimator=test_mc_estimator(sol,root);v.welfare=struct('passed',w.passed,'MC_SE',w.mc.SE,'differences',w.table.MC_minus_Bellman);
 timing.welfare=toc(wt);
 fprintf('8. Figure11 validation and disclosed assumption sensitivities\n');comparison=cgm_compare(sol,sim,targets,root,solutions);sensitivity=cgm_sensitivity(sol,root);
 timing.solver=sol.solve_seconds;timing.simulation=sim.seconds;timing.total=toc(start);
 v.clean_source_run=true;v.publication_reproduced=false;
 write_json(fullfile(root,'results/validation/dashboard.json'),v);write_json(fullfile(root,'results/validation/runtime.json'),timing);
 save(fullfile(root,'results','canonical_results.mat'),'sol','v','comparison','w','timing','sensitivity');
 source_end=cgm_provenance(root);assert(isequal(source_start,source_end),'Source changed during run');
 cgm_summary(root,v,comparison,w,timing);
 status=struct('completed',true,'stage','complete','seconds',toc(start),'publication_reproduced',false);
 write_json(fullfile(root,'results/validation/run_status.json'),status);
 fprintf('ALL NUMERICAL GATES PASSED. Full run %.2f seconds. Publication mismatches are reported.\n',status.seconds);
catch ex
 status.stage='failed';status.error=ex.message;write_json(fullfile(root,'results/validation/run_status.json'),status);rethrow(ex);
end
end
