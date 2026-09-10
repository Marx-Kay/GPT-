function run_cgm()
% One entry point, no Python and no precomputed workspace dependencies.
base=fileparts(mfilename('fullpath'));addpath(base);
p=cgm_parameters();sol=cgm_solve(p);save(fullfile(base,'cgm_solution.mat'),'sol','-v7');
summary=[];seeds=[20260909 20260910 20260911];
for seed=seeds
 sim=cgm_simulate(sol,seed);[peak,ix]=max(sim.levels(:,2));
 summary=[summary;seed,peak,p.ages(ix),sim.wealth_se(ix),sim.mean_alpha(11),sim.mean_alpha(41)];
 writematrix([p.ages,sim.levels,sim.mean_alpha,sim.wealth_se,sim.alpha_se],fullfile(base,sprintf('cgm_seed_%d.csv',seed)));
 if seed==p.seed,save(fullfile(base,'cgm_simulation.mat'),'sim','-v7');validate_cgm(sol,sim);end
end
writematrix(summary,fullfile(base,'seed_robustness.csv'));
checks=verify_analytical();fid=fopen(fullfile(base,'analytical_checks.json'),'w');fprintf(fid,'%s',jsonencode(checks,PrettyPrint=true));fclose(fid);
fprintf('CGM annual model solved %.3fs. Analytical check passed.\n',sol.solve_seconds);disp(array2table(summary,'VariableNames',{'seed','wealth_peak','peak_age','wealth_se','alpha30','alpha60'}));
run(fullfile(base,'build_figures.m'));
run_welfare();
end
