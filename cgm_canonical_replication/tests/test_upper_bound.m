function out=test_upper_bound(sol,sim,root)
n=sol.n;n.asset_max=n.upper_bound_check;
s=cgm_solve(sol.p,n);ss=cgm_simulate(s,n.simulation_N,n.seed);
out.maximum_observed_saving=max(sim.S./sim.P,[],'all');
out.wealth_change=max(abs(ss.table.wealth-sim.table.wealth))/max(sim.table.wealth);
e1=cgm_evaluate_policy(sol,n.validation_quadrature);e2=cgm_evaluate_policy(s,n.validation_quadrature);
out.initial_CE_change=e2.initial_CE/e1.initial_CE-1;
out.passed=out.wealth_change<n.wealth_convergence && abs(out.initial_CE_change)<n.consumption_convergence;
assert(out.passed,'Asset upper-bound sensitivity failed');
write_json(fullfile(root,'results/validation/upper_bound_check.json'),out);
end
