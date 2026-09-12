function out=test_simulation(sol,sim)
out.budget=max(abs(sim.W+sim.Y-sim.C-sim.S),[],'all');
out.terminal=max(abs(sim.S(end,:)));out.retirement=max(abs(diff(sim.P(46:end,:),1)),[],'all');
out.solver_policy=0;
for t=1:numel(sol.p.ages)
 c=cgm_policy(sol,t,sim.x(t,:));out.solver_policy=max(out.solver_policy,max(abs(c.*sim.P(t,:)-sim.C(t,:))));
end
assert(out.budget<1e-8&&out.terminal==0&&out.retirement==0&&out.solver_policy<1e-10);
out.passed=true;
end
