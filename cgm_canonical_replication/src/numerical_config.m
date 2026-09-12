function n=numerical_config()
% Frozen before publication comparison. Tolerances target numerical accuracy.
n.upper_bound_check=400; n.assets=1001; n.asset_max=200; n.asset_min=1e-6;
n.quadrature=9; n.validation_quadrature=11; n.bisections=30;
n.simulation_N=20000; n.seed=20260911;
n.mc_income_quadrature=21; n.welfare_N=100000; n.welfare_seed=731029; n.welfare_seed_offsets=[0 101 202];
n.fine_assets=2001; n.high_quadrature=11;
n.euler_tolerance=.002; n.kkt_tolerance=.001;
n.consumption_convergence=.003; n.share_convergence=.015;
n.wealth_convergence=.01; n.welfare_convergence=.03; % percentage points
n.welfare_numerical_tolerance=.03; n.mc_z_limit=4;
n.interpolation='linear'; n.value_interpolation='pchip';
end
