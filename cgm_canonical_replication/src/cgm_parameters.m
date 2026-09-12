function p=cgm_parameters()
% Economic source of truth. See docs/calibration.md for assumptions.
r=fileparts(fileparts(mfilename('fullpath')));
t=readtable(fullfile(r,'inputs','survival','survival.csv'));
p.ages=(20:100)'; p.survival=t.conditional_survival;
p.gamma=10; p.beta=.96; p.bequest=0; p.rf=1.02;
p.premium=.04; p.stock_sigma=.157;
p.permanent_variance=.0106; p.transitory_variance=.0738;
p.income_correlation=0; p.return_income_correlation=0;
p.last_work_age=65; p.replacement=.68212;
p.income_coefficients=[-2.1700,.1682,-.0323/10,.0020/100];
p.income_log_level=2.700381; % unprinted recovered scale, not a verified paper parameter
p.initial_wealth=0; p.initial_permanent_variance=0;
p.income_enabled=true;
p.diagnostic_full_precision_coefficients=[-2.170042,.16818,-.0323371/10,.0019704/100];
p.diagnostic_income_scale=1.1;

end
