function p=cgm_parameters()
% CGM (2005), Tables 2--4; high-school baseline, annual decisions.
root=fileparts(fileparts(fileparts(mfilename('fullpath'))));
s=fileread(fullfile(root,'Life_Cycle_Matlab','life_cycle.m'));
tokens=regexp(s,'survprob\((\d+),1\)\s*=\s*([0-9.]+)','tokens');
p.survival=zeros(80,1);for i=1:numel(tokens),p.survival(str2double(tokens{i}{1}))=str2double(tokens{i}{2});end
assert(numel(tokens)==80&&all(p.survival>0)&all(p.survival<1));
p.ages=(20:100)';p.last_work_age=65;p.gamma=10;p.beta=.96;p.rf=1.02;p.premium=.04;p.stock_sigma=.157;
p.permanent_variance=.0106;p.transitory_variance=.0738;p.replacement=.68212;
p.income_coefficients=[-2.170042+2.700381,.16818,-.0323371/10,.0019704/100];
p.n_assets=801;p.max_assets=200;p.quadrature=9;p.bisections=28;
p.nsim=10000;p.seed=20260909;p.initial_permanent_draw=true;
p.no_income=false;p.share_rule="optimal";
p.source='CGM (2005) baseline; survival and unrounded high-school polynomial from supplied template';
end
