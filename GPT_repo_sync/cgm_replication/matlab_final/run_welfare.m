function run_welfare()
base=fileparts(mfilename('fullpath'));addpath(base);p=cgm_parameters();r=load(fullfile(base,'cgm_solution.mat'));opt=r.sol;ev=cgm_evaluate_policy(opt);baseline=ev.initial_moment;
rules=["100_age","no_income_merton","zero","approximation"];paper=[.637,1.531,2.108,.084];rows=zeros(4,3);
for i=1:4
 p.share_rule=rules(i);sol=cgm_solve(p);value=cgm_evaluate_policy(sol);
 loss=100*(1-(value.initial_moment/baseline)^(1/(1-p.gamma)));rows(i,:)=[loss,paper(i),loss-paper(i)];
 assert(loss>-1e-3,'A restricted policy outperforms computed optimum; inspect evaluation accuracy.');
 save(fullfile(base,"policy_"+rules(i)+".mat"),'sol','value','-v7');
end
T=array2table(rows,'VariableNames',{'computed_loss_percent','paper_table6_percent','difference_percentage_points'},'RowNames',cellstr(rules));writetable(T,fullfile(base,'welfare_table6_partial.csv'),'WriteRowNames',true);disp(T);
end
