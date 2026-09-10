base=fileparts(mfilename('fullpath'));root=fileparts(base);set(groot,'defaultFigureVisible','off');
r=load(fullfile(base,'cgm_solution.mat'));sol=r.sol;rr=load(fullfile(base,'cgm_simulation.mat'));sim=rr.sim;
sol.W=sim.W./sim.P;sol.C=sim.C./sim.P;sol.A=sim.A;sol.P=sim.P;sol.levels=sim.levels(:,1:3);sol.mean_alpha=sim.mean_alpha;
d=struct('survprob',sol.parameters.survival,'simY',sim.Y./sim.P);age=sol.parameters.ages;
hist=struct('levels',nan(81,3),'mean_alpha',nan(81,1));
if isfile(fullfile(root,'historical_matlab','simulation.mat')),h=load(fullfile(root,'historical_matlab','simulation.mat'));hist=h.out;end
target=readtable(fullfile(root,'paper_targets','figure3_annual_targets.csv'));p2=readtable(fullfile(root,'paper_targets','figure2a_age99_targets.csv'));
f=@(a)exp(-2.170042+2.700381+.16818*a-.0323371/10*a.^2+.0019704/100*a.^3);
scale=f(65);p2fit=interp1(sol.x{80},sol.alpha{80},p2.cash_on_hand_thousand_usd/scale);
mask=isfinite(p2.value)&p2.value<=1.001;
metrics=struct('figure2a_share_max_abs_error',max(abs(p2fit(mask)-p2.value(mask))),'figure2a_share_rmse',sqrt(mean((p2fit(mask)-p2.value(mask)).^2)));
writetable(table(p2.cash_on_hand_thousand_usd,p2.value,p2fit,'VariableNames',{'cash','paper_alpha','computed_alpha'}),fullfile(base,'figure2a_comparison.csv'));
% Physical-unit policy panels using exp(random permanent component)=1.
X=linspace(.1,300,500)';fig=figure('Position',[0 0 1050 1050]);tiledlayout(3,1);
nexttile;plot(X,interp1(sol.x{80},sol.alpha{80},X/scale),'LineWidth',1.6);hold on;plot(p2.cash_on_hand_thousand_usd,p2.value,'k.');yline(.04/(10*.157^2),'--');ylim([0 1.02]);grid on;title('Figure 2A: age 99, verified two-period comparison');legend('Computed','Paper vector extraction','Merton approximation');xlabel('Cash on hand, thousands of 1992 USD');ylabel('Equity share');
nexttile;hold on;for a=[20 30 55 75],plot(X,interp1(sol.x{a-19},sol.alpha{a-19},X/f(min(a,65))),'LineWidth',1.5,'DisplayName',string(a));end;ylim([0 1.02]);legend('Location','northeast');grid on;title('Figure 2B: permanent-income model');xlabel('Cash on hand, thousands of 1992 USD');ylabel('Equity share');
nexttile;hold on;for a=[85 65 35 20],p=f(min(a,65));plot(X,p*interp1(sol.x{a-19},sol.c{a-19},X/p),'LineWidth',1.5,'DisplayName',string(a));end;legend('Location','northwest');grid on;title('Figure 2C: permanent-income model');xlabel('Cash on hand, thousands of 1992 USD');ylabel('Consumption, thousands of 1992 USD');
exportgraphics(fig,fullfile(base,'figure2_policies.png'),'Resolution',150);exportgraphics(fig,fullfile(base,'figure2_policies.pdf'),'ContentType','vector');close(fig);
fig=figure('Position',[0 0 1150 950]);tiledlayout(2,2);
nexttile;plot(age,sol.levels(:,2),'LineWidth',1.8);hold on;plot(target.age,target.wealth,'k--','LineWidth',1.5);plot(age,hist.levels(:,2),':','LineWidth',1.8);legend('Permanent-income optimum','Paper','Historical-policy diagnostic','Location','northwest');title('Wealth: differing models remain distinct');xlabel('Age');ylabel('Thousands of 1992 USD');grid on;
nexttile;plot(age,sol.mean_alpha,'LineWidth',1.8);hold on;plot(target.age,target.alpha_mean,'k--','LineWidth',1.5);plot(age,hist.mean_alpha,':','LineWidth',1.8);xlim([20 99]);ylim([0 1.02]);title('Mean equity share');xlabel('Age');grid on;
nexttile;plot(age,sol.levels(:,1),'LineWidth',1.6);hold on;plot(target.age,target.consumption,'k--','LineWidth',1.5);plot(age,sol.levels(:,3),'LineWidth',1.6);plot(target.age,target.income,'k:','LineWidth',1.5);legend('Computed consumption','Paper consumption','Computed income','Paper income','Location','northwest');title('Consumption and income');xlabel('Age');ylabel('Thousands of 1992 USD');grid on;
nexttile;plot(age(1:80),prctile(sol.A(1:80,:),[5 95],2),'LineWidth',1.4);hold on;plot(target.age,[target.alpha_p05,target.alpha_p95],'k--','LineWidth',1.2);ylim([0 1.02]);title('Equity-share 5th and 95th percentiles');xlabel('Age');legend('Computed p05','Computed p95','Paper p05','Paper p95','Location','southwest');grid on;
exportgraphics(fig,fullfile(base,'figure3_comparison.png'),'Resolution',150);exportgraphics(fig,fullfile(base,'figure3_comparison.pdf'),'ContentType','vector');close(fig);
% Human wealth: explicitly specified conditional expected future income,
% discounted at Rf and conditional survival. Not an asserted Figure 3B match.
H=zeros(81,1);
for t=80:-1:1
 a=t+19;if a>=65,G=1;yn=.68212;else,G=f(a+1)/f(a)*exp(.0106/2);yn=exp(.0738/2);end
 H(t)=d.survprob(t)/1.02*G*(yn+H(t+1));
end
meanH=H.*mean(sol.P,2);ratio=mean(H./(sol.W+d.simY),2);
writematrix([age,meanH,ratio],fullfile(base,'human_wealth_defined.csv'));
fig=figure('Position',[0 0 1000 400]);tiledlayout(1,2);nexttile;plot(age,meanH,'LineWidth',1.6);xlabel('Age');ylabel('Thousands of 1992 USD');title('Expected future income PV, survival-adjusted');grid on;nexttile;plot(age,ratio,'LineWidth',1.6);xlabel('Age');ylabel('Mean of individual H / cash on hand');title('Explicit aggregation; not yet matched to Figure 3B');grid on;exportgraphics(fig,fullfile(base,'human_wealth_defined.png'),'Resolution',150);close(fig);
% Monte Carlo uncertainty uses antithetic PAIRS as independent observations.
N=size(sol.W,2)/2;pairW=(sol.W(:,1:N).*sol.P(:,1:N)+sol.W(:,N+1:end).*sol.P(:,N+1:end))/2;
pairA=(sol.A(:,1:N)+sol.A(:,N+1:end))/2;
seW=std(pairW,0,2)/sqrt(N);seA=std(pairA,0,2)/sqrt(N);
writematrix([age,sol.levels,sol.mean_alpha,seW,seA],fullfile(base,'final_lifecycle_with_se.csv'));
metrics.persistent_wealth_peak=max(sol.levels(:,2));[~,ix]=max(sol.levels(:,2));metrics.peak_age=age(ix);metrics.peak_age_mean_wealth_se=seW(ix);
metrics.paper_wealth_peak=max(target.wealth);metrics.paper_mean_equity_age65=target.alpha_mean(target.age==65);
metrics.persistent_mean_equity_age65=sol.mean_alpha(46);metrics.historical_diagnostic_wealth_peak=max(hist.levels(:,2));
metrics.income_comparison_rmse=sqrt(mean((sol.levels(target.age-19,3)-target.income).^2,'omitnan'));
metrics.figure3_wealth_rmse=sqrt(mean((sol.levels(target.age-19,2)-target.wealth).^2,'omitnan'));
metrics.figure3_alpha_rmse=sqrt(mean((sol.mean_alpha(target.age-19)-target.alpha_mean).^2,'omitnan'));
validation=jsondecode(fileread(fullfile(base,'numerical_validation.json')));metrics.numeric_model_validated=validation.passed;metrics.published_figure3_reproduced=false;
fid=fopen(fullfile(base,'metrics.json'),'w');fprintf(fid,'%s',jsonencode(metrics,PrettyPrint=true));fclose(fid);disp(metrics);
