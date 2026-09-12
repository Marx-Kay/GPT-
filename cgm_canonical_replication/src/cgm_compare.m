function out=cgm_compare(sol,sim,targets,root,solutions)
p=sol.p;rows=table();detail=table();f=figure('Visible','off');tiledlayout(3,1);
for panel=["A","B","C"]
 nexttile;hold on;
 for i=1:numel(targets.fig2)
  d=targets.fig2{i};if d.panel~=panel,continue;end
  cash=(5:5:295)';v=d.points;valid=cash>=min(v(:,1))&cash<=max(v(:,1));cash=cash(valid);
  paper=interp1(v(:,1),v(:,2),cash);F=cgm_income(d.age,p);[c,a]=cgm_policy(sol,d.age-p.ages(1)+1,cash/F);
  if panel=="C",model=c*F;else,model=a;end
  e=model-paper;row=table(panel,d.age,sqrt(mean(e.^2)),max(abs(e)),'VariableNames',{'panel','age','RMSE','max_absolute_error'});rows=[rows;row];
  detail=[detail;table(repmat(panel,numel(cash),1),repmat(d.age,numel(cash),1),cash,paper,model,e,'VariableNames',{'panel','age','cash','paper','canonical','difference'})];
  h=plot(cash,paper,'--','DisplayName',sprintf('paper %d',d.age));plot(cash,model,'Color',h.Color,'DisplayName',sprintf('canonical %d',d.age));
 end
 title('Figure 2'+panel);xlabel('Cash, thousand USD');legend('Location','eastoutside');grid on;
end
exportgraphics(f,fullfile(root,'results/figure2/comparison.png'),'Resolution',150);close(f);
writetable(rows,fullfile(root,'results/figure2/errors.csv'));writetable(detail,fullfile(root,'results/figure2/selected_states.csv'));out.figure2=rows;
t=targets.fig3;s=sim.table;
[mp,ia]=max(t.wealth);[ms,ib]=max(s.wealth);
names=["wealth_peak";"wealth_peak_age";"wealth_age50";"wealth_age65";"income_age50";"income_age65";"consumption_peak";"alpha_age65"];
pap=[mp;t.age(ia);t.wealth(31);t.wealth(46);t.income(31);t.income(46);max(t.consumption);t.alpha_mean(46)];
can=[ms;s.age(ib);s.wealth(31);s.wealth(46);s.income(31);s.income(46);max(s.consumption);s.alpha(46)];
out.figure3=table(names,pap,can,can-pap,'VariableNames',{'statistic','paper','canonical','difference'});
writetable(out.figure3,fullfile(root,'results/figure3/comparison.csv'));
f=figure('Visible','off');tiledlayout(2,2);
for field=["wealth","income","consumption","alpha"]
 nexttile;plot(s.age,s.(field),'LineWidth',1.3);hold on;
 if field=="alpha",fieldp="alpha_mean";else,fieldp=field;end
 plot(t.age,t.(fieldp),'--');title(field);xlabel('Age');legend('canonical','paper');grid on;
end
exportgraphics(f,fullfile(root,'results/figure3/comparison.png'),'Resolution',150);close(f);
if nargin>=5
 f=figure('Visible','off');hold on;rr=table();
 for j=[1 2 3 4 6]
  rule=solutions{j}.rule;sm=cgm_simulate(solutions{j},sol.n.simulation_N,sol.n.seed);
  pa=targets.fig11.(rule);a=sm.table.alpha(1:80);e=a-pa;
  rr=[rr;table(rule,sqrt(mean(e.^2,'omitnan')),max(abs(e),[],'omitnan'),'VariableNames',{'rule','RMSE','max_absolute_error'})];
  plot(sm.table.age(1:80),a,'DisplayName',rule);plot(targets.fig11.age,pa,'--','HandleVisibility','off');
 end
 title('Figure 11: solid canonical; dashed publication');legend('Location','eastoutside');xlabel('Age');ylabel('Stock share');grid on;
 exportgraphics(f,fullfile(root,'results/table6/figure11.png'),'Resolution',150);close(f);writetable(rr,fullfile(root,'results/table6/figure11_errors.csv'));out.figure11=rr;
end
end
