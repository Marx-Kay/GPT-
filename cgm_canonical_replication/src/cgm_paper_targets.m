function targets=cgm_paper_targets(root)
% Regenerate data-coordinate targets from registered raw PDF vector paths.
% Drawing indices below are zero-based PDF extraction indices, NOT fit params.
r=fullfile(root,'inputs/paper_targets');raw=jsondecode(fileread(fullfile(r,'pdf_vector_paths.json')));
d=raw.page12;
A=axesfit(d,10,0:25:300,0:.2:1);B=axesfit(d,338,0:25:300,0:.2:1);C=axesfit(d,641,0:25:300,0:10:50);
spec={"A",99,11:324,true,A;"B",20,339:341,false,B;"B",30,342:343,false,B;"B",55,344:624,true,B;"B",75,625:627,false,B;"C",20,965:968,false,C;"C",35,962:964,false,C;"C",65,959:961,false,C;"C",85,642:958,true,C};
targets.fig2=cell(size(spec,1),1);
for i=1:size(spec,1)
 pts=getpoints(d,spec{i,3},spec{i,4});tr=spec{i,5};v=convert(pts,tr);
 targets.fig2{i}=struct('panel',spec{i,1},'age',spec{i,2},'points',v);
end
% Figure3 uses registered raw point identities and freshly fitted tick axes.
d=raw.page15;A=axesfit(d,10,20:5:100,0:50:250);C=axesfit(d,535,20:5:95,0:.2:1);
spec={"consumption",11:170,true,A;"income",171,false,A;"wealth",172:313,true,A;"alpha_mean",536:537,false,C};
targets.fig3=table((20:100)','VariableNames',{'age'});
for i=1:size(spec,1)
 v=convert(getpoints(d,spec{i,2},spec{i,3}),spec{i,4});
 vals=interp1(v(:,1),v(:,2),targets.fig3.age,'linear',NaN);
 targets.fig3.(spec{i,1})=vals;
end
old=readtable(fullfile(r,'figure3_annual_targets.csv'));
err=max(abs(old.wealth-targets.fig3.wealth),[],'omitnan');assert(err<1e-8,'Figure3 extraction regression failed');
% Figure11: identify by actual path indices, not line width alone. Optimal
% and approximation have SAME width; approximation also has cross markers.
d=raw.page33;A=axesfit(d,10,20:5:95,0:.2:1);
spec={"optimal",11:12,false;"100_age",13:170,true;"no_income",171:249,true;"no_income_risk",250,false};
targets.fig11=table((20:99)','VariableNames',{'age'});
for i=1:size(spec,1)
 v=convert(getpoints(d,spec{i,2},spec{i,3}),A);
 targets.fig11.(spec{i,1})=interp1(v(:,1),v(:,2),targets.fig11.age,'linear',NaN);
end
p=d(252).points(1:40,:);v=convert(p,A); % first 20 line segments; omit decorative crosses
 targets.fig11.approximation=interp1(v(:,1),v(:,2),targets.fig11.age,'linear',NaN);
assert(max(abs(targets.fig11.approximation-min(1,max(.5,2-.025*targets.fig11.age))),[],'omitnan')<.01);
writetable(targets.fig3,fullfile(root,'results/figure3/paper_targets.csv'));
writetable(targets.fig11,fullfile(root,'results/table6/figure11_targets.csv'));
write_json(fullfile(root,'results/figure2/paper_targets.json'),targets.fig2);
end
function tr=axesfit(d,idx,xv,yv)
p=d(idx+1).points;ny=numel(yv);
% each line has two endpoints; item 1 vertical spine, next ny are y ticks,
% next item horizontal spine, subsequent items are x ticks.
yt=p(3:2:2*(ny+1),2);xt=p(2*(ny+2)+1:2:end,1);
assert(numel(xt)==numel(xv)&&numel(yt)==numel(yv));
tr.x=polyfit(xt,xv,1);tr.y=polyfit(yt,yv,1);
end
function pts=getpoints(d,ids,filled)
pts=[];
for i=ids
 pp=d(i+1).points;
 if filled,assert(strcmp(d(i+1).kind,'f'));pp=mean(unique(pp,'rows'),1);end
 pts=[pts;pp];
end
end
function v=convert(p,tr)
a=polyval(tr.x,p(:,1));b=polyval(tr.y,p(:,2));
[a,~,g]=unique(a);b=accumarray(g,b,[],@mean);v=[a,b];
end
