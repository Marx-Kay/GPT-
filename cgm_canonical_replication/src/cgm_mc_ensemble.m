function out=cgm_mc_ensemble(solutions,root)
n=solutions{1}.n;p=solutions{1}.p;B=numel(n.welfare_seed_offsets);J=numel(solutions);mom=zeros(J,B);cv=zeros(J,J);seedrows=[];tim=0;concentration=0;
for b=1:B
 fprintf('Independent welfare MC seed %d\n',n.welfare_seed+n.welfare_seed_offsets(b));
 m=cgm_welfare_mc(solutions,n.welfare_N,n.welfare_seed+n.welfare_seed_offsets(b));
 mom(:,b)=m.moments;cv=cv+m.moment_covariance/B^2;tim=tim+m.seconds;
 concentration=max(concentration,max(m.max_pair_fraction));
 seedrows=[seedrows;[repmat(m.seed,J,1),(1:J)',m.loss,m.SE,m.effective_pairs,m.max_pair_fraction]];
end
means=mean(mom,2);loss=zeros(J,1);SE=loss;
for j=2:J
 rr=(means(1)/means(j))^(1/(1-p.gamma));loss(j)=100*(rr-1);
 g=zeros(J,1);g(1)=100*rr/(1-p.gamma)/means(1);g(j)=-100*rr/(1-p.gamma)/means(j);
 SE(j)=sqrt(max(0,g'*cv*g));
end
out=struct('moments',means,'moment_covariance',cv,'loss',loss,'SE',SE,'total_N',B*n.welfare_N,'seconds',tim,'maximum_pair_fraction',concentration);
writetable(array2table(seedrows,'VariableNames',{'seed','rule_index','loss','SE','ESS_pairs','max_pair_fraction'}),fullfile(root,'results/table6/mc_seeds.csv'));
assert(concentration<.01,'MC sample dominated by an individual pair');
assert(max(SE)<.02,'MC welfare precision insufficient');
% Seed replication: each pair of estimates should agree within joint SE.
for a=1:B
 for b=a+1:B
  aa=seedrows((a-1)*J+(1:J),:);bb=seedrows((b-1)*J+(1:J),:);
  assert(all(abs(aa(:,3)-bb(:,3))<=4*sqrt(aa(:,4).^2+bb(:,4).^2)+n.welfare_numerical_tolerance),'MC seed instability');
 end
end
end
