function out=cgm_welfare_mc(solutions,N,seed)
% Independent continuous-normal lifetime utility estimator.
% Adaptive permanent Gaussian tilt with exact prefix likelihood correction.
% Conditional integration removes current temporary-shock noise; past
% temporary shocks follow the ordinary measure. No path likelihood product. Antithetic trajectory pairs are the independent units.
start=tic;p=solutions{1}.p;T=numel(p.ages);J=numel(solutions);k=1-p.gamma;
assert(mod(N,2)==0);rng(seed,'twister');half=N/2;
w=zeros(J,N);total=zeros(J,N);discount=1;
logscale=k*log(cgm_income(p.ages(1),p))+zeros(J,N);
assert(p.initial_wealth==0 && p.initial_permanent_variance==0,'Welfare MC baseline requires W20=0 and deterministic initial permanent income');
for t=1:T
 % Integrate current temporary income conditional on entering wealth.
 % Fresh 21-node Gaussian rule is independent of solver/evaluator integration.
 if p.ages(t)<=p.last_work_age
  q=cgm_shocks(solutions{1}.n.mc_income_quadrature,false);
  yeval=exp(sqrt(p.transitory_variance)*q.z');weights=q.w;
 else,yeval=p.replacement;weights=1;end
 for j=1:J
  cc=cgm_policy(solutions{j},t,w(j,:)'+yeval);
  total(j,:)=total(j,:)+discount*exp(logscale(j,:)).*(cc.^k*weights)';
 end
 if t<T
  % Ordinary current temporary draw drives next wealth. Independent of
  % the Rao-Blackwell-style conditional utility evaluation draw above.
  z=randn(3,half);z=[z,-z];
  if p.ages(t)<=p.last_work_age,y=exp(sqrt(p.transitory_variance)*z(1,:));else,y=p.replacement+zeros(1,N);end
  % Adaptive Gaussian proposal uses future-income share as an exposure proxy.
  % Exact likelihood correction preserves the original economic process.
  H=solutions{1}.H(t);
  for j=1:J
   [~,alpha,saving]=cgm_policy(solutions{j},t,w(j,:)+y);
   exposure=H./(H+w(j,:)+1);
   shift=k*sqrt(p.permanent_variance)*exposure;
   zp=z(2,:)+shift;
   stockExposure=alpha.*saving./(p.rf*saving+H+1);
   shiftR=k*p.stock_sigma*stockExposure;zr=z(3,:)+shiftR;
   [xn,G,yn]=cgm_transition(p.ages(t),saving,alpha,zp,zeros(1,N),zr,p);
   w(j,:)=xn-yn;
   logscale(j,:)=logscale(j,:)-shiftR.*z(3,:)-.5*shiftR.^2;
   if p.ages(t)<p.last_work_age
    logscale(j,:)=logscale(j,:)+k*log(cgm_income(p.ages(t)+1,p)/cgm_income(p.ages(t),p))+k*sqrt(p.permanent_variance)*zp-shift.*z(2,:)-.5*shift.^2;
   end
  end
  discount=discount*p.beta*p.survival(t);
 end
end
paired=(total(:,1:half)+total(:,half+1:end))/2;means=mean(paired,2);
out.moment_covariance=cov(paired')/half;out.moments=means;out.loss=zeros(J,1);out.SE=zeros(J,1);out.effective_pairs=zeros(J,1);out.max_pair_fraction=zeros(J,1);
for j=1:J
 ratio=(means(1)/means(j))^(1/k);out.loss(j)=100*(ratio-1);
 influence=100*ratio/k*(paired(1,:)/means(1)-paired(j,:)/means(j));
 out.SE(j)=std(influence)/sqrt(half);
 out.effective_pairs(j)=sum(paired(j,:))^2/sum(paired(j,:).^2);
 out.max_pair_fraction(j)=max(paired(j,:))/sum(paired(j,:));
end
out.N=N;out.seed=seed;out.seconds=toc(start);
end
