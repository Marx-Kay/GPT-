function result=verify_analytical()
% Finite-horizon CRRA with no income has exact proportional consumption
% and a time-invariant portfolio for the discretized return distribution.
p=cgm_parameters();p.no_income=true;p.n_assets=41;p.quadrature=5;
s=cgm_solve(p);J=diag(sqrt(1:4),1)+diag(sqrt(1:4),-1);[Q,D]=eig(J);[z,ix]=sort(diag(D));q=Q(1,ix)'.^2;R=p.rf+p.premium+p.stock_sigma*z;
foc=@(a)sum(q.*(p.rf+a*(R-p.rf)).^(-p.gamma).*(R-p.rf));
if foc(0)<=0,astar=0;elseif foc(1)>=0,astar=1;else,astar=fzero(foc,[0 1]);end
M=sum(q.*(p.rf+astar*(R-p.rf)).^(1-p.gamma));kappa=ones(81,1);
for t=80:-1:1,kappa(t)=1/(1+(p.beta*p.survival(t)*M)^(1/p.gamma)/kappa(t+1));end
errC=0;errA=0;
for t=1:80
 errC=max(errC,max(abs(s.c{t}(2:end)./s.x{t}(2:end)-kappa(t))));
 errA=max(errA,max(abs(s.alpha{t}(2:end)-astar)));
end
result=struct('test','finite-horizon no-income CRRA with exact discrete-return solution','consumption_ratio_max_abs_error',errC,'equity_share_max_abs_error',errA,'exact_equity_share',astar,'passed',errC<1e-8&&errA<1e-7);
assert(result.passed,'Analytical model verification failed');
end
