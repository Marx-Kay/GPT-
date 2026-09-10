function sim=cgm_simulate(sol,seed)
p=sol.parameters;assert(~p.no_income,'No-income analytical test uses its own initial wealth.');
if nargin<2,seed=p.seed;end
rng(seed,'twister');T=numel(p.ages);N=p.nsim;assert(mod(N,2)==0);
eps=randn(T,N/2,3);eps=cat(2,eps,-eps);
P=zeros(T,N);Y=P;W=P;C=P;A=P;S=P;R=p.rf+p.premium+p.stock_sigma*eps(:,:,3);
for t=1:T
 age=p.ages(t);
 if t==1
 v=sqrt(p.permanent_variance)*eps(1,:,1)*double(p.initial_permanent_draw);
 P(t,:)=cgm_income(age,p)*exp(v);
 elseif age<=p.last_work_age
 P(t,:)=P(t-1,:)*cgm_income(age,p)/cgm_income(age-1,p).*exp(sqrt(p.permanent_variance)*eps(t,:,1));
 else,P(t,:)=P(t-1,:);end
 if age<=p.last_work_age,Y(t,:)=P(t,:).*exp(sqrt(p.transitory_variance)*eps(t,:,2));else,Y(t,:)=p.replacement*P(t,:);end
 cash=W(t,:)+Y(t,:);xn=cash./P(t,:);
 C(t,:)=min(cash,interp1(sol.x{t},sol.c{t},xn,'linear','extrap').*P(t,:));
 A(t,:)=min(1,max(0,interp1(sol.x{t},sol.alpha{t},xn,'linear','extrap')));S(t,:)=cash-C(t,:);
 if t<T
 rp=p.rf+A(t,:).*(R(t,:)-p.rf);assert(all(rp>0),'A nonpositive Gaussian return realization requires explicit treatment, not silent clipping.');
 W(t+1,:)=S(t,:).*rp;
 else,C(t,:)=cash;A(t,:)=0;S(t,:)=0;end
end
assert(all(C(:)>0)&&all(S(:)>=-1e-10)&&all(W(:)>=0));
levels=[mean(C,2),mean(W,2),mean(Y,2),mean(S,2)];mean_alpha=mean(A,2);
seW=std((W(:,1:N/2)+W(:,N/2+1:end))/2,0,2)/sqrt(N/2);
seA=std((A(:,1:N/2)+A(:,N/2+1:end))/2,0,2)/sqrt(N/2);
sim=struct('P',P,'Y',Y,'W',W,'C',C,'A',A,'S',S,'levels',levels,'mean_alpha',mean_alpha,'wealth_se',seW,'alpha_se',seA,'seed',seed);
end
