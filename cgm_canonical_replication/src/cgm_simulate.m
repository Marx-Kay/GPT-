function sim=cgm_simulate(sol,N,seed)
start=tic;p=sol.p;T=numel(p.ages);assert(mod(N,2)==0);rng(seed,'twister');
z=randn(T,N/2,3);z=cat(2,z,-z);
[P,y,W]=cgm_initial(z(1,:,1),z(1,:,2),p);x=W./P+y;
rows=zeros(T,10);X=zeros(T,N);CC=X;AA=X;WW=X;PP=X;YY=X;SS=X;
for t=1:T
 [c,alpha,s]=cgm_policy(sol,t,x);C=P.*c;S=P.*s;Y=P.*y;
 assert(max(abs(W+Y-C-S))<1e-8);
 WW(t,:)=W;PP(t,:)=P;YY(t,:)=Y;CC(t,:)=C;AA(t,:)=alpha;SS(t,:)=S;X(t,:)=x;
 wp=(W(1:N/2)+W(N/2+1:end))/2;
 rows(t,:)=[p.ages(t),mean(C),mean(W),mean(Y),mean(S),mean(alpha),mean(W+Y),std(wp)/sqrt(N/2),quantile(alpha,.05),quantile(alpha,.95)];
 if t<T
  [xn,G,yn,rp]=cgm_transition(p.ages(t),s,alpha,z(t+1,:,1),z(t+1,:,2),z(t,:,3),p);
  W=S.*rp;P=P.*G;y=yn;x=xn;
  assert(max(abs(x-(W./P+y)))<1e-10);
 end
end
sim.table=array2table(rows,'VariableNames',{'age','consumption','wealth','income','saving','alpha','cash','wealth_SE','alpha_p05','alpha_p95'});
sim.x=X;sim.W=WW;sim.P=PP;sim.Y=YY;sim.C=CC;sim.S=SS;sim.alpha=AA;sim.seconds=toc(start);
end
