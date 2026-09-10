function sol=cgm_solve(p)
% Euler/envelope EGM with a globally concave conditional portfolio problem.
% This is an independent check and faster solver for the SAME annual model.
% No precomputed simulation or old .mat output is required.
start=tic;nq=p.quadrature;J=diag(sqrt(1:nq-1),1)+diag(sqrt(1:nq-1),-1);
[Q,D]=eig(J);[z,ix]=sort(diag(D));q=Q(1,ix)'.^2;
[zr,zp,zt]=ndgrid(z,z,z);[qr,qp,qt]=ndgrid(q,q,q);weight=qr.*qp.*qt;weight=weight(:)';R=p.rf+p.premium+p.stock_sigma*zr(:)';
assert(all(R>0),'Quadrature includes nonpositive returns; revise distribution treatment explicitly.');
a=[0;logspace(-5,log10(p.max_assets),p.n_assets-1)'];
if p.no_income,a=a(2:end);end
N=numel(a);T=numel(p.ages);x=cell(T,1);c=x;alpha=x;x{T}=[0;1e6];c{T}=x{T};alpha{T}=[0;0];
for t=T-1:-1:1
 age=p.ages(t);
 if p.no_income,G=ones(size(R));Y=zeros(size(R));
 elseif age>=p.last_work_age,G=ones(size(R));Y=p.replacement*ones(size(R));
 else,G=cgm_income(age+1,p)/cgm_income(age,p)*exp(sqrt(p.permanent_variance)*zp(:)');Y=exp(sqrt(p.transitory_variance)*zt(:)');end
 lo=zeros(N,1);hi=ones(N,1);
 if p.share_rule=="zero",lo(:)=0;hi(:)=0;
 elseif p.share_rule=="no_income_merton",lo(:)=min(1,p.premium/(p.gamma*p.stock_sigma^2));hi=lo;
 elseif p.share_rule=="100_age",lo(:)=(100-age)/100;hi=lo;
 elseif p.share_rule=="approximation",lo(:)=min(1,max(.5,2-.025*age));hi=lo;
 end
 iterations=p.bisections;if p.share_rule~="optimal",iterations=0;end
 for k=1:iterations
 al=(lo+hi)/2;rp=p.rf+al.*(R-p.rf);xp=a.*rp./G+Y;
 cn=interp1(x{t+1},c{t+1},xp,'linear','extrap');assert(all(cn(:)>0));
 m=(G.*cn).^(-p.gamma);foc=(m.*(R-p.rf))*weight';
 lo(foc>0)=al(foc>0);hi(foc<=0)=al(foc<=0);
 end
 al=(lo+hi)/2;rp=p.rf+al.*(R-p.rf);xp=a.*rp./G+Y;
 cn=interp1(x{t+1},c{t+1},xp,'linear','extrap');expected=((G.*cn).^(-p.gamma).*rp)*weight';
 ct=(p.beta*p.survival(t)*expected).^(-1/p.gamma);xt=a+ct;
 assert(all(diff(xt)>0),'Nonmonotone endogenous grid');
 x{t}=[0;xt];c{t}=[0;ct];alpha{t}=[al(1);al];
end
sol=struct('x',{x},'c',{c},'alpha',{alpha},'parameters',p,'solve_seconds',toc(start));
end
