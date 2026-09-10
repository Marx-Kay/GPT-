function evaluation=cgm_evaluate_policy(sol,n)
% Direct Bellman policy evaluation, independent of the Euler solve.
% Store consumption-equivalent value: V_actual=CE^(1-gamma)/(1-gamma).
p=sol.parameters;if nargin<2,n=p.quadrature;end;J=diag(sqrt(1:n-1),1)+diag(sqrt(1:n-1),-1);[Q,D]=eig(J);[z,ix]=sort(diag(D));q=Q(1,ix)'.^2;
[zr,zp,zt]=ndgrid(z,z,z);[qr,qp,qt]=ndgrid(q,q,q);weights=qr.*qp.*qt;weights=weights(:)';R=p.rf+p.premium+p.stock_sigma*zr(:)';
T=numel(p.ages);k=1-p.gamma;ce=cell(T,1);ce{T}=sol.x{T};K=zeros(T,1);
for t=T-1:-1:1
 age=p.ages(t);x=sol.x{t}(2:end);c=sol.c{t}(2:end);alpha=sol.alpha{t}(2:end);s=x-c;
 if p.no_income,G=ones(size(R));Y=zeros(size(R));elseif age>=p.last_work_age,G=ones(size(R));Y=p.replacement*ones(size(R));else,G=cgm_income(age+1,p)/cgm_income(age,p)*exp(sqrt(p.permanent_variance)*zp(:)');Y=exp(sqrt(p.transitory_variance)*zt(:)');end
 xp=s.*(p.rf+alpha.*(R-p.rf))./G+Y;
 vn=ce_at(t+1,xp);assert(all(vn(:)>0));
 K(t)=p.beta*p.survival(t)*sum(weights.*(G.*ce_at(t+1,Y)).^k);
 value=(c.^k+p.beta*p.survival(t)*((G.*vn).^k)*weights').^(1/k);ce{t}=[0;value];
end
% Initial financial wealth=0, income realized before the first decision.
x0=exp(sqrt(p.transitory_variance)*z);vc=ce_at(1,x0);
initial_moment=sum(q.*vc.^k);evaluation=struct('ce',{ce},'initial_moment',initial_moment,'initial_ce_normalized',initial_moment^(1/k),'evaluation_quadrature',n,'borrowing_continuation',K);
 function out=ce_at(ti,xx)
  if ti<T && ~p.no_income
   out=interp1(sol.x{ti}(2:end),ce{ti}(2:end),xx,'pchip','extrap');
  else,out=interp1(sol.x{ti},ce{ti},xx,'pchip','extrap');end
  % On the borrowing boundary c=x, but CE is NOT linear in x.
  % The exact Bellman branch avoids extrapolating utility from the origin.
  if ti<T && ~p.no_income
   constrained=xx<=sol.c{ti}(2);
   out(constrained)=(xx(constrained).^k+K(ti)).^(1/k);
  end
 end
end
