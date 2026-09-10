function result=validate_cgm(sol,sim)
base=fileparts(mfilename('fullpath'));
% Fresh 11-node integration, different from training's 9 nodes.
J=diag(sqrt(1:10),1)+diag(sqrt(1:10),-1);[Q,D]=eig(J);[z,idx]=sort(diag(D));q=Q(1,idx)'.^2;
[zr,zp,zh]=ndgrid(z,z,z);[qr,qp,qh]=ndgrid(q,q,q);weight=qr.*qp.*qh;weight=weight(:)';R=1.06+.157*zr(:)';
gamma=sol.parameters.gamma;beta=sol.parameters.beta;rf=sol.parameters.rf;rows=[];
for age=[20 25 30 40 50 60 65 66 75 85 95 99]
 t=age-19;
 observed=sim.W(t,:)./sim.P(t,:)+sim.Y(t,:)./sim.P(t,:);
 cash=unique([quantile(observed,linspace(.001,.999,121)),.25,.5,1,2,5,10,20,50])';
 c=min(cash,interp1(sol.x{t},sol.c{t},cash,'linear','extrap'));
 a=min(1,max(0,interp1(sol.x{t},sol.alpha{t},cash,'linear','extrap')));s=cash-c;
 if age>=65,G=ones(size(R));Y=.68212*ones(size(R));else
 f=@(ag).16818*ag-.0323371/10*ag.^2+.0019704/100*ag.^3;
 G=exp(f(age+1)-f(age)+sqrt(.0106)*zp(:)');Y=exp(sqrt(.0738)*zh(:)');end
 rp=rf+a.*(R-rf);xp=s.*rp./G+Y;cn=interp1(sol.x{t+1},sol.c{t+1},xp,'linear','extrap');
 marginal=(G.*cn).^(-gamma);rhs=beta*sol.parameters.survival(t)*(marginal.*rp)*weight';
 implied=rhs.^(-1/gamma);rel=implied./c;
 cerror=abs(rel-1);bound=s<1e-8;cerror(bound)=max(0,1-rel(bound));
 raw=(marginal.*(R-rf))*weight';norm=marginal*weight';foc=raw./norm;
 aerror=abs(foc);aerror(a<1e-5)=max(0,foc(a<1e-5));aerror(a>1-1e-5)=max(0,-foc(a>1-1e-5));aerror(bound)=0;
 rows=[rows;age,max(cerror),median(cerror),max(aerror),max(abs(c+c*0+s-cash)),min(c),min(a),max(a)];
end
writematrix(rows,fullfile(base,'optimality_checks.csv'));
result=struct('cash_states','121 simulation quantiles plus explicit low/high cash states at 12 ages','integration_nodes_per_shock',11,'max_relative_consumption_euler_error',max(rows(:,2)),'max_normalized_portfolio_kkt_error',max(rows(:,4)),'rows',rows);

result.analytical=verify_analytical();
result.budget_max_abs_error=max(abs(sim.W+sim.Y-sim.C-sim.S),[],'all');
result.terminal_savings_max_abs=max(abs(sim.S(end,:)));
result.retirement_permanent_max_abs_change=max(abs(diff(sim.P(46:end,:),1)),[],'all');
result.value_boundary=verify_value_boundary();
result.passed=result.max_relative_consumption_euler_error<.002 && result.max_normalized_portfolio_kkt_error<.001 && result.budget_max_abs_error<1e-9 && result.terminal_savings_max_abs==0 && result.retirement_permanent_max_abs_change==0 && result.analytical.passed && result.value_boundary<1e-10;
assert(result.passed,'Numerical acceptance checks failed');
fid=fopen(fullfile(base,'numerical_validation.json'),'w');fprintf(fid,'%s',jsonencode(result,PrettyPrint=true));fclose(fid);
end
function err=verify_value_boundary()
% Two-period lognormal initial income, certain next income, gamma=2.
% At x<=1: consume x; CE=x/(1+x). At x>1: CE=(x+1)/4.
p=cgm_parameters();p.ages=[20;21];p.gamma=2;p.beta=1;p.survival=1;p.last_work_age=0;p.replacement=1;p.rf=1;p.premium=0;p.stock_sigma=0;p.quadrature=3;p.transitory_variance=.09;
s=struct('parameters',p,'x',{{[0;1;2;4;8],[0;1e6]}},'c',{{[0;1;1.5;2.5;4.5],[0;1e6]}},'alpha',{{zeros(5,1),zeros(2,1)}});
e=cgm_evaluate_policy(s);z=[-sqrt(3);0;sqrt(3)];q=[1/6;2/3;1/6];x=exp(.3*z);expected=x./(1+x);expected(x>1)=(x(x>1)+1)/4;
err=abs(e.initial_moment-sum(q./expected));
assert(err<1e-10,'Borrowing-boundary CE interpolation is incorrect');
end
