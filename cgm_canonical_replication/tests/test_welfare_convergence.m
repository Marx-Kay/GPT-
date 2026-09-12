function out=test_welfare_convergence(solutions,evals,fine,high,root)
p=solutions{1}.p;n=solutions{1}.n;mom=zeros(6,3);
for j=1:6,mom(j,1)=evals{j}.initial_moment;end
for col=2:3
 for j=1:6
  if col==2,nn=n;nn.assets=n.fine_assets;else,nn=n;nn.quadrature=n.high_quadrature;end
  if j==1
   if col==2,s=fine;else,s=high;end
  else,s=cgm_solve(p,nn,solutions{j}.rule);end
  ev=cgm_evaluate_policy(s,n.validation_quadrature);mom(j,col)=ev.initial_moment;
 end
end
loss=100*((mom(1,:)./mom).^(1/(1-p.gamma))-1);
out.loss=loss;out.max_difference=max(abs(loss(:,2:3)-loss(:,1)),[],'all');out.passed=out.max_difference<n.welfare_convergence;
writetable(array2table(loss,'VariableNames',{'baseline','fine_grid','higher_quadrature'}),fullfile(root,'results/validation/welfare_convergence.csv'));
assert(out.passed,'CGM:welfareConvergence','Welfare convergence tolerance failed');
end
