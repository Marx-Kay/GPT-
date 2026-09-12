function [c,alpha,s]=cgm_policy(sol,t,x)
c=interp1(sol.x{t},sol.c{t},x,'linear','extrap');c=min(x,max(0,c));s=x-c;
if sol.rule=="no_income_risk"
 alpha=cgm_welfare_rule(sol.rule,sol.p.ages(t),s,sol.H(t),sol.p);
else,alpha=min(1,max(0,interp1(sol.x{t},sol.alpha{t},x,'linear','extrap')));end
if t==numel(sol.p.ages),c=x;s=zeros(size(x));alpha=zeros(size(x));end
end
