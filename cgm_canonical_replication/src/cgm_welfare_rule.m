function [alpha,dstock]=cgm_welfare_rule(rule,age,s,H,p)
m=min(1,p.premium/(p.gamma*p.stock_sigma^2));
switch string(rule)
 case "zero",alpha=zeros(size(s));dstock=alpha;
 case "no_income",alpha=m+zeros(size(s));dstock=alpha;
 case "100_age",alpha=(100-age)/100+zeros(size(s));dstock=alpha;
 case "approximation",alpha=min(1,max(.5,2-.025*age))+zeros(size(s));dstock=alpha;
 case "no_income_risk"
  stock=min(s,m*(s+H));alpha=stock./max(s,realmin);alpha(s==0)=1;
  dstock=m+zeros(size(s));dstock(s<=m*H/max(1-m,eps))=1;
 otherwise,error('Unknown heuristic');
end
end
