function [P,y,W]=cgm_initial(zp,ze,p)
P=cgm_income(p.ages(1),p)*exp(sqrt(p.initial_permanent_variance)*zp);
y=exp(sqrt(p.transitory_variance)*ze);W=p.initial_wealth+zeros(size(y));
end
