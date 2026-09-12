function [G,y]=cgm_income_process(age,zp,ze,p)
% Transition FROM age TO age+1, in next-period permanent units.
if ~p.income_enabled,G=ones(size(zp));y=zeros(size(ze));
elseif age>=p.last_work_age,G=ones(size(zp));y=p.replacement+zeros(size(ze));
else,G=cgm_income(age+1,p)/cgm_income(age,p).*exp(sqrt(p.permanent_variance)*zp);y=exp(sqrt(p.transitory_variance)*ze);end
end
