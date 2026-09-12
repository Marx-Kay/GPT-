function [xp,G,y,rp]=cgm_transition(age,s,alpha,zp,ze,zr,p)
[G,y]=cgm_income_process(age,zp,ze,p);
[~,rp]=cgm_returns(zr,alpha,p);
assert(all(rp(:)>0),'CGM:negativeReturn','Nonpositive portfolio return: no silent truncation allowed.');
xp=s.*rp./G+y;
end
