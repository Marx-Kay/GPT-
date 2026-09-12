function H=cgm_human_capital(p)
% Future income excluding current income; survival-adjusted PDV, conditional
% on permanent state, future innovations set to zero (risk ignored).
T=numel(p.ages);H=zeros(T,1);
for t=T-1:-1:1
 [G,y]=cgm_income_process(p.ages(t),0,0,p);
 H(t)=p.survival(t)/p.rf*G*(y+H(t+1));
end
end
