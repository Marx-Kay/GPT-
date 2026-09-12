function [R,rp]=cgm_returns(z,alpha,p)
R=p.rf+p.premium+p.stock_sigma*z;
rp=p.rf+alpha.*(R-p.rf);
end
