function v=cgm_ce_at(sol,e,t,x)
T=numel(sol.p.ages);
if t==T,v=x;return;end
v=interp1(sol.x{t}(2:end),e.ce{t}(2:end),x,'pchip','extrap');
b=x<=sol.x{t}(2);k=1-sol.p.gamma;
v(b)=(x(b).^k+e.boundary(t)).^(1/k);
assert(all(v(:)>0),'Nonpositive CE extrapolation');
end
