function q=cgm_shocks(n,working)
J=diag(sqrt(1:n-1),1)+diag(sqrt(1:n-1),-1);
[Q,D]=eig(J);[z,i]=sort(diag(D));w=Q(1,i)'.^2;
q.z=z;q.w=w;
if working,[q.zr,q.zp,q.ze]=ndgrid(z,z,z);[wr,wp,we]=ndgrid(w,w,w);q.weight=wr.*wp.*we;
else,q.zr=z;q.zp=zeros(size(z));q.ze=q.zp;q.weight=w;end
q.zr=q.zr(:)';q.zp=q.zp(:)';q.ze=q.ze(:)';q.weight=q.weight(:);
end
