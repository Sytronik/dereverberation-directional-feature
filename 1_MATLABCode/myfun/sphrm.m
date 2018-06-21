function [Ymn,n,m] = sphrm(N,PHIv,THv,flag)

Ymn=[]; m=[]; n=[];
for ii=0:N,
Ymn = [Ymn; spharmc(ii,PHIv,THv,flag)];
m = [m; (-ii:ii)'];
n  = [n;  ii*ones(2*ii+1,1)]; 
end
