function [Aout,idxhr] = seltriag(Ain,nrord,shft)
%
% extract shifted triangular matrix 
%   shft : shift in [ vertical horizontal]
%   nrord : reduction order
ndim = min(size(Ain));  % smallest dim of input
shft=-shft(2:-1:1); % reverse horizontal shift direction

if ndim==1,
    Nh_max = sqrt(numel(Ain))-1;
else 
    Nh_max = size(Ain,1)-1;
end

[idxh,Az] = genhidx(Nh_max);  % generate 2d harmonics index and zero mtx
[idxhr,Azr] = genhidx(Nh_max-nrord);  % for indexing order-reduced mtx

if ndim ==1,
    A = Az; A(idxh)=Ain;   % 2D hrm matrix
else
    A = Ain;
end
    
    Asft = shiftmatrix(A,1,shft,0); % shift matrix
    Arsft = Asft(1:(end-nrord),(nrord+1):(end-nrord)); % select subset from the shifted matrix
    Aoutv = Arsft(idxhr);  % extract triangle elements from the shifted matrix
    
if ndim ==1,
    Aout = Aoutv;
else
    Aout = Azr; Aout(idxhr)=Aoutv;
end