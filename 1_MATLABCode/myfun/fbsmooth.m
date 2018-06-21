function Ab = fbsmooth(Ain)

ndim = min(size(Ain));  % smallest dim of input

if ndim==1,  %- for vector input
    Nh_max = sqrt(numel(Ain))-1;
else 
    Nh_max = size(Ain,1)-1;
end
[idxh,Az] = genhidx(Nh_max);  % generate 2d harmonics index and zero mtx

%- order & degree matrix
m = -Nh_max:Nh_max;     
n = 0:Nh_max; 
[NN,MM] = ndgrid(n,m);
% nv = NN(idxh); mv = MM(idxh);

%- flip horizontly
if ndim==1,
    Az(idxh) = Ain; 
    Ain = Az;   % convert to 2D matrix
end

Aout = (-1).^(NN+MM).*conj(fliplr(Ain));

%- output
if ndim==1, 
    Ab= Aout(idxh); % convert to vector
else
    Ab = Aout;
end
