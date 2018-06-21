function W = recurcoef(Nh_max,flag,mode)

m = -Nh_max:Nh_max;     
n = 0:Nh_max; 
[NN,MM] = ndgrid(n,m);

switch flag
    case 'h+' 
        W = sqrt( (NN-MM-1).*(NN-MM)./(4*NN.^2-1) );
    case 'h-'
        W = sqrt( (NN+MM-1).*(NN+MM)./(4*NN.^2-1) );
    case 'v'
        W = sqrt( (NN.^2 - MM.^2)./ (4*NN.^2-1) );
end

W(NN<abs(MM)) =0;
if mode==1, 
    [idxh,~] = genhidx(Nh_max);
    W = W(idxh);
end

