function [idxout,B] = genhidx(Nh_max)
a = 1:((Nh_max+1)^2);

idx = [];
    for nn=0:Nh_max,
        idx = [idx (2*Nh_max+1)*nn+(Nh_max-nn)+(1:(2*nn+1))];
    end

A= zeros(2*Nh_max+1,Nh_max+1); B=A';
A(idx) = a; A=A.';

newidx = find(A);

tmpout = sortrows([newidx(:) A(newidx(:))],2);
idxout = tmpout(:,1);
% B(idxout)=a