function y=sphr(x,n,flag)
% spherical radial function of order n
%   
% y=sphr(n,x,flag)
%
% flag: 'b' for the 1st kind Bessel, 'y' for the 2nd kind, 
%       'h' for the Hankel
spy = @(x,n) sqrt(pi ./ (2 * x)) .* bessely(n+0.5, x);
sph = @(x,n) sphj(x,n)+1i*spy(x,n);
spah = @(x,n) sphj(x,n)+1i*sin(x-pi/2*(n+1))./x;

[X,N]= ndgrid(x,n);

switch flag
    case 'j'
        y=sphj(X,N);
    case 'y'
        y=spy(X,N);
    case 'h'
        y=sph(X,N);
    case 'ah'
        y=spah(X,N);
    case 'dj'
        y= (N.*sphj(X,N-1)-(N+1).*sphj(X,N+1))./(2*N+1);
    case 'dy'
        y= (N.*spy(X,N-1)-(N+1).*spy(X,N+1))./(2*N+1);
    case 'dh'
        y= (N.*sph(X,N-1)-(N+1).*sph(X,N+1))./(2*N+1);
    case 'adh'
        y= (N.*sphj(X,N-1)-(N+1).*sphj(X,N+1))./(2*N+1) + 1i* (- sin(x-pi/2*(n+1))./x.^2 + cos(x-pi/2*(n+1))./x) ;
        
end
end

function y=sphj(x,n)
spj= @(x,n) sqrt(pi ./ (2 * x)) .* besselj(n+0.5, x);

        y = spj(x,n);
        id_xz_nz = eq(x,0) & eq(n,0);
        id_xz_nnz = eq(x,0) & ~eq(n,0);
        y(id_xz_nz) = 1; 
        y(id_xz_nnz) = 0;
end

