%
% [Ymn]=spharmc(L,PHIv,THv)
%
% Generate spherical harmonics

function [Ymn]=spharmc(L,PHIv,THv,flag)

PHIv = PHIv(:)'; 
THv= THv(:)'; 

% Gamn = @(L)  sqrt((L+1/2)/(4*pi));

Lmn=legendre(L,cos(THv),'norm');
switch flag 
    case 'complex'
        M = (-L:L)';
        if L~=0, 
        mDia = (-1).^(0:L);
        % Ymn = Gamn(L)*(sparse(diag(mDia))*[flipud(Lmn(2:end,:)); Lmn]).*exp(1i*M*PHIv);
        Ymn = (1/sqrt(2*pi))*([flipud(Lmn(2:end,:)); sparse(diag(mDia))*Lmn]).*exp(1i*M*PHIv);
        else 
            Ymn = (1/sqrt(2*pi))*Lmn;
        % add Ymn for minus index
        end
    case 'real'
        M = 1:L;
        if L~=0,
            Ymn = 1/sqrt(pi)* [flipud(cos(M(:)*PHIv).*Lmn(2:end,:)); Lmn(1,:)/sqrt(2); ...
                                     sin(M(:)*PHIv).*Lmn(2:end,:)];
        else 
            Ymn = (1/sqrt(2*pi))*Lmn;
        end
end
                                 
