function [Intensity_sph_norm,Intensity_sph] = sphsmooth_new(Asv,Wnv,Wpv,Vv,Nh_max)

nsrc = 1;

% extract eigenvectors of nsrc signal subspace
% [U,~]= eigs(R,nsrc,'LM');
% [U_temp,~]= eigs(R);
% U = U_temp(:,1:2);


%     x_dir = [pi/2 pi];
%     y_dir = [pi/2 -pi/2];
%     z_dir = [pi 0];
%     
%     Y_enc_x = sphrm(Nh_max-1,x_dir(2),x_dir(1),'complex');
%     Y_enc_y = sphrm(Nh_max-1,y_dir(2),y_dir(1),'complex');
%     Y_enc_z = sphrm(Nh_max-1,z_dir(2),z_dir(1),'complex');

for ii = 1:nsrc
%     Asv = U(:,ii);
    
    %- Order-reduced 2D hrm vector
    [Asrv,~] = seltriag(Asv,1,[0 0]);  % generate 2d harmonics index and zero mtx
    
    Aug{1} = Asrv;
    
    %-  check the recurrence relation
    Aug{2} =  seltriag(Wpv,1,[1 -1]).*seltriag(Asv,1,[1 -1]) - seltriag(Wnv,1,[0 0]).*seltriag(Asv,1,[-1 -1])  ;  % +exp(phi)
    
    Aug{3} =  seltriag(Wpv,1,[0 0]).*seltriag(Asv,1,[-1 1])  - seltriag(Wnv,1,[1 1]).*seltriag(Asv,1,[1 1]) ;       % -exp(phi)
    
    Aug{4} =  seltriag(Vv,1,[0 0]).*seltriag(Asv,1,[-1 0])   + seltriag(Vv,1,[1 0]).*seltriag(Asv,1,[1 0]) ;
    
    %- Build Intensity Vector in Spherical harmonics Domain
    partial_x = (Aug{2}+Aug{3})/2;
    partial_y = (Aug{2}-Aug{3})/2i;
    partial_z = Aug{4};
    
%     index_end = 9;
    
    % D_x = (Y_enc_x(2:index_end)).' * partial_x(2:index_end);
    % D_y = (Y_enc_y(2:index_end)).' * partial_y(2:index_end);
    % D_z = (Y_enc_z(2:index_end)).' * partial_z(2:index_end);
    
%     D_x = sum(partial_x);
%     D_y = sum(partial_y);
%     D_z = sum(partial_z);
%     
%     % D_x = Y_enc_x(2:index_end).' * Asv(2:4);
%     % D_y = Y_enc_y(2:index_end).' * Asv(2:4);
%     % D_z = Y_enc_z(2:index_end).' * Asv(2:4);
%     
%     Intensity_sph = 1/2*real( conj(Aug{1}(1)) .* [D_x D_y D_z]  ) ;
%     Intensity_sph_norm{ii}  = Intensity_sph./sqrt(sum(Intensity_sph.^2));

    D_x = Aug{1}'*partial_x;
    D_y = Aug{1}'*partial_y;
    D_z = Aug{1}'*partial_z;
    
%     Intensity_sph = 1/2*real( conj(Aug{1}(1)) .* [D_x D_y D_z]  ) ;
    Intensity_sph = 1/2*real([D_x D_y D_z] ) ;
    Intensity_sph_norm  = Intensity_sph./sqrt(sum(Intensity_sph.^2));
    
end

end