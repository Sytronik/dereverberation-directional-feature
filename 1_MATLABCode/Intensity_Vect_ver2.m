clear all;
close all;
addpath myfun
addpath polarch-Spherical-Array-Processing-4df5f23


%% Anonymous functions

angle_points = @(theta1, theta2, phi1, phi2) acos(cos(theta1)*cos(theta2)+...
    cos(phi1-phi2)*sin(theta1)*sin(theta2));
vect = @(x) reshape(x,numel(x),1);            % reshape vector form
dB10 = @(x) 10*log10(abs(x));                   % 10 dB

%% Source generation

Nrd = 1;  % smoothing order

src.n   = 3
src.mag = ones(src.n,1); % fully correlated case

% source direction
az_degree_buf = [];
elev_degree_buf = [];

for ii = 1
    
    src.phi =  ([45 68 135 277]/180*pi).';
    src.th  =  ([20 46 140 169]/180*pi).';
    
    src.phi = src.phi(1:src.n);
    src.th  = src.th (1:src.n);
    
    Aliasing = 0;  % Aliasing : 1 / Non-aliasing : 0
    kr = 2;         % target kr
    
    noise_on = 1;  % Noise: 1 / no-Noise : 0
    
    %- Spherical Fourier transform
    Nh_max = 3;
    Nh = (Nh_max+1)^2;
    Nhr = (Nh_max+1-Nrd)^2;
    
    [Ys, ns, ms] = sphrm(Nh_max,src.phi,src.th,'complex');
    Ys = Ys.'; % src x nm
    
    %% Microphone configuration
    
    load('pos_32.mat');
    mic.r = 0.075;
    mic.pos = pos_32; % microphone position
    mic.n = size(mic.pos,2);
    
    Fs = 8000;
    k = kr/mic.r;     % target k
    freq = 343*k/(2*pi); % target frequency
    
    %- SPH of microphones
    Xv = mic.pos(1,:)';    Yv = mic.pos(2,:)';    Zv = mic.pos(3,:)';
    [PHIv,THv,Rv] = cart2sphc(Xv,Yv,Zv);
    [Ym, nm, mm] = sphrm(Nh_max,PHIv,THv,'complex');
    
    w = Ym'*pinv(Ym*Ym')*[sqrt(4*pi) zeros(1,length(nm)-1)]'; % quadrature weight
    Gamma_w = diag(w);
      
    % - Encoding matrix
    Yenc = (Gamma_w * Ym').';         % encoding matrix Nmic x Nharm
    
    %- Recurrence coefficients (for spherical harmonic smoothing)
    Wnv = recurcoef(Nh_max,'h-',1); % generate horizontal coeff W
    Wpv = recurcoef(Nh_max,'h+',1); % generate horizontal coeff W
    Vv  = recurcoef(Nh_max,'v',1);  % generate vertical coeff V
    
    % For Aliasing analysis
    Nh_max_tilde = 20;
    [Ys_tilde, ns_tilde, ms_tilde] = sphrm(Nh_max_tilde,src.phi,src.th,'complex');
    Ys_tilde = Ys_tilde.';
    [Ym_tilde, nm_tilde, mm_tilde] = sphrm(Nh_max_tilde,PHIv,THv,'complex');
    Ym_tilde = Ym_tilde.';
    
    
    %% frequency config
    
    Nfft = Fs; 
    c = 343;
    ka = kr;
    
    bmn_ka  = (ka^-2*((1i).^(nm(:)'-1)))./conj(sphr(ka,nm,'dh'))*(4*pi);     % radial function (inNbuf/2 x prNn+1)
    Binv    = diag(1./bmn_ka);
    bmn_ka_ali  = (ka^-2 * ((1i).^(nm_tilde(:)'-1)))./conj(sphr(ka,nm_tilde,'dh'))*(4*pi);     % radial function (inNbuf/2 x prNn+1)
    
    reg = 0;
    
    bEQf     = exp(-1i*angle(bmn_ka))./(abs(bmn_ka) + reg); %*(abs(bmn_norm(1))+sqrt(lambda));	% frequency domain EQ
    bEQspec = diag(bEQf);
    
    %% Simulation start
    % 400 128
    navg     = 1;
    snapshot = 300;
    % SNR_0      = [0 5 10 15 20 25 30];
    SNR_0      = [30];
    
    fax = (0:Nfft-1)/Nfft*Fs;
    [~,idx_freq] = min(abs(fax - freq));
    
    tic
    
    for uu = 1 : length(SNR_0)
        
        SNR = SNR_0(uu);
        
        
        PW_noise0 = 10^(-SNR/10) * src.n / mic.n;
        
        for jj = 1 : navg
            
            
            Rsph = zeros(Nh,Nh);
            Rs     = zeros(src.n,src.n);
            
            for kk = 1 : snapshot
                [jj,kk]
                %- signal generation
                sig0      = randn(src.n,Nfft);
                sig0_fft  = fft(sig0,Nfft,2) / Nfft * 2;
                sig0_fft_trg = sig0_fft(:,idx_freq);
                
                %- noise generation
                noise     = sqrt(PW_noise0) * randn(mic.n,Nfft);
                noise_fft = fft(noise,Nfft,2) / Nfft * 2;         
               
                %- Non-aliasing case
                if Aliasing == 0
                    p_nm_ali = sig0_fft_trg.'  * conj(Ys);
                    Rs = Rs + sig0_fft_trg(:)*sig0_fft_trg(:)';
                    
                    
                elseif Aliasing == 1
                    
                    %- Aliasing case
                    p_ali      = Ym_tilde * diag(bmn_ka_ali) * Ys_tilde' * sig0_fft_trg;
                    p_nm_ali   = bEQspec * Yenc * p_ali;
                    
                end
                
                %- Spherical harmonic measurements
                
                Amnf_noise = bEQspec * Yenc * noise_fft(:,idx_freq);
                Amnf       = p_nm_ali(:) + Amnf_noise(:); % with noise
                
                
                if noise_on == 0
                    Amnf       = p_nm_ali(:); % without noise
                end
                
                %                 Rsph = Rsph + Amnf(:)*Amnf(:)';
                %                 [Inten,Aug]   = sphsmooth_new_intensity(Amnf,Wnv,Wpv,Vv,Nh_max);
                
                [IV_unit(:,kk),IV(:,kk)]  = sphsmooth_new_intensity(Amnf(:),Wnv,Wpv,Vv,Nh_max);
                
            end
            
            %             Rs     = Rs/snapshot;
            %             Rsph = Rsph/snapshot;
            %             Rsph = Ys'*diag(ones(src.n,1))*Ys;
            
            %             [Inten,Aug]   = sphsmooth_new_intensity_R(Rsph,Wnv,Wpv,Vv,Nh_max,2);
            
            
            %             [az1,elev1,~] = cart2sphc(IV(1),IV(2),IV(3));
            %             [az2,elev2,~] = cart2sphc(Inten{2}(1),Inten{2}(2),Inten{2}(3));
            
            %             az_degree1 = az1*180/pi;
            %             elev_degree1 = elev1*180/pi;
            
            %             az_degree2 = az2*180/pi;
            %             elev_degree2 = elev2*180/pi;
            
            %             candi_az = az_degree+180;
            %             candi_elev = 180-elev_degree;
            
            %             [src.phi src.th]*180/pi
            %             [az_degree1 elev_degree1]
            %             [az_degree2 elev_degree2]
            %             az_degree_buf = [az_degree_buf az_degree1];
            %             elev_degree_buf = [elev_degree_buf elev_degree1];
            
        end
        
    end
    
    
end
%%
grid_resol = 1;
grid_dirs = grid2dirs(grid_resol,grid_resol,0,0); % Grid of directions to evaluate DoA estimation

[I_hist1, est_dirs_iv] = sphIntensityHist(IV_unit.', grid_dirs, src.n);
est_dirs_iv = est_dirs_iv*180/pi

% plots results
hf1 = figure(1);
hf1.Position = [284 287 719 574];
plotDirectionalMapFromGrid(I_hist1, grid_resol, grid_resol, [], 0, 0);
src_dirs_deg = [src.phi pi/2-src.th]*180/pi
line_args = {'linestyle','none','marker','o','color','r', 'linewidth',1.5,'markersize',12};
line(src_dirs_deg(:,1), src_dirs_deg(:,2), line_args{:});
line_args = {'linestyle','none','marker','x','color','r', 'linewidth',1.5,'markersize',12};
line(est_dirs_iv(:,1), est_dirs_iv(:,2), line_args{:});
xlabel('Azimuth (deg)'), ylabel('Elevation (deg)'), title('Intensity DoA with RBIV, o: true directions, x: estimated')

% [I_hist2, est_dirs_iv2] = sphIntensityHist(IV.', grid_dirs, src.n);
% est_dirs_iv2 = est_dirs_iv2*180/pi
% 
% % plots results
% hf2 = figure(3);
% hf2.Position = [284 287 719 574];
% plotDirectionalMapFromGrid(I_hist2, grid_resol, grid_resol, [], 0, 0);
% src_dirs_deg = [src.phi pi/2-src.th]*180/pi
% line_args = {'linestyle','none','marker','o','color','r', 'linewidth',1.5,'markersize',12};
% line(src_dirs_deg(:,1), src_dirs_deg(:,2), line_args{:});
% line_args = {'linestyle','none','marker','x','color','r', 'linewidth',1.5,'markersize',12};
% line(est_dirs_iv2(:,1), est_dirs_iv2(:,2), line_args{:});
% xlabel('Azimuth (deg)'), ylabel('Elevation (deg)'), title('Intensity DoA with RBIV, o: true directions, x: estimated')
