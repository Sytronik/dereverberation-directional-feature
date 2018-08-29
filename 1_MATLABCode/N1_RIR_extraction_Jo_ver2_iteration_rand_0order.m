clear
close all

addpath SMIR-Generator-master  myfun polarch-Spherical-Array-Processing-4df5f23

load('Eigenmike_ph_th.mat')
mic_dirs_rad = mic_dirs_deg/180*pi;
mic_dirs_rad(:,2) = pi/2-mic_dirs_rad(:,2);

%% Initialize variables

Fs = 16000;
c = 343;                            % Sound velocity (m/s)
N_harm = 30;                        % Maximum order of harmonics to use in SHD
K = 1;                              % Oversampling factor
HP = 0;                             % Optional high pass filter (0/1)
src_type = 'o';                     % Directional source type ('o','c','s','h','b')
L = [10.52 7.1 3.02];                   % Room dimensions (x,y,z) in m

order = 0;                          % Reflection order (-1 is maximum reflection order)
refl_coeff_ang_dep = 0;             % Real reflection coeff(0) or angle dependent reflection coeff(1)
beta = 0.7*ones(1,6);                         % Reflection coeff

sphRadius = 0.042;                  % Radius of the spherical microphone array (m)
sphType = 'rigid';                  % Type of sphere (open/rigid)
V = L(1)*L(2)*L(3);
alpha = ((1-beta(1)^2)+(1-beta(2)^2))*L(2)*L(3) + ...
    ((1-beta(3)^2)+(1-beta(4)^2))*L(1)*L(3) + ...
    ((1-beta(5)^2)+(1-beta(6)^2))*L(1)*L(2);
TR = 24*log(10.0)*V/(c*alpha);
if (TR < 0.128)
    TR = 0.128;
end

load Uniformly_random_spk_mic_pos.mat

% nsample = ceil(TR * Fs)+1;            %Length of IR
nsample = Fs;
Nrand = 100;
% ph_ax = 0;
for xx = 1:49
    
    s_ = s{xx}                    % Source location(s) (x,y,z) in m
    sphLocation_ = sphLocation{xx}       % Receiver location (x,y,z) in m
    
%     [src_ang(1),src_ang(2)] = mycart2sph(s{xx}(1)-sphLocation(1),s{xx}(2)-sphLocation(2),s{xx}(3)-sphLocation(3)); % Towards the receiver
    [src_ang(1),src_ang(2)] = mycart2sph(sphLocation_(1)-s_(1),sphLocation_(2)-s_(2),sphLocation_(3)-s_(3)); % Towards the receiver az elev
    src_ang*180/pi
    
    [mToS_ang(1),mToS_ang(2)] = mycart2sph(s_(1)-sphLocation_(1),s_(2)-sphLocation_(2),s_(3)-sphLocation_(3)); % Towards the speaker
    mToS_ang*180/pi

    % %- Spherical Fourier transform
    Nh_max = 3;
    % Nh = (Nh_max+1)^2;
    [Ys, ns, ms] = sphrm(Nh_max,mToS_ang(1),mToS_ang(2),'complex');
    
    %
    %
    % [Ym, nm, mm] = sphrm(Nh_max,mic_dirs_rad(:,1),mic_dirs_rad(:,2),'complex');
    % Yenc = Ym'*pinv(Ym*Ym');
    %
    % %% source settings
    % 
    % src.n = 1;
    % src.signal = audioread('S_01_01.wav');
    % 
    % Anm_free = Ys * src.signal(:).';

    %% 2. Room SHS modeling 

    tic
    [h1, H1] = smir_generator(c, Fs, sphLocation_, s_, L, beta,...
                                            sphType, sphRadius, mic_dirs_rad, N_harm, nsample, K,...
                                            order, refl_coeff_ang_dep, HP, src_type, src_ang);
    toc
    h1=h1(:,1:end/2);
%     RIR = h1
%     save('RIR_0deg_beta0.8.mat','RIR')
    save(['Anm_FIR_0order_Rand_' num2str((xx-1)*5)  'azi_.mat'],'h1','H1','Fs','sphLocation','s','L','beta',...
                                            'sphType', 'sphRadius', 'mic_dirs_deg', 'N_harm', 'nsample', 'K',...
                                            'order', 'refl_coeff_ang_dep', 'HP', 'src_type', 'src_ang','Ys','mToS_ang');
end



