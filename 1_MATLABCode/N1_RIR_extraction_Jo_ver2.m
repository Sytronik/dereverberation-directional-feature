clear
close all

addpath SMIR-Generator-master  myfun polarch-Spherical-Array-Processing-4df5f23

load('Eigenmike_ph_th.mat')
mic_dirs_rad = mic_dirs_deg/180*pi;
mic_dirs_rad(:,2) = pi/2-mic_dirs_rad(:,2);

%% Initialize variables

Fs = 48000;
c = 343;                            % Sound velocity (m/s)
nsample = Fs*1;                     % Length of desired RIR
N_harm = 30;                        % Maximum order of harmonics to use in SHD
K = 1;                              % Oversampling factor

L = [10.52 7.1 3.02];                   % Room dimensions (x,y,z) in m
sphLocation = [5.56 3.63 1.41];       % Receiver location (x,y,z) in m
s = [2.68 3.63 1.41];                    % Source location(s) (x,y,z) in m

HP = 0;                             % Optional high pass filter (0/1)
src_type = 'o';                     % Directional source type ('o','c','s','h','b')
[src_ang(1),src_ang(2)] = mycart2sph(sphLocation(1)-s(1),sphLocation(2)-s(2),sphLocation(3)-s(3)); % Towards the receiver
src_ang*180/pi

order = 6;                         % Reflection order (-1 is maximum reflection order)
refl_coeff_ang_dep = 0;             % Real reflection coeff(0) or angle dependent reflection coeff(1)
beta = 0.15;                         % Reverbration time T_60 (s)

sphRadius = 0.042;                  % Radius of the spherical microphone array (m)
sphType = 'rigid';                  % Type of sphere (open/rigid)

% %- Spherical Fourier transform
% Nh_max = 3;
% Nh = (Nh_max+1)^2;
% [Ys, ns, ms] = sphrm(Nh_max,src_ang(1),src_ang(2),'complex');
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
[h1, H1] = smir_generator(c, Fs, sphLocation, s, L, beta,...
                                            sphType, sphRadius, mic_dirs_rad, N_harm, nsample, K,...
                                            order, refl_coeff_ang_dep, HP, src_type, src_ang);
toc

save('Anm_FIR_6order.mat','h1','H1','Fs','sphLocation','s','L','beta',...
                                            'sphType', 'sphRadius', 'mic_dirs_deg', 'N_harm', 'nsample', 'K',...
                                            'order', 'refl_coeff_ang_dep', 'HP', 'src_type', 'src_ang');

