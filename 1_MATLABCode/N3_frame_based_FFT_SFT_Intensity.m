clear

addpath myfun speech old_data

%% 1. Load 32 channel speech data
% load Filtered_Speech_6order.mat 
% [src_signal,Fs_signal] = audioread('ar-01.wav');

%% 2. Initialize variables

vect = @(x) reshape(x,numel(x),1); % reshape vector form  
dB10 = @(x) 10*log10(abs(x)); % 10 dB 

bfig = 0;

Nh_max = 3;
Nhrm = (Nh_max+1)^2;
Analysis_ms = 20;
Fs = 48000;
Nframe = Fs*Analysis_ms/1000;
Nfft = Nframe;
Nhop = Nframe/2;
% DataSize = length(FilteredSignal(:,1));
% AnalNum = floor(DataSize/Nhop)-1;
HanWin  = hann(Nframe,'periodic');
HanWin_32 = repmat(HanWin,1,32);

load Eigenmike_ph_th.mat 
mic_dirs_rad = mic_dirs_deg/180*pi;
mic_r = 0.042;
mic_n = 32;

fax = (0:Nfft/2)/Nfft*Fs;
c = 343;
kr = fax*2*pi/c*mic_r;

%- SPH of sources
for ii = 1:72
[Ys{ii}, ~, ~] = sphrm(Nh_max,(ii-1)*pi/36,pi/2,'complex');
end

%- SPH of microphones
[Ym, nm, mm] = sphrm(Nh_max,mic_dirs_rad(:,1),mic_dirs_rad(:,2),'complex');
Yenc = Ym'*pinv(Ym*Ym');

%- Radial function
bmn_ka  = (kr(:).^-2*((1i).^(nm(:)'-1)))./conj(sphr(kr,nm,'dh'))*(4*pi);     % radial function (inNbuf/2 x prNn+1)
bmn_ka(1,:) = abs(bmn_ka(2,:));

%- Design EQ filter (J. Daniel type regularization -> frequency dependent, no spatial or temp smoothing)
% AES: 3D Sound Field Recording with Higher Order Ambisonics -
%      Objective Measuremnets and validation of spherical Microphone
maxG_dB = 20;                                      % maximum noise amplification in dB
maxG = sqrt(mic_n)*10^(maxG_dB/20);
lambda = (1-sqrt(1-1/maxG^2))/(1+sqrt(1-1/maxG^2)); % Tikhonov reg. param
bmn_norm = bmn_ka/(4*pi);
bEQf     = exp(-1i*angle(bmn_norm))./(abs(bmn_norm) + sqrt(lambda))*(1+sqrt(lambda)); % frequency domain EQ
% bEQf     = exp(-1i*angle(bmn_ka))./abs(bmn_ka); % frequency domain EQ

bEQspec = [bEQf(1:end-1,:); real(bEQf(end,:)); flipud(conj(bEQf(2:end-1,:)))];

%- Recurrence coefficients (for spherical harmonic smoothing)
Wnv = recurcoef(Nh_max,'h-',1); % generate horizontal coeff W
Wpv = recurcoef(Nh_max,'h+',1); % generate horizontal coeff W
Vv  = recurcoef(Nh_max,'v',1);  % generate vertical coeff V

%-  Prediction points
Np_th  = 100;
Np_phi = 100;

phi_p = (0:Np_phi-1)/Np_phi*2*pi;
th_p = (0:Np_th-1)/Np_th*pi;

[PHIp,THp]= ndgrid(phi_p,th_p);
PHIv_p = vect(PHIp);
THv_p  = vect(THp);

[Ymn_p, ~, ~] = sphrm(Nh_max,PHIv_p,THv_p,'complex');  % harm x mic pos

%% Drawing graphs

if bfig
    Hf1 = figure(1);
    clf

    beampw      = zeros(Np_th,Np_phi);
    beampwtmp   = zeros(1,Np_th*Np_phi);

    set(gcf,'GraphicsSmoothing','off');
    Hsr_beam    = imagesc(phi_p/pi*180,th_p/pi*180,beampw);

    hold on
    xlabel('Azimuth (degree)','FontSize',12,'FontWeight','bold');
    ylabel('Elavation (degree)','FontSize',12,'FontWeight','bold');
    title('Beampower (MVDR)');
    set(gca,'ydir','normal');
    Hcb = colorbar;
    colormap('jet');
end

%% 3. Frame by frame analysis
IV_room_temp = zeros(1,3);
IV_room1 = zeros(AnalNum,Nfft/2);
IV_room2 = zeros(AnalNum,Nfft/2);
IV_room3 = zeros(AnalNum,Nfft/2);
IV_room4 = zeros(AnalNum,Nfft/2);

IV_free_temp = zeros(1,3);
IV_free1 = zeros(AnalNum,Nfft/2);
IV_free2 = zeros(AnalNum,Nfft/2);
IV_free3 = zeros(AnalNum,Nfft/2);
IV_free4 = zeros(AnalNum,Nfft/2);

tic
parpool(8)
parfor idx_Frame = 1:AnalNum
    
    [idx_Frame]        
    %- Room signal
    FilSig_Frame = FilteredSignal((1:Nframe)+(idx_Frame-1)*Nhop,:);
    FilSig_Frame_fft = fft(FilSig_Frame.*HanWin_32,Nfft);
    
    %- Free field signal
    src_signal_Frame = src_signal((1:Nframe)+(idx_Frame-1)*Nhop);
    src_signal_Frame_fft = fft(src_signal_Frame.*HanWin,Nfft);
    
    %- SFT Encoding
    FilSig_Frame_fft_SFT = FilSig_Frame_fft * Yenc;
    
    %- Bn(kr) normalization    
    FilSig_Frame_fft_SFT_anm = FilSig_Frame_fft_SFT .* bEQspec;
    src_signal_Frame_fft_anm = src_signal_Frame_fft * Ys';
    
    %- Intensity vector extraction
    
    
    for ff = 1:480
        IV_room_temp = sphsmooth_new_intensity(FilSig_Frame_fft_SFT_anm(ff,:),Wnv,Wpv,Vv);
        IV_room1(idx_Frame,ff) = IV_room_temp(1);
        IV_room2(idx_Frame,ff) = IV_room_temp(2);
        IV_room3(idx_Frame,ff) = IV_room_temp(3);
        IV_room4(idx_Frame,ff) = abs(FilSig_Frame_fft_SFT_anm(ff,1));
        
        IV_free_temp = sphsmooth_new_intensity(src_signal_Frame_fft_anm(ff,:),Wnv,Wpv,Vv);
        IV_free1(idx_Frame,ff) = IV_free_temp(1);
        IV_free2(idx_Frame,ff) = IV_free_temp(2);
        IV_free3(idx_Frame,ff) = IV_free_temp(3);
        IV_free4(idx_Frame,ff) = abs(src_signal_Frame_fft_anm(ff,1));
    end
    
    %% For validation
    % MVDR kr(27) == 1.002, kr(53) == 2.004
    
    % Covariance matrix (frequency averaging)
    
%     Cvm = zeros(Nhrm);
%     Cvm_temp = zeros(Nhrm);
%     prRspec = 0.1;
%     fcount = 0;
%     kr_rng = [27 53];
%     for ii = kr_rng(1):kr_rng(2) % for each frequency
%         Cvm_temp = (1-prRspec)*Cvm_temp + prRspec*(FilSig_Frame_fft_SFT_anm(ii,:).'*...
%                                           conj(FilSig_Frame_fft_SFT_anm(ii,:))); %covariance between channels
%         Cvm = Cvm + Cvm_temp;
%     end
%     
%     Cvm = Cvm/(kr_rng(2)-kr_rng(1)+1); % among frequency correlation mean value b.h.
%     Cvm_inv  = pinv(Cvm);
% %     [UU,~] = svd(Cvm);
% %     UU_noise = UU(:,2:end);
%     
%     % Beampower calculation
%     for jj=1:(Np_phi*Np_th)
% %         beampwtmp(jj) = Ymn_p(:,jj).'*Cvm*conj(Ymn_p(:,jj)); %DAS
%         beampwtmp(jj) = (Ymn_p(:,jj).' * Cvm_inv * conj(Ymn_p(:,jj))); % MVDR
% %         beampwtmp(jj) = 1/(abs(Ymn_p(:,jj).' * (UU_noise * UU_noise') * conj(Ymn_p(:,jj)))); % MUSIC
%     end
%     
%     beampw = reshape(dB10(beampwtmp),Np_phi,Np_th).'; % reshape the matrix form
%     set(Hsr_beam,'CData',beampw);      
%     drawnow limitrate;
    
    
end
toc
% save('IV_RGBA_noreg.mat','IV_room','IV_free');


