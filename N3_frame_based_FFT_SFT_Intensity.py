# Generated with SMOP  0.41
from libsmop import *
# N3_frame_based_FFT_SFT_Intensity.m

    clear
    addpath('myfun','speech','old_data')
    ## 1. Load 32 channel speech data
# load Filtered_Speech_6order.mat
    src_signal,Fs_signal=audioread('ar-01.wav',nargout=2)
# N3_frame_based_FFT_SFT_Intensity.m:7
    ## 2. Initialize variables
    
    vect=lambda x=None: reshape(x,numel(x),1)
# N3_frame_based_FFT_SFT_Intensity.m:11
    
    dB10=lambda x=None: dot(10,log10(abs(x)))
# N3_frame_based_FFT_SFT_Intensity.m:12
    
    bfig=0
# N3_frame_based_FFT_SFT_Intensity.m:14
    Nh_max=3
# N3_frame_based_FFT_SFT_Intensity.m:16
    Nhrm=(Nh_max + 1) ** 2
# N3_frame_based_FFT_SFT_Intensity.m:17
    Analysis_ms=20
# N3_frame_based_FFT_SFT_Intensity.m:18
    Fs=48000
# N3_frame_based_FFT_SFT_Intensity.m:19
    Nframe=dot(Fs,Analysis_ms) / 1000
# N3_frame_based_FFT_SFT_Intensity.m:20
    Nfft=copy(Nframe)
# N3_frame_based_FFT_SFT_Intensity.m:21
    Nhop=Nframe / 2
# N3_frame_based_FFT_SFT_Intensity.m:22
    DataSize=length(FilteredSignal(arange(),1))
# N3_frame_based_FFT_SFT_Intensity.m:23
    AnalNum=floor(DataSize / Nhop) - 1
# N3_frame_based_FFT_SFT_Intensity.m:24
    HanWin=hann(Nframe,'periodic')
# N3_frame_based_FFT_SFT_Intensity.m:25
    HanWin_32=repmat(HanWin,1,32)
# N3_frame_based_FFT_SFT_Intensity.m:26
    # load Eigenmike_ph_th.mat
    mic_dirs_rad=dot(mic_dirs_deg / 180,pi)
# N3_frame_based_FFT_SFT_Intensity.m:29
    mic_r=0.042
# N3_frame_based_FFT_SFT_Intensity.m:30
    mic_n=32
# N3_frame_based_FFT_SFT_Intensity.m:31
    fax=dot((arange(0,Nfft / 2)) / Nfft,Fs)
# N3_frame_based_FFT_SFT_Intensity.m:33
    c=343
# N3_frame_based_FFT_SFT_Intensity.m:34
    kr=dot(dot(dot(fax,2),pi) / c,mic_r)
# N3_frame_based_FFT_SFT_Intensity.m:35
    #- SPH of sources
    Ys,__,__=sphrm(Nh_max,0,pi / 2,'complex',nargout=3)
# N3_frame_based_FFT_SFT_Intensity.m:38
    #- SPH of microphones
    Ym,nm,mm=sphrm(Nh_max,mic_dirs_rad(arange(),1),mic_dirs_rad(arange(),2),'complex',nargout=3)
# N3_frame_based_FFT_SFT_Intensity.m:42
    Yenc=dot(Ym.T,pinv(dot(Ym,Ym.T)))
# N3_frame_based_FFT_SFT_Intensity.m:43
    #- Radial function
    bmn_ka=dot((dot(ravel(kr) ** - 2,((1j) ** (ravel(nm).T - 1)))) / conj(sphr(kr,nm,'dh')),(dot(4,pi)))
# N3_frame_based_FFT_SFT_Intensity.m:46
    
    bmn_ka[1,arange()]=abs(bmn_ka(2,arange()))
# N3_frame_based_FFT_SFT_Intensity.m:47
    #- Design EQ filter (J. Daniel type regularization -> frequency dependent, no spatial or temp smoothing)
# AES: 3D Sound Field Recording with Higher Order Ambisonics -
#      Objective Measuremnets and validation of spherical Microphone
    maxG_dB=20
# N3_frame_based_FFT_SFT_Intensity.m:52
    
    maxG=dot(sqrt(mic_n),10 ** (maxG_dB / 20))
# N3_frame_based_FFT_SFT_Intensity.m:53
    lambda_=(1 - sqrt(1 - 1 / maxG ** 2)) / (1 + sqrt(1 - 1 / maxG ** 2))
# N3_frame_based_FFT_SFT_Intensity.m:54
    
    bmn_norm=bmn_ka / (dot(4,pi))
# N3_frame_based_FFT_SFT_Intensity.m:55
    bEQf=dot(exp(dot(- 1j,angle(bmn_norm))) / (abs(bmn_norm) + sqrt(lambda_)),(1 + sqrt(lambda_)))
# N3_frame_based_FFT_SFT_Intensity.m:56
    
    # bEQf     = exp(-1i*angle(bmn_ka))./abs(bmn_ka); # frequency domain EQ
    
    bEQspec=concat([[bEQf(arange(1,end() - 1),arange())],[real(bEQf(end(),arange()))],[flipud(conj(bEQf(arange(2,end() - 1),arange())))]])
# N3_frame_based_FFT_SFT_Intensity.m:59
    #- Recurrence coefficients (for spherical harmonic smoothing)
    Wnv=recurcoef(Nh_max,'h-',1)
# N3_frame_based_FFT_SFT_Intensity.m:62
    
    Wpv=recurcoef(Nh_max,'h+',1)
# N3_frame_based_FFT_SFT_Intensity.m:63
    
    Vv=recurcoef(Nh_max,'v',1)
# N3_frame_based_FFT_SFT_Intensity.m:64
    
    #-  Prediction points
    Np_th=100
# N3_frame_based_FFT_SFT_Intensity.m:67
    Np_phi=100
# N3_frame_based_FFT_SFT_Intensity.m:68
    phi_p=dot(dot((arange(0,Np_phi - 1)) / Np_phi,2),pi)
# N3_frame_based_FFT_SFT_Intensity.m:70
    th_p=dot((arange(0,Np_th - 1)) / Np_th,pi)
# N3_frame_based_FFT_SFT_Intensity.m:71
    PHIp,THp=ndgrid(phi_p,th_p,nargout=2)
# N3_frame_based_FFT_SFT_Intensity.m:73
    PHIv_p=vect(PHIp)
# N3_frame_based_FFT_SFT_Intensity.m:74
    THv_p=vect(THp)
# N3_frame_based_FFT_SFT_Intensity.m:75
    Ymn_p,__,__=sphrm(Nh_max,PHIv_p,THv_p,'complex',nargout=3)
# N3_frame_based_FFT_SFT_Intensity.m:77
    
    ## Drawing graphs
    
    if bfig:
        Hf1=figure(1)
# N3_frame_based_FFT_SFT_Intensity.m:82
        clf
        beampw=zeros(Np_th,Np_phi)
# N3_frame_based_FFT_SFT_Intensity.m:85
        beampwtmp=zeros(1,dot(Np_th,Np_phi))
# N3_frame_based_FFT_SFT_Intensity.m:86
        set(gcf,'GraphicsSmoothing','off')
        Hsr_beam=imagesc(dot(phi_p / pi,180),dot(th_p / pi,180),beampw)
# N3_frame_based_FFT_SFT_Intensity.m:89
        hold('on')
        xlabel('Azimuth (degree)','FontSize',12,'FontWeight','bold')
        ylabel('Elavation (degree)','FontSize',12,'FontWeight','bold')
        title('Beampower (MVDR)')
        set(gca,'ydir','normal')
        Hcb=copy(colorbar)
# N3_frame_based_FFT_SFT_Intensity.m:96
        colormap('jet')
    
    ## 3. Frame by frame analysis
    IV_room_temp=zeros(1,3)
# N3_frame_based_FFT_SFT_Intensity.m:101
    IV_room1=zeros(AnalNum,Nfft / 2)
# N3_frame_based_FFT_SFT_Intensity.m:102
    IV_room2=zeros(AnalNum,Nfft / 2)
# N3_frame_based_FFT_SFT_Intensity.m:103
    IV_room3=zeros(AnalNum,Nfft / 2)
# N3_frame_based_FFT_SFT_Intensity.m:104
    IV_room4=zeros(AnalNum,Nfft / 2)
# N3_frame_based_FFT_SFT_Intensity.m:105
    IV_free_temp=zeros(1,3)
# N3_frame_based_FFT_SFT_Intensity.m:107
    IV_free1=zeros(AnalNum,Nfft / 2)
# N3_frame_based_FFT_SFT_Intensity.m:108
    IV_free2=zeros(AnalNum,Nfft / 2)
# N3_frame_based_FFT_SFT_Intensity.m:109
    IV_free3=zeros(AnalNum,Nfft / 2)
# N3_frame_based_FFT_SFT_Intensity.m:110
    IV_free4=zeros(AnalNum,Nfft / 2)
# N3_frame_based_FFT_SFT_Intensity.m:111
    tic
    parpool(8)
    for idx_Frame in arange(1,AnalNum).reshape(-1):
        concat([idx_Frame])
        #- Room signal
        FilSig_Frame=FilteredSignal((arange(1,Nframe)) + dot((idx_Frame - 1),Nhop),arange())
# N3_frame_based_FFT_SFT_Intensity.m:119
        FilSig_Frame_fft=fft(multiply(FilSig_Frame,HanWin_32),Nfft)
# N3_frame_based_FFT_SFT_Intensity.m:120
        src_signal_Frame=src_signal((arange(1,Nframe)) + dot((idx_Frame - 1),Nhop))
# N3_frame_based_FFT_SFT_Intensity.m:123
        src_signal_Frame_fft=fft(multiply(src_signal_Frame,HanWin),Nfft)
# N3_frame_based_FFT_SFT_Intensity.m:124
        FilSig_Frame_fft_SFT=dot(FilSig_Frame_fft,Yenc)
# N3_frame_based_FFT_SFT_Intensity.m:127
        FilSig_Frame_fft_SFT_anm=multiply(FilSig_Frame_fft_SFT,bEQspec)
# N3_frame_based_FFT_SFT_Intensity.m:130
        src_signal_Frame_fft_anm=dot(src_signal_Frame_fft,Ys.T)
# N3_frame_based_FFT_SFT_Intensity.m:131
        for ff in arange(1,480).reshape(-1):
            IV_room_temp=sphsmooth_new_intensity(FilSig_Frame_fft_SFT_anm(ff,arange()),Wnv,Wpv,Vv)
# N3_frame_based_FFT_SFT_Intensity.m:137
            IV_room1[idx_Frame,ff]=IV_room_temp(1)
# N3_frame_based_FFT_SFT_Intensity.m:138
            IV_room2[idx_Frame,ff]=IV_room_temp(2)
# N3_frame_based_FFT_SFT_Intensity.m:139
            IV_room3[idx_Frame,ff]=IV_room_temp(3)
# N3_frame_based_FFT_SFT_Intensity.m:140
            IV_room4[idx_Frame,ff]=abs(FilSig_Frame_fft_SFT_anm(ff,1))
# N3_frame_based_FFT_SFT_Intensity.m:141
            IV_free_temp=sphsmooth_new_intensity(src_signal_Frame_fft_anm(ff,arange()),Wnv,Wpv,Vv)
# N3_frame_based_FFT_SFT_Intensity.m:143
            IV_free1[idx_Frame,ff]=IV_free_temp(1)
# N3_frame_based_FFT_SFT_Intensity.m:144
            IV_free2[idx_Frame,ff]=IV_free_temp(2)
# N3_frame_based_FFT_SFT_Intensity.m:145
            IV_free3[idx_Frame,ff]=IV_free_temp(3)
# N3_frame_based_FFT_SFT_Intensity.m:146
            IV_free4[idx_Frame,ff]=abs(src_signal_Frame_fft_anm(ff,1))
# N3_frame_based_FFT_SFT_Intensity.m:147
        ## For validation
    # MVDR kr(27) == 1.002, kr(53) == 2.004
        # Covariance matrix (frequency averaging)
        #     Cvm = zeros(Nhrm);
#     Cvm_temp = zeros(Nhrm);
#     prRspec = 0.1;
#     fcount = 0;
#     kr_rng = [27 53];
#     for ii = kr_rng(1):kr_rng(2) # for each frequency
#         Cvm_temp = (1-prRspec)*Cvm_temp + prRspec*(FilSig_Frame_fft_SFT_anm(ii,:).'*...
#                                           conj(FilSig_Frame_fft_SFT_anm(ii,:))); #covariance between channels
#         Cvm = Cvm + Cvm_temp;
#     end
#     
#     Cvm = Cvm/(kr_rng(2)-kr_rng(1)+1); # among frequency correlation mean value b.h.
#     Cvm_inv  = pinv(Cvm);
# #     [UU,~] = svd(Cvm);
# #     UU_noise = UU(:,2:end);
#     
#     # Beampower calculation
#     for jj=1:(Np_phi*Np_th)
# #         beampwtmp(jj) = Ymn_p(:,jj).'*Cvm*conj(Ymn_p(:,jj)); #DAS
#         beampwtmp(jj) = (Ymn_p(:,jj).' * Cvm_inv * conj(Ymn_p(:,jj))); # MVDR
# #         beampwtmp(jj) = 1/(abs(Ymn_p(:,jj).' * (UU_noise * UU_noise') * conj(Ymn_p(:,jj)))); # MUSIC
#     end
#     
#     beampw = reshape(dB10(beampwtmp),Np_phi,Np_th).'; # reshape the matrix form
#     set(Hsr_beam,'CData',beampw);      
#     drawnow limitrate;
    
    toc
    # save('IV_RGBA_noreg.mat','IV_room','IV_free');
    