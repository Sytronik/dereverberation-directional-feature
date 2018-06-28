import pdb
# pdb.set_trace()

# import math
import numpy as np
import scipy as sc
import scipy.io as scio

import librosa

import os
from glob import glob
import soundfile as sf

def seltriag(Ain, nrord, shft):
    # pdb.set_trace()
    N = int(np.ceil(np.sqrt(Ain.size))-1)
    idx = 0
    len_new = (N-nrord+1)**2
    Aout = np.zeros(len_new)
    for ii in range(N-nrord+1):
        for jj in range(-ii,ii+1):
            n=shft[0]+ii; m=shft[1]+jj
            if -n <= m and m <= n and 0 <= n and n <= N and \
                                                        m+n*(n+1) < Ain.size:
                Aout[idx] = Ain[m+n*(n+1)]
            idx += 1
    return Aout

def get_intensity(Asv, Wnv, Wpv, Vv, Nh_max):
    Asrv = seltriag(Asv, 1, [0, 0])

    Aug_1 = Asrv
    Aug_2 =  seltriag(Wpv, 1, [1, -1])*seltriag(Asv,1,[1, -1]) - seltriag(Wnv, 1, [0, 0])*seltriag(Asv, 1, [-1, -1])# +exp(phi)
    Aug_3 =  seltriag(Wpv, 1, [0, 0]) *seltriag(Asv,1,[-1, 1]) - seltriag(Wnv, 1, [1, 1])*seltriag(Asv, 1, [1, 1])  # -exp(phi)
    Aug_4 =  seltriag(Vv , 1, [0, 0]) *seltriag(Asv,1,[-1, 0]) + seltriag(Vv , 1, [1, 0])*seltriag(Asv, 1, [1, 0])
    partial_x = (Aug_2+Aug_3)/2
    partial_y = (Aug_2-Aug_3)/2j
    partial_z = Aug_4

    D_x = np.dot(Aug_1, partial_x)
    D_y = np.dot(Aug_1, partial_y)
    D_z = np.dot(Aug_1, partial_z)

    return 1/2*([D_x.real, D_y.real, D_z.real])

speech_dir = './speech/data/lisa/data/timit/raw/TIMIT/TRAIN/'

#RIR Data
RIRmat = scio.loadmat('./1_MATLABCode/RIR_Data/RIR.mat', variable_names = 'RIR')
RIR = np.array(RIRmat['RIR'])
# pdb.set_trace()
RIR = RIR.transpose((2, 0, 1)) #72 x 32 x 48k
N_loc, N_ch = RIR.shape[0:2]

#bEqspec, Yenc Data
sph_mat = scio.loadmat('./11_MATLABCode/sph_data.mat', variable_names='bEQspec\nYenc\nWnv\nWpv\nVv')
bEQspec = np.matrix(sph_mat['bEQspec']).transpose()
Yenc = np.matrix(sph_mat['Yenc']).transpose()

N_speech = 0
fs = 48000
Lwin_ms = 20
Lframe = fs*Lwin_ms/1000
Nfft = Lframe
Lhop = Lframe/2
win = sc.signal.hamming(Lframe, sym=False)
for folder, _, _ in os.walk(speech_dir):
    files = glob(os.path.join(folder, '*_converted.wav'))
    if files is None:
        continue
    for file in files:
        N_speech += 1

        #File Open & Resample
        if fs_original == 0:
            data, fs_original = sf.read(file)
        else:
            data, _ = sf.read(file)
        resampled = librosa.core.resample(data, fs_original, fs)


        length = resampled.shape[0]
        Nframe = np.floor(length/Lhop)-1
        for i_loc in range(N_loc):
            # RIR Filtering
            filtered = np.zeros((N_ch, length))
            for i_ch in range(N_ch):
                filtered[i_ch] = sc.signal.fftconvolve(resampled, RIR[i_loc, i_ch], mode = 'same')

            IV_room = np.zeros((Nframe, Nfft/2, 4))
            IV_free = np.zeros((Nframe, Nfft/2, 4))
            # Intensity Vector Image
            for i_frame = range(Nframe):
                i_sample = i_frame*Lhop

                frame_room = filtered[:, i_sample+range(Lframe)]
                fft_room = np.fft.fft(frame_room*win, n=Nfft)

                frame_free = resampled[:, i_sample+range(Lframe)]
                fft_free = np.fft.fft(frame_free*win, n=Nfft)

                anm_room = np.matmul(Yenc, fft_room) *bEQspec
                anm_free = np.matmul(Ys, fft_free)

                for i_freq = range(Nfft/2):
                    IV_room[i_frame, ff, 0:3] \
                            = get_intensity(anm_room[:, i_freq], Wnv, Wpv, Vv)
                    IV_room[i_frame, ff, 3] \
                            = np.abs(anm_room[0, i_freq])

                    IV_free[i_frame, ff, 0:3] \
                            = get_intensity(anm_free[:, i_freq], Wnv, Wpv, Vv)
                    IV_free[i_frame, ff, 3] = np.abs(anm_free[0, i_freq])


print('Number of data: {}'.format(N_files))
print('Sample Rate: {}'.format(fs))
print('Number of location: {}'.format(N_loc))
print('Number of microphone channel: {}'.format(N_ch))
