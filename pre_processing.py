import pdb
# pdb.set_trace()

# import math
import numpy as np
import scipy as sc
import scipy.io as scio

import matplotlib.pyplot as plt

import librosa

import os
import time
from glob import glob
import soundfile as sf

def seltriag(Ain, nrord, shft):
    # pdb.set_trace()
    N = int(np.ceil(np.sqrt(Ain.size))-1)
    idx = 0
    len_new = (N-nrord+1)**2
    Aout = np.zeros(len_new, dtype=complex)
    for ii in range(N-nrord+1):
        for jj in range(-ii,ii+1):
            n=shft[0]+ii; m=shft[1]+jj
            if -n <= m and m <= n and 0 <= n and n <= N and \
                                                        m+n*(n+1) < Ain.size:
                Aout[idx] = Ain[m+n*(n+1)]
            idx += 1
    return Aout

def get_intensity(Asv, Wnv, Wpv, Vv):
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

    return 1/2*np.real([D_x, D_y, D_z])

dir_speech = './speech/data/lisa/data/timit/raw/TIMIT/TRAIN/'
dir_IV = './IV/TRAIN/'
if not os.path.exists(dir_IV):
    os.makedirs(dir_IV)

#RIR Data
RIRmat = scio.loadmat('./1_MATLABCode/RIR.mat', variable_names = 'RIR')
RIR = np.array(RIRmat['RIR'])
RIR = RIR.transpose((2, 0, 1)) #72 x 32 x 48k
N_loc, N_ch, L_RIR = RIR.shape

#SFT Data
sph_mat = scio.loadmat('./1_MATLABCode/sph_data.mat',
                        variable_names=['bEQspec','Yenc','Ys','Wnv','Wpv','Vv'])
bEQspec = np.array(sph_mat['bEQspec']).T
Yenc = np.array(sph_mat['Yenc']).T
Ys = np.array(sph_mat['Ys']).reshape(-1)
Wnv = np.array(sph_mat['Wnv']).reshape(-1)
Wpv = np.array(sph_mat['Wpv']).reshape(-1)
Vv = np.array(sph_mat['Vv']).reshape(-1)

N_speech = 0
fs = 48000
fs_original = 0
Lwin_ms = 20.
Lframe = int(fs*Lwin_ms/1000)
Nfft = Lframe
Lhop = int(Lframe/2)
win = sc.signal.hamming(Lframe, sym=False)
for folder, _, _ in os.walk(dir_speech):
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
        print(file)

        len_free = resampled.shape[0]
        Nframe_free = int(np.floor(len_free/Lhop)-1)

        len_room = len_free+L_RIR-1
        Nframe_room = int(np.floor(len_room/Lhop)-1)
        for i_loc in range(N_loc):
            t_start = time.time()
            # RIR Filtering
            filtered = np.zeros((N_ch, len_room))
            for i_ch in range(N_ch):
                filtered[i_ch] = sc.signal.fftconvolve(resampled, RIR[i_loc, i_ch])

            # Free-field Intensity Vector Image
            IV_free = np.zeros((int(Nfft/2), Nframe_free, 4))
            for i_frame in range(Nframe_free):
                i_sample = i_frame*Lhop

                frame_free = resampled[i_sample+np.arange(Lframe)]
                fft_free = np.fft.fft(frame_free*win, n=Nfft)
                anm_free = np.outer(Ys[i_loc].reshape(-1), fft_free.T)

                for i_freq in range(int(Nfft/2)):
                    IV_free[i_freq, i_frame, 0:3] \
                            = get_intensity(anm_free[:, i_freq], Wnv, Wpv, Vv)
                    IV_free[i_freq, i_frame, 3] = np.abs(anm_free[0, i_freq])

            # Room Intensity Vector Image
            IV_room = np.zeros((int(Nfft/2), Nframe_room, 4))
            for i_frame in range(Nframe_room):
                i_sample = i_frame*Lhop

                frame_room = filtered[:, i_sample+np.arange(Lframe)]
                fft_room = np.fft.fft(frame_room*win, n=Nfft)
                anm_room = Yenc @ fft_room * bEQspec

                for i_freq in range(int(Nfft/2)):
                    IV_room[i_freq, i_frame, 0:3] \
                            = get_intensity(anm_room[:, i_freq], Wnv, Wpv, Vv)
                    IV_room[i_freq, i_frame, 3] \
                            = np.abs(anm_room[0, i_freq])

            file_id='%06d_%2d'%(N_speech, i_loc)
            np.save(os.path.join(dir_IV,file_id+'_room.npy'), IV_room)
            np.save(os.path.join(dir_IV,file_id+'_free.npy'), IV_free)
            # scio.savemat(os.path.join(dir_IV,file_id+'.mat'), {'IV_room':IV_room, 'IV_free':IV_free}, appendmat=False)

            print(file_id)
            print('%.3f sec'%(time.time()-t_start))
            pdb.set_trace()


print('Number of data: {}'.format(N_speech))
print('Sample Rate: {}'.format(fs))
print('Number of source location: {}'.format(N_loc))
print('Number of microphone channel: {}'.format(N_ch))
