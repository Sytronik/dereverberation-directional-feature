import pdb

# import math
import numpy as np
import scipy as sc
import scipy.io as scio
import librosa
import matplotlib.pyplot as plt

import os
import time
from glob import glob

import soundfile as sf

from joblib import Parallel, delayed
import multiprocessing

def main():
    dir_speech = './speech/data/lisa/data/timit/raw/TIMIT/TRAIN/'
    dir_IV = './IV/TRAIN/'
    if not os.path.exists(dir_IV):
        os.makedirs(dir_IV)

    #RIR Data
    RIRmat = scio.loadmat('./1_MATLABCode/RIR.mat', variable_names = 'RIR')
    RIR = np.array(RIRmat['RIR'])
    RIR = RIR.transpose((2, 0, 1)) #72 x 32 x 48k
    Nloc, Nch, L_RIR = RIR.shape

    #SFT Data
    sph_mat = scio.loadmat('./1_MATLABCode/sph_data.mat',
                            variable_names=['bEQspec','Yenc','Ys','Wnv','Wpv','Vv'])
    bEQspec = np.array(sph_mat['bEQspec']).T
    Yenc = np.array(sph_mat['Yenc']).T
    Ys = np.array(sph_mat['Ys']).reshape(-1)
    Wnv = np.array(sph_mat['Wnv']).reshape(-1)
    Wpv = np.array(sph_mat['Wpv']).reshape(-1)
    Vv = np.array(sph_mat['Vv']).reshape(-1)

    Nwavfile = 0
    fs = 16000
    # fs_original = 0
    Lwin_ms = 20.
    Lframe = int(fs*Lwin_ms/1000)
    Nfft = Lframe
    Lhop = int(Lframe/2)
    win = sc.signal.hamming(Lframe, sym=False)
    # pdb.set_trace()
    num_cores = multiprocessing.cpu_count()

    for folder, _, _ in os.walk(dir_speech):
        files = glob(os.path.join(folder, '*_converted.wav'))
        if files is None:
            continue
        for file in files:
            Nwavfile += 1

            #File Open & Resample
            data, _ = sf.read(file)
            # data = librosa.core.resample(data, fs_original, fs)
            print(file)

            Ndata_free = data.shape[0]
            Nframe_free = int(np.floor(Ndata_free/Lhop)-1)

            Ndata_room = Ndata_free+L_RIR-1
            Nframe_room = int(np.floor(Ndata_room/Lhop)-1)
            t_start = time.time()

            # Parallel(n_jobs=num_cores)(
            #     delayed(save_IV)(data, RIR[i_loc],
            #         Nframe_free, Ndata_room, Nframe_room, Nloc, Nch, Nwavfile, Nfft,
            #         win, Lframe, Lhop,
            #         Yenc, Ys[i_loc], bEQspec,
            #         Wnv, Wpv, Vv,
            #         dir_IV, i_loc)
            #     for i_loc in range(Nloc)
            # )
            for i_loc in range(Nloc):
                save_IV(data, RIR[i_loc],
                        Nframe_free, Ndata_room, Nframe_room, Nloc, Nch, Nwavfile, Nfft,
                        win, Lframe, Lhop,
                        Yenc, Ys[i_loc], bEQspec,
                        Wnv, Wpv, Vv,
                        dir_IV, i_loc)
                pdb.set_trace()

            print('%.3f sec'%(time.time()-t_start))
    print('Number of data: {}'.format(Nwavfile))
    # print('Sample Rate: {}'.format(fs))
    # print('Number of source location: {}'.format(Nloc))
    # print('Number of microphone channel: {}'.format(Nch))

def seltriag(Ain, nrord, shft):
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

def calcIntensity(Asv, Wnv, Wpv, Vv):
    Asrv = seltriag(Asv, 1, [0, 0])

    Aug_1 = Asrv
    Aug_2 =  seltriag(Wpv, 1, [1, -1])*seltriag(Asv,1,[1, -1]) - seltriag(Wnv, 1, [0, 0])*seltriag(Asv, 1, [-1, -1])# +exp(phi)
    Aug_3 =  seltriag(Wpv, 1, [0, 0]) *seltriag(Asv,1,[-1, 1]) - seltriag(Wnv, 1, [1, 1])*seltriag(Asv, 1, [1, 1])  # -exp(phi)
    Aug_4 =  seltriag(Vv , 1, [0, 0]) *seltriag(Asv,1,[-1, 0]) + seltriag(Vv , 1, [1, 0])*seltriag(Asv, 1, [1, 0])
    partial_x = (Aug_2+Aug_3)/2
    partial_y = (Aug_2-Aug_3)/2j
    partial_z = Aug_4

    D_x = np.dot(Aug_1.conj(), partial_x)
    D_y = np.dot(Aug_1.conj(), partial_y)
    D_z = np.dot(Aug_1.conj(), partial_z)

    return 1/2*np.real([D_x, D_y, D_z])

def save_IV(data, RIR,
            Nframe_free, Ndata_room, Nframe_room, Nloc, Nch, Nwavfile, Nfft,
            win, Lframe, Lhop,
            Yenc, Ys, bEQspec,
            Wnv, Wpv, Vv,
            dir_IV, i_loc):
    # RIR Filtering
    filtered = np.zeros((Nch, Ndata_room))
    for i_ch in range(Nch):
        filtered[i_ch] = sc.signal.fftconvolve(data, RIR[i_ch])

    # Free-field Intensity Vector Image
    IV_free = np.zeros((int(Nfft/2), Nframe_free, 4))
    for i_frame in range(Nframe_free):
        i_sample = i_frame*Lhop

        frame_free = data[i_sample+np.arange(Lframe)]
        fft_free = np.fft.fft(frame_free*win, n=Nfft)
        anm_free = np.outer(Ys.reshape(-1).conj(), fft_free.T)

        for i_freq in range(int(Nfft/2)):
            IV_free[i_freq, i_frame, 0:3] \
                    = calcIntensity(anm_free[:, i_freq], Wnv, Wpv, Vv)
            IV_free[i_freq, i_frame, 3] = np.abs(anm_free[0, i_freq])

    # Room Intensity Vector Image
    IV_room = np.zeros((int(Nfft/2), Nframe_room, 4))
    for i_frame in range(Nframe_room):
        i_sample = i_frame*Lhop

        frame_room = filtered[:, i_sample+np.arange(Lframe)]
        fft_room = np.fft.fft(frame_room*win, n=Nfft)
        anm_room = (Yenc @ fft_room) * bEQspec

        for i_freq in range(int(Nfft/2)):
            IV_room[i_freq, i_frame, 0:3] \
                    = calcIntensity(anm_room[:, i_freq], Wnv, Wpv, Vv)
            IV_room[i_freq, i_frame, 3] \
                    = np.abs(anm_room[0, i_freq])

    file_id='%06d_%2d'%(Nwavfile, i_loc)
    # np.save(os.path.join(dir_IV,file_id+'_room.npy'), IV_room)
    # np.save(os.path.join(dir_IV,file_id+'_free.npy'), IV_free)
    scio.savemat(os.path.join(dir_IV,file_id+'.mat'),
                 {'IV_room_py':IV_room, 'IV_free_py':IV_free}, appendmat=False)

    print(file_id)

if __name__ == '__main__':
    main()
