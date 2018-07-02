import pdb

# import math
import numpy as np
import cupy as cp
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
    RIR = RIRmat['RIR']
    RIR = RIR.transpose((2, 0, 1)) #72 x 32 x 48k
    RIR = cp.array(RIR)
    Nloc, Nch, L_RIR = RIR.shape

    #SFT Data
    sph_mat = scio.loadmat('./1_MATLABCode/sph_data.mat',
                            variable_names=['bEQspec','Yenc','Ys','Wnv','Wpv','Vv'])
    bEQspec = cp.array(sph_mat['bEQspec']).T
    Yenc = cp.array(sph_mat['Yenc']).T
    Ys_original = sph_mat['Ys'].reshape(-1)
    Ys_np = np.zeros((Ys_original.size,Ys_original[0].size), dtype=complex)
    for ii in range(Ys_original.size):
        Ys_np[ii] = Ys_original[ii].reshape(-1)
    Ys = cp.array(Ys_np)
    Wnv = cp.array(sph_mat['Wnv']).reshape(-1)
    Wpv = cp.array(sph_mat['Wpv']).reshape(-1)
    Vv = cp.array(sph_mat['Vv']).reshape(-1)

    Nwavfile = 0
    fs = 16000
    # fs_original = 0
    Lwin_ms = 20.
    Lframe = int(fs*Lwin_ms/1000)
    Nfft = Lframe
    Lhop = int(Lframe/2)
    win = cp.array(sc.signal.hamming(Lframe, sym=False))
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
            data = cp.array(data)
            # data = librosa.core.resample(data, fs_original, fs)
            print(file)
            Ndata_free = data.shape[0]
            Nframe_free = int(np.floor(Ndata_free/Lhop)-1)

            Ndata_room = Ndata_free+L_RIR-1
            Nframe_room = int(np.floor(Ndata_room/Lhop)-1)


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
                t_start = time.time()
                save_IV(data, RIR[i_loc],
                        Nframe_free, Ndata_room, Nframe_room, Nloc, Nch, Nwavfile, Nfft,
                        win, Lframe, Lhop,
                        Yenc, Ys[i_loc], bEQspec,
                        Wnv, Wpv, Vv,
                        dir_IV, i_loc)
                print('%.3f sec'%(time.time()-t_start))

    print('Number of data: {}'.format(Nwavfile))
    # print('Sample Rate: {}'.format(fs))
    # print('Number of source location: {}'.format(Nloc))
    # print('Number of microphone channel: {}'.format(Nch))

def seltriag(Ain, nrord:int, shft):
    if Ain.ndim == 1:
        Nfreq = 1
    else:
        Nfreq=Ain.shape[1]
    N = int(np.ceil(np.sqrt(Ain.shape[0]))-1)
    idx = 0
    len_new = (N-nrord+1)**2

    Aout = cp.zeros((len_new, Nfreq), dtype=complex)
    for ii in range(N-nrord+1):
        for jj in range(-ii,ii+1):
            n=shft[0]+ii; m=shft[1]+jj
            if -n <= m and m <= n and 0 <= n and n <= N and \
                                                        m+n*(n+1) < Ain.size:
                Aout[idx] = Ain[m+n*(n+1)]
            idx += 1
    return Aout

def calcIntensity(Asv, Wnv, Wpv, Vv):
    Aug_1 = seltriag(Asv, 1, [0, 0])
    Aug_2 = seltriag(Wpv, 1, [1, -1])*seltriag(Asv, 1, [1, -1]) \
            - seltriag(Wnv, 1, [0, 0])*seltriag(Asv, 1, [-1, -1])# +exp(phi)
    Aug_3 = seltriag(Wpv, 1, [0, 0])*seltriag(Asv, 1, [-1, 1]) \
            - seltriag(Wnv, 1, [1, 1])*seltriag(Asv, 1, [1, 1])  # -exp(phi)
    Aug_4 = seltriag(Vv, 1, [0, 0])*seltriag(Asv, 1, [-1, 0]) \
            + seltriag(Vv, 1, [1, 0])*seltriag(Asv, 1, [1, 0])

    D_x = cp.sum(Aug_1.conj()*(Aug_2+Aug_3)/2, axis=0).reshape(-1,1)
    D_y = cp.sum(Aug_1.conj()*(Aug_2-Aug_3)/2j, axis=0).reshape(-1,1)
    D_z = cp.sum(Aug_1.conj()*Aug_4, axis=0).reshape(-1,1)

    return 1/2*cp.real(cp.concatenate((D_x, D_y, D_z), axis=1))

def save_IV(data, RIR,
            Nframe_free, Ndata_room, Nframe_room, Nloc, Nch, Nwavfile, Nfft,
            win, Lframe, Lhop,
            Yenc, Ys, bEQspec,
            Wnv, Wpv, Vv,
            dir_IV, i_loc):
    # RIR Filtering
    filtered = cp.array(sc.signal.fftconvolve(data.reshape(1,-1).get(), RIR.get()))
    print('filtered')
    
    # Free-field Intensity Vector Image
    IV_free = cp.zeros((int(Nfft/2), Nframe_free, 4))
    for i_frame in range(Nframe_free):
        i_sample = i_frame*Lhop

        fft_free = cp.fft.fft(data[i_sample+np.arange(Lframe)]*win, n=Nfft)
        anm_free = cp.outer(Ys.conj(), fft_free.T)

        IV_free[:, i_frame, 0:3] \
                = calcIntensity(anm_free[:, 0:int(Nfft/2)], Wnv, Wpv, Vv)
        IV_free[:, i_frame, 3] = cp.abs(anm_free[0, 0:int(Nfft/2)])
    print('IV_free calculated')

    # Room Intensity Vector Image
    IV_room = cp.zeros((int(Nfft/2), Nframe_room, 4))
    for i_frame in range(Nframe_room):
        i_sample = i_frame*Lhop

        fft_room = cp.fft.fft(filtered[:, i_sample+np.arange(Lframe)]*win, n=Nfft)
        anm_room = (Yenc @ fft_room) * bEQspec

        IV_room[:, i_frame, 0:3] \
                = calcIntensity(anm_room[:, 0:int(Nfft/2)], Wnv, Wpv, Vv)
        IV_room[:, i_frame, 3] \
                = np.abs(anm_room[0, 0:int(Nfft/2)])
    print('IV_room calculated')

    file_id='%06d_%2d'%(Nwavfile, i_loc)
    np.save(os.path.join(dir_IV,file_id+'_room.npy'), IV_room.get())
    np.save(os.path.join(dir_IV,file_id+'_free.npy'), IV_free.get())
    # scio.savemat(os.path.join(dir_IV,file_id+'.mat'),
    #              {'IV_room_py':IV_room, 'IV_free_py':IV_free}, appendmat=False)

    print(file_id)

if __name__ == '__main__':
    main()
