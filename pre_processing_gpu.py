import pdb

import numpy as np
import cupy as cp
import scipy as sc
import scipy.io as scio
import librosa
import matplotlib.pyplot as plt

import os
import time
import atexit
from glob import glob

import soundfile as sf

from joblib import Parallel, delayed
import multiprocessing

def main():
    global Nwavfile
    global fs
    global Nloc
    global Nch
    global DIR_IV

    #Constants
    DIR_WAVFILE = './speech/data/lisa/data/timit/raw/TIMIT/TRAIN/'
    DIR_IV = './IV/TRAIN/'

    fs = 16000
    Nwavfile = 0
    Nloc = 0    #determined by RIR
    Nch= 0      #determined by RIR
    Lwin_ms = 20.
    Lframe = int(fs*Lwin_ms/1000)
    Nfft = Lframe
    Lhop = int(Lframe/2)

    N_CORES = multiprocessing.cpu_count()

    if not os.path.exists(DIR_IV):
        os.makedirs(DIR_IV)

    win = cp.array(sc.signal.hamming(Lframe, sym=False))

    #RIR Data
    RIR = scio.loadmat('./1_MATLABCode/RIR.mat', variable_names = 'RIR')['RIR']
    RIR = RIR.transpose((2, 0, 1)) #72 x 32 x 48k
    RIR = cp.array(RIR)
    Nloc, Nch, L_RIR = RIR.shape

    N_exist=len(glob(os.path.join(DIR_IV, '*_%d_room.npy'%(Nloc-1))))
    print('{} wave files have been already processed'.format(N_exist))

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

    Wnv = cp.array(sph_mat['Wnv'], dtype=complex).reshape(-1)
    Wpv = cp.array(sph_mat['Wpv'], dtype=complex).reshape(-1)
    Vv = cp.array(sph_mat['Vv'], dtype=complex).reshape(-1)

    sph_mat = None

    for folder, _, _ in os.walk(DIR_WAVFILE):
        files = glob(os.path.join(folder, '*_converted.wav'))
        if files is None:
            continue
        for file in files:
            Nwavfile += 1
            if Nwavfile <= N_exist:
                continue

            #File Open & Resample
            data, _ = sf.read(file)
            # data = librosa.core.resample(data, fs_original, fs)
            data = cp.array(data)

            print(file)
            L_data_free = data.shape[0]
            Nframe_free = int(np.floor(L_data_free/Lhop)-1)

            L_data_room = L_data_free+L_RIR-1
            Nframe_room = int(np.floor(L_data_room/Lhop)-1)

            t_start = time.time()
            Parallel(n_jobs=int(N_CORES/2))(
                delayed(save_IV)(data, RIR[i_loc],
                                Nframe_free, Nframe_room,
                                Nwavfile,
                                Nfft, win, Lframe, Lhop,
                                Yenc, Ys[i_loc], bEQspec,
                                Wnv, Wpv, Vv,
                                DIR_IV, i_loc)
                for i_loc in range(Nloc)
            )
            # for i_loc in range(Nloc):
            #     t_start = time.time()
            #     save_IV(data, RIR[i_loc],
            #             Nframe_free, Nframe_room,
            #             Nwavfile,
            #             Nfft, win, Lframe, Lhop,
            #             Yenc, Ys[i_loc], bEQspec,
            #             Wnv, Wpv, Vv,
            #             DIR_IV, i_loc)
            print('%.3f sec'%(time.time()-t_start))

@atexit.register
def print_information():
    if 'Nwavfile' in globals():
        print('Number of data: {}'.format(Nwavfile-1))
    if 'fs' in globals():
        print('Sample Rate: {}'.format(fs))
    if 'Nloc' in globals():
        print('Number of source location: {}'.format(Nloc))
    if 'Nch' in globals():
        print('Number of microphone channel: {}'.format(Nch))
    if 'DIR_IV' in globals():
        print('Saved the result in \"'+os.path.abspath(DIR_IV)+'\"')

def seltriag(Ain, nrord:int, shft):
    if Ain.ndim == 1:
        Nfreq = 1
    else:
        Nfreq=Ain.shape[1]
    N = int(np.ceil(np.sqrt(Ain.shape[0]))-1)
    idx = 0
    len_new = (N-nrord+1)**2

    Aout = cp.zeros((len_new, Nfreq), dtype=Ain.dtype)
    for ii in range(N-nrord+1):
        for jj in range(-ii,ii+1):
            n=shft[0]+ii; m=shft[1]+jj
            idx_from = m+n*(n+1)
            if -n <= m and m <= n and 0 <= n and n <= N and \
                                                        idx_from < Ain.shape[0]:
                Aout[idx] = Ain[idx_from]
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
            Nframe_free:int, Nframe_room:int,
            Nwavfile:int,
            Nfft:int, win, Lframe:int, Lhop:int,
            Yenc, Ys, bEQspec,
            Wnv, Wpv, Vv,
            DIR_IV:str, i_loc:int):
    # Free-field Intensity Vector Image
    IV_free = cp.zeros((int(Nfft/2), Nframe_free, 4))
    for i_frame in range(Nframe_free):
        interval_frame = i_frame*Lhop + np.arange(Lframe)
        fft_free = cp.fft.fft(data[interval_frame]*win, n=Nfft)
        anm_free = cp.outer(Ys.conj(), fft_free.T)

        IV_free[:, i_frame, :3] \
                = calcIntensity(anm_free[:, :int(Nfft/2)], Wnv, Wpv, Vv)
        IV_free[:, i_frame, 3] = cp.abs(anm_free[0, :int(Nfft/2)])

    # RIR Filtering
    filtered \
        = cp.array(sc.signal.fftconvolve(data.reshape(1,-1).get(), RIR.get()))

    # Room Intensity Vector Image
    IV_room = cp.zeros((int(Nfft/2), Nframe_room, 4))
    for i_frame in range(Nframe_room):
        interval_frame = i_frame*Lhop + np.arange(Lframe)
        fft_room = cp.fft.fft(filtered[:, interval_frame]*win, n=Nfft)
        anm_room = (Yenc @ fft_room) * bEQspec

        IV_room[:, i_frame, :3] \
                = calcIntensity(anm_room[:, :int(Nfft/2)], Wnv, Wpv, Vv)
        IV_room[:, i_frame, 3] \
                = np.abs(anm_room[0, :int(Nfft/2)])

    #Save
    file_id='%06d_%2d'%(Nwavfile, i_loc)
    np.save(os.path.join(DIR_IV,file_id+'_room.npy'), IV_room.get())
    np.save(os.path.join(DIR_IV,file_id+'_free.npy'), IV_free.get())
    # scio.savemat(os.path.join(DIR_IV,file_id+'.mat'),
    #              {'IV_room_py':IV_room, 'IV_free_py':IV_free}, appendmat=False)
    print(file_id)

if __name__ == '__main__':
    main()
