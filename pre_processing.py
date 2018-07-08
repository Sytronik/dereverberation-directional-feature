import pdb

import numpy as np
import cupy as cp
import scipy as sc
import scipy.signal
import scipy.io as scio
# import librosa

import os
import time
import atexit
from glob import glob

import soundfile as sf

from joblib import Parallel, delayed
import multiprocessing

def process(DIR_WAVFILE:str, ID:str, N_START,
            DIR_IV:str, FORM_FREE:str, FORM_ROOM:str,
            RIR, bEQspec, Yenc, Ys, Wnv, Wpv, Vv):
    global N_wavfile
    global Fs
    global N_FFT
    global L_FRAME
    global L_HOP
    global N_LOC
    global N_MIC

    #Constants
    Lwin_ms = 20.

    Fs = 0
    N_wavfile = 0
    L_FRAME = 0
    N_FFT = 0
    L_HOP = 0
    win = []

    N_LOC, N_MIC, L_RIR = RIR.shape

    N_CORES = multiprocessing.cpu_count()

    if not os.path.exists(DIR_IV):
        os.makedirs(DIR_IV)

    print('Start processing from the {}-th wave file'.format(N_START))

    for folder, _, _ in os.walk(DIR_WAVFILE):
        files = glob(os.path.join(folder, ID))
        if not files:
            continue
        for file in files:
            N_wavfile += 1
            if N_wavfile < N_START:
                continue

            #File Open & Resample
            if Fs==0:
                data, Fs = sf.read(file)
                L_FRAME = int(Fs*Lwin_ms/1000)
                N_FFT = L_FRAME
                L_HOP = int(L_FRAME/2)
                win = cp.array(sc.signal.hamming(L_FRAME, sym=False))
                print_information()
            else:
                data, _ = sf.read(file)
            # data = librosa.core.resample(data, Fs_original, Fs)
            data = cp.array(data)

            print(file)
            L_data_free = data.shape[0]
            N_frame_free = int(np.floor(L_data_free/L_HOP)-1)

            L_data_room = L_data_free+L_RIR-1
            N_frame_room = int(np.floor(L_data_room/L_HOP)-1)

            t_start = time.time()
            Parallel(n_jobs=int(N_CORES/2))(
                delayed(save_IV)(data, RIR[i_loc],
                                 N_frame_free, N_frame_room,
                                 N_FFT, win, L_FRAME, L_HOP,
                                 Yenc, Ys[i_loc], bEQspec,
                                 Wnv, Wpv, Vv,
                                 DIR_IV,
                                 FORM_FREE%(N_wavfile, i_loc),
                                 FORM_ROOM%(N_wavfile, i_loc))
                for i_loc in range(N_LOC)
            )
            print('%.3f sec'%(time.time()-t_start))
    print('Done.')
    print_information()


def save_IV(data, RIR,
            N_frame_free:int, N_frame_room:int,
            N_FFT:int, win, L_FRAME:int, L_HOP:int,
            Yenc, Ys, bEQspec,
            Wnv, Wpv, Vv,
            DIR_IV:str, FNAME_FREE:str, FNAME_ROOM:str):
    # Free-field Intensity Vector Image
    IV_free = cp.zeros((int(N_FFT/2), N_frame_free, 4))
    for i_frame in range(N_frame_free):
        interval_frame = i_frame*L_HOP + np.arange(L_FRAME)
        fft_free = cp.fft.fft(data[interval_frame]*win, n=N_FFT)
        anm_free = cp.outer(Ys.conj(), fft_free)

        IV_free[:,i_frame,:3] \
                = calc_intensity(anm_free[:,:int(N_FFT/2)], Wnv, Wpv, Vv)
        IV_free[:,i_frame,3] = cp.abs(anm_free[0,:int(N_FFT/2)])

    # RIR Filtering
    filtered \
        = cp.array(sc.signal.fftconvolve(data.reshape(1,-1).get(), RIR.get()))

    # Room Intensity Vector Image
    IV_room = cp.zeros((int(N_FFT/2), N_frame_room, 4))
    for i_frame in range(N_frame_room):
        interval_frame = i_frame*L_HOP + np.arange(L_FRAME)
        fft_room = cp.fft.fft(filtered[:,interval_frame]*win, n=N_FFT)
        anm_room = (Yenc @ fft_room) * bEQspec

        IV_room[:,i_frame,:3] \
                = calc_intensity(anm_room[:,:int(N_FFT/2)], Wnv, Wpv, Vv)
        IV_room[:,i_frame,3] = np.abs(anm_room[0,:int(N_FFT/2)])

    #Save
    np.save(os.path.join(DIR_IV,FNAME_FREE), IV_free.get())
    print(FNAME_FREE)
    np.save(os.path.join(DIR_IV,FNAME_ROOM), IV_room.get())
    print(FNAME_ROOM)
    # scio.savemat(os.path.join(DIR_IV,file_id+'.mat'),
    #              {'IV_room_py':IV_room, 'IV_free_py':IV_free}, appendmat=False)


def seltriag(Ain, nrord:int, shft):
    if Ain.ndim == 1:
        Nfreq = 1
    else:
        Nfreq = Ain.shape[1]
    N = int(np.ceil(np.sqrt(Ain.shape[0]))-1)
    idx = 0
    len_new = (N-nrord+1)**2

    Aout = cp.zeros((len_new, Nfreq), dtype=Ain.dtype)
    for ii in range(N-nrord+1):
        for jj in range(-ii,ii+1):
            n, m = shft + (ii, jj)
            idx_from = m + n*(n+1)
            if -n <= m and m <= n and 0 <= n and n <= N and \
                                                        idx_from < Ain.shape[0]:
                Aout[idx] = Ain[idx_from]
            idx += 1
    return Aout


def calc_intensity(Asv, Wnv, Wpv, Vv):
    Aug_1 = seltriag(Asv, 1, (0, 0))
    Aug_2 = seltriag(Wpv, 1, (1, -1))*seltriag(Asv, 1, (1, -1)) \
            - seltriag(Wnv, 1, (0, 0))*seltriag(Asv, 1, (-1, -1))# +exp(phi)
    Aug_3 = seltriag(Wpv, 1, (0, 0))*seltriag(Asv, 1, (-1, 1)) \
            - seltriag(Wnv, 1, (1, 1))*seltriag(Asv, 1, (1, 1))  # -exp(phi)
    Aug_4 = seltriag(Vv, 1, (0, 0))*seltriag(Asv, 1, (-1, 0)) \
            + seltriag(Vv, 1, (1, 0))*seltriag(Asv, 1, (1, 0))

    D_x = cp.sum(Aug_1.conj()*(Aug_2+Aug_3)/2, axis=0).reshape(-1,1)
    D_y = cp.sum(Aug_1.conj()*(Aug_2-Aug_3)/2j, axis=0).reshape(-1,1)
    D_z = cp.sum(Aug_1.conj()*Aug_4, axis=0).reshape(-1,1)

    return 1/2*cp.real(cp.concatenate((D_x, D_y, D_z), axis=1))


def print_information():
    metadata={}
    if 'N_wavfile' in globals():
        print('Number of data: {}'.format(N_wavfile-1))
        metadata['N_WAVFILE'] = N_wavfile

    if 'Fs' in globals():
        print('Sample Rate: {}'.format(Fs))
        metadata['Fs'] = Fs

    if 'N_FFT' in globals():
        metadata['N_FFT'] = N_FFT

    if 'L_FRAME' in globals():
        metadata['L_FRAME'] = L_FRAME

    if 'L_HOP' in globals():
        metadata['L_HOP'] = L_HOP

    if 'N_LOC' in globals():
        print('Number of source location: {}'.format(N_LOC))
        metadata['N_LOC'] = N_LOC

    if 'N_MIC' in globals():
        print('Number of microphone channel: {}'.format(N_MIC))
        # metadata['N_MIC'] = N_MIC

    np.save('metadata.npy', metadata)

# if __name__ == '__main__':
#     print_information()
#     process()
