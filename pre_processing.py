import pdb

import numpy as np
import cupy as cp
import scipy as sc
import scipy.signal
# import librosa

import os
import time
from glob import glob

import soundfile as sf

from joblib import Parallel, delayed


class PreProcessor:
    def __init__(self, RIR, bEQspec, Yenc, Ys, Wnv, Wpv, Vv, L_WIN_MS=20.):
        # From Parameters
        self.RIR = RIR
        self.N_LOC, self.N_MIC, self.L_RIR = RIR.shape
        self.bEQspec = bEQspec
        self.Yenc = Yenc
        self.Ys = Ys
        self.Wnv = Wnv
        self.Wpv = Wpv
        self.Vv = Vv
        self.L_WIN_MS = L_WIN_MS

        # Determined during process
        self.DIR_IV = ''
        self.data = None
        self.N_frame_free = 0
        self.N_frame_room = 0

        # Common for all wav file
        self.Fs = 0
        self.N_wavfile = 0
        self.N_fft = 0
        self.L_frame = 0
        self.L_hop = 0
        self.win = None

    def process(self, DIR_WAVFILE:str, ID:str, N_START,
                DIR_IV:str, FORM:str, N_CORES:int):
        if not os.path.exists(DIR_IV):
            os.makedirs(DIR_IV)
        self.DIR_IV = DIR_IV

        print('Start processing from the {}-th wave file'.format(N_START))

        RIR = self.RIR
        Ys = self.Ys
        self.RIR = None
        self.Ys = None

        for folder, _, _ in os.walk(DIR_WAVFILE):
            files = glob(os.path.join(folder, ID))
            if not files:
                continue
            for file in files:
                if self.N_wavfile < N_START-1:
                    self.N_wavfile += 1
                    continue

                # File Open & Resample
                if self.Fs == 0:
                    data, self.Fs = sf.read(file)
                    self.L_frame = int(self.Fs*self.L_WIN_MS/1000)
                    self.N_fft = self.L_frame
                    self.L_hop = int(self.L_frame/2)
                    self.win \
                        = cp.array(sc.signal.hamming(self.L_frame, sym=False))
                    self.print_save_info()
                else:
                    data, _ = sf.read(file)
                # data = librosa.core.resample(data, Fs_original, Fs)
                self.data = cp.array(data)

                print(file)
                L_data_free = self.data.shape[0]
                self.N_frame_free = int(np.floor(L_data_free/self.L_hop)-1)

                L_data_room = L_data_free+self.L_RIR-1
                self.N_frame_room = int(np.floor(L_data_room/self.L_hop)-1)

                t_start = time.time()
                Parallel(n_jobs=N_CORES)(
                    delayed(self.save_IV)(RIR[i_loc], Ys[i_loc],
                                          FORM % (self.N_wavfile, i_loc))
                    for i_loc in range(self.N_LOC)
                )
                print('%.3f sec' % (time.time()-t_start))
                self.N_wavfile += 1
                self.print_save_info()

        self.RIR = RIR
        self.Ys = Ys
        print('Done.')
        self.print_save_info()

    def save_IV(self, RIR, Ys, FNAME:str):
        # Free-field Intensity Vector Image
        IV_free = cp.zeros((int(self.N_fft/2), self.N_frame_free, 4))
        norm_factor_free = float('-inf')
        for i_frame in range(self.N_frame_free):
            interval = i_frame*self.L_hop + np.arange(self.L_frame)
            fft = cp.fft.fft(self.data[interval]*self.win, n=self.N_fft)
            anm = cp.outer(Ys.conj(), fft)
            max_in_frame \
                = cp.max(cp.sqrt(cp.sum(cp.abs(anm)**2, axis=0))).get().item()
            norm_factor_free = np.max([norm_factor_free, max_in_frame])

            IV_free[:,i_frame,:3] \
                = PreProcessor.calc_intensity(anm[:,:int(self.N_fft/2)],
                                              self.Wnv, self.Wpv, self.Vv)
            IV_free[:,i_frame,3] = cp.abs(anm[0,:int(self.N_fft/2)])

        # RIR Filtering
        filtered \
            = cp.array(sc.signal.fftconvolve(self.data.reshape(1, -1).get(),
                                             RIR.get()))

        # Room Intensity Vector Image
        IV_room = cp.zeros((int(self.N_fft/2), self.N_frame_room, 4))
        norm_factor_room = float('-inf')
        for i_frame in range(self.N_frame_room):
            interval = i_frame*self.L_hop + np.arange(self.L_frame)
            fft = cp.fft.fft(filtered[:,interval]*self.win, n=self.N_fft)
            anm = (self.Yenc @ fft) * self.bEQspec
            max_in_frame \
                = cp.max(cp.sqrt(cp.sum(cp.abs(anm)**2, axis=0))).get().item()
            norm_factor_room = np.max([norm_factor_room, max_in_frame])

            IV_room[:,i_frame,:3] \
                = PreProcessor.calc_intensity(anm[:,:int(self.N_fft/2)],
                                              self.Wnv, self.Wpv, self.Vv)
            IV_room[:,i_frame,3] = np.abs(anm[0,:int(self.N_fft/2)])

        # Save
        dict = {'IV_free': IV_free.get(),
                'IV_room': IV_room.get(),
                'norm_factor_free': norm_factor_free,
                'norm_factor_room': norm_factor_room}
        np.save(os.path.join(self.DIR_IV, FNAME), dict)
        print(FNAME)

    def __str__(self):
        return 'Number of wave files already processed: {}\n'\
                    .format(self.N_wavfile) \
               + 'Sample Rate: {}\n'.format(self.Fs) \
               + 'Number of source location: {}'.format(self.N_LOC)

    def print_save_info(self):
        print(self)

        metadata = {'N_wavfile': self.N_wavfile,
                    'Fs': self.Fs,
                    'N_fft': self.N_fft,
                    'L_frame': self.L_frame,
                    'L_hop': self.L_hop,
                    'N_LOC': self.N_LOC}

        np.save('metadata.npy', metadata)

    @classmethod
    def seltriag(cls, Ain, nrord:int, shft):
        if Ain.ndim == 1:
            Nfreq = 1
        else:
            Nfreq = Ain.shape[1]
        N = int(np.ceil(np.sqrt(Ain.shape[0]))-1)
        idx = 0
        len_new = (N-nrord+1)**2

        Aout = cp.zeros((len_new, Nfreq), dtype=Ain.dtype)
        for ii in range(N-nrord+1):
            for jj in range(-ii, ii+1):
                n = shft[0] + ii
                m = shft[1] + jj
                idx_from = m + n*(n+1)
                if -n <= m and m <= n and 0 <= n and n <= N \
                        and idx_from < Ain.shape[0]:
                    Aout[idx] = Ain[idx_from]
                idx += 1
        return Aout

    @classmethod
    def calc_intensity(cls, Asv, Wnv, Wpv, Vv):
        Aug_1 = cls.seltriag(Asv, 1, (0, 0))
        Aug_2 = cls.seltriag(Wpv, 1, (1, -1))*cls.seltriag(Asv, 1, (1, -1)) \
                - cls.seltriag(Wnv, 1, (0, 0))*cls.seltriag(Asv, 1, (-1, -1))
        Aug_3 = cls.seltriag(Wpv, 1, (0, 0))*cls.seltriag(Asv, 1, (-1, 1)) \
                - cls.seltriag(Wnv, 1, (1, 1))*cls.seltriag(Asv, 1, (1, 1))
        Aug_4 = cls.seltriag(Vv, 1, (0, 0))*cls.seltriag(Asv, 1, (-1, 0)) \
                + cls.seltriag(Vv, 1, (1, 0))*cls.seltriag(Asv, 1, (1, 0))

        D_x = cp.sum(Aug_1.conj()*(Aug_2+Aug_3)/2, axis=0)
        D_y = cp.sum(Aug_1.conj()*(Aug_2-Aug_3)/2j, axis=0)
        D_z = cp.sum(Aug_1.conj()*Aug_4, axis=0)

        return 0.5*cp.real(cp.stack((D_x, D_y, D_z), axis=1))


if __name__ == '__main__':
    pass
