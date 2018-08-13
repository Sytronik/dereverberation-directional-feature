import pdb  # noqa: F401

import numpy as np
import cupy as cp
import scipy.signal as scsig
# import librosa

import os
import time
from glob import glob

from typing import Tuple, NamedTuple, TypeVar

import soundfile as sf

import multiprocessing as mp
import logging  # noqa: F401

N_CUDA_DEV = 4
NDARRAY = TypeVar('NDARRAY', np.ndarray, cp.ndarray)


class SFTData(NamedTuple):
    bEQspec:NDARRAY
    Yenc:NDARRAY
    Wnv:NDARRAY
    Wpv:NDARRAY
    Vv:NDARRAY

    def get_triags(self):
        return (self.Wnv, self.Wpv, self.Vv)


class PreProcessor:
    def __init__(self, RIR, Yenc, sftdata:SFTData, L_WIN_MS=20.):
        # Bug Fix
        np.fft.restore_all()
        # From Parameters
        self.RIR = RIR
        self.N_LOC, self.N_MIC, self.L_RIR = RIR.shape
        self.Ys = Ys
        self.sftdata = sftdata

        self.L_WIN_MS = L_WIN_MS

        # Determined during process
        self.DIR_IV = ''
        self.N_frame_free = 0
        self.N_frame_room = 0
        self.all_files = []

        # Common for all wave file
        self.Fs = 0
        self.N_wavfile = 0
        self.N_fft = 0
        self.N_freq = 0
        self.L_frame = 0
        self.L_hop = 0
        self.win = None

    def process(self, DIR_WAVFILE:str, ID:str, IDX_START:int,
                DIR_IV:str, FORM:str, N_CORES=mp.cpu_count()//4):
        if not os.path.exists(DIR_IV):
            os.makedirs(DIR_IV)
        self.DIR_IV = DIR_IV

        print(f'Start processing from the {IDX_START}-th wave file')

        # Search all wave files
        self.all_files = []
        for folder, _, _ in os.walk(DIR_WAVFILE):
            files = glob(os.path.join(folder, ID))
            if not files:
                continue
            self.all_files.extend(files)

        # Main Process
        for file in self.all_files:
            if self.N_wavfile < IDX_START-1:
                self.N_wavfile += 1
                continue

            # File Open (& Resample)
            if self.Fs == 0:
                data, self.Fs = sf.read(file)
                self.L_frame = self.Fs*self.L_WIN_MS//1000
                self.N_fft = self.L_frame
                if self.N_fft % 2 == 0:
                    self.N_freq = self.N_fft//2 + 1
                else:
                    self.N_freq = self.N_fft//2
                self.L_hop = self.L_frame//2

                self.win = scsig.hamming(self.L_frame, sym=False)
                self.print_save_info()
            else:
                data, _ = sf.read(file)

            print(file)

            # Data length
            L_data_free = data.shape[0]
            self.N_frame_free = L_data_free//self.L_hop - 1

            L_data_room = L_data_free+self.L_RIR-1
            self.N_frame_room = L_data_room//self.L_hop - 1

            t_start = time.time()

            # logger = mp.log_to_stderr()  #debugging subprocess
            # logger.setLevel(mp.SUBDEBUG) #debugging subprocess
            pool = mp.Pool(N_CORES)
            for i_proc in range(N_CORES):
                pool.apply_async(self.save_IV,
                                 (i_proc % N_CUDA_DEV,
                                  data,
                                  range(i_proc*(self.N_LOC//N_CORES),
                                        (i_proc+1)*(self.N_LOC//N_CORES)),
                                  FORM, self.N_wavfile+1))
            pool.close()
            pool.join()

            # Non-parallel
            # for i_dev in range(N_CUDA_DEV):
            #     self.save_IV(i_dev,
            #                  data,
            #                  range(i_dev*int(self.N_LOC/N_CUDA_DEV),
            #                        (i_dev+1)*int(self.N_LOC/N_CUDA_DEV)),
            #                  FORM, self.N_wavfile+1)

            print(f'{time.time()-t_start:.3f} sec')
            self.N_wavfile += 1
            self.print_save_info()

        print('Done.')
        self.print_save_info()

    def save_IV(self, i_dev:int, data, range_loc:iter, FORM:str, *args):
        # CUDA Ready
        cp.cuda.Device(i_dev).use()
        data = cp.array(data)
        win = cp.array(self.win)
        Ys = cp.array(self.Ys)
        sftdata = SFTData(*[cp.array(item) for item in self.sftdata])

        for i_loc in range_loc:
            # RIR Filtering
            filtered \
                = cp.array(scsig.fftconvolve(cp.asnumpy(data.reshape(1, -1)),
                                             self.RIR[i_loc]))

            # Free-field Intensity Vector Image
            IV_free = cp.zeros((self.N_freq, self.N_frame_free, 4))
            # norm_factor_free = float('-inf')
            for i_frame in range(self.N_frame_free):
                interval = i_frame*self.L_hop + np.arange(self.L_frame)
                fft = cp.fft.fft(data[interval]*win, n=self.N_fft)
                anm = cp.outer(Ys[i_loc].conj(), fft)

                IV_free[:,i_frame,:3] \
                    = PreProcessor.calc_intensity(anm[:,:self.N_freq],
                                                  *sftdata.get_triags())
                IV_free[:,i_frame,3] = cp.abs(anm[0,:self.N_freq])**2

                # max_in_frame \
                #     = cp.max(0.5*cp.sum(cp.abs(anm)**2, axis=0)).get().item()
                # norm_factor_free = np.max([norm_factor_free, max_in_frame])

            # Room Intensity Vector Image
            IV_room = cp.zeros((self.N_freq, self.N_frame_room, 4))
            # norm_factor_room = float('-inf')
            for i_frame in range(self.N_frame_room):
                interval = i_frame*self.L_hop + np.arange(self.L_frame)
                fft = cp.fft.fft(filtered[:,interval]*win, n=self.N_fft)
                anm = (sftdata.Yenc @ fft) * sftdata.bEQspec

                IV_room[:,i_frame,:3] \
                    = PreProcessor.calc_intensity(anm[:,:self.N_freq],
                                                  *sftdata.get_triags())
                IV_room[:,i_frame,3] = cp.abs(anm[0,:self.N_freq])**2

                # max_in_frame \
                #     = cp.max(0.5*cp.sum(cp.abs(anm)**2, axis=0)).get().item()
                # norm_factor_room = np.max([norm_factor_room, max_in_frame])

            # Save
            dic = {'IV_free': cp.asnumpy(IV_free),
                   'IV_room': cp.asnumpy(IV_room),
                   # 'norm_factor_free': norm_factor_free,
                   # 'norm_factor_room': norm_factor_room,
                   }
            FNAME = FORM % (*args, i_loc)
            np.save(os.path.join(self.DIR_IV, FNAME), dic)

            print(FORM % (*args, i_loc))

    def __str__(self):
        return ('Wave Files Processed/Total: '
                f'{self.N_wavfile}/{len(self.all_files)}\n'
                f'Sample Rate: {self.Fs}\n'
                f'Number of source location: {self.N_LOC}\n'
                )

    def print_save_info(self):
        print(self)

        metadata = {'N_wavfile': self.N_wavfile,
                    'Fs': self.Fs,
                    # 'N_fft': self.N_fft,
                    'N_freq': self.N_freq,
                    'L_frame': self.L_frame,
                    'L_hop': self.L_hop,
                    'N_LOC': self.N_LOC,
                    'path_wavfiles': self.all_files,
                    }

        np.save(os.path.join(self.DIR_IV, 'metadata.npy'), metadata)

    @staticmethod
    def seltriag(Ain:NDARRAY, nrord:int, shft:Tuple[int, int]) -> NDARRAY:
        xp = cp.get_array_module(Ain)
        if Ain.ndim == 1:
            N_freq = 1
        else:
            N_freq = Ain.shape[1]
        N = int(np.ceil(np.sqrt(Ain.shape[0]))-1)
        idx = 0
        len_new = (N-nrord+1)**2

        Aout = xp.zeros((len_new, N_freq), dtype=Ain.dtype)
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
    def calc_intensity(cls, Asv:NDARRAY,
                       Wnv:NDARRAY, Wpv:NDARRAY, Vv:NDARRAY) -> NDARRAY:
        xp = cp.get_array_module(Asv)
        
        aug1 = cls.seltriag(Asv, 1, (0, 0))
        aug2 = cls.seltriag(Wpv, 1, (1, -1))*cls.seltriag(Asv, 1, (1, -1)) \
            - cls.seltriag(Wnv, 1, (0, 0))*cls.seltriag(Asv, 1, (-1, -1))
        aug3 = cls.seltriag(Wpv, 1, (0, 0))*cls.seltriag(Asv, 1, (-1, 1)) \
            - cls.seltriag(Wnv, 1, (1, 1))*cls.seltriag(Asv, 1, (1, 1))
        aug4 = cls.seltriag(Vv, 1, (0, 0))*cls.seltriag(Asv, 1, (-1, 0)) \
            + cls.seltriag(Vv, 1, (1, 0))*cls.seltriag(Asv, 1, (1, 0))

        dx = xp.sum(aug1.conj()*(aug2+aug3)/2, axis=0)
        dy = xp.sum(aug1.conj()*(aug2-aug3)/2j, axis=0)
        dz = xp.sum(aug1.conj()*aug4, axis=0)

        return 0.5*xp.real(xp.stack((dx, dy, dz), axis=1))


if __name__ == '__main__':
    pass
