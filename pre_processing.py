import pdb  # noqa: F401

import numpy as np
import cupy as cp
import scipy.io as scio
import scipy.signal as scsig

import os
from argparse import ArgumentParser
import time

from typing import Tuple, NamedTuple, TypeVar, List

from glob import glob
import deepdish as dd
import soundfile as sf

import multiprocessing as mp
import logging  # noqa: F401

import mypath

NDArray = TypeVar('NDArray', np.ndarray, cp.ndarray)


class SFTData(NamedTuple):
    """
    Constant Matrices/Vectors for Spherical Fourier Analysis
    """
    bEQspec: NDArray
    Yenc: NDArray
    Wnv: NDArray
    Wpv: NDArray
    Vv: NDArray

    def get_triags(self) -> Tuple:
        return (self.Wnv, self.Wpv, self.Vv)


def search_all_files(DIR_WAVFILE: str, ID: str) -> List[str]:
    result = []
    for folder, _, _ in os.walk(DIR_WAVFILE):
        files = glob(os.path.join(folder, ID))
        if not files:
            continue
        result += files

    return result


# manually select
N_CUDA_DEV = 4
FORM = '%04d_%02d.h5'
L_WIN_MS = 20.
FN_WIN = scsig.hamming

# determined by wave files
N_freq = 0
L_hop = 0
win = None
L_frame = 0
N_wavfile = 0
Fs = 0
N_fft = 0

# determined by sys argv
parser = ArgumentParser()
parser.add_argument('kind_data', choices=('TRAIN', 'train', 'TEST', 'test'))
ARGS = parser.parse_args()

# determined by mypath
DIR_DATA = mypath.path('root')
DIR_IV = mypath.path(f'iv_{ARGS.kind_data.lower()}')
if not os.path.exists(DIR_IV):
    os.makedirs(DIR_IV)
DIR_WAVFILE = mypath.path(f'wav_{ARGS.kind_data.lower()}')

# RIR Data
transfer_dict = scio.loadmat(os.path.join(DIR_DATA, 'RIR_Ys.mat'),
                             squeeze_me=True)
RIRs = transfer_dict[f'RIR_{ARGS.kind_data}'].transpose((2, 0, 1))
N_LOC, N_MIC, L_RIR = RIRs.shape
Ys = transfer_dict[f'Ys_{ARGS.kind_data}'].T

# RIRs_0 = scio.loadmat(os.path.join(DIR_DATA, 'RIR_0_order.mat'),
#                       variable_names='RIR_'+ARGS.kind_data)
# RIRs_0 = RIRs_0['RIR_'+ARGS.kind_data].transpose((2, 0, 1))

# SFT Data
sft_dict = scio.loadmat(os.path.join(DIR_DATA, 'sft_data.mat'),
                        variable_names=('bEQspec', 'Yenc', 'Wnv', 'Wpv', 'Vv'),
                        squeeze_me=True)

bEQspec = sft_dict['bEQspec'].T
Yenc = sft_dict['Yenc'].T
Wnv = sft_dict['Wnv'].astype(complex)
Wpv = sft_dict['Wpv'].astype(complex)
Vv = sft_dict['Vv'].astype(complex)

sftdata = SFTData(bEQspec, Yenc, Wnv, Wpv, Vv)

# all_files = search_all_files(DIR_WAVFILE, '*.WAV')
all_files = dd.io.load(os.path.join(DIR_IV, 'metadata.h5'))['path_wavfiles']


def process():
    global Fs, N_freq, L_hop, win, N_wavfile, L_frame, N_fft

    N_CORES = N_LOC if N_LOC < mp.cpu_count() else mp.cpu_count()//4
    n_loc_per_core = int(np.ceil(N_LOC//N_CORES))

    max_n_pool = 1
    while max_n_pool*int(np.ceil(n_loc_per_core*N_CORES/N_CUDA_DEV)) < 30:
        max_n_pool += 1
    max_n_pool -= 1

    # The index of the first wave file that have to be processed
    idx_start \
        = len(glob(os.path.join(DIR_IV, f'*_{RIRs.shape[0]-1:02d}.h5')))+1
    print(f'Start processing from the {idx_start}-th wave file')

    pools = []
    for fname in all_files:
        if N_wavfile < idx_start-1:
            N_wavfile += 1
            continue

        if (N_wavfile - idx_start) % max_n_pool == max_n_pool-1:
            t_start = time.time()

        # File Open (& Resample)
        if not Fs:
            data, Fs = sf.read(fname)
            L_frame = int(Fs*L_WIN_MS//1000)
            N_fft = L_frame
            N_freq = N_fft//2 + 1
            L_hop = L_frame//2

            win = FN_WIN(L_frame, sym=False)
            print_save_info()
        else:
            data, _ = sf.read(fname)

        # print(fname)
        N_frame_free = int(np.ceil(data.shape[0] / L_hop) - 1)
        N_frame_room = int(np.ceil((data.shape[0]+L_RIR-1) / L_hop) - 1)

        # logger = mp.log_to_stderr()  # debugging subprocess
        # logger.setLevel(mp.SUBDEBUG)  # debugging subprocess
        pools.append(mp.Pool(N_CORES))
        for i_proc in range(N_CORES):
            if (i_proc + 1) * n_loc_per_core <= N_LOC:
                range_loc = range(i_proc * n_loc_per_core,
                                  (i_proc+1) * n_loc_per_core)
            elif i_proc * n_loc_per_core < N_LOC:
                range_loc = range(i_proc * n_loc_per_core, N_LOC)
            else:
                break
            pools[-1].apply_async(save_IV,
                                  (i_proc % N_CUDA_DEV,
                                   data, N_frame_free, N_frame_room,
                                   range_loc,
                                   FORM, N_wavfile+1))

        pools[-1].close()

        # Non-parallel
        # for i_proc in range(N_CORES):
        #     if (i_proc + 1) * n_loc_per_core <= N_LOC:
        #         range_loc = range(i_proc * n_loc_per_core,
        #                           (i_proc+1) * n_loc_per_core)
        #     elif i_proc * n_loc_per_core < N_LOC:
        #         range_loc = range(i_proc * n_loc_per_core, N_LOC)
        #     else:
        #         break
        #     save_IV(i_proc % N_CUDA_DEV,
        #                  data,
        #                  range_loc,
        #                  FORM, N_wavfile+1)

        if (N_wavfile - idx_start) % max_n_pool == max_n_pool-2:
            for pool in pools:
                pool.join()
            print(f'{time.time() - t_start:.3f} sec')
            pools = []
            N_wavfile += 1
            print_save_info()
        else:
            N_wavfile += 1

    for pool in pools:
        pool.join()
    print('Done.')
    print_save_info()


def save_IV(i_dev: int,
            data_np: NDArray, N_frame_free: int, N_frame_room: int,
            range_loc: iter,
            FORM: str, *args):
    global win, Ys, L_frame, N_fft, sftdata
    """
    Save IV files.

    i_dev: GPU Device No.
    range_loc: RIR Index Range(S/M Location Index Range)
    FORM: format of filename
    args: format string arguments

    return: None
    """
    # CUDA Ready
    cp.cuda.Device(i_dev).use()
    win_cp = cp.array(win)
    Ys_cp = cp.array(Ys)
    sftdata_cp = SFTData(*[cp.array(item) for item in sftdata])
    data = cp.array(
        np.append(data_np, np.zeros(L_hop - data_np.shape[0] % L_hop))
    ) if data_np.shape[0] % L_hop else cp.array(data_np)

    for i_loc in range_loc:
        # RIR Filtering
        # data_0 \
        #     = cp.array(scsig.fftconvolve(cp.asnumpy(data.reshape(1, -1)),
        #                                  RIRs_0[i_loc]))
        data_room = scsig.fftconvolve(data_np.reshape(1, -1), RIRs[i_loc])
        data_room = cp.array(
            np.append(data_room,
                      np.zeros((data_room.shape[0],
                                L_hop - data_room.shape[1] % L_hop)),
                      axis=1)
        ) if data_room.shape[1] % L_hop else cp.array(data_room)

        # Energy using 0-th Order RIR
        # iv_0 = cp.zeros((N_freq, N_frame_room, 4))
        # for i_frame in range(N_frame_room):
        #     interval = i_frame*L_hop + np.arange(L_frame)
        #     fft = cp.fft.fft(data_0[:, interval]*win_cp, n=N_fft)
        #     anm = (sftdata_cp.Yenc @ fft) * sftdata_cp.bEQspec
        #
        #     iv_0[:, i_frame, :3] \
        #         = calc_intensity(anm[:, :N_freq],
        #                                       *sftdata_cp.get_triags())
        #     iv_0[:, i_frame, 3] \
        #         = cp.sum(cp.abs(anm[:, :N_freq])**2, axis=0)

        # Free-field Intensity Vector Image
        iv_free = cp.zeros((N_freq, N_frame_free, 4))
        # norm_factor_free = float('-inf')
        for i_frame in range(N_frame_free):
            interval = i_frame*L_hop + np.arange(L_frame)
            fft = cp.fft.fft(data[interval]*win_cp, n=N_fft)
            anm = cp.outer(Ys_cp[i_loc].conj(), fft)
            # energy_free += cp.sum(cp.abs(anm[:, :N_freq])**2)

            iv_free[:, i_frame, :3] \
                = calc_intensity(anm[:, :N_freq], *sftdata_cp.get_triags())
            iv_free[:, i_frame, 3] \
                = cp.sum(cp.abs(anm[:, :N_freq])**2, axis=0)

            # max_in_frame \
            #     = cp.max(0.5*cp.sum(cp.abs(anm)**2, axis=0)).get().item()
            # norm_factor_free = np.max([norm_factor_free, max_in_frame])
        # iv_free /= iv_free[:, :, 3].mean()

        # Room Intensity Vector Image
        iv_room = cp.zeros((N_freq, N_frame_room, 4))
        # norm_factor_room = float('-inf')
        for i_frame in range(N_frame_room):
            interval = i_frame*L_hop + np.arange(L_frame)
            fft = cp.fft.fft(data_room[:, interval]*win_cp, n=N_fft)
            anm = (sftdata_cp.Yenc @ fft) * sftdata_cp.bEQspec

            iv_room[:, i_frame, :3] \
                = calc_intensity(anm[:, :N_freq], *sftdata_cp.get_triags())
            iv_room[:, i_frame, 3] \
                = cp.sum(cp.abs(anm[:, :N_freq])**2, axis=0)

            # max_in_frame \
            #     = cp.max(0.5*cp.sum(cp.abs(anm)**2, axis=0)).get().item()
            # norm_factor_room = np.max([norm_factor_room, max_in_frame])
        # iv_room /= iv_room[:, :, 3].mean()

        # Save
        dict_to_save = {'IV_free': cp.asnumpy(iv_free),
                        'IV_room': cp.asnumpy(iv_room),
                        # 'IV_0': cp.asnumpy(iv_0),
                        # 'data': cp.asnumpy(data),
                        # 'norm_factor_free': norm_factor_free,
                        # 'norm_factor_room': norm_factor_room,
                        }
        FNAME = FORM % (*args, i_loc)
        dd.io.save(os.path.join(DIR_IV, FNAME), dict_to_save,
                   compression=None)

        print(FORM % (*args, i_loc))


def seltriag(Ain: NDArray, nrord: int, shft: Tuple[int, int]) -> NDArray:
    xp = cp.get_array_module(Ain)
    N_freq = 1 if Ain.ndim == 1 else Ain.shape[1]
    N = int(np.ceil(np.sqrt(Ain.shape[0]))-1)
    idx = 0
    len_new = (N-nrord+1)**2

    Aout = xp.zeros((len_new, N_freq), dtype=Ain.dtype)
    for ii in range(N-nrord+1):
        for jj in range(-ii, ii+1):
            n, m = shft[0] + ii, shft[1] + jj
            idx_from = m + n*(n+1)
            if -n <= m <= n and 0 <= n <= N and idx_from < Ain.shape[0]:
                Aout[idx] = Ain[idx_from]
            idx += 1
    return Aout


def calc_intensity(Asv: NDArray,
                   Wnv: NDArray, Wpv: NDArray, Vv: NDArray) -> NDArray:
    """
    Asv(anm) -> IV
    """
    xp = cp.get_array_module(Asv)

    aug1 = seltriag(Asv, 1, (0, 0))
    aug2 = seltriag(Wpv, 1, (1, -1))*seltriag(Asv, 1, (1, -1)) \
        - seltriag(Wnv, 1, (0, 0))*seltriag(Asv, 1, (-1, -1))
    aug3 = seltriag(Wpv, 1, (0, 0))*seltriag(Asv, 1, (-1, 1)) \
        - seltriag(Wnv, 1, (1, 1))*seltriag(Asv, 1, (1, 1))
    aug4 = seltriag(Vv, 1, (0, 0))*seltriag(Asv, 1, (-1, 0)) \
        + seltriag(Vv, 1, (1, 0))*seltriag(Asv, 1, (1, 0))

    dx = (aug1.conj()*(aug2+aug3)/2).sum(axis=0)
    dy = (aug1.conj()*(aug2-aug3)/2j).sum(axis=0)
    dz = (aug1.conj()*aug4).sum(axis=0)

    return 0.5*xp.real(xp.stack((dx, dy, dz), axis=1))


def print_save_info():
    """
    Print __str__ and save metadata.
    """
    print(f'Wave Files Processed/Total: '
          f'{N_wavfile}/{len(all_files)}\n'
          f'Sample Rate: {Fs}\n'
          f'Number of source location: {N_LOC}\n'
          )

    metadata = {'N_wavfile': N_wavfile,
                'Fs': Fs,
                # 'N_fft': N_fft,
                'N_freq': N_freq,
                'L_frame': L_frame,
                'L_hop': L_hop,
                'N_LOC': N_LOC,
                'path_wavfiles': all_files,
                }

    dd.io.save(os.path.join(DIR_IV, 'metadata.h5'), metadata)


if __name__ == '__main__':
    process()
