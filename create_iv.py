# noinspection PyUnresolvedReferences
import logging
import multiprocessing as mp
import os
import time
from argparse import ArgumentParser
from glob import glob
from typing import List, NamedTuple, Tuple, TypeVar

import cupy as cp
import deepdish as dd
import numpy as np
import scipy.io as scio
import scipy.signal as scsig
import soundfile as sf

from mypath import DICT_PATH

NDArray = TypeVar('NDArray', np.ndarray, cp.ndarray)

# manually select
N_CUDA_DEV = 4
FORM = '%04d_%02d.h5'
L_WIN_MS = 32.
HOP_RATIO = 0.5
FN_WIN = scsig.windows.hann

# determined by wave files
N_freq = 0
L_hop = 0
win = None
L_frame = 0
N_wavfile = 0
Fs = 0
N_fft = 0


class SFTData(NamedTuple):
    """
    Constant Matrices/Vectors for Spherical Fourier Analysis
    """
    Yenc: NDArray
    Wnv: NDArray
    Wpv: NDArray
    Vv: NDArray
    bnkr: NDArray
    bEQf: NDArray
    bn_sel2_0: NDArray
    bn_sel2_1: NDArray
    bn_sel3_0: NDArray
    bn_sel3_1: NDArray
    bn_sel_4_0: NDArray
    bn_sel_4_1: NDArray

    def get_for_intensity(self) -> Tuple:
        return (self.Wnv, self.Wpv, self.Vv,
                self.bn_sel2_0, self.bn_sel2_1,
                self.bn_sel3_0, self.bn_sel3_1,
                self.bn_sel_4_0, self.bn_sel_4_1)


def search_all_files(directory: str, id_: str) -> List[str]:
    result = []
    for folder, _, _ in os.walk(directory):
        files = glob(os.path.join(folder, id_))
        if files:
            result += files

    return result


# noinspection PyShadowingNames
def seltriag(Ain: NDArray, nrord: int, shft: Tuple[int, int]) -> NDArray:
    xp = cp.get_array_module(Ain)
    N_freq = 1 if Ain.ndim == 1 else Ain.shape[1]
    N = int(np.ceil(np.sqrt(Ain.shape[0])) - 1)
    idx = 0
    len_new = (N - nrord + 1)**2

    Aout = xp.zeros((len_new, N_freq), dtype=Ain.dtype)
    for ii in range(N - nrord + 1):
        for jj in range(-ii, ii + 1):
            n, m = shft[0] + ii, shft[1] + jj
            idx_from = m + n * (n + 1)
            if -n <= m <= n and 0 <= n <= N and idx_from < Ain.shape[0]:
                Aout[idx] = Ain[idx_from]
            idx += 1
    return Aout


# noinspection PyShadowingNames
def calc_intensity(Asv: NDArray,
                   Wnv: NDArray, Wpv: NDArray, Vv: NDArray,
                   bn_sel2_0: NDArray, bn_sel2_1: NDArray,
                   bn_sel3_0: NDArray, bn_sel3_1: NDArray,
                   bn_sel_4_0: NDArray, bn_sel_4_1: NDArray) -> NDArray:
    """
    Asv(anm) -> IV
    """
    xp = cp.get_array_module(Asv)

    aug1 = seltriag(Asv, 1, (0, 0))
    aug2 = (bn_sel2_0 * seltriag(Wpv, 1, (1, -1)) * seltriag(Asv, 1, (1, -1))
            - bn_sel2_1 * seltriag(Wnv, 1, (0, 0)) * seltriag(Asv, 1, (-1, -1)))
    aug3 = (bn_sel3_0 * seltriag(Wpv, 1, (0, 0)) * seltriag(Asv, 1, (-1, 1))
            - bn_sel3_1 * seltriag(Wnv, 1, (1, 1)) * seltriag(Asv, 1, (1, 1)))
    aug4 = (bn_sel_4_0 * seltriag(Vv, 1, (0, 0)) * seltriag(Asv, 1, (-1, 0))
            + bn_sel_4_1 * seltriag(Vv, 1, (1, 0)) * seltriag(Asv, 1, (1, 0)))

    aug1 = aug1.conj()
    intensity = xp.empty((aug1.shape[1], 3))
    intensity[:, 0] = (xp.einsum('mf,mf->f', aug1, aug2 + aug3) / 2).real
    intensity[:, 1] = (xp.einsum('mf,mf->f', aug1, aug2 - aug3) / 2j).real
    intensity[:, 2] = (xp.einsum('mf,mf->f', aug1, aug4)).real

    return 0.5 * intensity


def calc_real_coeffs(anm: NDArray) -> NDArray:
    xp = cp.get_array_module(anm)
    N = int(np.ceil(np.sqrt(anm.shape[0])) - 1)
    trans = np.zeros(((N + 1)**2, (N + 1)**2), dtype=np.complex128)
    trans[0, 0] = 1
    if N > 0:
        idxs = (np.arange(N + 1) + 1)**2

        for n in range(1, N + 1):
            m1 = np.arange(n)
            diag = np.array([1j] * n + [0] + list(-(-1)**m1))

            m2 = m1[::-1]
            anti_diag = np.array(list(1j * (-1)**m2) + [0] + [1] * n)

            block = (np.diagflat(diag) + np.diagflat(anti_diag)[:, ::-1]) / np.sqrt(2)
            block[n, n] = 1.

            trans[idxs[n - 1]:idxs[n], idxs[n - 1]:idxs[n]] = block

    anm_real = (xp.asarray(trans.conj()) @ anm).real

    return anm_real


def calc_direction_vec(anm: NDArray) -> NDArray:
    """ Calculate direciton vector in DirAC

    using Complex Spherical Harmonics Coefficients

    :param anm:
    :return:
    """
    if anm.shape[0] > 4:
        anm = anm[:4]
    anm_real = calc_real_coeffs(anm)  # transform to real coefficient
    v = anm_real[[3, 1, 2]]  # DirAC particle velocity vector

    return (1 / np.sqrt(2) * anm_real[0] * v).T


if __name__ == '__main__':
    # determined by sys argv
    parser = ArgumentParser()
    parser.add_argument('kind_data',
                        choices=('TRAIN', 'train',
                                 'SEEN', 'seen',
                                 'UNSEEN', 'unseen',
                                 ),
                        )
    parser.add_argument('--dirac', action='store_true')
    parser.add_argument('--init', action='store_true')
    ARGS = parser.parse_args()
    USE_DIRAC = ARGS.dirac

    # Paths
    DIR_DATA = DICT_PATH['root']
    DIR_IV = DICT_PATH[f'iv_{ARGS.kind_data.lower()}']
    if not os.path.exists(DIR_IV):
        os.makedirs(DIR_IV)
    DIR_WAVFILE = DICT_PATH[f'wav_{ARGS.kind_data.lower()}']

    # RIR Data
    transfer_dict = scio.loadmat(os.path.join(DIR_DATA, 'RIR_Ys.mat'), squeeze_me=True)
    kind_RIR = 'TEST' if ARGS.kind_data.lower() == 'unseen' else 'TRAIN'
    RIRs = transfer_dict[f'RIR_{kind_RIR}'].transpose((2, 0, 1))
    N_LOC, N_MIC, L_RIR = RIRs.shape
    Ys = transfer_dict[f'Ys_{kind_RIR}'].T  # N_LOC x Order
    if USE_DIRAC:
        Ys = Ys[:, :4]

    t_peak = np.round(RIRs.argmax(axis=2).mean(axis=1)).astype(int)
    amp_peak = RIRs.max(axis=2).mean(axis=1)

    # RIRs_0 = scio.loadmat(os.path.join(DIR_DATA, 'RIR_0_order.mat'),
    #                       variable_names='RIR_'+ARGS.kind_data)
    # RIRs_0 = RIRs_0['RIR_'+ARGS.kind_data].transpose((2, 0, 1))

    # SFT Data
    sft_dict = scio.loadmat(
        os.path.join(DIR_DATA, 'sft_data.mat'),
        variable_names=('bmn_ka', 'bEQf', 'Yenc', 'Wnv', 'Wpv', 'Vv'),
        squeeze_me=True
    )
    bEQf = sft_dict['bEQf'].T  # Order x N_freq
    Yenc = sft_dict['Yenc'].T  # Order x N_MIC

    if USE_DIRAC:
        bnkr = None
        bEQf = bEQf[:4]
        Yenc = Yenc[:4]
        Wnv = None
        Wpv = None
        Vv = None
        bn_sel2_0 = None
        bn_sel2_1 = None

        bn_sel3_0 = None
        bn_sel3_1 = None

        bn_sel_4_0 = None
        bn_sel_4_1 = None
    else:
        bnkr = sft_dict['bmn_ka'].T  # Order x N_freq
        Wnv = sft_dict['Wnv'].astype(complex)
        Wpv = sft_dict['Wpv'].astype(complex)
        Vv = sft_dict['Vv'].astype(complex)

        bn_sel2_0 = seltriag(1. / bnkr, 1, (1, -1)) * seltriag(bnkr, 1, (0, 0))
        bn_sel2_1 = seltriag(1. / bnkr, 1, (-1, -1)) * seltriag(bnkr, 1, (0, 0))

        bn_sel3_0 = seltriag(1. / bnkr, 1, (-1, 1)) * seltriag(bnkr, 1, (0, 0))
        bn_sel3_1 = seltriag(1. / bnkr, 1, (1, 1)) * seltriag(bnkr, 1, (0, 0))

        bn_sel_4_0 = seltriag(1. / bnkr, 1, (-1, 0)) * seltriag(bnkr, 1, (0, 0))
        bn_sel_4_1 = seltriag(1. / bnkr, 1, (1, 0)) * seltriag(bnkr, 1, (0, 0))

    sftdata = SFTData(
        Yenc, Wnv, Wpv, Vv, bnkr, bEQf,
        bn_sel2_0, bn_sel2_1, bn_sel3_0, bn_sel3_1, bn_sel_4_0, bn_sel_4_1
    )

    del (sft_dict, Yenc, Wnv, Wpv, Vv, bnkr, bEQf,
         bn_sel2_0, bn_sel2_1, bn_sel3_0, bn_sel3_1, bn_sel_4_0, bn_sel_4_1)

    f_metadata = os.path.join(DIR_IV, 'metadata.h5')
    if os.path.isfile(f_metadata):
        all_files = dd.io.load(f_metadata)['path_wavfiles']
    else:
        all_files = search_all_files(DIR_WAVFILE, '*.WAV')


def process():
    global Fs, N_freq, L_hop, win, N_wavfile, L_frame, N_fft

    N_CORES = N_LOC if N_LOC < mp.cpu_count() else mp.cpu_count() // N_CUDA_DEV
    n_loc_per_core = int(np.ceil(N_LOC // N_CORES))

    max_n_pool = 1
    while max_n_pool * int(np.ceil(n_loc_per_core * N_CORES / N_CUDA_DEV)) < 30:
        max_n_pool += 1
    max_n_pool -= 1

    # The index of the first wave file that have to be processed
    if ARGS.init:
        idx_start = 1
    else:
        idx_start = len(glob(os.path.join(DIR_IV, f'*_{N_LOC - 1:02d}.h5'))) + 1
    # idx_start = 1
    print(f'Start processing from the {idx_start}-th wave file')

    pools = []
    t_start = 0
    for fname in all_files:
        if N_wavfile < idx_start - 1:
            N_wavfile += 1
            continue

        if (N_wavfile - idx_start) % max_n_pool == max_n_pool - 1:
            t_start = time.time()

        # File Open (& Resample)
        if not Fs:
            data, Fs = sf.read(fname)
            L_frame = int(Fs * L_WIN_MS // 1000)
            N_fft = L_frame
            N_freq = N_fft // 2 + 1
            L_hop = int(L_frame * HOP_RATIO)

            win = FN_WIN(L_frame, sym=False)
            print_save_info()
        else:
            data, _ = sf.read(fname)

        # logger = mp.log_to_stderr()  # debugging subprocess
        # logger.setLevel(mp.SUBDEBUG)  # debugging subprocess
        pools.append(mp.Pool(N_CORES))
        for i_proc in range(N_CORES):
            start_idx = i_proc * n_loc_per_core
            if start_idx > N_LOC:
                break
            end_idx = min((i_proc + 1) * n_loc_per_core, N_LOC)
            range_loc = range(start_idx, end_idx)

            pools[-1].apply_async(
                save_IV,
                (i_proc % N_CUDA_DEV, data, range_loc, fname, N_wavfile + 1)
            )

        pools[-1].close()

        # Non-parallel
        # for i_proc in range(N_CORES):
        #     if (i_proc + 1) * n_loc_per_core <= N_LOC:
        #         range_loc = range(i_proc * n_loc_per_core,
        #                           (i_proc + 1) * n_loc_per_core)
        #     elif i_proc * n_loc_per_core < N_LOC:
        #         range_loc = range(i_proc * n_loc_per_core, N_LOC)
        #     else:
        #         break
        #     save_IV(i_proc % N_CUDA_DEV, data, range_loc, fname, N_wavfile + 1)

        if (N_wavfile - idx_start) % max_n_pool == max_n_pool - 2:
            for pool in pools:
                pool.join()
            print(f'{time.time() - t_start:.1f} sec')
            pools = []
            N_wavfile += 1
            print_save_info()
        else:
            N_wavfile += 1

    for pool in pools:
        pool.join()
    print('Done.')
    print_save_info()


def save_IV(i_dev: int, data_np: np.ndarray, range_loc: iter, fname_wav: str, *args):
    """ Save IV files.

    :param i_dev: GPU Device No.
    :param data_np: original wave data
    :param range_loc: RIR Index Range(S/M Location Index Range)
    :param fname_wav: the file path of the original wave(data_np)
    :param args: format string arguments

    :return: None
    """

    global win, Ys, L_frame, N_fft, sftdata, N_freq, FORM, L_RIR, L_hop, t_peak, USE_DIRAC
    # Ready CUDA
    cp.cuda.Device(i_dev).use()
    win_cp = cp.array(win)
    Ys_cp = cp.array(Ys)
    sftdata_cp = SFTData(*[cp.array(item) if item is not None else None for item in sftdata])
    # data = cp.array(data_np)

    N_frame_room = int(np.ceil((data_np.shape[0] + L_RIR - 1) / L_hop) - 1)

    for i_loc in range_loc:
        # RIR Filtering
        data_room = scsig.fftconvolve(data_np.reshape(1, -1), RIRs[i_loc])
        if data_room.shape[1] % L_hop:
            data_room = np.append(
                data_room,
                np.zeros((data_room.shape[0], L_hop - data_room.shape[1] % L_hop)),
                axis=1
            )
        # fname = '%04d_%02d_room.wav' % (*args, i_loc)
        # sf.write(os.path.join(DIR_IV, fname), data_room.T, Fs)
        # print(fname)
        data_room = cp.array(data_room)

        # Propagation
        data = np.append(np.zeros(t_peak[i_loc]), data_np*amp_peak[i_loc])
        if data.shape[0] % L_hop:
            data = np.append(data, np.zeros(L_hop - data.shape[0] % L_hop))
        data = cp.array(data)

        N_frame_free = data.shape[0] // L_hop - 1

        # data_0 \
        #     = cp.array(scsig.fftconvolve(cp.asnumpy(data.reshape(1, -1)),
        #                                  RIRs_0[i_loc]))

        # Energy using 0-th Order RIR
        # iv_0 = cp.zeros((N_freq, N_frame_room, 4))
        # for i_frame in range(N_frame_room):
        #     interval = i_frame*L_hop + np.arange(L_frame)
        #     fft = cp.fft.fft(data_0[:, interval]*win_cp, n=N_fft)
        #     anm = (sftdata_cp.Yenc @ fft)  # * sftdata_cp.bEQspec
        #
        #     iv_0[:, i_frame, :3] \
        #         = calc_intensity(anm[:, :N_freq],
        #                                       *sftdata_cp.get_triags())
        #     iv_0[:, i_frame, 3] \
        #         = cp.sum(cp.abs(anm[:, :N_freq])**2, axis=0)

        # Free-field Intensity Vector Image
        anm_time = cp.outer(Ys_cp[i_loc].conj(), data)

        iv_free = cp.empty((N_freq, N_frame_free, 4))
        phase_free = cp.empty((N_freq, N_frame_free, 1))
        for i_frame in range(N_frame_free):
            interval = i_frame * L_hop + np.arange(L_frame)
            anm = cp.fft.fft(anm_time[:, interval] * win_cp, n=N_fft)[:, :N_freq]

            if USE_DIRAC:
                # DirAC and a00
                iv_free[:, i_frame, :3] = calc_direction_vec(anm)
                iv_free[:, i_frame, 3] = cp.abs(anm[0])**2
                phase_free[:, i_frame, 0] = cp.angle(anm[0])
            else:
                # IV and p00
                pnm = anm * sftdata_cp.bnkr
                iv_free[:, i_frame, :3] = calc_intensity(
                    pnm, *sftdata_cp.get_for_intensity()
                )
                iv_free[:, i_frame, 3] = cp.abs(pnm[0])**2
                phase_free[:, i_frame, 0] = cp.angle(pnm[0])

        # Room Intensity Vector Image
        pnm_time = sftdata_cp.Yenc @ data_room

        iv_room = cp.empty((N_freq, N_frame_room, 4))
        phase_room = cp.empty((N_freq, N_frame_room, 1))
        for i_frame in range(N_frame_room):
            interval = i_frame * L_hop + np.arange(L_frame)
            pnm = cp.fft.fft(pnm_time[:, interval] * win_cp, n=N_fft)[:, :N_freq]

            if USE_DIRAC:
                # DirAC and a00
                anm = (pnm * sftdata_cp.bEQf)
                iv_room[:, i_frame, :3] = calc_direction_vec(anm)
                iv_room[:, i_frame, 3] = cp.abs(anm[0])**2
                phase_room[:, i_frame, 0] = cp.angle(anm[0])
            else:
                # IV and p00
                iv_room[:, i_frame, :3] = calc_intensity(
                    pnm, *sftdata_cp.get_for_intensity()
                )
                iv_room[:, i_frame, 3] = cp.abs(pnm[0])**2
                phase_room[:, i_frame, 0] = cp.angle(pnm[0])

        # Save
        dict_to_save = dict(fname_wav=fname_wav,
                            IV_free=cp.asnumpy(iv_free),
                            IV_room=cp.asnumpy(iv_room),
                            phase_free=cp.asnumpy(phase_free),
                            phase_room=cp.asnumpy(phase_room),
                            # IV_0=cp.asnumpy(iv_0),
                            )
        fname = FORM % (*args, i_loc)
        dd.io.save(os.path.join(DIR_IV, fname), dict_to_save, compression=None)

        print(fname)


def print_save_info():
    """
    Print __str__ and save metadata.
    """
    print(f'Wave Files Processed/Total: '
          f'{N_wavfile}/{len(all_files)}\n'
          f'Sample Rate: {Fs}\n'
          f'Number of source location: {N_LOC}\n'
          )

    metadata = dict(N_wavfile=N_wavfile,
                    Fs=Fs,
                    N_fft=N_fft,
                    N_freq=N_freq,
                    L_frame=L_frame,
                    L_hop=L_hop,
                    N_LOC=N_LOC,
                    path_wavfiles=all_files,
                    )

    dd.io.save(os.path.join(DIR_IV, 'metadata.h5'), metadata)


if __name__ == '__main__':
    process()
