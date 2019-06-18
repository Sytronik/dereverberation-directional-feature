""" create directional spectrogram.

--init option forces to start from the first data.
--dirac option is for using dirac instead of spatially average intensity.

Ex)
python create_dirspec.py TRAIN
python create_dirspec.py UNSEEN --init
python create_dirspec.py SEEN --dirac
python create_dirspec.py TRAIN --dirac --init
"""

# noinspection PyUnresolvedReferences
import logging
import multiprocessing as mp
import os
from argparse import ArgumentParser, ArgumentError
from collections import defaultdict
from pathlib import Path
from typing import Tuple, TypeVar, Optional
from dataclasses import dataclass, asdict
import cupy as cp
import deepdish as dd

import librosa
import numpy as np
import scipy.io as scio
import scipy.signal as scsig
import soundfile as sf
from tqdm import tqdm

from hparams import hp
NDArray = TypeVar('NDArray', np.ndarray, cp.ndarray)


@dataclass
class SFTData:
    """ Constant Matrices/Vectors for Spherical Fourier Analysis

    """
    Yenc: NDArray = None
    bnkr_inv: NDArray = None
    bnkr: Optional[NDArray] = None
    Wnv: Optional[NDArray] = None
    Wpv: Optional[NDArray] = None
    Vv: Optional[NDArray] = None
    T_real: Optional[NDArray] = None
    bn_sel2_0: Optional[NDArray] = None
    bn_sel2_1: Optional[NDArray] = None
    bn_sel3_0: Optional[NDArray] = None
    bn_sel3_1: Optional[NDArray] = None
    bn_sel_4_0: Optional[NDArray] = None
    bn_sel_4_1: Optional[NDArray] = None

    def get_for_intensity(self) -> Tuple:
        return (self.Wnv, self.Wpv, self.Vv,
                self.bn_sel2_0, self.bn_sel2_1,
                self.bn_sel3_0, self.bn_sel3_1,
                self.bn_sel_4_0, self.bn_sel_4_1)


def stft(data: NDArray, N_frame: int, _win: NDArray):
    xp = cp.get_array_module(data)
    spec = xp.empty((data.shape[0], hp.n_freq, N_frame), dtype=xp.complex128)
    interval = np.arange(hp.l_frame)
    for i_frame in range(N_frame):
        spec[:, :, i_frame] \
            = xp.fft.fft(data[:, interval] * _win, n=hp.n_fft)[:, :hp.n_freq]
        interval += hp.l_hop

    return spec


# noinspection PyShadowingNames
def seltriag(Ain: NDArray, nrord: int, shft: Tuple[int, int]) -> NDArray:
    """ select spherical harmonics coefficients from Ain
     with $N$-`nrord` order, $m$+`shft[0]`, $n$+`shift[1]`

    :param Ain:
    :param nrord:
    :param shft:
    :return:
    """
    xp = cp.get_array_module(Ain)
    other_shape = Ain.shape[1:] if Ain.ndim > 1 else tuple()
    N = int(np.ceil(np.sqrt(Ain.shape[0])) - 1)
    idx = 0
    len_new = (N - nrord + 1)**2

    Aout = xp.zeros((len_new, *other_shape), dtype=Ain.dtype)
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
    """ Asv(anm) (Order x ...) -> Intensity (... x 3)

    :param Asv:
    :param Wnv:
    :param Wpv:
    :param Vv:
    :param bn_sel2_0:
    :param bn_sel2_1:
    :param bn_sel3_0:
    :param bn_sel3_1:
    :param bn_sel_4_0:
    :param bn_sel_4_1:
    :return:
    """

    xp = cp.get_array_module(Asv)
    other_shape = Asv.shape[1:] if Asv.ndim > 1 else tuple()

    aug1 = seltriag(Asv, 1, (0, 0))
    aug2 = (bn_sel2_0 * seltriag(Wpv, 1, (1, -1)) * seltriag(Asv, 1, (1, -1))
            - bn_sel2_1 * seltriag(Wnv, 1, (0, 0)) * seltriag(Asv, 1, (-1, -1)))
    aug3 = (bn_sel3_0 * seltriag(Wpv, 1, (0, 0)) * seltriag(Asv, 1, (-1, 1))
            - bn_sel3_1 * seltriag(Wnv, 1, (1, 1)) * seltriag(Asv, 1, (1, 1)))
    aug4 = (bn_sel_4_0 * seltriag(Vv, 1, (0, 0)) * seltriag(Asv, 1, (-1, 0))
            + bn_sel_4_1 * seltriag(Vv, 1, (1, 0)) * seltriag(Asv, 1, (1, 0)))

    aug1 = aug1.conj()
    intensity = xp.empty((*other_shape, 3))
    intensity[..., 0] = (aug1 * (aug2 + aug3)).real.sum(axis=0) / 2
    intensity[..., 1] = (aug1 * (aug2 - aug3) / 2j).real.sum(axis=0)
    intensity[..., 2] = (aug1 * aug4).real.sum(axis=0)

    return 0.5 * intensity


def calc_mat_for_real_coeffs(N: int) -> np.ndarray:
    """ calculate matrix to convert complex SH coeffs to real

    :param N: n-order
    :return: (Order x Order)
    """
    matrix = np.zeros(((N + 1)**2, (N + 1)**2), dtype=np.complex128)
    matrix[0, 0] = 1
    if N > 0:
        idxs = (np.arange(N + 1) + 1)**2

        for n in range(1, N + 1):
            m1 = np.arange(n)
            diag = np.concatenate((np.full(n, 1j), (0,), -(-1)**m1))

            m2 = m1[::-1]
            anti_diag = np.concatenate((1j * (-1)**m2, (0,), np.ones(n)))

            block = (np.diagflat(diag) + np.diagflat(anti_diag)[:, ::-1]) / np.sqrt(2)
            block[n, n] = 1.

            matrix[idxs[n - 1]:idxs[n], idxs[n - 1]:idxs[n]] = block

    return matrix.conj()


def calc_direction_vec(anm: NDArray) -> NDArray:
    """ Calculate direciton vector in DirAC
     using Complex Spherical Harmonics Coefficients

    :param anm: (Order x ...)
    :return: (... x 3)
    """
    direction = 1. / np.sqrt(2) * (anm[0].conj() * anm[[3, 1, 2]]).real

    return direction.transpose([*range(1, anm.ndim), 0])


def process(idx_start: int):
    global pbar

    print_save_info(idx_start)
    pool_propagater = mp.Pool(mp.cpu_count() - n_cuda_dev - hp.num_disk_workers - 1)
    pool_creator = mp.Pool(n_cuda_dev)
    pool_saver = mp.Pool(hp.num_disk_workers)
    with mp.Manager() as manager:
        q_data = [manager.Queue(int(40 / 0.02135 / n_cuda_dev)) for _ in hp.device]
        q_dirspec = manager.Queue()

        # apply creater first
        # creater gets data from q_data, and sends dirspec to q_dirspec
        pool_creator.starmap_async(
            create_dirspecs,
            [(dev,
              q_data[idx],
              len(all_files[idx_start + idx::n_cuda_dev]) * n_loc,
              q_dirspec)
             for idx, dev in enumerate(hp.device)]
        )
        pool_creator.close()

        # apply propagater
        # propagater sends data to q_data
        pbar = tqdm(range(num_wavs),
                    desc='apply', dynamic_ncols=True, initial=idx_start)
        range_file = range(idx_start, num_wavs)
        for i_wav, f_wav in zip(range_file, all_files[idx_start:]):
            data, _ = sf.read(str(f_wav))

            for i_loc, RIR in enumerate(RIRs):
                pool_propagater.apply_async(
                    propagate,
                    (i_wav, f_wav,
                     data, i_loc, RIR,
                     q_data[(i_wav - idx_start) % n_cuda_dev])
                )
            pbar.update()
        pool_propagater.close()

        # apply saver
        # saver gets dirspec from q_dirspec
        pbar = tqdm(range(num_wavs),
                    desc='create', dynamic_ncols=True, initial=idx_start)
        for idx in range(len(range_file) * n_loc):
            pool_saver.apply_async(save_dirspec, q_dirspec.get(),
                                   callback=update_pbar)
            str_qsizes = ' '.join([f'{q.qsize()}' for q in q_data])
            pbar.set_postfix_str(f'[{str_qsizes}], {q_dirspec.qsize()}')
        pool_saver.close()

        pool_propagater.join()
        pool_creator.join()
        pool_saver.join()

    print_save_info(idx_start
                    + sum([1 for v in dict_count.values() if v >= n_loc]))


def propagate(i_wav: int, f_wav: Path,
              data: np.ndarray, i_loc: int, RIR: np.ndarray,
              queue: mp.Queue):
    # if (DIR_DIRSPEC / FORM % (i_wav, i_loc)).exists():
    #     return
    N_frame_room = int(np.ceil((data.shape[0] + len_RIR - 1) / hp.l_hop) - 1)

    # RIR Filtering
    data_room = scsig.fftconvolve(data[np.newaxis, :], RIR)
    if data_room.shape[1] % hp.l_hop:
        data_room = np.append(
            data_room,
            np.zeros((data_room.shape[0], hp.l_hop - data_room.shape[1] % hp.l_hop)),
            axis=1
        )

    # Propagation
    data = np.append(np.zeros(t_peak[i_loc]), data * amp_peak[i_loc])
    if data.shape[0] % hp.l_hop:
        data = np.append(data, np.zeros(hp.l_hop - data.shape[0] % hp.l_hop))

    N_frame_free = data.shape[0] // hp.l_hop - 1

    queue.put((i_wav, f_wav, i_loc, data, data_room, N_frame_free, N_frame_room))


def create_dirspecs(i_dev: int, q_data: mp.Queue, N_data: int, q_dirspec: mp.Queue):
    """ create directional spectrogram.

    :param i_dev: GPU Device No.
    :param q_data:
    :param N_data:
    :param q_dirspec:

    :return: None
    """

    # Ready CUDA
    cp.cuda.Device(i_dev).use()
    win_cp = cp.array(win)
    Ys_cp = cp.array(Ys)
    sftdata_cp = SFTData(
        **{k: cp.array(v) for k, v in asdict(sftdata).items() if v is not None}
    )

    for _ in range(N_data):
        i_wav, f_wav, i_loc, data, data_room, N_frame_free, N_frame_room = q_data.get()
        data_cp = cp.array(data)
        data_room_cp = cp.array(data_room)

        # Free-field Intensity Vector Image
        anm_time = cp.outer(Ys_cp[i_loc].conj(), data_cp) * np.sqrt(4 * np.pi)
        if use_dirac:  # real coefficients
            anm_time = cp.einsum('ij,jt->it', sftdata_cp.T_real, anm_time).real

        anm_spec = stft(anm_time, N_frame_free, win_cp)

        dirspec_free = cp.empty((hp.n_freq, N_frame_free, 4))
        if use_dirac:
            # DirAC and a00
            dirspec_free[:, :, :3] = calc_direction_vec(anm_spec)
            dirspec_free[:, :, 3] = cp.abs(anm_spec[0])
            phase_free = cp.angle(anm_spec[0])
        else:
            # IV and p00
            pnm_spec = anm_spec * sftdata_cp.bnkr

            dirspec_free[:, :, :3] = calc_intensity(
                pnm_spec, *sftdata_cp.get_for_intensity()
            )
            dirspec_free[:, :, 3] = cp.abs(pnm_spec[0])
            phase_free = cp.angle(pnm_spec[0])

        # Room Intensity Vector Image
        pnm_time = sftdata_cp.Yenc @ data_room_cp
        dirspec_room = cp.empty((hp.n_freq, N_frame_room, 4))
        if use_dirac:
            # DirAC and a00
            # bnkr equalization in frequency domain
            length = pnm_time.shape[1]
            pnm_time = cp.pad(pnm_time,
                              ((0, 0), (int(hp.n_fft // 2), int(hp.n_fft // 2))),
                              mode='reflect')
            N_frame = int(np.ceil(length / hp.l_hop))
            anm_time = cp.zeros((pnm_time.shape[0], (N_frame + 1) * hp.l_hop),
                                dtype=cp.complex128)
            interval = cp.arange(hp.l_frame)
            for i_frame in range(N_frame):
                pnm_spec = cp.fft.fft(pnm_time[:, interval] * win_cp, n=hp.n_fft)
                anm_time[:, interval] \
                    += cp.fft.ifft(pnm_spec * sftdata_cp.bnkr_inv, n=hp.n_fft) * win_cp
                interval += hp.l_hop

            # compensate artifact of stft/istft
            # noinspection PyTypeChecker
            artifact = librosa.filters.window_sumsquare(
                'hann',
                N_frame, win_length=hp.l_frame, n_fft=hp.n_fft, hop_length=hp.l_hop,
                dtype=np.float64
            )
            idxs_artifact = artifact > librosa.util.tiny(artifact)
            artifact = cp.array(artifact[idxs_artifact])

            anm_time[:, idxs_artifact] /= artifact
            anm_time = anm_time[:, int(hp.n_fft // 2):int(hp.n_fft // 2) + length]

            # real coefficients
            anm_t_real = cp.einsum('ij,jt->it', sftdata_cp.T_real, anm_time).real
            anm_spec_real = stft(anm_t_real, N_frame_room, win_cp)

            dirspec_room[:, :, :3] = calc_direction_vec(anm_spec_real)
            dirspec_room[:, :, 3] = cp.abs(anm_spec_real[0])
            phase_room = cp.angle(anm_spec_real[0])
        else:
            # IV and p00
            pnm_spec = stft(pnm_time, N_frame_room, win_cp)

            dirspec_room[:, :, :3] = calc_intensity(
                pnm_spec, *sftdata_cp.get_for_intensity()
            )
            dirspec_room[:, :, 3] = cp.abs(pnm_spec[0])
            phase_room = cp.angle(pnm_spec[0])

        # Save
        dict_dirspec = dict(fname_wav=f_wav,
                            dirspec_free=cp.asnumpy(dirspec_free),
                            dirspec_room=cp.asnumpy(dirspec_room),
                            phase_free=cp.asnumpy(phase_free)[..., np.newaxis],
                            phase_room=cp.asnumpy(phase_room)[..., np.newaxis],
                            )
        q_dirspec.put((i_wav, i_loc, dict_dirspec))


def save_dirspec(i_wav: int, i_loc: int, dict_dirspec: dict) -> Tuple[int, int]:
    dd.io.save(path_dirspec / (hp.form_dirspec.format(i_wav, i_loc)), dict_dirspec,
               compression=None)
    return i_wav, i_loc


def update_pbar(tup):
    global dict_count
    i_wav, i_loc = tup
    # pbar.display(FORM % (i_wav, i_loc))
    dict_count[i_wav] += 1
    if dict_count[i_wav] >= n_loc:
        pbar.update()


def print_save_info(i_wav):
    """ Print and save metadata.

    """
    print(f'Wave Files Processed/Total: {i_wav}/{len(all_files)}\n'
          f'Number of source location: {n_loc}\n')

    metadata = dict(Fs=hp.fs,
                    N_fft=hp.n_fft,
                    N_freq=hp.n_freq,
                    L_frame=hp.l_frame,
                    L_hop=hp.l_hop,
                    N_LOC=n_loc,
                    path_wavfiles=all_files,
                    )

    scio.savemat(f_metadata, metadata)


if __name__ == '__main__':
    # determined by sys argv
    parser = ArgumentParser()
    parser.add_argument('room_create')
    parser.add_argument('kind_data',
                        choices=('TRAIN', 'train',
                                 'SEEN', 'seen',
                                 'UNSEEN', 'unseen',
                                 ),
                        )
    parser.add_argument('-t', dest='dirspec_folder', default='')
    parser.add_argument('--dirac', action='store_true')
    parser.add_argument('--from', type=int, default=-1,
                        dest='from_idx')
    args = hp.parse_argument(parser)
    use_dirac = hp.DF == 'DirAC'
    n_cuda_dev = len(hp.device)

    # Paths
    if args.dirspec_folder:
        path_dirspec = hp.dict_path['path_feature'] / args.dirspec_folder
    else:
        path_dirspec = hp.dict_path[f'dirspec_{args.kind_data.lower()}']

    if args.kind_data.lower() == 'train':
        path_wavs = hp.dict_path['wav_train']
    else:
        path_dirspec = path_dirspec / 'TEST'
        path_wavs = hp.dict_path['wav_test']

    path_dirspec = path_dirspec / args.kind_data.upper()
    os.makedirs(path_dirspec, exist_ok=True)

    # RIR Data
    transfer_dict = scio.loadmat(str(hp.dict_path['RIR_Ys']), squeeze_me=True)
    kind_RIR = 'TEST' if args.kind_data.lower() == 'unseen' else 'TRAIN'
    RIRs = transfer_dict[f'RIR_{kind_RIR}'].transpose((2, 0, 1))
    n_loc, n_mic, len_RIR = RIRs.shape
    Ys = transfer_dict[f'Ys_{kind_RIR}'].T  # N_LOC x Order

    t_peak = np.round(RIRs.argmax(axis=2).mean(axis=1)).astype(int)
    amp_peak = RIRs.max(axis=2).mean(axis=1)

    # SFT Data
    sftdata = SFTData()
    sft_dict = scio.loadmat(
        str(hp.dict_path['sft_data']),
        variable_names=('bmn_ka', 'bEQf', 'Yenc', 'Wnv', 'Wpv', 'Vv'),
        squeeze_me=True
    )
    sftdata.bnkr_inv = sft_dict['bEQf'].T[:, :, np.newaxis]  # Order x N_freq x 1
    sftdata.Yenc = sft_dict['Yenc'].T  # Order x N_MIC

    if use_dirac:
        Ys = Ys[:, :4]
        sftdata.Yenc = sftdata.Yenc[:4]
        bnkr_inv = sftdata.bnkr_inv[:4]
        sftdata.bnkr_inv = np.concatenate(
            (sftdata.bnkr_inv, sftdata.bnkr_inv[:, -2:0:-1].conj()), axis=1
        ).squeeze()  # Order x N_fft
        sftdata.T_real = calc_mat_for_real_coeffs(1)
    else:
        sftdata.bnkr = sft_dict['bmn_ka'].T[:, :, np.newaxis] / (4 * np.pi)  # Order x N_freq
        sftdata.Wnv = sft_dict['Wnv'].astype(complex)[:, np.newaxis, np.newaxis]
        sftdata.Wpv = sft_dict['Wpv'].astype(complex)[:, np.newaxis, np.newaxis]
        sftdata.Vv = sft_dict['Vv'].astype(complex)[:, np.newaxis, np.newaxis]

        sftdata.bn_sel2_0 = (seltriag(1. / sftdata.bnkr, 1, (1, -1))
                             * seltriag(sftdata.bnkr, 1, (0, 0)))
        sftdata.bn_sel2_1 = (seltriag(1. / sftdata.bnkr, 1, (-1, -1))
                             * seltriag(sftdata.bnkr, 1, (0, 0)))

        sftdata.bn_sel3_0 = (seltriag(1. / sftdata.bnkr, 1, (-1, 1))
                             * seltriag(sftdata.bnkr, 1, (0, 0)))
        sftdata.bn_sel3_1 = (seltriag(1. / sftdata.bnkr, 1, (1, 1))
                             * seltriag(sftdata.bnkr, 1, (0, 0)))

        sftdata.bn_sel_4_0 = (seltriag(1. / sftdata.bnkr, 1, (-1, 0))
                              * seltriag(sftdata.bnkr, 1, (0, 0)))
        sftdata.bn_sel_4_1 = (seltriag(1. / sftdata.bnkr, 1, (1, 0))
                              * seltriag(sftdata.bnkr, 1, (0, 0)))

    del sft_dict

    win = scsig.windows.hann(hp.l_frame, sym=False)

    f_metadata = path_dirspec / 'metadata.h5'
    if f_metadata.exists():
        all_files = dd.io.load(f_metadata)['path_wavfiles']
    else:
        all_files = list(path_wavs.glob('**/*.WAV')) + list(path_wavs.glob('**/*.wav'))

    num_wavs = len(all_files)
    if num_wavs < args.from_idx:
        raise ArgumentError

    # The index of the first wave file that have to be processed
    idx_exist = -2
    should_ask_cont = False
    for i_wav in range(num_wavs):
        if len(list(path_dirspec.glob(f'{i_wav:04d}_*.h5'))) < n_loc:
            idx_exist = i_wav - 1
            break
    if args.from_idx == -1:
        if idx_exist == -2:
            print_save_info(num_wavs)
            exit(0)
        idx_start = idx_exist + 1
    else:
        idx_start = args.from_idx
        should_ask_cont = True

    print(f'Start processing from the {idx_start}-th wave file.')
    if should_ask_cont:
        ans = input(f'{idx_exist} wave files were already processed. continue? (y/n)')
        if not ans.startswith('y'):
            exit(0)

    # objects
    pbar = None
    dict_count = defaultdict(lambda: 0)

    process(idx_start)
