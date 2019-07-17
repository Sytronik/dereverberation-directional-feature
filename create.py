""" create directional spectrogram.

--init option forces to start from the first data.
--dirac option is for using dirac instead of spatially average intensity.

Ex)
python create.py TRAIN
python create.py UNSEEN --init
python create.py SEEN --dirac
python create.py TRAIN --dirac --init
"""

# noinspection PyUnresolvedReferences
import logging
import multiprocessing as mp
import os
from argparse import ArgumentParser, ArgumentError
from pathlib import Path
from typing import Tuple, TypeVar, Optional, List
from dataclasses import dataclass, asdict
from itertools import product
import cupy as cp

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
    Wnv: Optional[NDArray] = None
    Wpv: Optional[NDArray] = None
    Vv: Optional[NDArray] = None
    T_real: Optional[NDArray] = None

    def get_for_intensity(self) -> Tuple:
        return self.Wnv, self.Wpv, self.Vv


def stft(data: NDArray, _win: NDArray):
    xp = cp.get_array_module(data)
    data = xp.pad(data,
                  ((0, 0), (hp.n_fft // 2, hp.n_fft // 2)),
                  mode='reflect')

    n_frame = (data.shape[1] - hp.l_frame) // hp.l_hop + 1

    spec = xp.empty((data.shape[0], hp.n_freq, n_frame), dtype=xp.complex128)
    interval = np.arange(hp.l_frame)
    for i_frame in range(n_frame):
        spec[:, :, i_frame] \
            = xp.fft.fft(data[:, interval] * _win, n=hp.n_fft)[:, :hp.n_freq]
        interval += hp.l_hop

    return spec


def apply_iir_filter(wave, filter_fft, _win):
    # bnkr equalization in frequency domain
    xp = cp.get_array_module(wave)

    len_original = wave.shape[1]
    wave = xp.pad(wave,
                  ((0, 0), (hp.n_fft // 2, hp.n_fft // 2)),
                  mode='reflect')

    n_frame = len_original // hp.l_hop
    len_istft = hp.n_fft + hp.l_hop * (n_frame - 1)

    filtered = xp.zeros((wave.shape[0], len_istft), dtype=xp.complex128)
    interval = np.arange(hp.l_frame)
    for i_frame in range(n_frame):
        spectrum = xp.fft.fft(wave[:, interval] * _win, n=hp.n_fft)
        filtered[:, interval] += xp.fft.ifft(
            spectrum * filter_fft, n=hp.n_fft
        ) * _win
        interval += hp.l_hop

    # compensate artifact of stft/istft
    # noinspection PyTypeChecker
    artifact = librosa.filters.window_sumsquare(
        'hann',
        n_frame, win_length=hp.l_frame, n_fft=hp.n_fft, hop_length=hp.l_hop,
        dtype=np.float64
    )
    idxs_artifact = artifact > librosa.util.tiny(artifact)
    artifact = xp.array(artifact[idxs_artifact])

    filtered[:, idxs_artifact] /= artifact
    filtered = filtered[:, hp.n_fft // 2:]
    filtered = filtered[:, :len_original]

    return filtered.astype(wave.dtype)


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
def calc_intensity(Asv: NDArray, Wnv: NDArray, Wpv: NDArray, Vv: NDArray) \
        -> NDArray:
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
    aug2 = (seltriag(Wpv, 1, (1, -1)) * seltriag(Asv, 1, (1, -1))
            - seltriag(Wnv, 1, (0, 0)) * seltriag(Asv, 1, (-1, -1)))
    aug3 = (seltriag(Wpv, 1, (0, 0)) * seltriag(Asv, 1, (-1, 1))
            - seltriag(Wnv, 1, (1, 1)) * seltriag(Asv, 1, (1, 1)))
    aug4 = (seltriag(Vv, 1, (0, 0)) * seltriag(Asv, 1, (-1, 0))
            + seltriag(Vv, 1, (1, 0)) * seltriag(Asv, 1, (1, 0)))

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


def process():
    global pbar

    print_save_info(idx_start)
    os.cpu_count()
    pool_propagater = mp.Pool(mp.cpu_count()//2 - n_cuda_dev - hp.num_disk_workers - 1)
    pool_creator = mp.Pool(n_cuda_dev)
    pool_saver = mp.Pool(hp.num_disk_workers)
    with mp.Manager() as manager:
        q_data = [manager.Queue(hp.num_disk_workers * 3) for _ in hp.device]
        q_result = manager.Queue()

        # open speech files
        speech = []
        for f_speech in flist_speech:
            speech.append(sf.read(str(f_speech))[0])

        # apply creater first
        # creater gets data from q_data, and sends the result to q_result
        pool_creator.starmap_async(
            create_dirspecs,
            [(dev,
              q_data[idx],
              len(list_feature[idx_start + idx::n_cuda_dev]),
              q_result)
             for idx, dev in enumerate(hp.device)]
        )
        pool_creator.close()

        # apply propagater
        # propagater sends data to q_data
        pbar = tqdm(range(n_feature),
                    desc='apply', dynamic_ncols=True, initial=idx_start)
        range_feature = range(idx_start, n_feature)
        for idx, (i_speech, _, i_loc) in zip(range_feature, list_feature[idx_start:]):
            pool_propagater.apply_async(
                propagate,
                (idx, i_speech, flist_speech[i_speech],
                 speech[i_speech], i_loc,
                 q_data[(idx - idx_start) % n_cuda_dev])
            )
            # propagate(idx, i_speech, flist_speech[i_speech],
            #           speech[i_speech], i_loc,
            #           q_data[(idx - idx_start) % n_cuda_dev])
            pbar.update()
        pool_propagater.close()

        # apply saver
        # saver gets results from q_result
        pbar = tqdm(range(n_feature),
                    desc='create', dynamic_ncols=True, initial=idx_start)
        for _ in range_feature:
            pool_saver.apply_async(save_result, q_result.get())
            str_qsizes = ' '.join([f'{q.qsize()}' for q in q_data])
            pbar.set_postfix_str(f'[{str_qsizes}], {q_result.qsize()}')
            pbar.update()
        pool_saver.close()

        pool_propagater.join()
        pool_creator.join()
        pool_saver.join()

    print_save_info(n_feature)


def propagate(idx: int, i_speech: int, f_speech: Path,
              data: np.ndarray, i_loc: int,
              queue: mp.Queue):
    # RIR Filtering
    data_room = scsig.fftconvolve(data[np.newaxis, :], RIRs[i_loc])

    # Propagation
    data = np.append(np.zeros(t_peak[i_loc]), data * amp_peak[i_loc])

    queue.put((idx, i_speech, f_speech, i_loc, data, data_room))


def create_dirspecs(i_dev: int, q_data: mp.Queue, n_data: int, q_result: mp.Queue):
    """ create directional spectrogram.

    :param i_dev: GPU Device No.
    :param q_data:
    :param n_data:
    :param q_result:

    :return: None
    """

    # Ready CUDA
    cp.cuda.Device(i_dev).use()
    win_cp = cp.array(win)
    Ys_cp = cp.array(Ys)
    sftdata_cp = SFTData(
        **{k: cp.array(v) for k, v in asdict(sftdata).items() if v is not None}
    )

    for _ in range(n_data):
        idx, i_speech, f_speech, i_loc, data, data_room = q_data.get()
        data_cp = cp.array(data)
        data_room_cp = cp.array(data_room)

        # Free-field
        anm_time_cp = cp.outer(Ys_cp[i_loc].conj(), data_cp)
        if use_dirac:  # real coefficients
            anm_time_cp = (sftdata_cp.T_real @ anm_time_cp).real

        anm_spec_cp = stft(anm_time_cp, win_cp)

        # dirspec_free = np.empty((hp.n_freq, n_frame_free, 4))
        if use_dirac:
            # DirAC and a00
            df_free = cp.asnumpy(calc_direction_vec(anm_spec_cp))
        else:
            # IV and p00
            df_free = cp.asnumpy(
                calc_intensity(anm_spec_cp, *sftdata_cp.get_for_intensity())
            )
        mag_free = cp.asnumpy(cp.abs(anm_spec_cp[0]))
        phase_free = cp.asnumpy(cp.angle(anm_spec_cp[0]))
        dirspec_free = np.concatenate((df_free, mag_free[..., np.newaxis]), axis=2)

        # Room Intensity Vector Image
        pnm_time_cp = sftdata_cp.Yenc @ data_room_cp
        # dirspec_room = np.empty((hp.n_freq, n_frame_room, 4))
        if use_dirac:
            # DirAC and a00
            # bnkr equalization in frequency domain
            anm_time_cp = apply_iir_filter(pnm_time_cp, sftdata_cp.bnkr_inv[..., 0], win_cp)

            # real coefficients
            anm_t_real_cp = (sftdata_cp.T_real @ anm_time_cp).real
            anm_spec_real_cp = stft(anm_t_real_cp, win_cp)

            df_room = cp.asnumpy(calc_direction_vec(anm_spec_real_cp))
            mag_room = cp.asnumpy(cp.abs(anm_spec_real_cp[0]))
            phase_room = cp.angle(anm_spec_real_cp[0])
        else:
            # IV and p00
            pnm_spec_cp = stft(pnm_time_cp, win_cp)
            anm_spec_cp = pnm_spec_cp * sftdata_cp.bnkr_inv[:, :hp.n_freq]
            df_room = cp.asnumpy(
                calc_intensity(anm_spec_cp, *sftdata_cp.get_for_intensity())
            )
            mag_room = cp.asnumpy(cp.abs(anm_spec_cp[0]))
            phase_room = cp.asnumpy(cp.angle(anm_spec_cp[0]))
        dirspec_room = np.concatenate((df_room, mag_room[..., np.newaxis]), axis=2)

        # Save
        dict_result = dict(speech_fname=str(f_speech),
                           dirspec_free=dirspec_free,
                           dirspec_room=dirspec_room,
                           phase_free=phase_free[..., np.newaxis],
                           phase_room=phase_room[..., np.newaxis],
                           )
        q_result.put((idx, i_speech, i_loc, dict_result))


def save_result(idx: int, i_speech: int, i_loc: int, dict_result: dict) -> Tuple[int, int]:
    np.savez(path_result / hp.form_feature.format(idx, i_speech, hp.room_create, i_loc),
             **dict_result)
    return i_speech, i_loc


def print_save_info(i_feature: int):
    """ Print and save metadata.

    """
    print(f'Feature files processed/total: {i_feature}/{len(list_feature)}\n'
          f'Number of source location: {n_loc}\n')

    metadata = dict(fs=hp.fs,
                    n_fft=hp.n_fft,
                    n_freq=hp.n_freq,
                    l_frame=hp.l_frame,
                    l_hop=hp.l_hop,
                    n_loc=n_loc,
                    path_all_speech=[str(p) for p in flist_speech],
                    list_fname=list_feature_to_fname(list_feature),
                    )

    scio.savemat(f_metadata, metadata)


def list_feature_to_fname(list_feature: List[Tuple]) -> List[str]:
    return [
        hp.form_feature.format(i, *tup) for i, tup in enumerate(list_feature)
    ]


def list_fname_to_feature(list_fname: List[str]) -> List[Tuple]:
    list_feature = []
    for f in list_fname:
        f = f.rstrip().rstrip('.npz')
        _, i_speech, _, i_loc = f.split('_')
        list_feature.append((int(i_speech), hp.room_create, int(i_loc)))
    return list_feature


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
    parser.add_argument('-t', dest='target_folder', default='')
    parser.add_argument('--from', type=int, default=-1,
                        dest='from_idx')
    args = hp.parse_argument(parser)
    use_dirac = hp.DF == 'DirAC'
    n_cuda_dev = len(hp.device)
    is_train = args.kind_data.lower() == 'train'

    # Paths
    path_speech = hp.dict_path['speech_train' if is_train else 'speech_test']

    if args.target_folder:
        path_result = hp.path_feature / args.target_folder
        if not is_train:
            path_result = path_result / 'TEST'
        path_result = path_result / args.kind_data.upper()
    else:
        path_result = hp.dict_path[f'feature_{args.kind_data.lower()}']
    os.makedirs(path_result, exist_ok=True)

    # RIR Data
    transfer_dict = scio.loadmat(str(hp.dict_path['RIR_Ys']), squeeze_me=True)
    kind_RIR = 'TEST' if args.kind_data.lower() == 'unseen' else 'TRAIN'
    RIRs = transfer_dict[f'RIR_{kind_RIR}'].transpose((2, 0, 1))
    n_loc, n_mic, len_RIR = RIRs.shape
    Ys = transfer_dict[f'Ys_{kind_RIR}'].T * np.sqrt(4 * np.pi)  # N_LOC x Order

    # SFT Data
    sftdata = SFTData()
    sft_dict = scio.loadmat(
        str(hp.dict_path['sft_data']),
        variable_names=('bEQf', 'Yenc', 'Wnv', 'Wpv', 'Vv'),
        squeeze_me=True
    )
    sftdata.Yenc = sft_dict['Yenc'].T / np.sqrt(4 * np.pi) / n_mic  # Order x N_MIC
    sftdata.bnkr_inv = sft_dict['bEQf'].T[..., np.newaxis]  # Order x N_freq x 1
    sftdata.bnkr_inv = np.concatenate(
        (sftdata.bnkr_inv, sftdata.bnkr_inv[:, -2:0:-1].conj()), axis=1
    )  # Order x N_fft x 1

    if use_dirac:
        Ys = Ys[:, :4]
        sftdata.Yenc = sftdata.Yenc[:4]
        sftdata.bnkr_inv = sftdata.bnkr_inv[:4]
        sftdata.T_real = calc_mat_for_real_coeffs(1)
    else:
        sftdata.Wnv = sft_dict['Wnv'].astype(complex)[:, np.newaxis, np.newaxis]
        sftdata.Wpv = sft_dict['Wpv'].astype(complex)[:, np.newaxis, np.newaxis]
        sftdata.Vv = sft_dict['Vv'].astype(complex)[:, np.newaxis, np.newaxis]

    del sft_dict

    # propagation
    win = scsig.windows.hann(hp.l_frame, sym=False)
    p00_RIRs = np.einsum('ijk,j->ik', RIRs, sftdata.Yenc[0])  # n_loc x time
    a00_RIRs = apply_iir_filter(p00_RIRs, sftdata.bnkr_inv[0, :, 0], win)

    t_peak = a00_RIRs.argmax(axis=1)
    amp_peak = a00_RIRs.max(axis=1)

    f_metadata = path_result / 'metadata.mat'
    if hp.s_path_metadata:
        f_reference_meta = Path(hp.s_path_metadata)
        if not f_reference_meta.exists():
            raise ArgumentError
    elif f_metadata.exists():
        f_reference_meta = f_metadata
    else:
        f_reference_meta = None

    if f_reference_meta:
        metadata = scio.loadmat(str(f_reference_meta),
                                variable_names=('path_all_speech', 'list_fname'),
                                chars_as_strings=True,
                                squeeze_me=True)
        flist_speech = metadata['path_all_speech']
        flist_speech = [Path(p.rstrip()) for p in flist_speech]
        n_speech = len(flist_speech)
        list_fname = metadata['list_fname']
        list_feature: List[Tuple] = list_fname_to_feature(list_fname)
        n_feature = len(list_feature)
    else:
        flist_speech = list(path_speech.glob('**/*.WAV')) + list(path_speech.glob('**/*.wav'))
        n_speech = len(flist_speech)
        list_feature = [(i_speech, hp.room_create, i_loc)
                        for i_speech, i_loc in product(range(n_speech), range(n_loc))]

        if args.kind_data.lower() == 'train':
            n_feature = hp.n_data_per_room
        else:
            n_feature = hp.n_test_per_room
        idx_choice = np.random.choice(len(list_feature), n_feature, replace=False)
        idx_choice.sort()
        list_feature: List[Tuple] = [list_feature[i] for i in idx_choice]

    if n_feature < args.from_idx:
        raise ArgumentError

    # The index of the first speech file that have to be processed
    idx_exist = -2  # -2 means all files already exist
    for idx, tup in enumerate(list_feature):
        fname = hp.form_feature.format(idx, *tup)
        if not (path_result / fname).exists():
            idx_exist = idx - 1
            break

    if args.from_idx == -1:
        if idx_exist == -2:
            print_save_info(n_speech)
            exit(0)
        idx_start = idx_exist + 1
        should_ask_cont = False
    else:
        idx_start = args.from_idx
        should_ask_cont = True

    print(f'Start processing from the {idx_start}-th speech file.')
    if should_ask_cont:
        while True:
            ans = input(f'{idx_exist} speech files were already processed. continue? (y/n)')
            if ans.lower() == 'y':
                break
            elif ans.lower() == 'n':
                exit(0)

    process()
