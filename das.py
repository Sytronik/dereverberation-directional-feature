import multiprocessing as mp
import os
import shutil
from argparse import ArgumentParser

import numpy as np
import scipy.io as scio
from numpy import newaxis as newax
from tqdm import tqdm

from hparams import hp

rooms = [1, 2, 3, 4, 5, 6, 7, 9]
kinds = ['TEST/UNSEEN']


def take_conj_symm(spec, freq_axis=0):
    """ take conjugate symmetric part of STFT data.

    spec: (..., n_fft, ...) -- the length of the `freq_axis`-th axis is `n_fft`.

    """
    spec = np.moveaxis(spec, freq_axis, 0)  # n_fft, ...

    n_fft = len(spec)
    half = int(np.ceil(n_fft / 2))
    n_freq = n_fft // 2 + 1

    spec[1:half] += spec[:-half:-1].conj()
    spec[1:half] /= 2
    spec[0] = spec[0].real
    spec = spec[:n_freq]  # n_freq, ...

    spec = np.moveaxis(spec, 0, freq_axis)  # ..., n_freq, ...

    return spec


def process(i_room: int, queue):
    path_result \
        = hp.path_feature / f'{hp.folder_das_phase}_room{i_room}{hp.das_err}'
    path_result.mkdir(exist_ok=True)

    Ys_dict = scio.loadmat(
        str(hp.path_feature / f'RIR_Ys_room{i_room}{hp.das_err}.mat'),
        variable_names=('Ys_TEST'),
        squeeze_me=True,
    )
    Ys = dict()
    Ys['TEST/UNSEEN'] = Ys_dict['Ys_TEST'].T.astype(np.complex64)[:, :n_hrm]
    # --> n_loc x n_hrm
    # Ys = {k: v / np.linalg.norm(v, axis=1, keepdims=True) for k, v in Ys.items()}

    for kind in kinds:
        path = hp.path_feature / f'mulspec_room{i_room}' / kind
        if not path.exists():
            queue.put(f'"mulspec_room{i_room}/{kind}" does not exists.')
            continue

        os.makedirs(path_result / kind, exist_ok=True)

        metadata = scio.loadmat(path / 'metadata.mat',
                                chars_as_strings=True, squeeze_me=True)
        metadata['list_fname'] = [
            f.rstrip().replace('.npz', '.npy') for f in metadata['list_fname']
        ]
        scio.savemat(path_result / kind / 'metadata.mat', metadata)

        flist = list(path.glob('*.npz'))
        for fpath in flist:
            idx, i_speech, room, i_loc = fpath.stem.split('_')
            i_loc = int(i_loc)

            with np.load(fpath) as npz:
                specs = npz['dirspec_room']  # 32ch mag - 32ch phase
            specs = specs[..., :n_mic] * np.exp(1j * specs[..., n_mic:])

            specs = np.concatenate(
                (specs, specs[-2:0:-1].conj()), axis=0
            )  # n_fft x T x n_mic

            # SFT and bnkr equalization
            spec_shd = specs @ Yenc[:, :n_hrm]  # n_fft x T x n_hrm
            spec_shd *= bnkr_inv[..., :n_hrm]

            # DAS beamforming
            spec_est = spec_shd[..., newax, :] @ Ys[kind][i_loc][:, newax]
            # --> n_fft x T x 1 x 1

            spec_est = spec_est.squeeze()  # n_fft x T

            # take phase of the conj. symm. part of the freq-domain signal
            # which is phase of the real part of the time-domain signal
            spec_est = take_conj_symm(spec_est)
            phase_est = np.angle(spec_est)

            new_fpath = path_result / kind / f'{fpath.stem}.npy'
            np.save(new_fpath, phase_est)
            queue.put(new_fpath)
    queue.put('done')


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--num_threads', type=int, default=mp.cpu_count())
    args = hp.parse_argument(parser, print_argument=False)

    if hp.feature == 'DV' or hp.feature == 'mulspec4':
        n_hrm = (1 + 1)**2  # 4
    else:
        n_hrm = (3 + 1)**2  # 16

    np_env_vars = (
        'OMP_NUM_THREADS',
        'OPENBLAS_NUM_THREADS',
        'MKL_NUM_THREADS',
        'VECLIB_MAXIMUM_THREADS',
        'NUMEXPR_NUM_THREADS',
    )
    for var in np_env_vars:
        os.environ[var] = str(args.num_threads // len(rooms))

    # bnkr_inv, Yenc
    sft_dict = scio.loadmat(
        str(hp.path_feature / 'sft_data_32ms.mat'),
        # 'sft_data.mat',
        variable_names=('bEQf', 'Yenc'),
        squeeze_me=True,
    )
    Yenc = sft_dict['Yenc']
    n_mic = Yenc.shape[0]
    Yenc /= np.sqrt(4 * np.pi) / n_mic
    Yenc = Yenc.astype(np.complex64)  # n_mic x n_hrm

    bnkr_inv = sft_dict['bEQf'][:, newax, :]  # n_freq x 1 x n_hrm
    bnkr_inv = bnkr_inv.astype(np.complex64)
    bnkr_inv = np.concatenate(
        (bnkr_inv, bnkr_inv[-2:0:-1].conj()), axis=0
    )  # n_fft x 1 x n_hrm

    with mp.Manager() as manager:
        queue = manager.Queue()

        pool = mp.pool.Pool(len(rooms))
        pool.starmap_async(process, [(i, queue) for i in rooms])
        # for i in rooms:
        #     process(i, queue)
        pool.close()

        pbar = tqdm(dynamic_ncols=True)
        count = 0
        while count < len(rooms):
            # new_fpath, phase_est = queue.get()
            # np.save(new_fpath, phase_est)
            message = queue.get()
            pbar.set_postfix_str(message)
            pbar.update()
            if message == 'done':
                count += 1

        pbar.close()

        pool.join()
