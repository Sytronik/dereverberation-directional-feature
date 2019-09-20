import os
import multiprocessing as mp
from pathlib import Path
from argparse import ArgumentParser

import numpy as np
from tqdm import tqdm
import scipy.io as scio

from hparams import hp


def get_i_loc(fname: str):
    return int(fname.replace('.npz', '').split('_')[-1])


def symlink(_src_path, src_fnames, _dst_path, q, idx):
    for f in src_fnames:
        # f = f.replace('.h5', '')
        # f_new = f"{f.split('_')[0]}_{int(f.split('_')[1]) + _n_loc_shift:02d}"
        src = _src_path / f
        dst = _dst_path / f
        if dst.is_symlink():
            dst_resolve = dst.resolve()
            src_resolve = src
            same = False
            while dst.resolve().name == src.name:
                if dst_resolve.parent.name != src_resolve.parent.name:
                    if dst_resolve.name.startswith(args.DF):
                        same = True
                        break
                    else:
                        break
                dst_resolve = dst_resolve.parent
                src_resolve = src_resolve.parent
            if same:
                q.put(idx)
                continue
            os.remove(dst)
        os.symlink(src, dst)
        q.put(idx)


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('DF', choices=('IV', 'DirAC', 'mulspec'))
    parser.add_argument('ROOMS', type=str, nargs='+')  # multiple rooms (at least 1 room)
    parser.add_argument('--n_loc', type=int, nargs=2,
                        default=[-1, -1])  # multiple rooms (at least 1 room)

    args = parser.parse_args()

    PATHS_ROOMS = [hp.path_feature / f'{args.DF}_{ROOM}' for ROOM in args.ROOMS]
    s_num_rooms = [room.lstrip('room') for room in args.ROOMS]
    s_folder = f'{args.DF}_room' + '+'.join(s_num_rooms)
    s_folder += f'_{args.n_loc[0]}_{args.n_loc[1]}' if args.n_loc != [-1, -1] else ''
    PATH_MERGED = hp.path_feature / s_folder
    if not PATH_MERGED.is_dir():
        os.makedirs(PATH_MERGED)
    new_n_locs = {'TRAIN': args.n_loc[0],
                  'TEST/UNSEEN': args.n_loc[1],
                  'TEST/SEEN': args.n_loc[1],
                  }

    pool = mp.Pool(mp.cpu_count())
    pbars = []
    with mp.Manager() as manager:
        queue = manager.Queue()

        for kind in ('TRAIN', 'TEST/UNSEEN', 'TEST/SEEN'):
            if not all((p / kind).exists() for p in PATHS_ROOMS):
                continue
            dst_path = PATH_MERGED / kind
            os.makedirs(dst_path, exist_ok=True)
            # fnames_rooms = [
            #     [f.name for f in os.scandir(p / kind) if hp.is_featurefile(f)]
            #     for p in PATHS_ROOMS
            # ]
            metadata = dict()
            list_n_loc = []
            merged_list_fname = []
            for path in PATHS_ROOMS:
                metadata = scio.loadmat(path / kind / 'metadata.mat',
                                        chars_as_strings=True,
                                        squeeze_me=True)
                if new_n_locs[kind] == -1:
                    new_n_locs[kind] = metadata['n_loc']
                else:
                    assert new_n_locs[kind] <= metadata['n_loc']
                list_n_loc.append(new_n_locs[kind])
                fnames = [f.rstrip() for f in metadata['list_fname'] if
                          get_i_loc(f) < new_n_locs[kind]]
                merged_list_fname += fnames

                src_path = path / kind
                src_path = Path(
                    '../' * (list(dst_path.parents).index(hp.path_feature) + 1),
                    src_path.relative_to(hp.path_feature)
                )

                pbar = tqdm(total=len(fnames),
                            position=len(pbars),
                            desc=str(path / kind),
                            dynamic_ncols=True)
                pool.apply_async(
                    symlink,
                    (src_path, fnames, dst_path, queue, len(pbars))
                )
                pbars.append(pbar)
                # symlink(src_path, fnames, dst_path, n_loc_shift, queue, len(pbars))

            metadata['n_loc'] = list_n_loc
            metadata['rooms'] = args.ROOMS
            metadata['list_fname'] = merged_list_fname
            for k in list(metadata.keys()):
                if k.startswith('__'):
                    metadata.pop(k)
            scio.savemat(dst_path / 'metadata.mat', metadata)

        pool.close()
        for _ in range(sum((pbar.total for pbar in pbars))):
            pbars[queue.get()].update()

        for pbar in pbars:
            pbar.close()
        pool.join()
