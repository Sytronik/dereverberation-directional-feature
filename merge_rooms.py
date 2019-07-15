import os
import multiprocessing as mp
from pathlib import Path
from argparse import ArgumentParser

import numpy as np
import deepdish as dd
from tqdm import tqdm

from hparams import hp


def symlink(_src_path, src_fnames, _dst_path, _n_loc_shift, q, idx):
    for f in src_fnames:
        f = f.replace('.h5', '')
        f_new = f"{f.split('_')[0]}_{int(f.split('_')[1]) + _n_loc_shift:02d}"
        src = _src_path / (f + '.h5')
        dst = _dst_path / (f_new + '.h5')
        if dst.is_symlink():
            dst_resolve = dst.resolve()
            src_resolve = src
            same = False
            while dst.resolve().name == src.name:
                if dst_resolve.parent.name != src_resolve.parent.name:
                    if dst_resolve.name.startswith(ARGS.DF):
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

    parser.add_argument('DF', choices=('IV', 'DirAC'))
    parser.add_argument('ROOMS', type=str, nargs='+')
    ARGS = parser.parse_args()

    PATHS_ROOMS = [hp.path_feature / f'{ARGS.DF}_{ROOM}' for ROOM in ARGS.ROOMS]
    PATH_MERGED = hp.path_feature / (f'{ARGS.DF}_' + '_'.join(ARGS.ROOMS))
    if not PATH_MERGED.is_dir():
        os.makedirs(PATH_MERGED)

    pool = mp.Pool(mp.cpu_count())
    pbars = []
    with mp.Manager() as manager:
        queue = manager.Queue()

        for kind in ('TRAIN', 'TEST/UNSEEN', 'TEST/SEEN'):
            dst_path = PATH_MERGED / kind
            if not dst_path.is_dir():
                os.makedirs(dst_path)
            fnames_rooms = [
                [f.name for f in os.scandir(p / kind) if hp.is_featurefile(f)]
                for p in PATHS_ROOMS
            ]
            metadata = dict()
            n_loc_shift = 0
            for path, fnames in zip(PATHS_ROOMS, fnames_rooms):
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
                    (src_path, fnames, dst_path, n_loc_shift, queue, len(pbars))
                )
                pbars.append(pbar)
                # symlink(src_path, fnames, dst_path, n_loc_shift, queue, len(pbars))
                metadata = dd.io.load(path / kind / 'metadata.h5')
                n_loc_shift += metadata['N_LOC']

            metadata['N_LOC'] = n_loc_shift
            dd.io.save(dst_path / 'metadata.h5', metadata)

        pool.close()
        for _ in range(sum((pbar.total for pbar in pbars))):
            pbars[queue.get()].update()

        for pbar in pbars:
            pbar.close()
        pool.join()
