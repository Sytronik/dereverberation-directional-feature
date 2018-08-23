"""
<<Usage>>
python convert_db.py [--no-duplicate]
    {
        {--mat | --h5 | --npy}
        {
            {directory path 1 | file path 1}
            {directory path 2 | file path 2}
            ...
        }
    }
    {
        {--mat | --h5 | --npy}
        {
            {directory path 1 | file path 1}
            {directory path 2 | file path 2}
            ...
        }
    }
    ...

* "--no-duplicate option" is applied to all files.
* Avaiable types of original file: .mat, .h5, .npy, .pt
"""

import numpy as np
import scipy.io as scio
import deepdish.io as ddio
import torch

import sys
import os
from glob import glob

import multiprocessing as mp


def main():
    argv = sys.argv
    for arg in sys.argv[1:]:
        if arg == '--no-duplicate':
            convert.duplicate = False
            argv.remove(arg)

    for arg in argv:
        if arg.startswith('-'):
            convert.to = '.' + arg.replace('-', '')
            if convert.to not in OPEN.keys():
                raise 'Choose a right file type(--mat, --h5, --npy).'
        elif os.path.isdir(arg):
            if not convert.to:
                raise 'Choose a file type.'
            pool = mp.Pool(mp.cpu_count())
            for folder, _, _ in os.walk(arg):
                files = glob(os.path.join(folder, '*.*'))
                files = [f for f in files if is_db(f)]
                if files:
                    pool.map_async(convert, files)
            pool.close()
            pool.join()
        elif os.path.isfile(arg):
            if not convert.to:
                raise 'Choose a file type.'
            convert(arg)
        else:
            raise 'File or directory does not exist.'


def open_mat(fname: str):
    return scio.loadmat(fname, squeeze_me=True)


def open_h5(fname: str):
    return ddio.load(fname)


def open_npy(fname: str):
    contents = np.load(fname)
    if contents.size == 1:
        contents = contents.item()
    return contents


def open_pt(fname: str):
    contents = torch.load(fname, map_location=torch.device('cpu'))

    contents = {key.replace('.', '_'): value.numpy()
                for key, value in contents.items()}

    length = max([len(k) for k in contents.keys()])
    for key, value in contents:
        print(f'{key:<{length}}: ndarray of shape {value.shape}')

    return contents


def save_mat(fname: str, contents):
    if type(contents) != dict:  # make contents dict
        contents = {os.path.basename(fname).replace('.mat', ''): contents}
    scio.savemat(fname, contents)


def save_h5(fname: str, contents):
    if type(contents) == dict and len(contents) == 1:
        exec(f'{list(contents)[0]} = {contents[list(contents)[0]]}')
    ddio.save(fname, eval(list(contents)[0]), compression=None)


def save_npy(fname: str, contents):
    if type(contents) == dict and len(contents) == 1:
        contents = contents[list(contents)[0]]
    np.save(fname, contents)


OPEN = {'.mat': open_mat,
        '.h5': open_h5,
        '.npy': open_npy,
        '.pt': open_pt,
        }

SAVE = {'.mat': save_mat,
        '.h5': save_h5,
        '.npy': save_npy,
        }


def is_db(fname: str):
    return any([fname.endswith(ext) for ext in OPEN.keys()])


def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate


@static_vars(to='', duplicate=True)
def convert(fname: str, *args):
    if args:
        convert.to, convert.duplicate = args

    ext = '.' + fname.split('.')[-1]
    contents = OPEN[ext](fname)
    print(contents)
    fname_new = fname.replace(ext, convert.to)
    if os.path.isfile(fname_new) and not convert.duplicate:
        print('Didn\'t duplicate')
        return
    else:
        SAVE[convert.to](fname_new, contents)
        print(fname_new)


if __name__ == '__main__':
    main()
