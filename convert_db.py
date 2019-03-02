"""
<<Usage>>
python convert_db.py [--no-duplicate]
    {
        {--mat | --h5 | --npy | --show | --show-full | --txt}
        {
            {directory path 1 | file path 1}
            {directory path 2 | file path 2}
            ...
        }
    }
    {
        {--mat | --h5 | --npy | --show | --show-full | --txt}
        {
            {directory path 1 | file path 1}
            {directory path 2 | file path 2}
            ...
        }
    }
    ...

* "--show" means "do not convert, just show the file contents."
* "--no-duplicate option" is applied to all files.
* Avaiable types of original file: .mat, .h5, .npy, .pt
"""
import multiprocessing as mp
import os
from argparse import ArgumentParser
from glob import glob

import deepdish as dd
import numpy as np
import scipy.io as scio
import torch

from utils import static_vars


def main():
    parser = ArgumentParser()
    for arg_for_path in LIST_ARGS_FOR_PATH:
        parser.add_argument(*arg_for_path, nargs='*', metavar='PATH')
    parser.add_argument('--no-duplicate', '--nd', action='store_true')

    args = parser.parse_args()
    convert.duplicate = not args.no_duplicate
    del args.no_duplicate

    for to, paths in args.__dict__.items():
        convert.to = to
        if not paths:
            continue
        for path in paths:
            if os.path.isdir(path):
                pool = mp.Pool(mp.cpu_count())
                for folder, _, _ in os.walk(path):
                    files = glob(os.path.join(folder, '*.*'))
                    files = [f for f in files if is_db(f)]
                    if files:
                        pool.map_async(convert, files)
                pool.close()
                pool.join()
            elif os.path.isfile(path):
                convert(path, show=True)
            else:
                raise FileExistsError


def open_mat(fname: str):
    contents = scio.loadmat(fname, squeeze_me=True)
    return {key: value
            for key, value in contents.items()
            if not (key.startswith('__') and key.endswith('__'))
            }


def open_h5(fname: str):
    return dd.io.load(fname)


def open_npy(fname: str):
    contents = np.load(fname)
    if contents.size == 1:
        contents = contents.item()
    return contents


def open_pt(fname: str):
    def construct_dict(data):
        if hasattr(data, 'items'):
            if len(data) == 1:
                data = construct_dict(data.popitem()[1])
            else:
                keytype = type(list(data.keys())[0])
                if keytype != str:
                    def fieldname(key):
                        return f'{keytype.__name__}{key}'.replace('.', '_')
                else:
                    def fieldname(key):
                        return key.replace('.', '_')
                data = {fieldname(key): construct_dict(value)
                        for key, value in data.items()}
        elif type(data) == torch.Tensor:
            data = data.numpy()
        elif hasattr(data, '__len__'):
            if len(data) == 1:
                data = construct_dict(data[0])
            else:
                data = [construct_dict(item) for item in data]

        return data

    contents = torch.load(fname, map_location=torch.device('cpu'))
    contents = construct_dict(contents)

    return contents


def remove_none(a):
    if a is None:
        return []
    else:
        if hasattr(a, 'items'):
            return {remove_none(k): remove_none(v) for k, v in a.items()}
        elif type(a) == list:
            return [remove_none(item) for item in a]
        elif type(a) == tuple:
            return tuple([remove_none(item) for item in a])
        elif type(a) == set:
            return set([remove_none(item) for item in a])
        else:
            return a


def save_mat(fname: str, contents):
    if type(contents) != dict:  # make contents dict
        contents = {os.path.basename(fname).replace('.mat', ''): remove_none(contents)}
    scio.savemat(fname, contents, long_field_names=True)


def save_h5(fname: str, contents):
    if type(contents) == dict and len(contents) == 1:
        exec(f'{list(contents)[0]} = {contents[list(contents)[0]]}')
        dd.io.save(fname, eval(list(contents)[0]), compression=None)
    else:
        dd.io.save(fname, contents, compression=None)


def save_npy(fname: str, contents):
    if type(contents) == dict and len(contents) == 1:
        contents = contents[list(contents)[0]]
    np.save(fname, contents)


def save_txt(fname: str, contents):
    with open(fname, 'w') as f:
        f.write(str(contents))


LIST_ARGS_FOR_PATH = {('--show', '-s'),
                      ('--show-full', '-sf'),
                      ('--mat', '-m'),
                      ('--h5', '-h5'),
                      ('--npy', '-n'),
                      ('--txt', '-t'),
                      }
OPEN = {'.mat': open_mat,
        '.h5': open_h5,
        '.npy': open_npy,
        '.pt': open_pt,
        }
SAVE = {'.mat': save_mat,
        '.h5': save_h5,
        '.npy': save_npy,
        '.txt': save_txt,
        }


def is_db(fname: str):
    return any([fname.endswith(ext) for ext in OPEN.keys()])


def str_simple(contents) -> str:
    if type(contents) == dict:
        length = max([len(k) for k in contents.keys()])
        spaces = '\n' + ' ' * (length + 2)
        result = ''
        for key, value in contents.items():
            value_simple = str_simple(value).replace('\n', spaces)
            result += (f'{key:<{length}}: '
                       f'{value_simple}\n')
        result = result[:-1]
    elif type(contents) == list:
        result = f'list of len {len(contents)}'
    elif type(contents) == tuple:
        result = f'tuple of len {len(contents)}'
    elif type(contents) == np.ndarray:
        result = f'ndarray of shape {contents.shape}'
    else:
        result = str(contents)

    return result


@static_vars(to='', duplicate=True)
def convert(fname: str, show=False,
            # *args
            ):
    # if args:
    #     convert.to, convert.duplicate = args

    # Open
    ext = os.path.splitext(fname)[-1]
    contents = OPEN[ext](fname)

    # Print
    if convert.to == 'show_full':
        print(contents)
        return
    elif convert.to == 'show':
        print(str_simple(contents))
    else:
        if not convert.to.startswith('.'):
            convert.to = f'.{convert.to}'
        if show:
            print(str_simple(contents))
        fname_new = fname.replace(ext, convert.to)
        if os.path.isfile(fname_new) and not convert.duplicate:
            print('Didn\'t convert for avoiding duplicate')
            return
        else:
            SAVE[convert.to](fname_new, contents)
            print(fname_new)


if __name__ == '__main__':
    main()
