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
from argparse import ArgumentParser
from pathlib import Path

import deepdish as dd
import numpy as np
import scipy.io as scio
import torch


def main():
    parser = ArgumentParser()
    for arg_for_path in LIST_ARGS_FOR_PATH:
        parser.add_argument(*arg_for_path, action='append', nargs='+', metavar='PATH')
    parser.add_argument('--no-duplicate', '--nd', action='store_false')

    args = parser.parse_args()
    duplicate = args.no_duplicate
    del parser, args.no_duplicate

    pool = mp.Pool(3)
    for to, paths in args.__dict__.items():
        if not paths:
            continue
        paths = [p for path in paths for p in path]
        for path in paths:
            path = Path(path)
            if path.is_dir():
                file_args = (
                    (f, to, duplicate) for f in path.glob('**/*.*') if is_db(f)
                )
                if not file_args:
                    continue
                pool.starmap_async(convert, file_args)
            elif path.is_file():
                pool.apply_async(
                    convert,
                    (path,), dict(to=to, show=True, duplicate=duplicate)
                )
            else:
                raise FileExistsError

    pool.close()
    pool.join()


def open_mat(fpath: Path):
    contents = scio.loadmat(str(fpath), squeeze_me=True)
    return {key: value
            for key, value in contents.items()
            if not (key.startswith('__') and key.endswith('__'))
            }


def open_h5(fpath: Path):
    return dd.io.load(fpath)


def open_npy(fpath: Path):
    contents = np.load(str(fpath))
    if contents.size == 1:
        contents = contents.item()
    return contents


def open_pt(fpath: Path):
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

    contents = torch.load(fpath, map_location=torch.device('cpu'))
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


def save_mat(fpath: Path, contents):
    if type(contents) != dict:  # make contents dict
        contents = {fpath.stem: remove_none(contents)}
    scio.savemat(fpath, contents, long_field_names=True)


def save_h5(fpath: Path, contents):
    if type(contents) == dict and len(contents) == 1:
        key = list(contents)[0]
        exec(f'{key} = contents[key]')
        dd.io.save(fpath, eval(key), compression=None)
    else:
        dd.io.save(fpath, contents, compression=None)


def save_npy(fpath: Path, contents):
    if type(contents) == dict:
        if len(contents) == 1:
            contents = contents[list(contents)[0]]
        else:
            np.savez(str(fpath), **contents)
            return

    np.save(str(fpath), contents)


def save_txt(fpath: Path, contents):
    with fpath.open('w', encoding='utf-8'):
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


def is_db(f: Path):
    return any([f.suffix == ext for ext in OPEN.keys()])


def str_simple(contents) -> str:
    if type(contents) == dict:
        length = max([len(k) for k in contents.keys()])
        spaces = '\n' + ' ' * (length + 2)
        result = [''] * len(contents)
        for idx, (key, value) in enumerate(contents.items()):
            value_simple = str_simple(value).replace('\n', spaces)
            result[idx] = f'{key:<{length}}: {value_simple}'
        result = '\n'.join(result)
    elif type(contents) == list:
        result = f'list of len {len(contents)}'
    elif type(contents) == tuple:
        result = f'tuple of len {len(contents)}'
    elif type(contents) == np.ndarray:
        result = f'ndarray of shape {contents.shape}'
    else:
        result = str(contents)

    return result


def convert(fpath: Path, to: str, show=False, duplicate=True):
    # Open
    ext = fpath.suffix
    contents = OPEN[ext](fpath)

    # Print
    print(fpath)
    if to == 'show_full':
        print(contents)
    elif to == 'show':
        print(str_simple(contents))
    else:
        # Convert and Print
        if not to.startswith('.'):
            to = f'.{to}'
        if show:
            print(str_simple(contents))
        f_new = fpath.with_suffix(to)
        if f_new.exists() and not duplicate:
            print("Didn't convert for avoiding duplicate")
        else:
            SAVE[to](f_new, contents)
            print(f_new)
    print()


if __name__ == '__main__':
    main()
