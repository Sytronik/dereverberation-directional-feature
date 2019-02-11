# noinspection PyUnresolvedReferences
import matlab.engine

import atexit
import contextlib
import itertools
import os
import sys
from datetime import datetime
from os.path import join as pathjoin
import gc
# from multiprocessing import Process
#
# import keyboard as kb

import config as cfg

experiment = {
    'cfg.HyperParameters.weight_decay': [
        1e-5,
        1e-3,
    ],
    'cfg.HyperParameters.weight_loss': [
        (1, 0.1, 0),
        (0.1, 0.1, 0),
        (0.01, 0.01, 0),
    ],
    'cfg.NORM_CLASS': [
        'ReImMeanStdNormalization',
        'LogReImMeanStdNormalization',
    ],
}

main_argvs = 'python main.py UNet --train'.split(' ')[1:]


@contextlib.contextmanager
def redirect_argv(*args):
    sys.argv_backup = sys.argv[:]
    sys.argv = list(args)
    yield
    sys.argv = sys.argv_backup


def run_main():
    with open('main.py') as fmain:
        with redirect_argv(*main_argvs):
            code = compile(fmain.read(), 'main.py', 'exec')
            exec(code)
            gc.collect()


def _exit(exp: str):
    with open(pathjoin(cfg.DICT_PATH['UNet'], f'autoexp.txt'), 'w') as f:
        f.write(exp)
    # os.rename(cfg.DICT_PATH['UNet'], f'{cfg.DICT_PATH["UNet"]} Done')


if __name__ == '__main__':
    keys = list(experiment.keys())
    values = list(experiment.values())
    # cfg.DICT_PATH['UNet'] += '_auto'
    DIR_RESULT_backup = cfg.DICT_PATH['UNet']

    # save configurations
    keys_shorten = [k.replace('cfg.', '').replace('HyperParameters', 'hp') for k in keys]
    str_now = datetime.now().strftime('%y-%m-%d %Hh %Mm')
    str_exps = []
    for idx, vs in enumerate(itertools.product(*values)):
        # vs = [f'{v:.0e}' if type(v) == float else f'{v}' for v in vs]
        str_exps.append(
            f'{idx:2d}: ' + ', '.join([f'{k}={v}' for k, v in zip(keys_shorten, vs)])
        )

    with open(pathjoin(cfg.PATH_RESULT, f'autoexp {str_now}.txt'), 'w') as f:
        f.write('\n'.join(str_exps))

    # iterate main
    for idx, (vs, str_exp) in enumerate(zip(itertools.product(*values), str_exps)):
        # warning
        # if idx == 0:
        #     continue

        # apply configurations
        # str_now = datetime.now().strftime('%y-%m-%d %Hh')
        cfg.DICT_PATH['UNet'] = f'{DIR_RESULT_backup} {str_now} (autoexp {idx})'
        for k, v in zip(keys, vs):
            exec(f'{k} = "{v}"' if type(v) == str else f'{k} = {v}')

        print(str_exp)

        atexit.register(_exit, str_exp)
        # process = Process(target=main)
        # kb.add_hotkey('ctrl+q', process.terminate)
        # process.start()
        # process.join()
        run_main()
        atexit.unregister(_exit)
