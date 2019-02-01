# noinspection PyUnresolvedReferences
import contextlib
import itertools
import os
import sys
from datetime import datetime
from os.path import join as pathjoin

import matlab.engine

import config as cfg

experiment = {
    'cfg.HyperParameters.weight_decay':
        [1e-5, 1e-3],
    'cfg.NORM_CLASS':
        ['ReImMeanStdNormalization',
         'LogReImMeanStdNormalization',
         ],
    'cfg.HyperParameters.weight_loss':
        [(0.1, 0.1, 1),
         (0, 0, 1),
         (0.1, 0.01, 1),
         # (0.1, 0.1, 1),
         # (0.001, 0.001, 1),
         # (0.01, 0.01, 1),
         ],
}

main_argvs = 'python main.py UNet --train'.split(' ')[1:]


@contextlib.contextmanager
def redirect_argv(*args):
    sys.argv_backup = sys.argv[:]
    sys.argv = list(args)
    yield
    sys.argv = sys.argv_backup


if __name__ == '__main__':
    keys = list(experiment.keys())
    values = list(experiment.values())
    # cfg.DICT_PATH['UNet'] += '_auto'
    DIR_RESULT_backup = cfg.DICT_PATH['UNet']

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

    for idx, (vs, str_exp) in enumerate(zip(itertools.product(*values), str_exps)):
        # str_now = datetime.now().strftime('%y-%m-%d %Hh')
        cfg.DICT_PATH['UNet'] = f'{DIR_RESULT_backup} {str_now} (autoexp {idx})'
        for k, v in zip(keys, vs):
            exec(f'{k} = "{v}"' if type(v) == str else f'{k} = {v}')

        print(str_exp)

        with open('main.py') as f:
            with redirect_argv(*main_argvs):
                code = compile(f.read(), 'main.py', 'exec')
                exec(code)

        with open(pathjoin(cfg.DICT_PATH['UNet'], f'autoexp.txt'), 'w') as f:
            f.write(str_exp)
        os.rename(cfg.DICT_PATH['UNet'], f'{cfg.DICT_PATH["UNet"]} Done')