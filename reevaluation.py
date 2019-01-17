# noinspection PyUnresolvedReferences
import matlab.engine

import os
import time
from argparse import ArgumentError, ArgumentParser
from glob import glob

import scipy.io as scio

import config as cfg
from train import CustomWriter
from utils import arr2str

cfg.N_GRIFFIN_LIM = 10
TITLE = f'iter{cfg.N_GRIFFIN_LIM}'

parser = ArgumentParser()
parser.add_argument(
    '--test', action='store_true',
)
parser.add_argument(
    '--valid', action='store_true',
)
ARGS = parser.parse_args()
if ARGS.test:
    if ARGS.valid:
        raise ArgumentError
    else:
        group = 'test'
else:
    if ARGS.valid:
        group = 'valid'
    else:
        group = 'test'

DIR_MODEL = './result/UNet 19-01-14 (amp fix)'
DIR_ORIGINAL = os.path.join(DIR_MODEL, group)
DIR_NEW = os.path.join(DIR_MODEL, f'{group}_{TITLE}')
if not os.path.isdir(DIR_NEW):
    os.makedirs(DIR_NEW)
writer = CustomWriter(DIR_NEW)

iv_files = glob(os.path.join(DIR_ORIGINAL, 'IV_*.mat'))
names = {k: v[1:] for k, v in cfg.IV_DATA_NAME.items() if k != 'fname_wav'}
avg_measure = None

print(' 0.00%:')
for idx, file in enumerate(iv_files):
    t_start = time.time()
    iv_dict = scio.loadmat(file)
    iv_dict = {k: iv_dict[v] for k, v in names.items()}

    measure = writer.write_one(idx, group=group, **iv_dict)
    if avg_measure is None:
        avg_measure = measure
    else:
        avg_measure += measure

    str_measure = arr2str(measure).replace('\n', '; ')
    tt = time.strftime('(%Ss)', time.gmtime(time.time() - t_start))
    print(f'{(idx + 1) / len(iv_files) * 100:5.2f}%: {str_measure}\t{tt}')

avg_measure /= len(iv_files)

writer.add_text(f'{group}/Average Measure/Proposed', str(avg_measure[0]))
writer.add_text(f'{group}/Average Measure/Reverberant', str(avg_measure[1]))

print()
str_avg_measure = arr2str(avg_measure).replace('\n', '; ')
print(f'Average: {str_avg_measure}')
