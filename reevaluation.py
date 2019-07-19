# noinspection PyUnresolvedReferences
import matlab.engine

import os
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import scipy.io as scio
from numpy import ndarray
from tqdm import tqdm

from hparams import hp
from train import CustomWriter
from utils import arr2str

hp.n_glim_iter = 10  # info: the objective of reevaluation
TITLE = f'iter{hp.n_glim_iter}'
# TITLE = 'with_anechoic_ph'

parser = ArgumentParser()
parser.add_argument('kind_data', type=str, choices=('seen', 'unseen'))
args = hp.parse_argument(parser)
group = args.kind_data

path_original = hp.logdir / group
path_new = hp.logdir / f'{group}_{TITLE}'
os.makedirs(path_new, exist_ok=True)
writer = CustomWriter(str(path_new), group=group)

path_results = list(path_original.glob(hp.form_result.format('*')))
path_results = sorted(path_results)
names = {k: v for k, v in hp.spec_data_names.items()
         if k != 'speech_fname' and k != 'out_phase' and not k.endswith('bpd')}
avg_measure: ndarray = None


pbar = tqdm(path_results, desc=f'{group}_{TITLE}', dynamic_ncols=True)
for idx, file in enumerate(pbar):
    data_dict = scio.loadmat(str(file))
    data_dict = {k: data_dict[v] for k, v in names.items()}

    data_dict = {k: (np.maximum(v, 0) if not k.endswith('phase') else v)
                 for k, v in data_dict.items()}

    measure = writer.write_one(idx,
                               # eval_with_y_ph=True,  # info: the objective of reevaluation
                               **data_dict)
    if avg_measure is None:
        avg_measure = measure
    else:
        avg_measure += measure

    str_measure = arr2str(measure).replace('\n', '; ')
    pbar.write(str_measure)

avg_measure /= len(path_results)

writer.add_text(f'{group}/Average Measure/Proposed', str(avg_measure[0]))
writer.add_text(f'{group}/Average Measure/Reverberant', str(avg_measure[1]))

print()
str_avg_measure = arr2str(avg_measure).replace('\n', '; ')
print(f'Average: {str_avg_measure}')
