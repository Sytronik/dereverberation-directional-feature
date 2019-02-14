import csv
import json
import os
from argparse import ArgumentParser
from glob import glob
from itertools import product as iterprod
from os.path import join as pathjoin
from typing import List, NamedTuple, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
from tensorboard.backend.event_processing.event_multiplexer import EventMultiplexer
from tqdm import tqdm

from mypath import *

parser = ArgumentParser()
parser.add_argument('--save-scalars', '-s', action='store_true')
parser.add_argument('--force', '-f', action='store_true')
parser.add_argument('--merge-scalars', '-m', action='store_true')
parser.add_argument('--tbcommand', '-t', action='store_true')
ARGS = parser.parse_args()

TITLE_BASELINE = 'mag'
BASELINE = 'UNet 19-01-17 (w.dcy 1e-5)'
TITLE_PRIMARY = 'complex'
PRIMARY = 'UNet*'
COND_PRIMARY = lambda p: p.endswith('autoexp 8')

SUFFIX = 'train'
# SUFFIX = 'seen'
# SUFFIX = 'unseen'
need_all_scalars = False if SUFFIX == 'train' else True

Y_MAX = dict(PESQ=4.5, STOI=1., )

NEED_REVERB = False

class IdxSuffixCol(NamedTuple):
    idx: int
    suffix: str
    col: str

    def __contains__(self, item):
        if super().__contains__(item):
            return True
        else:
            return item in self.col

    def mutated_ver(self, *, _idx=0, _suffix='', _col=''):
        _idx = _idx if _idx else self.idx
        _suffix = _suffix if _suffix else self.suffix
        _col = _col if _col else self.col
        return IdxSuffixCol(_idx, _suffix, _col)


def save_scalars_from_tfevents(path: str, suffix: str):
    DIR_EVENTS = pathjoin(path, suffix)
    suffix = suffix.split('_')[0]

    dict_logdirs = {}
    for folder, _, _ in os.walk(DIR_EVENTS):
        files = glob(pathjoin(folder, 'events.out.tfevents.*'))
        if files:
            assert len(files) == 1
            key = folder.replace(DIR_EVENTS, '')
            if key.startswith(os.sep):
                key = key[1:]
            if not key:
                key = '.'
            dict_logdirs[key] = files[0]

    events = EventMultiplexer(dict_logdirs, size_guidance=dict(scalars=10000))
    events.Reload()

    scalars = {}
    step_longest = None
    for key in dict_logdirs:
        event = events.GetAccumulator(key)
        for tag in event.Tags()['scalars']:
            _, step, value = zip(*event.Scalars(tag))
            if tag.replace('_', ' ') in key:
                key = key.replace(f'{suffix}{os.sep}', '')
                scalars[key] = value
            else:
                tag = tag.replace('_', ' ').replace(f'{suffix}{os.sep}', '')
                scalars[tag] = value
            if step_longest is None or len(step) > len(step_longest):
                step_longest = step

    fjson = pathjoin(DIR_EVENTS, 'scalars.json')
    if not ARGS.force and os.path.isfile(fjson):
        print(f'{fjson} already exists.')
    with open(fjson, 'w') as f:
        json.dump(dict(step=step_longest, **scalars), f)


def draw_bar_graph(titles: Union[str, Sequence[str]],
                   means: ndarray, stds: ndarray = None, xticklabels: Sequence = None,
                   ylabel: str = ''):
    # constants
    bar_width = 0.5 / len(titles)
    ndigits = 3 if means.max() - means.min() < 0.1 else 2

    fig, ax = plt.subplots()

    # args
    if means.ndim == 1:
        if type(titles) == str:
            titles = (titles,)
        means = means[np.newaxis, :]
        stds = stds[np.newaxis, :] if stds is not None else (None,)
    if stds is None:
        stds = (None,) * len(titles)

    common = None
    if xticklabels:
        common = {x.split('_')[0] for x in xticklabels}
        if len(common) == 1:
            xticklabels = ['_'.join(x.split('_')[1:]) for x in xticklabels]
        ax.set_xticklabels(xticklabels)
    if len(titles) == 1:
        if titles[0] == '.' and common and len(common) == 1:
            titles = tuple(common)
            draw_legend = True
        else:
            draw_legend = False
    else:
        draw_legend = True

    # colors
    cmap = plt.get_cmap('tab20c')
    colors = [cmap.colors[(1 + 4*i//16)%4 + (4*i)%16] for i in range(len(titles))]
    colors[-1] = cmap.colors[17]

    # draw bar & text
    range_ = np.arange(means.shape[1])
    for ii, (title, mean, std) in enumerate(zip(titles, means, stds)):
        bar = ax.bar(range_ + bar_width * ii, mean,
                     bar_width,
                     yerr=std,
                     error_kw=dict(capsize=5),
                     label=title,
                     color=colors[ii])

        for b in bar:
            x = b.get_x() + b.get_width()*0.55
            y = b.get_height()
            ax.text(x, y, f'{b.get_height():.{ndigits}f}',
                    # horizontalalignment='center',
                    verticalalignment='bottom')

    ax.grid(True, axis='y')
    ax.set_axisbelow(True)

    # xlim
    xlim = list(ax.get_xlim())
    xlim[0] -= bar_width
    xlim[1] += bar_width
    ax.set_xlim(*xlim)

    # ylim
    ylim = list(ax.get_ylim())
    max_ = (means + stds).max()
    min_ = (means - stds).min()
    ylim[0] = min_ - (max_ - min_)*0.2
    if ylabel in Y_MAX:
        ylim[1] = min(Y_MAX[ylabel], ylim[1])
    ax.set_ylim(*ylim)

    # axis label
    ax.set_xlabel('RIR')
    ax.set_ylabel(ylabel)

    # ticks
    ax.set_xticks(range_ + bar_width * (len(titles) - 1) / 2)

    ax.tick_params('x', length=0)
    ax.tick_params('y', direction='in')

    # legend
    if draw_legend:
        ax.legend()

    fig.tight_layout()
    return fig


if __name__ == '__main__':
    if TITLE_PRIMARY:
        paths = [p for p in glob(pathjoin(PATH_RESULT, PRIMARY)) if os.path.isdir(p)]
        paths = [p for p in paths if COND_PRIMARY(p)]
        paths = sorted(paths)
    else:
        paths = []

    if ARGS.tbcommand:
        BASELINE = pathjoin(BASELINE, SUFFIX)
        paths = [pathjoin(os.path.basename(p), SUFFIX) for p in paths]
        if len(paths) == 1 and not TITLE_BASELINE:
            logdir = paths[0]
        else:
            list_logdir: List[str] = []
            if TITLE_BASELINE:
                list_logdir.append(f'{TITLE_BASELINE}:{BASELINE}')

            if len(paths) == 1:
                list_logdir.append(f'{TITLE_PRIMARY}:{paths[0]}')
            else:
                temp = ('log', '')  # warning
                for idx, path in enumerate(paths):

                    list_logdir.append(f'{TITLE_PRIMARY}{temp[idx]}:{path}')  # warning

            logdir = ','.join(list_logdir)

        command = f'tensorboard --logdir="{logdir}"'
        if need_all_scalars:
            command += ' --samples_per_plugin "scalars=10000,images=1,audio=1"'

        print(command)
        # for item in dir():
        #     if not item.startswith('_') and item != 'command':
        #         exec(f'del {item}')
        #
        # exec('del item')
        # os.system(f'echo \'{command}\' | pbcopy')
    else:
        if len(paths) > 1:
            titles = (([TITLE_BASELINE] if TITLE_BASELINE else [])
                      + [f'{TITLE_PRIMARY}{idx}' for idx in range(len(paths))])
        else:
            titles = [TITLE_BASELINE, TITLE_PRIMARY]
            titles = [t for t in titles if t]
        if NEED_REVERB:
            titles += ['reverberant']

        length = len(titles)
        if TITLE_BASELINE:
            paths.insert(0, pathjoin(PATH_RESULT, BASELINE))
        if ARGS.save_scalars:
            for path in tqdm(paths):
                save_scalars_from_tfevents(path, SUFFIX)

        elif ARGS.merge_scalars:
            # gather data
            data = {}
            suffixes: list = None
            for idx, path in enumerate(paths):
                def cond(p):
                    return (p.is_dir()
                            and p.name != 'train'
                            and p.name.startswith('unseen')  # warning
                            and os.path.isfile(pathjoin(p.path, 'scalars.json')))


                paths_suffix, suffixes = zip(*((p.path, p.name)
                                               for p in os.scandir(path)
                                               if cond(p)))
                paths_suffix = sorted(paths_suffix)
                suffixes = sorted(suffixes)
                assert suffixes, f'{path} is empty.'
                # suffixes = [os.path.basename(p) for p in paths_suffix]
                for suffix, p in zip(suffixes, paths_suffix):
                    try:
                        with open(pathjoin(p, 'scalars.json'), 'r') as f:
                            temp = json.loads(f.read())
                            for k, v in temp.items():
                                data[IdxSuffixCol(idx, suffix, k)] = v
                    except:
                        pass

            # arrange data
            data_arranged = {}
            cols = set()
            step_longest = None
            for key, value in data.items():
                if key.col == 'step':
                    if step_longest is None or len(step_longest) < len(value):
                        step_longest = value
                else:
                    col_new = key.col.split('/')[0].split('. ')[-1]
                    cols.add(col_new)
                    if 'Reverberant' in key:
                        if NEED_REVERB:
                            key_rev = IdxSuffixCol(-1, key.suffix.split('_')[0], col_new)
                            if key_rev not in data_arranged:
                                data_arranged[key_rev] = value
                    else:
                        data_arranged[key.mutated_ver(_col=col_new)] = value
            data = data_arranged
            cols: list = sorted(cols)
            idxs = list(range(length))
            if NEED_REVERB:
                idxs[-1] = -1
            del data_arranged

            # calculate mean, std
            means = {}
            stds = {}
            for key, value in data.items():
                means[key] = np.mean(value)
                stds[key] = np.std(value, ddof=1)


            # save data, mean and std to csv
            def make_rows(stats):
                rows: List[List[Any]] = [None] * (len(cols) + 2)
                rows[0] = [''] * (length * len(suffixes) + 1)
                for idx, suffix in enumerate(suffixes):
                    rows[0][length * idx + 1] = suffix
                rows[1] = [''] + titles * len(suffixes)
                for idx, col in zip(range(2, 2 + len(cols)), cols):
                    rows[idx] = [col]
                    for s in suffixes:
                        rows[idx] += [stats[IdxSuffixCol(i, s, col)] for i in idxs]
                return rows

            if len(paths) > 1:
                fresult = ', '.join([f'{os.path.basename(p)} ({t})'
                                     for p, t in zip(paths, titles[:-1] if NEED_REVERB else titles)])
            else:
                fresult = f'{os.path.basename(paths[0])} ' + ', '.join(['_'.join(s.split('_')[1:]) for s in suffixes])
            fmerged = pathjoin(PATH_RESULT, fresult + '.csv')
            with open(fmerged, 'w', newline='') as f:
                writer = csv.writer(f)
                rows = []
                for idx, title in zip(idxs, titles):
                    rows.append([title])
                    for suffix, (ii, col) in iterprod(suffixes, enumerate(cols)):
                        rows.append([suffix if ii == 0 else '',
                                     col,
                                     *data[IdxSuffixCol(idx, suffix, col)],
                                     ]
                                    )
                rows += [['means'], *make_rows(means)]
                rows += [['stds'], *make_rows(stds)]
                for r in rows:
                    writer.writerow(r)

            # draw bar graph
            fbar = pathjoin(PATH_FIG, fresult + ' ({}).png')
            for col in cols:
                mean = np.empty((length, len(suffixes)))
                std = np.empty((length, len(suffixes)))
                for (i, idx), (j, suffix) in iterprod(enumerate(idxs), enumerate(suffixes)):
                    mean[i, j] = means[IdxSuffixCol(idx, suffix, col)]
                    std[i, j] = stds[IdxSuffixCol(idx, suffix, col)]

                figbar = draw_bar_graph(titles, mean, std, suffixes, col)
                figbar.savefig(fbar.format(col), dpi=300)
