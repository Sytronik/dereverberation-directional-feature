# %%
import re
import csv
import json
from itertools import product as iterprod
from pathlib import Path
from typing import List, NamedTuple, Sequence, Union, Dict

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
from numpy import ndarray
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


# %% constants
# path_root = Path('./result')
path_root = Path('./backup/result_23_10')
path_fig = Path('./figures')
room_trained = 'room1+2+3'
foldername_results = {
    'SIV': f'SIV_23_10',
    'DV': f'DV_23_10',
    'Mulspec': f'mulspec_23_10',
    'No-DF': f'No-DF_23_10',
}
kind_folders = [
    'seen', 'unseen', 'unseen_room4+5+6+7',
]
# how to inform measurements of reverberant data
# method_rev = 'delta'  # as a difference
method_rev = 'sep'  # as a separated item
# method_rev = False  # not shown

metric_need_legend = 'PESQ'
legend_loc = 'upper right'
legend_col = 4 if method_rev == 'delta' else 3


def select_kinds(all_kinds):
    return dict(
        unseens=['room6', 'room1 (unseen)', 'room4', 'room2 (unseen)',
                 'room5', 'room3 (unseen)', 'room7'],
        seen_vs_unseen=[k for k in all_kinds if 'seen)' in k],
    )


def convert_xticklabels(suffix, selected_kinds):
    if suffix == 'unseens':
        return [re.sub('\d', str(i+1), item)
                for i, item in enumerate(selected_kinds)
                ]
    else:
        return [re.sub('\d', str(i // 2 * 2 + 2), item)
                for i, item in enumerate(selected_kinds)
                ]


def set_yticks_get_ylim(ax: plt.Axes, ylabel: str):
    if ylabel == 'SegSNR [dB]':
        ylim = (-10, 10)
        # ax.set_yticks(np.linspace(*ylim, num=6))
    elif ylabel == 'fwSegSNR [dB]':
        ylim = (5, 17)
        ax.set_yticks(np.linspace(*ylim, num=5))
        # ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    elif ylabel == 'PESQ':
        ylim = (1.5, 4.5)
        ax.set_yticks(np.linspace(*ylim, num=7))
    elif ylabel == 'STOI':
        ylim = (0.5, 1)
        # ax.set_yticks(np.linspace(*ylim, num=7))
    elif ylabel == 'ΔSegSNR [dB]':
        ylim = (0, 16)
        ax.set_yticks(np.linspace(*ylim, num=5))
    elif ylabel == 'ΔfwSegSNR [dB]':
        ylim = (0, 8)
    elif ylabel == 'ΔPESQ':
        ylim = (0, 1.5)
    elif ylabel == 'ΔSTOI':
        ylim = (0, 0.4)
    else:
        ylim = None
    return ylim


# %% Definitions
class MethodKindMetric(NamedTuple):
    method: str
    kind: str
    metric: str


# %% dependent constants

path_results = {k: path_root / v for k, v in foldername_results.items()}

fstem_analysis = '['
for method, path in path_results.items():
    fstem_analysis += path.name
    if method not in path.name:
        fstem_analysis += f' ({method})'
    fstem_analysis += ', '

fstem_analysis = fstem_analysis[:-2]
fstem_analysis += '] '
fstem_analysis += f'[{", ".join(kind_folders)}]'
fstem_analysis += f' [rev={method_rev}]'

# %% save scalars

force_save = False

all_scalars: Dict[str, Dict[str, List]] = dict()
for method, path_result in path_results.items():
    exist_kinds = [
        kind for kind in kind_folders
        if bool((path_result/kind).glob('events.out.tfevents.*'))
    ]
    path_json = path_result / ('scalars_' + '_'.join(exist_kinds) + '.json')
    if path_json.exists() and not force_save:
        print(f'"{path_json}" already exists.')
        continue

    all_scalars[method] = dict()
    for kind in exist_kinds:
        path_test = path_result / kind
        eventacc = EventAccumulator(str(path_test),
                                    size_guidance=dict(scalars=10000))
        eventacc.Reload()
        for tag in eventacc.Tags()['scalars']:
            if 'Reverberant' in tag or 'Proposed' in tag:
                _, _, value = zip(*eventacc.Scalars(tag))
                tag = tag.replace(tag.split('/')[0], kind)
                all_scalars[method][tag] = value

    with path_json.open('w') as f:
        json.dump(dict(**all_scalars[method]), f)

# %% scalars to array

if 'all_scalars' not in dir() or not bool(all_scalars):
    all_scalars: Dict[str, Dict[str, List]] = dict()
    for method, path_result in path_results.items():
        path_json = path_result / f'scalars_{"_".join(kind_folders)}.json'

        if not path_json.exists():
            raise Exception(f'scalar file does not exist in {path_result}.')

        with path_json.open() as f:
            all_scalars[method] = json.loads(f.read())

# Dictionary of MethodKindMetric - ndarray
# handle measurements of reverberant data
all_arrays: Dict[MethodKindMetric, ndarray] = dict()
all_methods: List[str] = list(all_scalars.keys())
if method_rev == 'sep':
    all_methods.append('Unproc.')
all_metrics = set()
for method, scalars in all_scalars.items():
    for k, v in scalars.items():
        kind, metric, rev_or_prop = k.split('/')
        # kind
        if '_' in kind:  # unseen room
            kind = kind.split('_')[1]
        else:  # seen room
            kind = f'{room_trained} ({kind})'

        # metric
        metric = metric.split('_')[-1]
        if 'SNRseg' in metric:
            metric = metric.replace('SNRseg', 'SegSNR')
        if 'SNR' in metric:
            metric += ' [dB]'
        if method_rev == 'delta':
            metric = 'Δ'+metric
        all_metrics.add(metric)

        if rev_or_prop == 'Reverberant':
            if method_rev == 'sep':
                new_key = MethodKindMetric('Unproc.', kind, metric)
                if new_key not in all_arrays:
                    all_arrays[new_key] = np.array(v)
            elif method_rev == 'delta':
                new_key = MethodKindMetric(method, kind, metric)
                if new_key in all_arrays:
                    all_arrays[new_key] -= np.array(v)
                else:
                    all_arrays[new_key] = -np.array(v)
        else:
            new_key = MethodKindMetric(method, kind, metric)
            if new_key in all_arrays:
                all_arrays[new_key] += np.array(v)
            else:
                all_arrays[new_key] = np.array(v)

        scalars[k] = np.array(v)

all_metrics: List[str] = list(all_metrics)
all_kinds: List[str] = kind_folders.copy()
all_kinds_backup = all_kinds.copy()
all_arrays_backup = all_arrays.copy()

# %% split into each room

if '+' in room_trained or any('+' in kind for kind in all_kinds):
    all_arrays = dict()
    all_kinds = set()
    for key, value in all_arrays_backup.items():
        # room numbers
        if ' (' in key.kind:  # seen room
            kind_paren = key.kind.split(' ')[1]
            rooms = room_trained.replace('room', '').split('+')
        else:  # unseen room
            kind_paren = ''
            rooms = key.kind.split(' ')[0].replace('room', '').split('+')
        rooms = [f'room{n_room}' for n_room in rooms]

        for i, room in enumerate(rooms):
            new_kind = f'{room} {kind_paren}'.rstrip()
            new_key = MethodKindMetric(key.method, new_kind, key.metric)
            all_arrays[new_key] = value[i::len(rooms)]
            all_kinds.add(new_kind)

    all_kinds: List[str] = sorted(list(all_kinds))

# %% mean / std

all_means: Dict[MethodKindMetric, np.float64] = dict()
all_stds: Dict[MethodKindMetric, np.float64] = dict()
for key, value in all_arrays.items():
    all_means[key] = np.mean(value)
    all_stds[key] = np.std(value, ddof=1)

# %% save to csv

""" csv example
METHOD1
KIND1   METRIC1 0       0       0       ...
        METRIC2 0       0       0       ...
KIND2   METRIC1 0       0       0       ...
        METRIC2 0       0       0       ...
METHOD2
KIND1   METRIC1 0       0       0       ...
        METRIC2 0       0       0       ...
KIND2   METRIC1 0       0       0       ...
        METRIC2 0       0       0       ...
means
        KIND1           KIND2
        METHOD1 METHOD2 METHOD1 METHOD2
METRIC1 0       0       0       0
METRIC2 0       0       0       0
stds
        KIND1           KIND2
        METHOD1 METHOD2 METHOD1 METHOD2
METRIC1 0       0       0       0
METRIC2 0       0       0       0
"""


def make_rows_for_stats(stats):
    _rows = [None] * 2
    _rows[0] = [''] * (1 + len(all_kinds) * len(all_methods))
    for i_kind, kind in enumerate(all_kinds):
        _rows[0][1 + len(all_methods) * i_kind] = kind
    _rows[1] = [''] + all_methods * len(all_kinds)
    for metric in all_metrics:
        _row = [metric]
        for kind, method in iterprod(all_kinds, all_methods):
            _row.append(stats[MethodKindMetric(method, kind, metric)])
        _rows.append(_row)
    return _rows


path_csv = (path_root / fstem_analysis).with_suffix('.csv')
with path_csv.open('w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    rows = []
    for method in all_methods:
        rows.append([method])
        for kind, (i, metric) in iterprod(all_kinds, enumerate(all_metrics)):
            rows.append([kind if i == 0 else '',
                         metric,
                         *all_arrays[MethodKindMetric(method, kind, metric)],
                         ])

    rows += [['means'], *make_rows_for_stats(all_means)]
    rows += [['stds'], *make_rows_for_stats(all_stds)]
    for r in rows:
        writer.writerow(r)


# %% plotting functions

def _graph_initialize(means: ndarray, stds: ndarray,
                      legends: Union[str, Sequence[str]],
                      xticklabels: Sequence,
                      ylabel: str):
    global room_trained, method_rev, metric_need_legend
    plt.rc('font', family='Arial', size=18)

    fig, ax = plt.subplots(
        figsize=(len(xticklabels)*1.2 if xticklabels else 5.3, 4),
    )

    # whitespace -> line ending
    if xticklabels and any(' (' in label for label in xticklabels):
        for ii, label in enumerate(xticklabels):
            xticklabels[ii] = label.replace(' ', '\n')

    if metric_need_legend in ylabel or ylabel == '':
        draw_legend = True
    else:
        draw_legend = False

    # colors
    cmap = plt.get_cmap('tab20c')
    colors = [cmap.colors[(1 + 4 * i // 16) % 4 + (4 * i) % 16]
              for i in range(len(legends))]
    if method_rev == 'sep':
        colors[-1] = cmap.colors[17]

    # ylim
    # ylim = list(ax.get_ylim())
    # max_ = (means + stds).max()
    # min_ = (means - stds).min()
    # if method_rev == 'delta' and min_ >= 0:
    #     ylim[0] = 0
    # else:
    #     ylim[0] = min_ - (max_ - min_) * 0.2
    #
    # ylim[1] = ylim[1] + (max_ - ylim[0]) * 0.25
    # if ylabel in Y_MAX:
    #     ylim[1] = min(Y_MAX[ylabel], ylim[1])

    ylim = set_yticks_get_ylim(ax, ylabel)

    return fig, ax, xticklabels, draw_legend, cmap, colors, ylim


def draw_lineplot(means: ndarray, stds: ndarray = None,
                  legends: Union[str, Sequence[str]] = None,
                  xticklabels: Sequence = None,
                  ylabel: str = '',
                  n_connect: int = -1):
    if legends is None:
        legends = ('',) * means.shape[0]
    if stds is None:
        stds = (None,) * len(legends)
    if n_connect == -1:
        n_connect = means.shape[1]
    ax: plt.Axes = None
    fig, ax, xticklabels, draw_legend, cmap, colors, ylim \
        = _graph_initialize(means, stds,
                            legends=legends,
                            xticklabels=xticklabels,
                            ylabel=ylabel)

    # draw
    x_range = np.arange(means.shape[1])
    lineplots = []
    for ii, (label, mean, std) in enumerate(zip(legends, means, stds)):
        for jj in x_range[::n_connect]:
            line, = ax.plot(x_range[jj:jj+n_connect], mean[jj:jj+n_connect],
                            color=colors[ii],
                            marker='o')
        lineplots.append(line)

    ax.set_xticks(x_range)
    ax.set_xticklabels(xticklabels)
    ax.set_xlim(x_range[0] - 0.5, x_range[-1] + 0.5)

    ax.grid(True, axis='y')
    if ylim:
        ax.set_ylim(*ylim)

    # axis label
    # ax.set_xlabel('RIR')
    ax.set_ylabel(ylabel)

    # ticks
    ax.tick_params('x', direction='in')
    ax.tick_params('y', direction='in')

    # legend
    if draw_legend:
        # ax.legend(loc='lower right', bbox_to_anchor=(1, 1),
        #           ncol=4, fontsize='small', columnspacing=1)
        ax.legend(lineplots, legends,
                  loc=legend_loc,
                  ncol=legend_col,
                  fontsize='small',
                  columnspacing=1)

    fig.tight_layout()
    return fig


def draw_bar_graph(means: ndarray, stds: ndarray = None,
                   legends: Union[str, Sequence[str]] = None,
                   xticklabels: Sequence = None,
                   ylabel: str = ''):
    if legends is None:
        legends = ('',) * means.shape[0]
    if stds is None:
        stds = (None,) * len(legends)
    # constants
    bar_width = 0.5 / len(legends)
    ndigits = 3 if means.max() - means.min() < 0.1 else 2

    fig, ax, xticklabels, draw_legend, cmap, colors, ylim = _graph_initialize(
        legends, means, stds, xticklabels, ylabel)

    # draw bar & text
    x_range = np.arange(means.shape[1])
    for ii, (title, mean, std) in enumerate(zip(legends, means, stds)):
        bar = ax.bar(x_range + bar_width * ii, mean,
                     bar_width,
                     yerr=std,
                     error_kw=dict(capsize=5),
                     label=title,
                     color=colors[ii])

        for b in bar:
            x = b.get_x() + b.get_width() * 0.55
            y = b.get_height()
            ax.text(x, y, f' {b.get_height():.{ndigits}f}',
                    # horizontalalignment='center',
                    rotation=40,
                    rotation_mode='anchor',
                    verticalalignment='center')

    ax.set_xticklabels(xticklabels)

    ax.grid(True, axis='y')
    ax.set_axisbelow(True)

    # xlim
    xlim = list(ax.get_xlim())
    xlim[0] -= bar_width
    xlim[1] += bar_width
    ax.set_xlim(*xlim)

    if ylim:
        ax.set_ylim(*ylim)

    # axis label
    # ax.set_xlabel('RIR')
    ax.set_ylabel(ylabel)

    # ticks
    ax.set_xticks(x_range + bar_width * (len(legends) - 1) / 2)

    ax.tick_params('x', length=0)
    ax.tick_params('y', direction='in')

    # legend
    if draw_legend:
        # ax.legend(loc='lower right', bbox_to_anchor=(1, 1),
        #           ncol=4, fontsize='small', columnspacing=1)
        ax.legend(loc=legend_loc,
                  ncol=2,
                  fontsize='small',
                  columnspacing=1)

    fig.tight_layout()
    return fig


# %% draw graphs

plt.style.use('default')
for suffix, selected_kinds in select_kinds(all_kinds).items():

    s_path_fig = str(path_fig /
                     fstem_analysis) + f' [{suffix}] [{{}}].png'
    for metric in all_metrics:
        means = np.empty((len(all_methods), len(selected_kinds)))
        stds = np.empty((len(all_methods), len(selected_kinds)))
        for (i, method), (j, kind) \
                in iterprod(enumerate(all_methods), enumerate(selected_kinds)):
            key = MethodKindMetric(method, kind, metric)
            means[i, j] = all_means[key]
            stds[i, j] = all_stds[key]

        # fig = draw_bar_graph(titles, mean, std, sfxs, col)
        fig = draw_lineplot(means, stds,
                            legends=all_methods,
                            xticklabels=convert_xticklabels(suffix, selected_kinds),
                            ylabel=metric,
                            n_connect=2 if suffix == 'seen_vs_unseen' else -1,
                            )
        fig.savefig(s_path_fig.format(metric.replace('Δ', '')), dpi=300)


# %%
