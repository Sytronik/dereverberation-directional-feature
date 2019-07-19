# %%
import csv
import json
from itertools import product as iterprod
from pathlib import Path
from typing import List, NamedTuple, Sequence, Union, Dict, Optional

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
from numpy import ndarray
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from hparams import hp


# %%
class MethodKindMetric(NamedTuple):
    method: str
    kind: str
    metric: str

    # def to(self, *,
    #        method: Optional[str] = None,
    #        kind: Optional[str] = None,
    #        metric: Optional[str] = None):
    #     method = method if method else self.method
    #     kind = kind if kind else self.kind
    #     metric = metric if metric else self.metric
    #     return MethodKindMetric(method, kind, metric)


def _graph_initialize(means: ndarray, stds: ndarray,
                      legends: Union[str, Sequence[str]],
                      xticklabels: Sequence,
                      ylabel: str):
    global room_trained, NEED_REVERB, metric_need_legend
    plt.rc('font', family='Arial', size=18)

    fig, ax = plt.subplots(figsize=(5.3, 4))

    # seen -> roomX (seen)
    # unseen_roomY -> roomY
    if xticklabels and any('_' in label for label in xticklabels):
        for ii, label in enumerate(xticklabels):
            if '_' in label:

                xticklabels[ii] = label.split('_')[1]
            else:
                xticklabels[ii] = f'{room_trained}\n({label})'

    if metric_need_legend in ylabel or ylabel == '':
        draw_legend = True
    else:
        draw_legend = False

    # colors
    cmap = plt.get_cmap('tab20c')
    colors = [cmap.colors[(1 + 4 * i // 16) % 4 + (4 * i) % 16]
              for i in range(len(legends))]
    if NEED_REVERB == 'sep':
        colors[-1] = cmap.colors[17]

    # ylim
    # ylim = list(ax.get_ylim())
    # max_ = (means + stds).max()
    # min_ = (means - stds).min()
    # if NEED_REVERB == 'delta' and min_ >= 0:
    #     ylim[0] = 0
    # else:
    #     ylim[0] = min_ - (max_ - min_) * 0.2
    #
    # ylim[1] = ylim[1] + (max_ - ylim[0]) * 0.25
    # if ylabel in Y_MAX:
    #     ylim[1] = min(Y_MAX[ylabel], ylim[1])

    if ylabel == 'SegSNR [dB]':
        ylim = (-2.5, 12.5)
        ax.set_yticks(np.linspace(*ylim, num=7))
    elif ylabel == 'fwSegSNR [dB]':
        ylim = (6, 16)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    elif ylabel == 'PESQ':
        ylim = (2, 4.5)
        # ax.set_yticks(np.linspace(*ylim, num=7))
    elif ylabel == 'STOI':
        ylim = (0.7, 1)
        ax.set_yticks(np.linspace(*ylim, num=7))
    elif ylabel == 'ΔSegSNR [dB]':
        ylim = (0, 15)
        ax.set_yticks(np.linspace(*ylim, num=9))
    elif ylabel == 'ΔfwSegSNR [dB]':
        ylim = (-1, 8)
    elif ylabel == 'ΔPESQ':
        ylim = (0, 1.5)
    elif ylabel == 'ΔSTOI':
        ylim = (0, 0.3)
    else:
        ylim = None

    return fig, ax, xticklabels, draw_legend, cmap, colors, ylim


def draw_lineplot(means: ndarray, stds: ndarray = None,
                  legends: Union[str, Sequence[str]] = None,
                  xticklabels: Sequence = None,
                  ylabel: str = ''):
    if legends is None:
        legends = ('',) * means.shape[0]
    if stds is None:
        stds = (None,) * len(legends)
    ax: plt.Axes = None
    fig, ax, xticklabels, draw_legend, cmap, colors, ylim \
        = _graph_initialize(means, stds,
                            legends=legends,
                            xticklabels=xticklabels,
                            ylabel=ylabel)

    # draw
    x_range = np.arange(means.shape[1])
    for ii, (label, mean, std) in enumerate(zip(legends, means, stds)):
        ax.plot(x_range, mean,
                label=label,
                color=colors[ii],
                marker='o')

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
        ax.legend(loc='upper center',
                  ncol=2, fontsize='small', columnspacing=1)

    fig.tight_layout()
    return fig


# noinspection PyUnusedLocal
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
        plt.Axes.bar
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
        ax.legend(loc='upper center',
                  ncol=2, fontsize='small', columnspacing=1)

    fig.tight_layout()
    return fig


# %% constants
path_root = Path('./result')
# room_trained = 'room1+2+3'
# foldername_results = {
#     'No-DF': f'No-DF ({room_trained})',
#     'DV': f'DV ({room_trained})',
#     'IV': f'IV ({room_trained})',
# }
# room_trained = 'room1'
foldername_results = {
    'No-DF': f'debug',
    # 'DV': f'UNet (DirAC+p00 {room_trained})',
    # 'IV': f'UNet (IV+p00 {room_trained})',
}
kind_test = [
    # 'seen', 'unseen', 'unseen_room4+5',
    'train',
]
# how to inform measurements of reverberant data
# NEED_REVERB = 'delta'  # as a difference
NEED_REVERB = 'sep'  # as a separated item
# NEED_REVERB = False  # not shown

metric_need_legend = 'PESQ'

# %% dependent constants

hp.init_dependent_vars()

path_results = {k: path_root / v for k, v in foldername_results.items()}

fstem_analysis = '['
for method, path in path_results.items():
    fstem_analysis += path.name
    if method not in path.name:
        fstem_analysis += f' ({method})'
    fstem_analysis += ', '

fstem_analysis = fstem_analysis[:-2]
fstem_analysis += '] '
fstem_analysis += f'[{", ".join(kind_test)}]'
fstem_analysis += f' [rev={NEED_REVERB}]'

# %% save scalars

force_save = False

all_scalars: Dict[str, Dict[str, List]] = dict()
for method, path_result in path_results.items():
    all_scalars[method] = dict()
    exist_kinds = []
    for kind in kind_test:
        path_test = path_result / kind
        if list(path_test.glob('events.out.tfevents.*')):
            exist_kinds.append(kind)
        else:
            continue
        eventacc = EventAccumulator(str(path_test),
                                    size_guidance=dict(scalars=10000))
        eventacc.Reload()
        for tag in eventacc.Tags()['scalars']:
            if 'Reverberant' in tag or 'Proposed' in tag:
                _, _, value = zip(*eventacc.Scalars(tag))
                all_scalars[method][tag] = value

    path_json = path_result / ('scalars_' + '_'.join(exist_kinds) + '.json')
    if path_json.exists() and not force_save:
        print(f'"{path_json}" already exists.')
        continue

    with path_json.open('w') as f:
        json.dump(dict(**all_scalars[method]), f)

# %% scalars to array

if 'all_scalars' not in dir():
    all_scalars: Dict[str, Dict[str, List]] = dict()
    for method, path_result in path_results.items():
        path_json = path_result / f'scalars_{"_".join(kind_test)}.json'

        if not path_json.exists():
            raise Exception(f'scalar file does not exist in {path_result}.')

        with path_json.open('w') as f:
            all_scalars[method] = json.loads(f.read())

# Dictionary of MethodKindMetric - ndarray
# handle measurements of reverberant data
all_arrays: Dict[MethodKindMetric, ndarray] = dict()
all_methods: List[str] = list(all_scalars.keys())
if NEED_REVERB == 'sep':
    all_methods.append('Unproc.')
all_metrics = set()
for method, scalars in all_scalars.items():
    for k, v in scalars.items():
        kind, metric, rev_or_prop = k.split('/')

        # metric
        metric = metric.split('_')[-1]
        if 'SNRseg' in metric:
            metric = metric.replace('SNRseg', 'SegSNR')
        if 'SNR' in metric:
            metric += ' [dB]'
        if NEED_REVERB == 'delta':
            metric = 'Δ'+metric
        all_metrics.add(metric)

        if rev_or_prop == 'Reverberant':
            if NEED_REVERB == 'sep':
                new_key = MethodKindMetric('Unproc.', kind, metric)
                if new_key not in all_arrays:
                    all_arrays[new_key] = np.array(v)
            elif NEED_REVERB == 'delta':
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
    _rows[0] = [''] * (1 + len(kind_test) * len(all_methods))
    for i_kind, kind in enumerate(kind_test):
        _rows[0][1 + len(all_methods) * i_kind] = kind
    _rows[1] = [''] + all_methods * len(kind_test)
    for metric in all_metrics:
        _row = [metric]
        for kind, method in iterprod(kind_test, all_methods):
            _row.append(stats[MethodKindMetric(method, kind, metric)])
        _rows.append(_row)
    return _rows


path_csv = (path_root / fstem_analysis).with_suffix('.csv')
with path_csv.open('w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    rows = []
    for method in all_methods:
        rows.append([method])
        for kind, (i, metric) in iterprod(kind_test, enumerate(all_metrics)):
            rows.append([kind if i == 0 else '',
                         metric,
                         *all_arrays[MethodKindMetric(method, kind, metric)],
                         ])

    rows += [['means'], *make_rows_for_stats(all_means)]
    rows += [['stds'], *make_rows_for_stats(all_stds)]
    for r in rows:
        writer.writerow(r)


# %% draw graphs

s_path_fig = str(hp.dict_path['figures'] / fstem_analysis) + ' [{}].png'
for metric in all_metrics:
    means = np.empty((len(all_methods), len(kind_test)))
    stds = np.empty((len(all_methods), len(kind_test)))
    for (i, method), (j, kind) \
            in iterprod(enumerate(all_methods), enumerate(kind_test)):
        key = MethodKindMetric(method, kind, metric)
        means[i, j] = all_means[key]
        stds[i, j] = all_stds[key]

    # fig = draw_bar_graph(titles, mean, std, sfxs, col)
    fig = draw_lineplot(means, stds,
                        legends=all_methods,
                        xticklabels=kind_test,
                        ylabel=metric,
                        )
    fig.savefig(s_path_fig.format(metric.replace('Δ', '')), dpi=300)