# %%
import pickle
from itertools import product as iterprod
from pathlib import Path
from typing import Dict, Set

import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray

path_root = Path('./result')
# path_root = Path('./backup/result_23_10')
room_trained = 'room1+2+3'
all_methods = ['SIV', 'DV', 'Mulspec32', 'Mulspec4']

filename_pickles = {
    'Unprocessed': 'SIV,DV,Mulspec32,Mulspec4-59_glim20-delta.pickle',
    'DAS': 'SIV,DV,Mulspec32,Mulspec4,Single,WPE-tuned,WPE-59-delta.pickle',
    'True': 'SIV,DV,Mulspec32,Mulspec4,Single-59_true_ph-delta.pickle',
    # 'DAS w/ DoA Error': 'SIV,DV,Mulspec32,Mulspec4-59_err5-delta.pickle',
}

bar_width = 0.6 / len(filename_pickles)
gray = (0.5,) * 3
cmap = ['k', gray, 'w', gray]
hatches = [None, None, None, '//']


def set_yticks_get_ylim(ax: plt.Axes, ylabel: str):
    if ylabel == 'ΔPESQ':
        ylim = (0, 1.5)
    elif ylabel == 'ΔSTOI':
        ylim = (0, 0.25)
    elif ylabel == 'ΔfwSegSNR':
        ylim = (0, 10)
    else:
        ylim = None
    return ylim


path_fig = path_root / 'figures'
path_fig.mkdir(exist_ok=True)


# %% get array from pickle files and calculate means

all_metrics: Set[str] = set()
all_means: Dict[str, Dict[str, ndarray]] = dict()
metric_is_delta = False
for phasetype, fname in filename_pickles.items():
    if 'delta' in fname:
        metric_is_delta = True
    with (path_root / fname).open('rb') as f:
        dict_pickled = pickle.load(f)

    rooms = set()
    kinds = set()
    for key, value in dict_pickled.items():
        _, room, kind, metric = key
        rooms.add(room)
        kinds.add(kind)
        all_metrics.add(metric)

    for metric in all_metrics:
        means = []
        for i_method, method in enumerate(all_methods):
            array = []
            for room, kind in iterprod(rooms, kinds):
                array += dict_pickled[method, room, kind, metric]
            means.append(np.mean(array))

        if metric not in all_means:
            all_means[metric] = dict()
        all_means[metric][phasetype] = np.array(means)


# %% draw bar plots

plt.style.use('default')
plt.rc('font', family='Arial', size=18)

figs = []
for metric, means in all_means.items():
    fig: plt.Figure = None
    ax: plt.Axes = None
    fig, ax = plt.subplots()
    xaxis = (np.arange(len(all_methods))
             - bar_width * (len(filename_pickles) - 1) / 2)
    for i_row, row in enumerate(means.values()):
        ax.bar(xaxis + bar_width * i_row, row,
               color=cmap[i_row],
               width=bar_width,
               edgecolor='k',
               hatch=hatches[i_row],
               )
    ax.set_xticks(np.arange(len(all_methods)))
    ax.set_xticklabels(all_methods)
    ax.set_xlim(
        - bar_width * (len(filename_pickles) - 1),
        (len(all_methods) - 1) + bar_width * (len(filename_pickles) - 1),
    )

    if metric_is_delta:
        metric = f'Δ{metric}'
    ylim = set_yticks_get_ylim(ax, metric)
    if ylim:
        ax.set_ylim(ylim)
    ax.set_ylabel(metric)

    ax.grid(True, axis='y')
    ax.set_axisbelow(True)

    ax.tick_params('x', length=0)
    ax.tick_params('y', direction='in')

    if 'fwSegSNR' in metric:
        ax.legend(
            list(filename_pickles.keys()),
            fontsize='small',
            ncol=2,
            columnspacing=1
        )

    # fig.tight_layout()
    figs.append(fig)


# %% save figures
for metric, fig in zip(all_metrics, figs):
    fig.savefig(
        path_fig / f'{",".join(all_methods)}-phase_diff-{metric}.png',
        bbox_inches='tight',
        dpi=300,
    )


# %%
