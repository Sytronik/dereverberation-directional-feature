# %%
import csv
import json
import pickle
from difflib import SequenceMatcher
from itertools import product as iterprod
from pathlib import Path
from typing import List, NamedTuple, Sequence, Union, Dict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter
from numpy import ndarray
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from utils import full_extent


# %% constants
path_root = Path('./result')
# path_root = Path('./backup/result_23_10')
room_trained = 'room1+2+3'
foldername_methods = {
    'SIV': 'SIV',
    'DV': 'DV',
    'Mulspec32': 'Mulspec32',
    'Mulspec4': 'Mulspec4',
    'Single': 'Single',
    'WPE-tuned': 'WPE-tuned',
    'WPE': 'WPE',
}

# uncomment one line only
kind_folders = [
    # 'unseen_11', 'unseen_room4+5+6+7_11',  # 11 epoch room1+2+3 and room4+5+6+7
    'unseen_59', 'unseen_room4+5+6+7_59',  # 59 epoch room1+2+3 and room4+5+6+7
    # 'unseen_59_true_ph', 'unseen_room4+5+6+7_59_true_ph',  # true (anechoic) phase
    # 'unseen_59_glim20', 'unseen_room4+5+6+7_59_glim20',  # without das phase
    # 'unseen_59_err5', 'unseen_room4+5+6+7_59_err5',  # das with 5 deg azimuth error
    # 'unseen_59_err10', 'unseen_room4+5+6+7_59_err10',  # das with 5 deg azimuth error
    # 'unseen_room9_59'  # real room
]

# set what to do
force_save_scalars = False
save_csv = True
plot_graphs = False

# set which information will be plotted
# key = information type name, value = (selected rooms, selected kinds)
info_types = {
    'unseens': (['6', '1', '4', '2', '5', '3', '7'], ['unseen']),
    # 'seen_vs_unseen': (['1', '2', '3'], ['seen', 'unseen']),
}

metric_need_legend = 'PESQ'  # which plot of metric needs legend

# how to inform measurements of reverberant data
method_rev = 'delta'  # as a difference
# method_rev = 'sep'  # as a separated item
# method_rev = False  # not shown

# legend style
kwargs_legend = dict(
    loc='upper left',
    bbox_to_anchor=(1.05, 0.98),
    ncol=1,
)

# figure size
figsize = (6, 4)  # line plot
# figsize = (11, 4)  # bar plot (room9) method_rev=delta
# figsize = (12.5, 4)  # bar plot (room9) method_rev=sep

# line styles
tab20 = list(plt.get_cmap('tab20').colors)
tab20c = list(plt.get_cmap('tab20c').colors)
tab20b = list(plt.get_cmap('tab20b').colors)
# cmap = tab20c[1::4][:-1] + [tab20b[10], tab20[12]]
# cmap = [tab20c[1], tab20c[2], tab20c[9], tab20c[10], tab20c[5], tab20c[13], tab20[12]]
# cmap = [tab20c[1], tab20c[1], tab20c[5], tab20c[5], tab20b[10], tab20c[13], tab20c[9]]
grey = (0.5, 0.5, 0.5)
cmap = [tab20c[1]] * 2 + [tab20c[5]] * 3 + [tab20c[9], tab20c[13]]
markers = ['o', 'o', 'o', 'o', 'o', 'x', 'x', '.']
linestyles = ['-', '--', '-', '--', ':', '-', '-', ':']


# overwrite room number in ascending order
def convert_xticklabels(suffix, selected_rooms, selected_kinds):
    if suffix == 'unseens':
        return [str(i + 1) for i in range(len(selected_rooms))]
    else:
        return [f'room{i // 2 * 2 + 2}\n({kind})'
                for i, kind in iterprod(range(len(selected_rooms)), selected_kinds)
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
        ylim = (0, 15)
        # ax.set_yticks(np.linspace(*ylim, num=5))
    elif ylabel == 'ΔfwSegSNR [dB]':
        ylim = (-1, 7)
        ax.set_yticks(np.linspace(*ylim, num=9))
    elif ylabel == 'ΔPESQ':
        ylim = (0, 1.5)
        # ylim = (-1, 3)
    elif ylabel == 'ΔSTOI':
        ylim = (-0.05, 0.35)
        # ylim = (-0.5, 0.7)
        ax.set_yticks(np.linspace(*ylim, num=9))
    else:
        ylim = None
    return ylim


class Metadata(NamedTuple):
    method: str
    room: str
    kind: str
    metric: str


# get the longest substring of args
def longest_substr(*args):
    def _longest(str1, str2):
        # initialize SequenceMatcher object with
        # input string
        seqMatch = SequenceMatcher(None, str1, str2)

        # find match of longest sub-string
        # output will be like Match(a=0, b=0, size=5)
        match = seqMatch.find_longest_match(0, len(str1), 0, len(str2))

        # print longest substring
        if match.size > 0:
            return str1[match.a: match.a + match.size]
        else:
            return ''
    args = list(args)
    while len(args) > 1:
        a = args.pop()
        b = args.pop()
        args.append(_longest(a, b))
    return args[0]


# dependent variable
path_methods = {k: path_root / v for k, v in foldername_methods.items()}

# result filenames
# ex)
#  SIV,DV,...,WPE-59-delta.csv
#  SIV,DV,...,WPE-59-delta-unseens-PESQ.png
#  SIV,DV,...,WPE-59-sep-seen_vs_unseen-STOI.png
#  SIV,DV,...,WPE-room9_59-sep-bar-STOI.png
temp = [s.replace(f'{s.split("_")[0]}_', '') for s in kind_folders]
substr = longest_substr(*temp)
fname_result = ','.join(method for method in path_methods.keys())
fname_result = f'{fname_result}-{substr}-{method_rev}'

# create figure folder if not exists
path_fig = path_root / 'figures'
path_fig.mkdir(exist_ok=True)

# %% extract, merge, and save scalars
"""
Open tf event files of all "kinds" and merge the scalars to one file per "methods".
If a tf event file doesn't exist in a path, the path is ignored.
If `force_save_scalars==True`, overwrite the merged scalar file already exists,
else, print "blahblah json file already exists." and do nothing.

<Input>
tf event files

<Output>
all_scalars: key = method, value = dict of {key = tag, value = list of evaluation results}
save all_scalars to 'scalars-{}-{}-...-{}.json'
{} is kind_folder

[examples of tag]
"unseen_room4+5+6+7/1_SNRseg/Reverberant"
"unseen/4_STOI/Proposed"

"""
all_scalars: Dict[str, Dict[str, List]] = dict()
for method, path_method in path_methods.items():
    exist_kind_folders = [
        folder for folder in kind_folders
        if next((path_method / folder).glob('events.out.tfevents.*'), None) is not None
    ]
    if not exist_kind_folders:
        print(f'Any of {", ".join(kind_folders)} exists in {path_method.name}. '
              f'Skip extracting scalars.')
        continue

    path_json = path_method / (f'scalars-{"-".join(exist_kind_folders)}.json')
    if path_json.exists() and not force_save_scalars:
        print(f'"{path_json}" already exists. Skip saving merged scalar file.')
        continue

    all_scalars[method] = dict()
    for kind_folder in exist_kind_folders:
        path = path_method / kind_folder
        eventacc = EventAccumulator(
            str(path),
            size_guidance=dict(scalars=0, images=1, audio=1),
        )
        eventacc.Reload()
        for tag in eventacc.Tags()['scalars']:
            if 'Reverberant' in tag or 'Proposed' in tag:
                _, _, value = zip(*eventacc.Scalars(tag))

                kind_split = kind_folder.split('_')
                if kind_split[1].startswith('room'):
                    kind = '_'.join(kind_split[:2])
                else:
                    kind = '_'.join(kind_split[:1])

                tag = tag.replace(tag.split('/')[0], kind)
                all_scalars[method][tag] = value

    with path_json.open('w') as f:
        json.dump(dict(**all_scalars[method]), f)

# %% scalars to array
"""
all_scalars --> all_arrays, all_methods, all_kinds, all_metrics

all_arrays: key = Metadata, value = np.ndarray of evaluation results
all_methods: list of methods (without numbers (1-4).)
all_rooms: list of rooms (split into each room)
all_kinds: list of kinds (seen/unseen)
all_metrics: list of metrics

Only the first occurence of "Reverberant" tag is used.
If `method_rev == delta`, all ndarrays becomes the difference between "Proposed" and "Reverberant",
and all metrics becomes `f'Δ{metric}'`,
else, the first occurence of "Reverberant" tag becomes "Unproc." method.

SNRseg --> SegSNR

"""

# Load missed scalar data.
# If the scalar file doesn't exist, similar files can be used with warning.
if 'all_scalars' not in dir() or len(all_scalars) < len(foldername_methods):
    all_scalars: Dict[str, Dict[str, List]] = dict()
    for method, path_method in path_methods.items():
        path_json = path_method / f'scalars-{"-".join(kind_folders)}.json'
        if not path_json.exists():
            for p in path_method.glob('scalars-*'):
                if all(k in p.stem for k in kind_folders):
                    path_json = p
                    break
            if path_json.exists():
                print(f'"{path_json.name}" will be used.')
            else:
                raise Exception(
                    f'No scalar file for {", ".join(kind_folders)} exists '
                    f'in {path_method.name}.'
                )

        with path_json.open() as f:
            all_scalars[method] = json.loads(f.read())

all_arrays: Dict[Metadata, ndarray] = dict()
all_methods: List[str] = list(all_scalars.keys())
if method_rev == 'sep':
    all_methods.append('Unproc.')
all_rooms = set()
all_kinds = set()
all_metrics = set()
for method, scalars in all_scalars.items():
    for k, v in scalars.items():
        room_kind, metric, rev_or_prop = k.split('/')

        # room kind
        if '_' in room_kind:  # room not in training set
            rooms = room_kind.split('_')[1]
            kind = 'unseen'
        else:  # room in training set
            rooms = room_trained
            kind = room_kind
        all_kinds.add(kind)

        # metric
        metric = metric.split('_')[-1]  # remove number
        if 'SNRseg' in metric:
            metric = metric.replace('SNRseg', 'SegSNR')
        all_metrics.add(metric)

        rooms = rooms.replace('room', '').split('+')
        for i, room in enumerate(rooms):
            all_rooms.add(room)
            if rev_or_prop == 'Reverberant':
                new_key = Metadata('Unproc.', room, kind, metric)
            else:
                new_key = Metadata(method, room, kind, metric)

            if new_key not in all_arrays:
                all_arrays[new_key] = np.array(v[i::len(rooms)])

if method_rev == 'delta':
    for k, v in all_arrays.items():
        if k.method == 'Unproc.':
            continue
        rev_key = Metadata('Unproc.', k.room, k.kind, k.metric)
        all_arrays[k] -= all_arrays[rev_key]
    all_arrays = {k: v for k, v in all_arrays.items() if k.method != 'Unproc.'}

all_kinds: List[str] = list(all_kinds)
all_kinds.sort()
all_rooms: List[str] = list(all_rooms)
all_rooms.sort()
all_metrics: List[str] = list(all_metrics)

# %% pickle array
"""
save all_arrays for analysis_phase.py
"""
with (path_root / fname_result).with_suffix('.pickle').open('wb') as f:
    pickle.dump(
        {(k.method, k.room, k.kind, k.metric): v.tolist()
         for k, v in all_arrays.items()
         },
        f
    )

# %% mean / std

all_means: Dict[Metadata, np.float64] = dict()
all_stds: Dict[Metadata, np.float64] = dict()
for key, value in all_arrays.items():
    all_means[key] = np.mean(value)
    all_stds[key] = np.std(value, ddof=1)

# %% save to csv
""" 
save csv file for human-friendly data expression.

[csv example]

METHOD1
ROOM/KIND1  METRIC1  0        0        0       ...
            METRIC2  0        0        0       ...
ROOM/KIND2  METRIC1  0        0        0       ...
            METRIC2  0        0        0       ...
METHOD2
ROOM/KIND1  METRIC1  0        0        0       ...
            METRIC2  0        0        0       ...
ROOM/KIND2  METRIC1  0        0        0       ...
            METRIC2  0        0        0       ...
means
            ROOM/KIND1        ROOM/KIND2
            METHOD1  METHOD2  METHOD1  METHOD2
METRIC1     0        0        0        0
METRIC2     0        0        0        0
stds
            ROOM/KIND1        ROOM/KIND2
            METHOD1  METHOD2  METHOD1  METHOD2
METRIC1     0        0        0        0
METRIC2     0        0        0        0
"""

if save_csv:
    def make_rows_for_stats(stats):
        _rows = [None] * 2
        _rows[0] = [''] * (1 + len(all_rooms) * len(all_kinds) * len(all_methods))
        for i_roomkind, (room, kind) in enumerate(iterprod(all_rooms, all_kinds)):
            _rows[0][1 + len(all_methods) * i_roomkind] = f'room{room} ({kind})'
        _rows[1] = [''] + all_methods * len(all_rooms) * len(all_kinds)
        for metric in all_metrics:
            _row = [f'Δ{metric}' if method_rev == 'delta' else metric]
            for room, kind, method in iterprod(all_rooms, all_kinds, all_methods):
                _row.append(stats.get(Metadata(method, room, kind, metric), '-'))
            _rows.append(_row)
        return _rows

    path_csv = (path_root / fname_result).with_suffix('.csv')
    with path_csv.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        rows = []
        for method in all_methods:
            rows.append([method])
            for room, kind, (i, metric) in iterprod(all_rooms, all_kinds, enumerate(all_metrics)):
                values = all_arrays.get(Metadata(method, room, kind, metric), ['-'])
                rows.append([
                    f'room{room} ({kind})' if i == 0 else '',
                    f'Δ{metric}' if method_rev == 'delta' else metric,
                    *values,
                ])

        rows += [['means'], *make_rows_for_stats(all_means)]
        rows += [['stds'], *make_rows_for_stats(all_stds)]
        for r in rows:
            writer.writerow(r)


# %% plotting functions

def _graph_initialize(means: ndarray, stds: ndarray,
                      legends: Union[str, Sequence[str]],
                      xticklabels: Sequence,
                      ylabel: str,
                      set_yticks=True,
                      ):
    global room_trained, method_rev, metric_need_legend
    plt.style.use('default')
    plt.rc('font', family='Arial', size=18)

    fig, ax = plt.subplots(figsize=figsize)

    # colors
    n_comparisons = len(legends) if legends else len(xticklabels)
    if method_rev == 'sep':
        colors = cmap[:n_comparisons - 1]
        colors.append(tab20c[-3])
        _markers = markers[:n_comparisons - 1]
        _markers.append(markers[-1])
        _linestyles = linestyles[:n_comparisons - 1]
        _linestyles.append(linestyles[-1])
    else:
        colors = cmap
        _markers = markers
        _linestyles = linestyles

    if method_rev == 'delta':
        ylabel = f'Δ{ylabel}'
    if 'SNR' in ylabel:
        ylabel = f'{ylabel} [dB]'

    ylim = set_yticks_get_ylim(ax, ylabel) if set_yticks else None

    ax.grid(True, axis='y')
    ax.tick_params('y', direction='in')
    ax.set_ylabel(ylabel)

    return fig, ax, ylabel, ylim, colors, _markers, _linestyles


def draw_lineplot(means: ndarray, stds: ndarray = None,
                  legends: Union[str, Sequence[str]] = None,
                  xticklabels: Sequence = None,
                  xlabel: str = '',
                  ylabel: str = '',
                  n_connect: int = -1):
    if legends is None:
        legends = ('',) * means.shape[0]
    if stds is None:
        stds = (None,) * len(legends)
    if n_connect == -1:
        n_connect = means.shape[1]
    fig: plt.Figure = None
    ax: plt.Axes = None
    fig, ax, ylabel, ylim, colors, markers, linestyles \
        = _graph_initialize(means, stds,
                            legends=legends,
                            xticklabels=xticklabels,
                            ylabel=ylabel,
                            set_yticks=True,
                            )

    # draw
    x_range = np.arange(means.shape[1])
    lineplots = []
    for ii, (label, mean, std) in enumerate(zip(legends, means, stds)):
        for jj in x_range[::n_connect]:
            line, = ax.plot(x_range[jj:jj + n_connect], mean[jj:jj + n_connect],
                            color=colors[ii],
                            marker=markers[ii],
                            linestyle=linestyles[ii],
                            )
        lineplots.append(line)

    ax.set_xticks(x_range)
    ax.set_xticklabels(xticklabels)
    ax.set_xlim(x_range[0] - 0.5, x_range[-1] + 0.5)
    ax.set_xlabel(xlabel)
    ax.tick_params('x', direction='in')

    ax.set_ylim(*ylim)

    legend = ax.legend(lineplots, legends,
                       shadow=False,
                       fontsize='small', columnspacing=1,
                       **kwargs_legend)

    # fig.tight_layout()
    return fig, ax, legend


def draw_bar_graph(means: ndarray, stds: ndarray = None,
                   xticklabels: Sequence = None,
                   ylabel: str = ''):
    if stds is None:
        stds = (None,) * len(xticklabels)

    # constants
    bar_width = 0.5

    fig: plt.Figure = None
    ax: plt.Axes = None
    fig, ax, colors, ylabel, *_, \
        = _graph_initialize(means, stds,
                            legends=None,
                            xticklabels=xticklabels,
                            ylabel=ylabel,
                            set_yticks=False,
                            )

    # draw bar & text
    ax.bar(xticklabels, means,
           width=bar_width,
           #  yerr=stds,
           error_kw=dict(capsize=5),
           color='k',
           )
    ax.set_axisbelow(True)

    ax.set_xlim([-bar_width, len(xticklabels) - 1 + bar_width])
    ax.tick_params('x', length=0)

    # ax.set_ylim(*ylim)

    # fig.tight_layout()
    return fig


# %% draw line plots about room/position vs metric

if '9' not in all_rooms and plot_graphs:
    for info_type, selected in info_types.items():
        rooms, kinds = selected

        s_p_fig = str(path_fig / f'{fname_result}-{info_type}-{{}}.png')
        for metric in all_metrics:
            means = np.empty((len(all_methods), len(rooms) * len(kinds)))
            stds = np.empty((len(all_methods), len(rooms) * len(kinds)))
            for i, method in enumerate(all_methods):
                for j, (room, kind) in enumerate(iterprod(rooms, kinds)):
                    key = Metadata(method, room, kind, metric)
                    means[i, j] = all_means.get(key, np.float64('nan'))
                    stds[i, j] = all_stds.get(key, np.float64('nan'))

            fig, ax, legend = draw_lineplot(
                means, stds,
                legends=all_methods,
                xticklabels=convert_xticklabels(info_type, rooms, kinds),
                xlabel='' if info_type == 'seen_vs_unseen' else 'Room No.',
                ylabel=metric,
                n_connect=2 if info_type == 'seen_vs_unseen' else -1,
            )
            bbox = full_extent(
                fig, ax, legend if metric_need_legend == metric else None
            )
            fig.savefig(s_p_fig.format(metric), bbox_inches=bbox, dpi=300)

# %% draw bar plots about method vs metric

if '9' in all_rooms and plot_graphs:
    s_p_fig = str(path_fig / f'{fname_result}-bar-{{}}.png')
    xticklabels = list(foldername_methods.keys())
    if method_rev == 'sep':
        xticklabels.append('Unproc.')
    for metric in all_metrics:
        means = np.empty(len(all_methods))
        stds = np.empty(len(all_methods))
        for (i, method) in enumerate(all_methods):
            key = Metadata(method, all_rooms[0], all_kinds[0], metric)
            means[i] = all_means[key]
            stds[i] = all_stds[key]

        fig = draw_bar_graph(
            means, stds,
            xticklabels=xticklabels,
            ylabel=metric,
        )
        fig.savefig(s_p_fig.format(metric), bbox_inches='tight', dpi=300)
