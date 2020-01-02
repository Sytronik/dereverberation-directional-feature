# %%
import re
import csv
import json
from itertools import product as iterprod
from pathlib import Path
from typing import List, NamedTuple, Sequence, Union, Dict

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.transforms import Bbox
import numpy as np
from numpy import ndarray
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


# %% constants
# path_root = Path('./result')
path_root = Path('./backup/result_23_10')
path_fig = Path('./figures')
room_trained = 'room1+2+3'
foldername_methods = {
    'SIV': f'SIV_23_10',
    'DV': f'DV_23_10',
    'Mulspec32': f'mulspec_23_10',
    'Mulspec4': f'mulspec4_23_10',
    'Single': f'No-DF_23_10',
    'WPE-tuned': f'wpe',
    'WPE': f'wpe_3_3_5',
}
kind_folders = [
    'unseen_11', 'unseen_room4+5+6+7_11',
    # 'unseen_59', 'unseen_room4+5+6+7_59',
    # 'unseen_59_glim20', 'unseen_room4+5+6+7_59_glim20',
    # 'unseen_59_err5', 'unseen_room4+5+6+7_59_err5',
    # 'unseen_59_err10', 'unseen_room4+5+6+7_59_err10',
    # 'unseen_59_true_ph', 'unseen_room4+5+6+7_59_true_ph',
    # 'unseen_room9_59'
]

# set what to do
force_save_scalars = False
save_csv = False
plot_graphs = True

# set which information will be plotted
# key = plot type name, value = (selected rooms, selected kinds)
type_plots = dict(
    unseens=(['6', '1', '4', '2', '5', '3', '7'], ['unseen']),
    # seen_vs_unseen=(['1', '2', '3'], ['seen', 'unseen']),
)

# --------------------------------------------------
# how to inform measurements of reverberant data
method_rev = 'delta'  # as a difference
metric_need_legend = 'PESQ'

# method_rev = 'sep'  # as a separated item
# metric_need_legend = 'PESQ'

# method_rev = False  # not shown
# --------------------------------------------------

# legend style
kwargs_legend = dict(
    loc='upper left',
    bbox_to_anchor=(1.05, 0.98),
    ncol=1,
)

# figure size
figsize = (6, 4)  # line plot
# figsize = (7 * 1.55, 4)  # bar plot (room9)

# line styles
tab20 = list(plt.get_cmap('tab20').colors)
tab20c = list(plt.get_cmap('tab20c').colors)
tab20b = list(plt.get_cmap('tab20b').colors)
# cmap = tab20c[1::4][:-1] + [tab20b[10], tab20[12]]
# cmap = [tab20c[1], tab20c[2], tab20c[9], tab20c[10], tab20c[5], tab20c[13], tab20[12]]
# cmap = [tab20c[1], tab20c[1], tab20c[5], tab20c[5], tab20b[10], tab20c[13], tab20c[9]]
grey = (0.5, 0.5, 0.5)
cmap = [tab20c[1]] * 2 + [tab20c[5]] * 3 + [tab20c[9], tab20c[13]]
markers = ['o', 'o', 'o', 'o', 'o', 'x', 'x']
linestyles = ['-', '--', '-', '--', ':', '-', '-']


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


# dependent variable
path_methods = {k: path_root / v for k, v in foldername_methods.items()}

# ex)
#  [SIV_23_10, Mulspec32 (mulspec_23_10), ...] [unseen unseen_room4+5+6+7] [rev=delta]
# figure file:
#  [SIV_23_10, ...] [unseen unseen_room4+5+6+7] [rev=delta] [unseens] [PESQ]
#  [SIV_23_10, ...] [unseen unseen_room4+5+6+7] [rev=delta] [seen_vs_unseen] [PESQ]
#  [SIV_23_10, ...] [unseen unseen_room4+5+6+7] [rev=delta] [bar] [PESQ]

fname_result = ', '.join(
    path.name + f' ({method})' if method not in path.name else path.name
    for method, path in path_methods.items()
)
fname_result = f'[{fname_result}] [{", ".join(kind_folders)}] [rev={method_rev}]'

# %% save scalars
"""
Open tf event files of all "kinds" and merge the scalars to one file per "methods".
If a tf event file doesn't exist in a path, the path is ignored.
If `force_save_scalars==True`, overwrite the merged scalar file already exists,
else, print "blahblah json file already exists." and do nothing.

<Input>
tf event files

<Output>
all_scalars: key = method, value = dict of {key = tag, value = list of evaluation results}

[examples of tag]
"unseen_room4+5+6+7/1_SNRseg/Reverberant"
"unseen/4_STOI/Proposed"

"""
all_scalars: Dict[str, Dict[str, List]] = dict()
for method, path_result in path_methods.items():
    exist_kind_folders = [
        kind_folder for kind_folder in kind_folders
        if bool((path_result / kind_folder).glob('events.out.tfevents.*'))
    ]
    path_json \
        = path_result / ('scalars_' + '_'.join(exist_kind_folders) + '.json')
    if path_json.exists() and not force_save_scalars:
        print(f'"{path_json}" already exists.')
        continue

    all_scalars[method] = dict()
    for kind_folder in exist_kind_folders:
        path_test = path_result / kind_folder
        eventacc = EventAccumulator(
            str(path_test),
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
    for method, path_result in path_methods.items():
        path_json = path_result / f'scalars_{"_".join(kind_folders)}.json'
        if not path_json.exists():
            for p in path_result.glob('scalars_*'):
                if all(k in p.stem for k in kind_folders):
                    path_json = p
                    break
            if path_json.exists():
                print(f'"{path_json.name}" will be used.')
            else:
                raise Exception(f'scalar file does not exist in {path_result}.')

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


# %% mean / std

all_means: Dict[Metadata, np.float64] = dict()
all_stds: Dict[Metadata, np.float64] = dict()
for key, value in all_arrays.items():
    all_means[key] = np.mean(value)
    all_stds[key] = np.std(value, ddof=1)

# %% save to csv

if save_csv:
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
                rows.append([kind if i == 0 else '', metric, *values])

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
    plt.style.use('default')
    plt.rc('font', family='Arial', size=18)

    fig, ax = plt.subplots(figsize=figsize)

    # colors
    n_comparisons = len(legends) if legends else len(xticklabels)
    if method_rev == 'sep':
        colors = cmap[:n_comparisons - 1]
        colors.append(tab20c[-3])
    else:
        colors = cmap[:n_comparisons]

    if method_rev == 'delta':
        ylabel = f'Δ{ylabel}'
    if 'SNR' in ylabel:
        ylabel = f'{ylabel} [dB]'

    ylim = set_yticks_get_ylim(ax, ylabel)

    return fig, ax, colors, ylim, ylabel


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
    fig, ax, colors, ylim, ylabel \
        = _graph_initialize(means, stds,
                            legends=legends,
                            xticklabels=xticklabels,
                            ylabel=ylabel)

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

    ax.grid(True, axis='y')

    ax.set_ylim(*ylim)

    # axis label
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # ticks
    ax.tick_params('x', direction='in')
    ax.tick_params('y', direction='in')

    # legend
    # if draw_legend:
    # ax.legend(loc='lower right', bbox_to_anchor=(1, 1),
    #           ncol=4, fontsize='small', columnspacing=1)
    # ax.legend(lineplots, legends,
    #           loc=legend_loc,
    #           ncol=legend_col,
    #           fontsize='small',
    #           columnspacing=1)

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
    fig, ax, xticklabels, colors, ylim \
        = _graph_initialize(means, stds,
                            legends=None,
                            xticklabels=xticklabels,
                            ylabel=ylabel)

    # draw bar & text
    ax.bar(xticklabels, means,
           width=bar_width,
           #  yerr=stds,
           error_kw=dict(capsize=5),
           color='k',
           )

    ax.grid(True, axis='y')
    ax.set_axisbelow(True)

    ax.set_xlim([-bar_width, len(xticklabels) - 1 + bar_width])
    # ax.set_ylim(*ylim)

    ax.set_ylabel(ylabel)

    ax.tick_params('x', length=0)
    ax.tick_params('y', direction='in')

    # fig.tight_layout()
    return fig


def full_extent(ax, extra=None, pad=0.01):
    """Get the full extent of an axes, including axes labels, tick labels, and
    titles."""
    # For text objects, we need to draw the figure first, otherwise the extents
    # are undefined.
    ax.figure.canvas.draw()
    items = ax.get_xticklabels() + ax.get_yticklabels()
    items += [ax, ax.title, ax.xaxis.label, ax.yaxis.label]
    if extra is not None:
        items += [extra]
        if hasattr(extra, 'get_yticklabels'):
            items += extra.get_yticklabels()
    bbox = Bbox.union([item.get_window_extent() for item in items])

    return bbox.expanded(1.0 + pad, 1.0 + pad)


# %% draw graphs

if 'room9' not in all_kinds and plot_graphs:
    for suffix, selected in type_plots.items():
        rooms, kinds = selected

        s_path_fig = str(path_fig / fname_result) + f' [{suffix}] [{{}}].png'
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
                xticklabels=convert_xticklabels(suffix, rooms, kinds),
                xlabel='' if suffix == 'seen_vs_unseen' else 'Room No.',
                ylabel=metric,
                n_connect=2 if suffix == 'seen_vs_unseen' else -1,
            )
            bbox = full_extent(
                ax, legend if metric_need_legend == metric else None
            ).transformed(fig.dpi_scale_trans.inverted())
            fig.savefig(s_path_fig.format(metric),
                        bbox_inches=bbox,
                        dpi=300,
                        )

# %% draw bar graph per methods

if 'room9' in all_kinds and plot_graphs:
    s_path_fig = str(path_fig / fname_result) + f' [bar] [{{}}].png'
    for metric in all_metrics:
        means = np.empty(len(all_methods))
        stds = np.empty(len(all_methods))
        for (i, method) in enumerate(all_methods):
            key = Metadata(method, all_rooms[0], all_kinds[0], metric)
            means[i] = all_means[key]
            stds[i] = all_stds[key]

        fig = draw_bar_graph(
            means, stds,
            xticklabels=list(foldername_methods.keys()),
            ylabel=metric,
        )
        fig.savefig(s_path_fig.format(metric), dpi=300)
