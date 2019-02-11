import json
import os
from argparse import ArgumentParser
from glob import glob
from os.path import join as pathjoin
from typing import List, NamedTuple

import numpy as np
from tensorboard.backend.event_processing.event_multiplexer import EventMultiplexer
from tqdm import tqdm
import csv

import matplotlib.pyplot as plt

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


if __name__ == '__main__':
    paths = [p for p in glob(pathjoin(PATH_RESULT, PRIMARY)) if os.path.isdir(p)]
    paths = [p for p in paths if COND_PRIMARY(p)]
    paths = sorted(paths)

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
            titles = ([TITLE_BASELINE]
                      + [f'{TITLE_PRIMARY}{idx}' for idx in range(len(paths))]
                      + ['Reverberant'])
        else:
            titles = [TITLE_BASELINE, TITLE_PRIMARY, 'Reverberant']
        length = len(titles)
        paths.insert(0, pathjoin(PATH_RESULT, BASELINE))
        if ARGS.save_scalars:
            for path in tqdm(paths):
                save_scalars_from_tfevents(path, SUFFIX)

        elif ARGS.merge_scalars:
            # gather data
            data = {}
            suffixes: list = None
            for idx, path in enumerate(paths):
                paths_suffix, suffixes = zip(*((p.path, p.name) for p in os.scandir(path)
                                               if p.is_dir() and p.name != 'train'))
                assert suffixes, f'{path} is empty.'
                # suffixes = [os.path.basename(p) for p in paths_suffix]
                for suffix, p in zip(suffixes, paths_suffix):
                    with open(pathjoin(p, 'scalars.json'), 'r') as f:
                        temp = json.loads(f.read())
                        for k, v in temp.items():
                            data[IdxSuffixCol(idx, suffix, k)] = v

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
                        key_rev = IdxSuffixCol(-1, key.suffix, col_new)
                        if key_rev not in data_arranged:
                            data_arranged[key_rev] = value
                    else:
                        data_arranged[key.mutated_ver(_col=col_new)] = value
            data = data_arranged
            cols: list = sorted(cols)
            idxs = list(range(length-1)) + [-1]
            del data_arranged

            # calculate mean, std
            means = {}
            stds = {}
            upper = {}
            lower = {}
            for key, value in data.items():
                means[key] = np.mean(value)
                stds[key] = np.std(value, ddof=1)
                upper[key] = means[key] + stds[key]
                lower[key] = means[key] - stds[key]

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

            fmerged = ', '.join([f'{os.path.basename(p)} ({t})'
                                 for p, t in zip(paths, titles[:-1])])
            fmerged = pathjoin(PATH_RESULT, fmerged + '.csv')
            with open(fmerged, 'w', newline='') as f:
                writer = csv.writer(f)
                rows = []
                for idx, title in zip(idxs, titles):
                    rows.append([title])
                    for suffix in suffixes:
                        for ii, col in enumerate(cols):
                            rows.append([suffix if ii == 0 else '',
                                         col,
                                         *data[IdxSuffixCol(idx, suffix, col)],
                                         ]
                                        )
                rows += [['means'], *make_rows(means)]
                rows += [['stds'], *make_rows(stds)]
                for r in rows:
                    writer.writerow(r)

            # plot data

            # suffixes = sorted(list({k.suffix for k in means}))
            # indices = sorted(list({k.idx for k in means}))
            # titles_sorted = [titles[idx] for idx in indices]
            # length = len(titles_sorted)
            # cols = sorted(list({k.col for k in means}))
            # # SuffIdxs = [(suffix, idx) for suffix in suffixes for idx in indices]
            # # x = [(suffix, title) for suffix in suffixes for title in titles_sorted]
            # for col in cols:
            #     # top = [means[IdxSuffixCol(i, s, col)] for s, i in SuffIdxs]
            #     # data = {'suffixes': suffixes}
            #     data = {titles[i]: [means[IdxSuffixCol(i, s, col)]
            #                         for s in suffixes]
            #             for i in indices}
            #     data_upper = {titles[i]: [upper[IdxSuffixCol(i, s, col)]
            #                               for s in suffixes]
            #                   for i in indices}
            #     data_lower = {titles[i]: [lower[IdxSuffixCol(i, s, col)]
            #                               for s in suffixes]
            #                   for i in indices}
            #     data['suffixes'] = suffixes
            #     source = ColumnDataSource(data=data)
            #     plt.output_file(pathjoin(PATH_FIG, f'{col}.html'))
            #     p = plt.figure(plot_height=350,
            #                    x_range=suffixes,
            #                    x_axis_label='RIR',
            #                    y_axis_label=col,
            #                    toolbar_location=None, tools=''
            #                    )
            #     max_ = max(sum(zip(*list(data.values())[:-1]), ()))
            #     min_ = min(sum(zip(*list(data.values())[:-1]), ()))
            #     if max_ - min_ < 1:
            #         ndigits = 3
            #     else:
            #         ndigits = 2
            #     for idx, title in enumerate(titles_sorted):
            #         x_dodged = dodge(
            #             'suffixes',
            #             1 / (length + 1) * (idx - (length - 1) / 2),
            #             range=p.x_range
            #         )
            #         p.vbar(source=source,
            #                x=x_dodged, top=title,
            #                width=0.8 / (length + 1),
            #                color=Palette[length][idx],
            #                legend=bkcp.value(title),
            #                )
            #
            #         # text
            #         text = [f'{round(d, ndigits):.{ndigits}f}'
            #                 for d in data[title]
            #                 ]
            #         label_source = ColumnDataSource(
            #             data=dict(x=suffixes, y=data[title], text=text)
            #         )
            #         x_text_dodged = dodge(
            #             'x',
            #             1 / (length + 1) * (idx - (length - 1) / 2),
            #             range=p.x_range
            #         )
            #         labels = LabelSet(source=label_source,
            #                           x=x_text_dodged, y='y', text='text',
            #                           level='glyph',
            #                           x_offset=15,
            #                           y_offset=2 if data[title][0] > 0 else -40,
            #                           angle=45,
            #                           render_mode='canvas',
            #                           )
            #         p.add_layout(labels)
            #
            #         # error bar
            #         whisker_source = ColumnDataSource(
            #             data=dict(base=suffixes,
            #                       upper=data_upper[title],
            #                       lower=data_lower[title],
            #                       )
            #         )
            #         base_dodged = dodge(
            #             'base',
            #             1 / (length + 1) * (idx - (length - 1) / 2),
            #             range=p.x_range
            #         )
            #         p.add_layout(
            #             Whisker(source=whisker_source,
            #                     base=base_dodged, upper="upper", lower="lower",
            #                     level="overlay",
            #                     )
            #         )
            #     p.xgrid.grid_line_color = None
            #     p.x_range.range_padding = 0.1
            #
            #     max_ = max(sum(zip(*list(data_upper.values())[:-1]), ()))
            #     min_ = min(sum(zip(*list(data_lower.values())[:-1]), ()))
            #
            #     p.y_range.start = min_
            #     p.y_range.start -= 0.2 * (max_ - min_) if min_ > 0 else 0.1 * (max_ - min_)
            #     p.y_range.end = max_ + 0.2 * (max_ - min_)
            #     if col in Y_MAX:
            #         p.y_range.end = min(p.y_range.end, Y_MAX[col])
            #
            #     # p.legend.orientation = 'horizontal'
            #     p.legend.location = 'top_right'
            #     p.xaxis.major_tick_line_color = None
            #     p.axis.axis_label_text_font_style = 'normal'
            #     p.xaxis.major_label_text_font_size = '10pt'
            #     p.legend.label_text_font_size = '9pt'
            #     p.legend.orientation = 'horizontal'
            #     plt.show(p)
            #     bkio.export_png(p, filename=pathjoin(PATH_FIG, f'{col}.png'))
