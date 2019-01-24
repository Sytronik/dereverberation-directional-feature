import openpyxl
import csv
import numpy as np
import json
import os
from collections import defaultdict
from argparse import ArgumentParser

parser = ArgumentParser()
# parser.add_argument('title2', nargs=1, type=str)
# parser.add_argument('--title1', nargs=1, type=str, metavar='title1')
parser.add_argument('path', nargs='?', type=str, const='./')
parser.add_argument('--tag', nargs=1, type=str, default=('',))
ARGS = parser.parse_args()
files = [f.path
         for f in os.scandir(ARGS.path)
         if f.name.endswith('.json')]
if not files:
    print('No files.')
    exit()

step = None
measures = dict()
col_title = ''

common = defaultdict(lambda: 0)
for fname in files:
    fname = os.path.basename(fname)
    fname_split = fname.replace('.json', '').split('_')
    if ARGS.tag[0] not in fname_split:
        continue

    print(fname)

    if not col_title:
        col_title = fname_split[2]

    for item in fname_split:
        if '.' not in item:
            common[item] += 1

if not col_title:
    print('No files.')
    exit()

print()

col_names = [k for k, v in common.items() if v == 1]
title = ARGS.tag[0]

for fname, col_name in zip(files, col_names):
    with open(fname) as f:
        data = json.loads(f.read())
        data = np.array(data)
    if not step:  # and ARGS.title1:
        step = data[:, 1].tolist()
    measures[col_name] = data[:, 2]


wb = openpyxl.Workbook()
ws = wb.active
# cols = ['B', 'C', 'D']  # if step else ['A', 'B', 'C']
cols = [chr(ord('B')+i) for i in range(len(measures))]

ws['A1'] = title
ws[f'{cols[0]}1'] = col_title
ws.merge_cells(f'{cols[0]}1:{cols[-1]}1')

ws['A2'] = 'Step' if step else ''
for idx, (cell,) in enumerate(ws[f'A3:A{3 + len(step) - 1}']):
    cell.value = step[idx]

for ii, (key, value) in enumerate(measures.items()):
    ws[f'{cols[ii]}2'] = key
    for jj, (cell,) in enumerate(ws[f'{cols[ii]}3:{cols[ii]}{3 + value.shape[0] - 1}']):
        cell.value = value[jj]

fname_result = os.path.join(ARGS.path, f'merged_{title}_{col_title}.xlsx')
while True:
    try:
        wb.save(fname_result)
        print(fname_result)
        break
    except PermissionError:
        fname_result = fname_result.replace('.xlsx', '_.xlsx')
