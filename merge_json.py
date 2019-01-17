import openpyxl
import csv
import numpy as np
import json
import os
from argparse import ArgumentParser

parser = ArgumentParser()
# parser.add_argument('title2', nargs=1, type=str)
# parser.add_argument('--title1', nargs=1, type=str, metavar='title1')
parser.add_argument('path', nargs='?', type=str, const='./')
parser.add_argument('--tag', nargs=1, type=str, default=('',))
ARGS = parser.parse_args()
files = [f.path
         for f in os.scandir(ARGS.path)
         if f.name.endswith('.json') and ARGS.tag[0] in f.name]
if not files:
    print('No files.')
    exit()
print(files)
print()

step = None
measures = dict()
title = ''
col_title = ''

for fname in files:
    with open(fname) as f:
        data = json.loads(f.read())
        data = np.array(data)
    fname = fname.replace('.json', '')
    if not step:  # and ARGS.title1:
        step = data[:, 1].tolist()
        title = fname.split('_')[1] if ARGS.tag[0] == 'Proposed' else ARGS.tag[0]
        col_title = fname.split('_')[2]
    measures[fname.split('_')[-1]] = data[:, 2]


wb = openpyxl.Workbook()
ws = wb.active
cols = ['B', 'C', 'D']  # if step else ['A', 'B', 'C']

ws['A1'] = title
ws[f'{cols[0]}1'] = col_title
ws.merge_cells(f'{cols[0]}1:{cols[2]}1')

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
