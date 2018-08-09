import pdb  # noqa: F401

import torch
import scipy.io as scio
import sys

file = sys.argv[1]
if not file.endswith('.pt'):
    file += '.pt'

a = torch.load(file, map_location=torch.device('cpu'))

a_np = {}

for key, value in a.items():
    key = key.replace('.', '_')
    a_np[key] = value.numpy()
    print('{}\t\t: ndarray of shape {}'.format(key, a_np[key].shape))

scio.savemat(file.replace('.pt', '.mat'), a_np)
