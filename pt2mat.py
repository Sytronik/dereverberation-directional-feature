import pdb  # noqa: F401

import torch
import scipy.io as scio
import sys

file = sys.argv[1]
if not file.endswith('.pt'):
    file += '.pt'

state_dict = torch.load(file, map_location=torch.device('cpu'))

dict_np = {key.replace('.', '_'):value.numpy()
           for key, value in state_dict.items()}

length = max([len(k) for k in dict_np.keys()])
for key, value in dict_np:
    print(f'{key:<{length}}: ndarray of shape {value.shape}')

scio.savemat(file.replace('.pt', '.mat'), dict_np)
