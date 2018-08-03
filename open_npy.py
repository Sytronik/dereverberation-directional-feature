import numpy as np
import scipy.io as scio
import sys

doSave = False
fnames = []
for arg in sys.argv[1:]:
    if arg == '--save' or arg == '-s':
        doSave = True
    else:
        fnames.append(arg)

for fname in fnames:
    contents = np.load(fname)
    if contents.size == 1:
        contents = contents.item()
    print(contents)
    fname_mat = fname.replace('.npy', '')
    if type(contents) == dict:
        scio.savemat(fname_mat, contents, oned_as='column')
    else:
        scio.savemat(fname_mat, {fname_mat:contents}, oned_as='column')
