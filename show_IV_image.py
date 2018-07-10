import pdb

import numpy as np
import matplotlib.pyplot as plt

import os

from joblib import Parallel, delayed
import multiprocessing

# mean + n*std -> upper
# mean - n*std -> lower
def norm_by_std(a, n, lower=0, upper=1):
    return (a-np.mean(a))/(n*np.std(a)) * (upper-lower)/2 + (lower+upper)/2


def show(*args, **kargs):
    # Load IVs
    IVs=[]
    fname=[]
    for arg in args:
        fname.append(arg)
        IVs.append(np.load(arg))
    N_FIG = len(IVs)

    L=np.zeros(N_FIG)
    for i in range(N_FIG):
        L[i] = IVs[i].shape[1]

    # Initialize
    H_CHESS = 100
    axis = [1, np.max(L), 1, IVs[0].shape[0]]
    for key, value in kargs.items():
        if key == 'H_CHESS':
            H_CHESS = value
        elif key == 'axis':
            axis = value
        elif key == 'xlim':
            axis[0:2] = value
        elif key == 'ylim':
            axis[2:4] = value
        elif key == 'xmin':
            axis[0] = value
        elif key == 'xmax':
            axis[1] = value
        elif key == 'ymin':
            axis[2] = value
        elif key == 'ymax':
            axis[3] = value
    chess = (np.add.outer(range(H_CHESS), range(H_CHESS*2)) % 2)*0.3+0.7
    extent = axis[:]

    plt.figure(frameon=False)
    for i in range(N_FIG):
        #Normalize
        IVs[i][:,:,:3] = norm_by_std(IVs[i][:,:,:3], 2).clip(0, 1)
        IVs[i][:,:,3] = norm_by_std(IVs[i][:,:,3], 2).clip(0, 1)

        min_alpha = IVs[i][:,:,3].min()
        if min_alpha > 0:
            IVs[i][:,:,3] = (IVs[i][:,:,3] - min_alpha)/(1. - min_alpha)

        # flip frequency axis
        IVs[i][:,:,:] = IVs[i][::-1,:,:]

        extent[1] = L[i]

        plt.subplot(N_FIG,1,i+1)
        plt.imshow(chess, cmap=plt.cm.gray, interpolation='nearest',
                   vmin=0, vmax=1, extent=axis, aspect='auto')
        plt.imshow(IVs[i], interpolation='bilinear',
                   extent=extent[:], aspect='auto')
        plt.axis(axis)
        plt.title(os.path.basename(fname[i]))
        plt.xlabel('Frame Index')
        plt.ylabel('Frequency (Hz)')
    plt.show()
