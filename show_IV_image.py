import pdb

import numpy as np
import matplotlib.pyplot as plt

import os

from joblib import Parallel, delayed
import multiprocessing

def norm_by_std(a, n, lower=0, upper=1):
    # mean + n*std -> upper
    # mean - n*std -> lower
    return (a-np.mean(a))/(n*np.std(a)) * (upper-lower)/2 + (lower+upper)/2


def norm_by_minmax(a, upper=1.):
    return upper*(a-a.min())/(a.max()-a.min())


def hist_eq(a):
    dtype = a.dtype
    if dtype==float:
        a = (255*a[:]).astype('uint8')
    else:
        a = a[:]

    hist,bins = np.histogram(a.flatten(),256,[0,256])
    cdf = hist.cumsum()
    cdf_m = np.ma.masked_equal(cdf,0)
    cdf_m = norm_by_minmax(cdf_m, upper=255)
    cdf = np.ma.filled(cdf_m,0).astype('uint8')

    if dtype==float:
        a=cdf[a]/255.
    else:
        a=cdf[a]

    return a,cdf


def show(*args, title=[], norm_factor=[], **kargs):
    # Load IVs
    IVs=[]
    for arg in args:
        IVs.append(arg)
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

    plt.figure(frameon=False, figsize=(9,9))
    for i in range(N_FIG):
        try:
            title_i=title[i]
            print(title[i])
        except:
            title_i=''
            print('no title')

        #RGB Channel
        #Normalize
        try:
            IVs[i][:,:,:3] /= (2*norm_factor[i])
            IVs[i][:,:,:3] += 0.5
            IVs[i][:,:,:3] = IVs[i][:,:,:3].clip(0, 1)
        except:
            IVs[i][:,:,:3] = norm_by_minmax(IVs[i][:,:,:3])

        #Correction
        IVs[i][:,:,:3] = 1./(1.+np.exp(-30*(IVs[i][:,:,:3]-0.5)))
        # Histogram Equalization
        # IV_gray = np.sqrt(np.sum(IVs[i][:,:,:3]**2, axis=2))
        # IV_gray /= IV_gray.max()
        # _,cdf = hist_eq(IV_gray)
        # for ch in range(3):
        #     IVs[i][:,:,ch] = cdf[(255*IVs[i][:,:,ch]).astype('uint8')]/255.

        #Alpha Channel
        #Normalize
        IVs[i][:,:,3] = norm_by_minmax(IVs[i][:,:,3])

        #Correction
        # IVs[i][:,:,3] = 1
        # IVs[i][:,:,3] = IVs[i][:,:,3]**(1/5)

        # IVs[i][:,:,3] = IVs[i][:,:,3]/2. + 0.5
        # IVs[i][:,:,3] = 1/(1+np.exp(-10*(IVs[i][:,:,3]-0.5)))
        # IVs[i][:,:,3] = (IVs[i][:,:,3]-0.5)*2

        IVs[i][:,:,3],_ = hist_eq(IVs[i][:,:,3])

        # flip frequency axis
        IVs[i][:,:,:] = IVs[i][::-1,:,:]

        extent[1] = L[i]

        plt.subplot(N_FIG,1,i+1)
        plt.imshow(chess, cmap=plt.cm.gray, interpolation='nearest',
                   vmin=0, vmax=1, extent=axis, aspect='auto')
        plt.imshow(IVs[i], interpolation='nearest',
                   vmin=0, vmax=1, extent=extent[:], aspect='auto')
        plt.axis(axis)
        plt.title(title_i)
        plt.xlabel('Frame Index')
        plt.ylabel('Frequency (Hz)')
    plt.tight_layout()
    plt.show()
    # plt.savefig(title[0].split()[0]+'.png', dip=300)
