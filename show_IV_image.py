import pdb  # noqa: F401

import numpy as np
import matplotlib.pyplot as plt


# Not used
def norm_by_std(a, n, lower=0., upper=1.):
    # mean + n*std -> upper
    # mean - n*std -> lower
    return (a-np.mean(a))/(n*np.std(a)) * (upper-lower)/2 + (lower+upper)/2


def norm_by_minmax(a, upper=1., useAbs=False):
    if useAbs:
        max = np.abs(a).max()
        min = -max
    else:
        max = a.max()
        min = a.min()
    return upper*(a-min)/(max-min)


# Histogram Equalization
def hist_eq(a):
    dtype = a.dtype
    if dtype == np.float16 or dtype == np.float32 or dtype == np.float64:
        a = (255*a[:]).astype('uint8')
    else:
        a = a[:]

    hist, bins = np.histogram(a.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = norm_by_minmax(cdf_m, upper=255)
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')

    if dtype == np.float16 or dtype == np.float32 or dtype == np.float64:
        a = cdf[a]/255.
    else:
        a = cdf[a]

    return a, cdf


def show(IVs, title=[], norm_factor=[], **kargs):
    N_FIG = len(IVs)
    length = np.zeros(N_FIG)
    for i in range(N_FIG):
        length[i] = IVs[i].shape[1]

    scaling_separately = [False]*N_FIG
    for i in range(N_FIG):
        if 'room' in title[i]:
            scaling_separately[i] = True

    # Initialize
    H_CHESS = 100  # Number of cells of a column in the chess board
    axis = [1, np.max(length), 1, IVs[0].shape[0]]
    doSave = False
    for key, value in kargs.items():
        if key == 'axis':
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
        elif key == 'doSave':
            doSave = value
        elif key == 'H_CHESS':
            H_CHESS = value
    chess = (np.add.outer(range(H_CHESS), range(H_CHESS*N_FIG)) % 2)*0.3+0.7
    extent = axis[:]

    # =========================Image Processing=========================
    max_XYZ = float('-inf')
    max_a00 = float('-inf')
    min_a00 = float('inf')
    for i in range(N_FIG):
        if scaling_separately[i]:
            continue
        max_XYZ = np.abs(IVs[i][:,:,:3]).max() \
            if np.abs(IVs[i][:,:,:3]).max() > max_XYZ \
            else max_XYZ
        max_a00 = IVs[i][:,:,3].max() \
            if IVs[i][:,:,3].max() > max_a00 \
            else max_a00
        min_a00 = IVs[i][:,:,3].min() \
            if IVs[i][:,:,3].min() < min_a00 \
            else min_a00

    min_XYZ = -max_XYZ

    for i in range(N_FIG):
        # ======================XYZ Channel======================
        # Normalize
        try:
            IVs[i][:,:,:3] /= (2*norm_factor[i])
            IVs[i][:,:,:3] += 0.5
            IVs[i][:,:,:3] = IVs[i][:,:,:3].clip(0, 1)
        except (IndexError, TypeError):
            if scaling_separately[i]:
                IVs[i][:,:,:3] = norm_by_minmax(IVs[i][:,:,:3], useAbs=True)
            else:
                IVs[i][:,:,:3] = (IVs[i][:,:,:3] - min_XYZ)/(max_XYZ - min_XYZ)

        # Contrast Enhancement
        IVs[i][:,:,:3] = 1./(1.+np.exp(-50*(IVs[i][:,:,:3]-0.5)))
        # -Histogram Equalization
        # IV_gray = np.sqrt(np.sum(IVs[i][:,:,:3]**2, axis=2))
        # IV_gray /= IV_gray.max()
        # _,cdf = hist_eq(IV_gray)
        # for ch in range(3):
        #     IVs[i][:,:,ch] = cdf[(255*IVs[i][:,:,ch]).astype('uint8')]/255.

        # ======================a00 Channel======================
        # Normalize
        # if scaling_separately[i]:
        #     IVs[i][:,:,3] = norm_by_minmax(IVs[i][:,:,3])
        # else:
        #     IVs[i][:,:,3] = (IVs[i][:,:,3] - min_a00)/(max_a00 - min_a00)

        # Contrast Enhancement
        if scaling_separately[i]:
            eps = -IVs[i][:,:,3].min() + 1e-3
            IVs[i][:,:,3] = np.log10(IVs[i][:,:,3]+eps)
            IVs[i][:,:,3] = norm_by_minmax(IVs[i][:,:,3])
        else:
            eps = -min_a00 + 1e-3
            IVs[i][:,:,3] = np.log10(IVs[i][:,:,3]+eps)
            IVs[i][:,:,3] = (IVs[i][:,:,3] - np.log10(eps)) \
                / (np.log10(max_a00+eps) - np.log10(eps))
        # IVs[i][:,:,3], _ = hist_eq(IVs[i][:,:,3])
        # IVs[i][:,:,3] = IVs[i][:,:,3]**(1/5)

        # flip frequency axis
        IVs[i][:,:,:] = IVs[i][::-1,:,:]

    # =========================Show Figure=========================
    plt.figure(frameon=False, figsize=(9, 9))
    for i in range(N_FIG):
        try:
            title_i = title[i]
            print(title[i])
        except (IndexError, TypeError):
            title_i = ''
            print('no title')

        extent[1] = length[i]

        # for j in range(2):
        #     IV = IVs[i][:,:,:3] if j == 0 else IVs[i][:,:,3]
        #     plt.subplot(N_FIG, 2, 2*i+1+j)
        IV = IVs[i]
        plt.subplot(N_FIG, 1, i+1)
        plt.imshow(chess, cmap=plt.cm.gray, interpolation='nearest',
                   vmin=0, vmax=1, extent=axis, aspect='auto')
        plt.imshow(IV, interpolation='nearest',
                   vmin=0, vmax=1, extent=extent[:], aspect='auto')
        # plt.colorbar()
        plt.axis(axis)
        plt.title(title_i)
        plt.xlabel('Frame Index')
        plt.ylabel('Frequency (Hz)')
    plt.tight_layout()
    if doSave:
        plt.savefig(title[0].split()[0]+'.png', dpi=300)
    plt.show()
