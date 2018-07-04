import pdb

import numpy as np
import cupy as cp
import scipy as sc
import scipy.io as scio
import librosa
import matplotlib.pyplot as plt

import os
import time
import atexit
from glob import glob

import soundfile as sf

from joblib import Parallel, delayed
import multiprocessing

Metadata = scio.loadmat('Metadata.mat',
                        variable_names = ['fs','Nfft','Lframe','Lhop',
                                            'Nwavfile','Nloc', 'DIR_IV'])
fs = Metadata['fs'].reshape(-1)[0]
Nfft = int(Metadata['Nfft'].reshape(-1)[0])
Lframe = int(Metadata['Lframe'].reshape(-1)[0])
Lhop = int(Metadata['Lhop'].reshape(-1)[0])
Nwavfile = int(Metadata['Nwavfile'].reshape(-1)[0])
Nloc = int(Metadata['Nloc'].reshape(-1)[0])
DIR_IV = Metadata['DIR_IV'].reshape(-1)[0]

IV_free = np.load(os.path.join(DIR_IV, '000001_ 0_free.npy'))
IV_room = np.load(os.path.join(DIR_IV, '000001_ 0_room.npy'))

_,L_free,_ = IV_free.shape
_,L_room,_ = IV_room.shape

plt.figure(frameon=False)

chess = (np.add.outer(range(50), range(50*2)) % 2)*0.3+0.7  # chessboard
axis = [1, L_room, 0, fs*(Nfft/2.-1)/Nfft]

dx, dy = 1, fs/Nfft

x_free = np.arange(axis[0], L_free, dx)
y_free = np.arange(axis[2], axis[3], dy)
X_free, Y_free = np.meshgrid(x_free, y_free)
extent_free = x_free.min(), x_free.max(), y_free.min(), y_free.max()

x_room = np.arange(axis[0], L_room, dx)
y_room = np.arange(axis[2], axis[3], dy)
X_room, Y_room = np.meshgrid(x_room, y_room)
extent_room = x_room.min(), x_room.max(), y_room.min(), y_room.max()

IV_free = (IV_free+IV_free.min())/(IV_free.max()-IV_free.min())
IV_room = (IV_room+IV_room.min())/(IV_room.max()-IV_room.min())

plt.subplot(2,1,1)
im_chess1 = plt.imshow(chess, cmap=plt.cm.gray, interpolation='nearest', vmin=0, vmax=1, extent=axis, aspect='auto')
im_free = plt.imshow(IV_free[::-1,:,:]*10, extent=extent_free, aspect='auto')
plt.axis(axis)

plt.subplot(2,1,2)
im_chess2 = plt.imshow(chess, cmap=plt.cm.gray, interpolation='nearest', vmin=0, vmax=1, extent=axis, aspect='auto')
im_room = plt.imshow(IV_room[::-1,:,:]*10, extent=extent_room, aspect='auto')
plt.axis(axis)

plt.show()
