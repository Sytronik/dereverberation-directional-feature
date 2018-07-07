import pdb
import sys

import numpy as np
import cupy as cp
import scipy as sc
import scipy.io as scio

import matplotlib.pyplot as plt

import os
from glob import glob

import pre_processing as pre
import show_IV_image as showIV

if __name__ == '__main__':
    DIR_WAVFILE = './speech/data/lisa/data/timit/raw/TIMIT/TRAIN/'
    DIR_IV = './IV/TRAIN/'
    FORM = '%04d_%02d'
    FORM_FREE = FORM+'_free.npy'
    FORM_ROOM = FORM+'_room.npy'
    ID = '*_converted.wav'

    for arg in sys.argv[1:]:
        if arg == 'pre_processing':
            #RIR Data
            RIR = scio.loadmat('./1_MATLABCode/RIR.mat',
                                variable_names = 'RIR')['RIR']
            RIR = RIR.transpose((2, 0, 1)) #72 x 32 x 48k
            RIR = cp.array(RIR)

            #SFT Data
            sph_mat = scio.loadmat('./1_MATLABCode/sph_data.mat',
                        variable_names=['bEQspec','Yenc','Ys','Wnv','Wpv','Vv'])
            bEQspec = cp.array(sph_mat['bEQspec']).T
            Yenc = cp.array(sph_mat['Yenc']).T

            Ys_original = sph_mat['Ys'].reshape(-1)
            Ys_np = np.zeros((Ys_original.size,Ys_original[0].size),
                                                                dtype=complex)
            for ii in range(Ys_original.size):
                Ys_np[ii] = Ys_original[ii].reshape(-1)
            Ys = cp.array(Ys_np)
            Ys_original = None

            Wnv = cp.array(sph_mat['Wnv'], dtype=complex).reshape(-1)
            Wpv = cp.array(sph_mat['Wpv'], dtype=complex).reshape(-1)
            Vv = cp.array(sph_mat['Vv'], dtype=complex).reshape(-1)

            sph_mat = None

            N_START = len(glob(os.path.join(DIR_IV,
                                            '*_%02d_room.npy'%(RIR.shape[0]-1)))
                          )+1

            pre.process(DIR_WAVFILE, ID, N_START,
                        DIR_IV, FORM_FREE, FORM_ROOM,
                        RIR, bEQspec, Yenc, Ys, Wnv, Wpv, Vv)

        elif arg.startswith('show_IV_image') or arg.startswith('histogram'):
            Metadata = np.load('Metadata.npy').item()

            Fs = Metadata['Fs']
            # N_WAVFILE = Metadata['N_WAVFILE']
            # N_FFT = Metadata['N_FFT']
            # L_FRAME = Metadata['L_FRAME']
            # L_HOP = Metadata['L_HOP']
            # N_LOC = Metadata['N_LOC']

            Metadata = None

            IDX_WAV = 1
            IDX_LOC = 0
            arg_sp = arg.split()
            if len(arg_sp)>1:
                IDX_WAV = int(arg_sp[1])
                if len(arg_sp)>2:
                    IDX_LOC = int(arg_sp[2])

            FNAME_FREE = os.path.join(DIR_IV, FORM_FREE%(IDX_WAV,IDX_LOC))
            FNAME_ROOM = os.path.join(DIR_IV, FORM_ROOM%(IDX_WAV,IDX_LOC))

            if arg.startswith('show_IV_image'):
                showIV.show(FNAME_FREE, FNAME_ROOM, ylim=[0, Fs/2])

            elif arg.startswith('histogram'):
                IV_free = np.load(FNAME_FREE)
                IV_room = np.load(FNAME_ROOM)
                bins = 200

                plt.figure()
                plt.subplot(2,2,1)
                plt.hist(IV_free[:,:,:3].reshape(-1), bins=bins)
                plt.xlim(IV_free[:,:,:3].min(), IV_free[:,:,:3].max())
                plt.title('Histogram for RGB (Free-space)')
                plt.subplot(2,2,2)
                plt.hist(IV_free[:,:,3].reshape(-1), bins=bins)
                plt.xlim(IV_free[:,:,3].min(), IV_free[:,:,3].max())
                plt.title('Histogram for alpha (Free-space)')
                plt.subplot(2,2,3)
                plt.hist(IV_room[:,:,:3].reshape(-1), bins=bins)
                plt.xlim(IV_room[:,:,:3].min(), IV_room[:,:,:3].max())
                plt.title('Histogram for RGB (Room)')
                plt.subplot(2,2,4)
                plt.hist(IV_room[:,:,3].reshape(-1), bins=bins)
                plt.xlim(IV_room[:,:,3].min(), IV_room[:,:,3].max())
                plt.title('Histogram for alpha (Room))')
                plt.show()
