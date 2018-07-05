import pdb
import sys

import numpy as np
import cupy as cp
import scipy as sc
import scipy.io as scio

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
                                        '*_%02d_room.npy'%(RIR.shape[0]-1))))+1

            pre.process(DIR_WAVFILE, ID, N_START,
                        DIR_IV, FORM_FREE, FORM_ROOM,
                        RIR, bEQspec, Yenc, Ys, Wnv, Wpv, Vv)

        elif arg == 'show_IV_image':
            Metadata = scio.loadmat('Metadata.mat',
                            variable_names = ['Fs','N_FFT','L_FRAME','L_HOP',
                                                'N_WAVFILE','N_LOC'])
            Fs = Metadata['Fs'].reshape(-1)[0]
            N_WAVFILE = int(Metadata['N_WAVFILE'].reshape(-1)[0])
            N_FFT = int(Metadata['N_FFT'].reshape(-1)[0])
            L_FRAME = int(Metadata['L_FRAME'].reshape(-1)[0])
            L_HOP = int(Metadata['L_HOP'].reshape(-1)[0])
            N_LOC = int(Metadata['N_LOC'].reshape(-1)[0])

            showIV.show(FORM_FREE%(1,0), FORM_ROOM%(1,0), Fs, N_FFT)
