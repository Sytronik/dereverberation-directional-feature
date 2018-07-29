import pdb  # noqa: F401

import numpy as np
import scipy.io as scio

# import matplotlib.pyplot as plt

import os
from glob import glob
import sys

from pre_processing import PreProcessor as Pre
import show_IV_image as showIV
from train_nn import NNTrainer

if __name__ == '__main__':
    DIR_DATA = '../../De-Reverberation Data'
    DIR_WAVFILE = DIR_DATA + '/speech/data/lisa/data/timit/raw/TIMIT/'
    DIR_IV_DIC = {'TRAIN': DIR_DATA + '/IV/TRAIN',
                  'TEST': DIR_DATA + '/IV/TEST'}
    FORM = '%04d_%02d.npy'
    ID = '*.WAV'  # The common name of wave file

    # main needs the arguments
    # python main.py FUNCTION [TRAIN(default) | TEST] [ADDITIONAL_ARG]
    if len(sys.argv) == 1:
        print('Arguments are needed')
        exit()

    # the second argument is 'TRAIN' or 'TEST'
    if len(sys.argv) >= 3:
        KIND_DATA = sys.argv[2].upper()
    else:
        KIND_DATA = 'TRAIN'
    DIR_IV = DIR_IV_DIC[KIND_DATA]

    if sys.argv[1] == 'pre_processing':
        DIR_WAVFILE += KIND_DATA

        # RIR Data
        RIR = scio.loadmat(os.path.join(DIR_DATA, 'RIR.mat'),
                           variable_names='RIR')['RIR']
        RIR = RIR.transpose((2, 0, 1))  # 72 x 32 x 48k

        # SFT Data
        sph_mat = scio.loadmat(os.path.join(DIR_DATA, 'sph_data.mat'),
                               variable_names=['bEQspec', 'Yenc', 'Ys',
                                               'Wnv', 'Wpv', 'Vv'],
                               squeeze_me=True)

        bEQspec = sph_mat['bEQspec'].T
        Yenc = sph_mat['Yenc'].T

        Ys_original = sph_mat['Ys']
        Ys = np.zeros((Ys_original.size, Ys_original[0].size), dtype=complex)
        for ii in range(Ys_original.size):
            Ys[ii] = Ys_original[ii]

        Wnv = sph_mat['Wnv'].astype(complex)
        Wpv = sph_mat['Wpv'].astype(complex)
        Vv = sph_mat['Vv'].astype(complex)

        # The index of the first wave file that have to be processed
        IDX_START = len(glob(
            os.path.join(DIR_IV, '*_%02d.npy' % (RIR.shape[0]-1))
        ))+1

        p = Pre(RIR, bEQspec, Yenc, Ys, Wnv, Wpv, Vv)
        p.process(DIR_WAVFILE, ID, IDX_START, DIR_IV, FORM)

    else:  # the functions that need metadata
        metadata = np.load(os.path.join(DIR_IV, 'metadata.npy')).item()

        if sys.argv[1] == 'show_IV_image':
            # The default file to be shown is 0001_00.npy
            FNAME = FORM % (1, 0)
            if len(sys.argv) >= 4:
                FNAME = sys.argv[3]
                if not FNAME.endswith('.npy'):
                    FNAME += '.npy'

            needToSave = False
            if len(sys.argv) >= 5:
                if sys.argv[4] == '--save' or sys.argv[4] == '-S':
                    needToSave = True

            data_dic = np.load(os.path.join(DIR_IV, FNAME)).item()

            showIV.show(data_dic['IV_free'], data_dic['IV_room'],
                        title=[FNAME+' (free)', FNAME+' (room)'],
                        norm_factor=(data_dic['norm_factor_free'],
                                     data_dic['norm_factor_room']),
                        ylim=[0, metadata['Fs']/2],
                        needToSave=needToSave)

        elif sys.argv[1] == 'train_nn':
            trainer = NNTrainer(DIR_IV_DIC['TRAIN'], DIR_IV_DIC['TEST'],
                                'IV_room', 'IV_free',
                                metadata['N_fft'],
                                metadata['L_frame'],
                                metadata['L_hop'])
            trainer.train()
        elif sys.argv[1] == 'test_nn':
            trainer = NNTrainer(DIR_IV_DIC['TRAIN'], DIR_IV_DIC['TEST'],
                                'IV_room', 'IV_free',
                                metadata['N_fft'],
                                metadata['L_frame'],
                                metadata['L_hop'],
                                # DIR_DATA+'/MLP/MLP_epoch_9.pth'
                                )
            trainer.eval(FNAME=DIR_DATA + '/MLP/MLP_result.npy')
