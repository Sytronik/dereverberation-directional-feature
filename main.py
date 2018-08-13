import pdb  # noqa: F401

import numpy as np
import scipy.io as scio

import os
from glob import glob
import sys

from pre_processing import PreProcessor as Pre
from pre_processing import SFTData
# import show_IV_image as showIV
from train_nn import NNTrainer

if __name__ == '__main__':
    DIR_DATA = '../../De-Reverberation Data'
    DIR_WAVFILE = DIR_DATA + '/speech/data/lisa/data/timit/raw/TIMIT/'
    DIR_IV_dict = {'TRAIN': DIR_DATA + '/IV/TRAIN',
                   'TEST': DIR_DATA + '/IV/TEST'}
    FORM = '%04d_%02d.npy'
    ID = '*.WAV'  # The common name of wave file

    # main needs the arguments
    # python main.py FUNCTION [TRAIN(default) | TEST] [ADDITIONAL_ARG]
    if len(sys.argv) == 1:
        print('Arguments are needed')
        exit()

    if sys.argv[1] == 'pre_processing':
        # the second argument is 'TRAIN' or 'TEST'
        if len(sys.argv) >= 3:
            KIND_DATA = sys.argv[2].upper()
        else:
            KIND_DATA = 'TRAIN'
        DIR_IV = DIR_IV_dict[KIND_DATA]
        DIR_WAVFILE += KIND_DATA

        # RIR Data
        transfer_dict = scio.loadmat(os.path.join(DIR_DATA, 'transfer.mat'),
                           variable_names=('RIR', 'Ys'))
        RIR = transfer_dict['RIR'].transpose((2, 0, 1))
        Ys_original = transfer_dict['Ys']
        Ys = np.zeros((Ys_original.size, Ys_original[0].size), dtype=complex)
        for ii in range(Ys_original.size):
            Ys[ii] = Ys_original[ii]

        # SFT Data
        sft_dict = scio.loadmat(os.path.join(DIR_DATA, 'sft_data.mat'),
                               variable_names=('bEQspec', 'Yenc',
                                               'Wnv', 'Wpv', 'Vv'),
                               squeeze_me=True)

        bEQspec = sft_dict['bEQspec'].T
        Yenc = sft_dict['Yenc'].T
        Wnv = sft_dict['Wnv'].astype(complex)
        Wpv = sft_dict['Wpv'].astype(complex)
        Vv = sft_dict['Vv'].astype(complex)

        sftdata = SFTData(bEQspec, Yenc, Wnv, Wpv, Vv)

        # The index of the first wave file that have to be processed
        IDX_START = len(glob(
            os.path.join(DIR_IV, f'*_{RIR.shape[0]-1:02d}.npy')
        ))+1

        p = Pre(RIR, Ys, sftdata)
        p.process(DIR_WAVFILE, ID, IDX_START, DIR_IV, FORM)

    else:  # the functions that need metadata
        metadata = np.load(os.path.join(DIR_IV_dict['TRAIN'],
                                        'metadata.npy')).item()
        # if sys.argv[1] == 'show_IV_image':
        #     doSave = False
        #     FNAME = FORM % (1, 0)  # The default file is 0001_00.npy
        #     DIR_IV = ''
        #     for arg in sys.argv[2:]:
        #         if arg == '--save' or arg == '-S':
        #             doSave = True
        #         elif arg.upper() == 'TRAIN' or arg.upper() == 'TEST':
        #             KIND_DATA = arg.upper()
        #             DIR_IV = DIR_IV_dict[KIND_DATA]
        #         else:
        #             FNAME = arg
        #
        #     if not FNAME.endswith('.npy'):
        #         FNAME += '.npy'
        #
        #     IV_dict = np.load(os.path.join(DIR_IV, FNAME)).item()
        #
        #     IVnames = [key for key in IV_dict if key.startswith('IV')]
        #     title = ['{} ({})'.format(FNAME.replace('.npy',''),
        #                               name.split('_')[-1],
        #                               )
        #              for name in IVnames]
        #     IVs = [IV_dict[k] for k in IVnames]
        #     showIV.show(IVs,
        #                 title=title,
        #                 ylim=[0., metadata['Fs']/2],
        #                 doSave=doSave,
        #                 # norm_factor=(IV_dict['norm_factor_free'],
        #                 #              IV_dict['norm_factor_room']),
        #                 )

        if sys.argv[1] == 'train_nn':
            trainer = NNTrainer(DIR_IV_dict['TRAIN'], DIR_IV_dict['TEST'],
                                'IV_room', 'IV_free',
                                metadata['N_freq'],
                                metadata['L_frame'],
                                metadata['L_hop'])
            trainer.train()
        elif sys.argv[1] == 'test_nn':
            trainer = NNTrainer(DIR_IV_dict['TRAIN'], DIR_IV_dict['TEST'],
                                'IV_room', 'IV_free',
                                metadata['N_freq'],
                                metadata['L_frame'],
                                metadata['L_hop'],
                                F_MODEL_STATE='MLP_26.pt'
                                )

            loss_test, snr_test_dB = trainer.eval(
                FNAME='MLP_result_26_test.mat'
            )
            print(f'\nTest Loss: {loss_test:.2e}', end='\t')
            print(f'Test SNR (dB): {snr_test_dB:.2e}')
