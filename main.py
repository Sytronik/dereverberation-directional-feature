"""
<USAGE>
python main.py FUNCTION [ARGS]

<FUNCTION>
- pre_processing
    ARGS: [TRAIN(default) | TEST]
- train_nn
- test_nn
    ARGS: [n] (The parameters of the n-th epoch will be used.)
"""

import pdb  # noqa: F401

# import numpy as np
import scipy.io as scio

from os import path
from glob import glob
import sys

from pre_processing import PreProcessor as Pre, SFTData
from pre_processing_anm_check import PreProcessor as AnmCheck
# import show_IV_image as showIV

import mypath

if __name__ == '__main__':
    DIR_DATA = mypath.path('root')
    FORM = '%04d_%02d.h5'
    ID = '*.WAV'  # The common name of wave file

    if len(sys.argv) == 1:
        print('Arguments are needed')
        exit()

    if sys.argv[1] == 'pre_processing' or sys.argv[1] == 'anm_check':
        # the second argument is 'TRAIN' or 'TEST'
        if len(sys.argv) >= 3:
            KIND_DATA = sys.argv[2].upper()
        else:
            KIND_DATA = 'TRAIN'
        DIR_IV = mypath.path(f'iv_{KIND_DATA.lower()}')
        DIR_WAVFILE = mypath.path(f'wav_{KIND_DATA.lower()}')

        # RIR Data
        transfer_dict = scio.loadmat(path.join(DIR_DATA, 'RIR_Ys.mat'),
                                     squeeze_me=True)
        RIRs = transfer_dict['RIR_'+KIND_DATA].transpose((2, 0, 1))
        Ys = transfer_dict['Ys_'+KIND_DATA].T

        RIRs_0 = scio.loadmat(path.join(DIR_DATA, 'RIR_0_order.mat'),
                              variable_names='RIR_'+KIND_DATA)
        RIRs_0 = RIRs_0['RIR_'+KIND_DATA].transpose((2, 0, 1))

        # SFT Data
        sft_dict = scio.loadmat(path.join(DIR_DATA, 'sft_data.mat'),
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
        idx_start \
            = len(glob(path.join(DIR_IV, f'*_{RIRs.shape[0]-1:02d}.h5')))+1

        if sys.argv[1] == 'pre_processing':
            # p = Pre(RIRs, Ys, sftdata, RIRs_0=RIRs_0)
            p = Pre(RIRs, Ys, sftdata)
            p.process(DIR_WAVFILE, ID, idx_start, DIR_IV, FORM)
        else:
            p = AnmCheck(RIRs, Ys, sftdata, RIRs_0=RIRs_0)
            p.process(DIR_WAVFILE, ID, 1, DIR_IV, '%04d_%02d_anm_check.h5')

        # elif sys.argv[1] == 'show_IV_image':
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
        #     IV_dict = np.load(path.join(DIR_IV, FNAME)).item()
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
