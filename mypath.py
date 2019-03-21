_PATH_WAV = './data/speech/data/lisa/data/timit/raw/TIMIT'
_PATH_DIRSPEC = './backup/DirAC_room1_fix'
# _PATH_DIRSPEC = './backup/IV_room1'
PATH_RESULT = './result'
PATH_FIG = './figures'

F_NORMCONST = 'normconst_{}_{}.h5'

DICT_PATH = dict(
    root='./backup',
    sft_data='./backup/sft_data_32ms.mat',
    RIR_Ys='./backup/RIR_Ys_TRAIN20_TEST20_[10.52 7.1 3.02].mat',

    wav_train=f'{_PATH_WAV}/TRAIN',
    wav_test=f'{_PATH_WAV}/TEST',

    dirspec_train=f'{_PATH_DIRSPEC}/TRAIN',
    dirspec_seen=f'{_PATH_DIRSPEC}/TEST/SEEN',
    dirspec_unseen=f'{_PATH_DIRSPEC}/TEST/UNSEEN',

    normconst_train=f'{_PATH_DIRSPEC}/TRAIN/{F_NORMCONST}',
    normconst_seen=f'{_PATH_DIRSPEC}/TEST/{F_NORMCONST}',
    normconst_unseen=f'{_PATH_DIRSPEC}/TEST/{F_NORMCONST}',

    UNet=f'{PATH_RESULT}/UNet DirAC+p00'
    # UNet=f'{PATH_RESULT}/UNet IV+p00'
    # UNet=f'{PATH_RESULT}/UNet p00'
)
