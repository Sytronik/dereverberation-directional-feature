_PATH_WAV = './data/speech/data/lisa/data/timit/raw/TIMIT'
_PATH_IV = './data/IV'
F_NORMCONST = 'normconst_{}_log_meanstd.h5'

DICT_PATH = dict(
    root='./data',
    sft_data='./data/sft_data.mat',

    wav_train=f'{_PATH_WAV}/TRAIN',
    wav_test=f'{_PATH_WAV}/TEST',

    iv_train=f'{_PATH_IV}/TRAIN',
    iv_seen=f'{_PATH_IV}/TEST/SEEN',
    iv_unseen=f'{_PATH_IV}/TEST/UNSEEN',

    normconst_train=f'{_PATH_IV}/TRAIN/{F_NORMCONST}',
    normconst_seen=f'{_PATH_IV}/TEST/{F_NORMCONST}',
    normconst_unseen=f'{_PATH_IV}/TEST/{F_NORMCONST}',

    UNet=f'./result/UNet'
)
