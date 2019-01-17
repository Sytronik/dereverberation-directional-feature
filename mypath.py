_PATH_WAV = './data/speech/data/lisa/data/timit/raw/TIMIT'
_PATH_IV = './data/IV'

DICT_PATH = dict(
    root='./data',
    sft_data='./data/sft_data.mat',
    wav_train=f'{_PATH_WAV}/TRAIN',
    wav_test=f'{_PATH_WAV}/TEST',
    iv_train=f'{_PATH_IV}/TRAIN',
    iv_test=f'{_PATH_IV}/TEST',
    UNet=f'./result/UNet'
)
