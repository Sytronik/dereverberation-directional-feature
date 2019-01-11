PATH_WAV = './data/speech/data/lisa/data/timit/raw/TIMIT'
PATH_IV = './data/IV'

DICT_PATH = dict(
    root='./data',
    sft_data='./data/sft_data.mat',
    wav_train=f'{PATH_WAV}/TRAIN',
    wav_test=f'{PATH_WAV}/TEST',
    iv_train=f'{PATH_IV}/TRAIN',
    iv_test=f'{PATH_IV}/TEST',
    UNet=f'./result/UNet'
)