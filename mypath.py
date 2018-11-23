PATH_WAV = './Data/speech/data/lisa/data/timit/raw/TIMIT'
PATH_IV = './Data/IV'

DICT_PATH = {
    'root': './Data',
    'wav_train': f'{PATH_WAV}/TRAIN',
    'wav_test': f'{PATH_WAV}/TEST',
    'iv_train': f'{PATH_IV}/TRAIN',
    'iv_test': f'{PATH_IV}/TEST',

}


def path(to):
    _path = DICT_PATH.get(to)
    if _path:
        return _path
    else:
        print(f'Path to {to} is not available.')
        raise NotImplementedError
