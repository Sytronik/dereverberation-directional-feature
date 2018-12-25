PATH_WAV = './data/speech/data/lisa/data/timit/raw/TIMIT'
PATH_IV = './data/IV'

DICT_PATH = dict(
    root='./data',
    wav_train=f'{PATH_WAV}/TRAIN',
    wav_test=f'{PATH_WAV}/TEST',
    iv_train=f'{PATH_IV}/TRAIN',
    iv_test=f'{PATH_IV}/TEST',
    UNet=f'./result/UNet'
)


def path(to: str) -> str:
    _path = DICT_PATH.get(to)
    if _path:
        return _path
    else:
        raise ValueError(f'Path to {to} is not available.')
