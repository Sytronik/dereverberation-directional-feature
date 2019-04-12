from pathlib import Path

DF = 'IV'
# DF = 'DirAC'

ROOM_TRAINED = 'room2'
ROOM = 'room1'

# ROOM_TRAINED = 'room1'
# ROOM = 'room1'

PATH_DATAROOT = Path('./backup')
_PATH_WAV = Path('./data/speech/data/lisa/data/timit/raw/TIMIT')
# _PATH_WAV = Path('./data')
_PATH_DIRSPEC = Path(f'./backup/{DF}_{ROOM}')
_PATH_TRAINED = Path(f'./backup/{DF}_{ROOM_TRAINED}') if ROOM_TRAINED else _PATH_DIRSPEC
# _PATH_DIRSPEC = Path(f'./backup/{DF}_impulse_{ROOM}')
PATH_RESULT = Path('./result')

_F_NORMCONST = 'normconst_{}_{}.h5'

DICT_PATH = dict(
    sft_data=PATH_DATAROOT / 'sft_data_32ms.mat',
    RIR_Ys=PATH_DATAROOT / f'RIR_Ys_TRAIN20_TEST20_{ROOM}.mat',

    wav_train=_PATH_WAV / 'TRAIN',
    wav_test=_PATH_WAV / 'TEST',

    # dirspec_train=_PATH_DIRSPEC / 'TRAIN',
    dirspec_train=_PATH_TRAINED / 'TRAIN',
    dirspec_seen=_PATH_DIRSPEC / 'TEST/SEEN',
    dirspec_unseen=_PATH_DIRSPEC / 'TEST/UNSEEN',

    s_normconst_train=str(_PATH_TRAINED / 'TRAIN' / _F_NORMCONST),
    s_normconst_seen=str(_PATH_DIRSPEC / 'TEST' / _F_NORMCONST),
    s_normconst_unseen=str(_PATH_DIRSPEC / 'TEST' / _F_NORMCONST),

    UNet=PATH_RESULT / f'UNet ({DF}+p00 {ROOM_TRAINED if ROOM_TRAINED else ROOM})',
    # UNet=PATH_RESULT / f'UNet (p00 {ROOM_TRAINED if ROOM_TRAINED else ROOM})',
    # UNet=PATH_RESULT / f'UNet-CBAM (p00 {ROOM_TRAINED if ROOM_TRAINED else ROOM})',
    UNetNALU=PATH_RESULT / f'UNetNALU 2019-04-03 (p00 {ROOM_TRAINED if ROOM_TRAINED else ROOM})',

    figures=Path('./figures'),
)
