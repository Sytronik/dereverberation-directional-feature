path_wav = './Data/speech/data/lisa/data/timit/raw/TIMIT'
path_iv = './Data/IV'


def path(to):
    if to == 'root':
        return './Data'
    elif to == 'wav_train':
        return f'{path_wav}/TRAIN'
    elif to == 'wav_test':
        return f'{path_wav}/TEST'
    elif to == 'iv_train':
        return f'{path_iv}/TRAIN'
    elif to == 'iv_test':
        return f'{path_iv}/TEST'

    else:
        print('Path to {} is not available.'.format(to))
        raise NotImplementedError
