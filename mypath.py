def path(to):
    if to == 'root':
        return './Data'
    elif to == 'wav_train':
        return './Data/speech/data/lisa/data/timit/raw/TIMIT/TRAIN'
    elif to == 'wav_test':
        return './Data/speech/data/lisa/data/timit/raw/TIMIT/TEST'
    elif to == 'iv_train':
        return './Data/IV/TRAIN'
    elif to == 'iv_test':
        return './Data/IV/TEST'

    else:
        print('Path to {} is not available.'.format(to))
        raise NotImplementedError
