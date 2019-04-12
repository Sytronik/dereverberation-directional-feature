# noinspection PyUnresolvedReferences
import matlab.engine

import os
import shutil
from argparse import ArgumentError, ArgumentParser

import deepdish as dd
# noinspection PyCompatibility
from dataclasses import asdict
from torch.utils.data import DataLoader

import config as cfg
from dirspecgram import DirSpecDataset
from train import Trainer

parser = ArgumentParser()

parser.add_argument(
    'model_name', type=str,
    metavar='MODEL'
)
parser.add_argument(
    '--train', action='store_true',
)
parser.add_argument(
    '--test', type=str, choices=('valid', 'seen', 'unseen'),
    metavar='DATASET'
)
parser.add_argument(
    '--from', type=int, default=-1,
    dest='epoch', metavar='EPOCH',
)
parser.add_argument(
    '--disk-worker', '-dw', type=int, nargs='?', const=1, default=3,
    dest='num_workers',
)  # number of subprocesses for dataloaders
ARGS = parser.parse_args()
del parser
if ARGS.train and ARGS.test or ARGS.epoch < -1:
    raise ArgumentError

model_name = ARGS.model_name

# directory
DIR_TRAIN = cfg.DICT_PATH[model_name] / 'train'
if DIR_TRAIN.exists():
    if ARGS.train and list(DIR_TRAIN.glob('events.out.tfevents.*')):
            print(f'The folder "{DIR_TRAIN}" already has tfevent files. Continue? [y/n]')
            ans = input()
            if ans.lower().startswith('y'):
                shutil.rmtree(DIR_TRAIN)
            else:
                exit()
else:
    os.makedirs(DIR_TRAIN)
if ARGS.test:
    DIR_TEST = cfg.DICT_PATH[model_name]
    if cfg.ROOM == cfg.ROOM_TRAINED:
        DIR_TEST /= ARGS.test
    else:
        DIR_TEST /= f'{ARGS.test}_{cfg.ROOM}'
    if DIR_TEST.exists():
        if list(DIR_TEST.glob('events.out.tfevents.*')):
            print(f'The folder "{DIR_TEST}" already has tfevent files. Continue? [y/n]')
            ans = input()
            if ans.lower().startswith('y'):
                shutil.rmtree(DIR_TEST)
            else:
                exit()
    else:
        os.makedirs(DIR_TEST)
else:
    DIR_TEST = ''

# epoch, state dict
FIRST_EPOCH = ARGS.epoch + 1
if FIRST_EPOCH > 0:
    F_STATE_DICT = DIR_TRAIN / f'{model_name}_{ARGS.epoch}.pt'
else:
    F_STATE_DICT = None

if F_STATE_DICT and not F_STATE_DICT.exists():
    raise FileNotFoundError(F_STATE_DICT)

# Training + Validation Set
dataset_temp = DirSpecDataset('train',
                              n_file=cfg.hp.n_file,
                              keys_trannorm=cfg.KEYS_TRANNORM,
                              )
dataset_train, dataset_valid \
    = DirSpecDataset.split(dataset_temp, (0.7, -1))
dataset_train.set_needs(**cfg.hp.CHANNELS)
dataset_valid.set_needs(**cfg.CH_WITH_PHASE)

loader_valid = DataLoader(dataset_valid,
                          batch_size=cfg.hp.batch_size if ARGS.train else 1,
                          shuffle=False,
                          num_workers=ARGS.num_workers,
                          collate_fn=dataset_valid.pad_collate,
                          pin_memory=True,
                          )

# run
if ARGS.train:
    loader_train = DataLoader(dataset_train,
                              batch_size=cfg.hp.batch_size,
                              shuffle=True,
                              num_workers=ARGS.num_workers,
                              collate_fn=dataset_train.pad_collate,
                              pin_memory=True,
                              )

    dd.io.save(DIR_TRAIN / cfg.F_HPARAMS, asdict(cfg.hp))
    with (DIR_TRAIN / cfg.F_HPARAMS).with_suffix('.txt').open('w') as f:
        f.write(repr(asdict(cfg.hp)))

    trainer = Trainer.create(model_name)
    trainer.train(loader_train, loader_valid, DIR_TRAIN,
                  FIRST_EPOCH, F_STATE_DICT)
elif ARGS.test:
    if ARGS.test == 'valid':
        loader = loader_valid
    else:
        # Test Set
        dataset_test = DirSpecDataset(ARGS.test,
                                      n_file=cfg.hp.n_file // 4,
                                      random_by_utterance=False,
                                      **cfg.CH_WITH_PHASE,
                                      )
        dataset_test.normalize_on_like(dataset_temp)
        loader = DataLoader(dataset_test,
                            batch_size=1,
                            shuffle=False,
                            num_workers=ARGS.num_workers,
                            collate_fn=dataset_test.pad_collate,
                            )

    trainer = Trainer.create(model_name, use_cuda=True)
    trainer.test(loader, DIR_TEST, F_STATE_DICT)
else:
    raise ArgumentError
