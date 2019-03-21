# noinspection PyUnresolvedReferences
import matlab.engine

import os
from argparse import ArgumentError, ArgumentParser
from os.path import join as pathjoin

import deepdish as dd
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
    '--disk-worker', '-dw', type=int, nargs='?', const=0, default=3,
    dest='num_workers',
)  # number of subprocesses for dataloaders
ARGS = parser.parse_args()
del parser
if ARGS.train and ARGS.test or ARGS.epoch < -1:
    raise ArgumentError

model_name = ARGS.model_name

# directory
DIR_TRAIN = pathjoin(cfg.DICT_PATH[model_name], 'train')
if not os.path.isdir(DIR_TRAIN):
    os.makedirs(DIR_TRAIN)
if ARGS.test:
    DIR_TEST = pathjoin(cfg.DICT_PATH[model_name], ARGS.test)
    if not os.path.isdir(DIR_TEST):
        os.makedirs(DIR_TEST)
else:
    DIR_TEST = ''

# epoch, state dict
FIRST_EPOCH = ARGS.epoch + 1
if FIRST_EPOCH > 0:
    F_STATE_DICT = pathjoin(DIR_TRAIN, f'{model_name}_{ARGS.epoch}.pt')
else:
    F_STATE_DICT = ''

if F_STATE_DICT and not os.path.isfile(F_STATE_DICT):
    raise FileNotFoundError

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
    # noinspection PyProtectedMember
    dd.io.save(pathjoin(DIR_TRAIN, cfg.F_HPARAMS),
               dict(cfg.hp.asdict()))

    trainer = Trainer.create('UNet')
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

    trainer = Trainer.create('UNet', use_cuda=True)
    trainer.test(loader, DIR_TEST, F_STATE_DICT)
else:
    raise ArgumentError
