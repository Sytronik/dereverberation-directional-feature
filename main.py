# noinspection PyUnresolvedReferences
import matlab.engine

import os
import shutil
from argparse import ArgumentError, ArgumentParser

import deepdish as dd
# noinspection PyCompatibility
from dataclasses import asdict
from torch.utils.data import DataLoader

from hparams import hp
from dirspecgram import DirSpecDataset
from train import Trainer

parser = ArgumentParser()

parser.add_argument('--train', action='store_true', )
parser.add_argument('--test', choices=('seen', 'unseen'), metavar='DATASET')
parser.add_argument('--from', type=int, default=-1, dest='epoch', metavar='EPOCH')

args = hp.parse_argument(parser)
del parser
if not (args.train ^ args.test is not None) or args.epoch < -1:
    raise ArgumentError

# directory
logdir_train = hp.logdir / 'train'
if (args.train and args.epoch == -1 and
        logdir_train.exists() and list(logdir_train.glob('events.out.tfevents.*'))):
    ans = input(f'The folder "{logdir_train}" already has tfevent files. Continue? [y/n]')
    if ans.lower() == 'y':
        shutil.rmtree(logdir_train)
    else:
        exit()
os.makedirs(logdir_train, exist_ok=True)

if args.test:
    logdir_test = hp.logdir
    if hp.room_test == hp.room_train:
        logdir_test /= args.test
    else:
        logdir_test /= f'{args.test}_{hp.room_test}'
    if logdir_test.exists() and list(logdir_test.glob('events.out.tfevents.*')):
        print(f'The folder "{logdir_test}" already has tfevent files. Continue? [y/n]')
        ans = input()
        if ans.lower().startswith('y'):
            shutil.rmtree(logdir_test)
            os.makedirs(logdir_test)
        else:
            exit()
    os.makedirs(logdir_test, exist_ok=True)

# epoch, state dict
first_epoch = args.epoch + 1
if first_epoch > 0:
    path_state_dict = logdir_train / f'{args.epoch}.pt'
    if not path_state_dict.exists():
        raise FileNotFoundError(path_state_dict)
else:
    path_state_dict = None

# Training + Validation Set
dataset_temp = DirSpecDataset('train',
                              n_file=hp.n_file,
                              keys_trannorm=hp.keys_trannorm,
                              )
dataset_train, dataset_valid = DirSpecDataset.split(dataset_temp, (hp.train_ratio, -1))
dataset_train.set_needs(**hp.channels)
dataset_valid.set_needs(**hp.channels_w_ph)

# run
trainer = Trainer.create(path_state_dict)
if args.train:
    loader_train = DataLoader(dataset_train,
                              batch_size=hp.batch_size,
                              num_workers=hp.num_disk_workers,
                              collate_fn=dataset_train.pad_collate,
                              pin_memory=True,
                              shuffle=True,
                              )
    loader_valid = DataLoader(dataset_valid,
                              batch_size=hp.batch_size,
                              num_workers=hp.num_disk_workers,
                              collate_fn=dataset_valid.pad_collate,
                              pin_memory=True,
                              shuffle=False,
                              )



    trainer.train(loader_train, loader_valid, logdir_train, first_epoch)
else:  # args.test
    # Test Set
    dataset_test = DirSpecDataset(args.test,
                                  n_file=hp.n_file // 4,
                                  random_by_utterance=False,
                                  **hp.channels_w_ph,
                                  )
    dataset_test.normalize_on_like(dataset_temp)
    loader = DataLoader(dataset_test,
                        batch_size=1,
                        num_workers=hp.num_disk_workers,
                        collate_fn=dataset_test.pad_collate,
                        shuffle=False,
                        )

    # noinspection PyUnboundLocalVariable
    trainer.test(loader, logdir_test)
