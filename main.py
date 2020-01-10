""" Train or Test DNN

Usage:
```
    python main.py {--train, --test={seen, unseen}}
                   [--feature {SIV, DV}]
                   [--room_train ROOM_TRAIN]
                   [--channels--VAR CH]
                   [--room_test ROOM_TEST]
                   [--logdir LOGDIR]
                   [--n_epochs MAX_EPOCH]
                   [--from START_EPOCH]
                   [--device DEVICES] [--out_device OUT_DEVICE]
                   [--batch_size B]
                   [--learning_rate LR]
                   [--weight_decay WD]
```

More parameters are in `hparams.py`.
- specify `--train` or `--test {seen, unseen}`.
- feature: "SIV" for using spatially-averaged intensity,
           "DV" for using direction vector.
- VAR, CH: VAR can be path_speech, x, x_mag, y.
           CH can be Channel.NONE, Channel.ALL, Channel.LAST.
           Read _HyperParameters.__post_init__
- ROOM_TRAIN: room used to train
- ROOM_TEST: room used to test
- LOGDIR: log directory
- MAX_EPOCH: maximum epoch
- START_EPOCH: start epoch (Default: -1)
- DEVICES, OUT_DEVICE, B, LR, WD: read `hparams.py`.
"""
# noinspection PyUnresolvedReferences
import matlab.engine

import os
import shutil
from argparse import ArgumentError, ArgumentParser

from torch.utils.data import DataLoader

from dataset import DirSpecDataset
from hparams import hp
from train import Trainer

tfevents_fname = 'events.out.tfevents.*'
form_overwrite_msg = 'The folder "{}" already has tfevent files. Continue? [y/n]\n'

parser = ArgumentParser()

parser.add_argument('--train', action='store_true', )
parser.add_argument('--test', choices=('seen', 'unseen'), metavar='DATASET')

# save output for deep griffin-lim
parser.add_argument('--save', choices=('train', 'valid', 'seen', 'unseen'), metavar='DATASET')

# epoch (test mode: this determines which .pt file is used for model)
parser.add_argument('--from', type=int, default=-1, dest='epoch', metavar='EPOCH')

# no. of threads per process
parser.add_argument('--num_threads', type=int, default=0)

args = hp.parse_argument(parser)
del parser
if not (args.train ^ bool(args.test) ^ bool(args.save)) or args.epoch < -1:
    raise ArgumentError
if args.num_threads > 0:
    np_env_vars = (
        'OMP_NUM_THREADS',
        'OPENBLAS_NUM_THREADS',
        'MKL_NUM_THREADS',
        'VECLIB_MAXIMUM_THREADS',
        'NUMEXPR_NUM_THREADS',
    )
    for var in np_env_vars:
        os.environ[var] = str(args.num_threads)
        # os.environ[var] = str(12)

# directory
logdir_train = hp.logdir / 'train'
if (args.train and args.epoch == -1
    and logdir_train.exists() and list(logdir_train.glob(tfevents_fname))):
    # ask if overwrite
    ans = input(form_overwrite_msg.format(logdir_train))
    if ans.lower() == 'y':
        shutil.rmtree(logdir_train)
        os.remove(hp.logdir / 'summary.txt')
        os.remove(hp.logdir / 'hparams.txt')
    else:
        exit()
os.makedirs(logdir_train, exist_ok=True)

if args.test:
    logdir_test = hp.logdir
    foldername = args.test
    if hp.room_test != hp.room_train:
        foldername += f'_{hp.room_test}'
    foldername += f'_{args.epoch}'
    if hp.n_save_block_outs > 0:
        foldername += '_blockouts'
    if hp.eval_with_y_ph:
        foldername += '_true_ph'
    if hp.use_das_phase and hp.das_err != '':
        foldername += hp.das_err
    logdir_test /= foldername
    if logdir_test.exists() and list(logdir_test.glob(tfevents_fname)):
        # ask if overwrite
        ans = input(form_overwrite_msg.format(logdir_test))
        if ans.lower().startswith('y'):
            shutil.rmtree(logdir_test)
            os.makedirs(logdir_test)
        else:
            exit()
    os.makedirs(logdir_test, exist_ok=True)

if args.save:
    logdir_save = hp.logdir
    foldername = args.save
    if args.save == 'unseen' or args.save == 'seen' and hp.room_test != hp.room_train:
        foldername += f'_{hp.room_test}'
    foldername += f'_{args.epoch}_save'
    logdir_save /= foldername

    os.makedirs(logdir_save, exist_ok=True)
    # hp.batch_size /= 2

# epoch, state dict
first_epoch = args.epoch + 1
if first_epoch > 0:
    path_state_dict = logdir_train / f'{args.epoch}.pt'
    if not path_state_dict.exists():
        raise FileNotFoundError(path_state_dict)
else:
    path_state_dict = None

# Training + Validation Set
dataset_temp = DirSpecDataset('train')
dataset_train, dataset_valid = DirSpecDataset.split(dataset_temp, (hp.train_ratio, -1))
dataset_train.set_needs(**(hp.channels if not args.save else hp.channels_w_ph))
dataset_valid.set_needs(**hp.channels_w_ph)

loader_train = DataLoader(dataset_train,
                          batch_size=hp.batch_size,
                          num_workers=hp.num_workers,
                          collate_fn=dataset_train.pad_collate,
                          pin_memory=(hp.device != 'cpu'),
                          shuffle=(not args.save),
                          )
loader_valid = DataLoader(dataset_valid,
                          batch_size=hp.batch_size,
                          num_workers=hp.num_workers,
                          collate_fn=dataset_valid.pad_collate,
                          pin_memory=(hp.device != 'cpu'),
                          shuffle=False,
                          )

# run
trainer = Trainer(path_state_dict)
if args.train:
    trainer.train(loader_train, loader_valid, logdir_train, first_epoch)
elif args.test:  # args.test
    # Test Set
    dataset_test = DirSpecDataset(args.test,
                                  dataset_temp.norm_modules,
                                  **hp.channels_w_ph)
    loader = DataLoader(dataset_test,
                        batch_size=1,
                        num_workers=hp.num_workers,
                        collate_fn=dataset_test.pad_collate,
                        pin_memory=(hp.device != 'cpu'),
                        shuffle=False,
                        )

    # noinspection PyUnboundLocalVariable
    trainer.test(loader, logdir_test)
else:
    if args.save == 'train':
        loader = loader_train
    elif args.save == 'valid':
        loader = loader_valid
    else:
        dataset_test = DirSpecDataset(args.save,
                                      dataset_temp.norm_modules,
                                      **hp.channels_w_ph)
        loader = DataLoader(dataset_test,
                            batch_size=hp.batch_size,
                            num_workers=hp.num_workers,
                            collate_fn=dataset_test.pad_collate,
                            pin_memory=(hp.device != 'cpu'),
                            shuffle=False,
                            )
    # noinspection PyUnboundLocalVariable
    trainer.save_result(loader, logdir_save)
