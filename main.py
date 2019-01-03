# noinspection PyUnresolvedReferences
import matlab.engine
from argparse import ArgumentParser, ArgumentError
import config as cfg
import os
from iv_dataset import IVDataset
from train import Trainer
from multiprocessing import cpu_count
import deepdish as dd
from torch.utils.data import DataLoader

if __name__ == '__main__':
    PERIOD_SAVE_STATE = cfg.hp.CosineLRWithRestarts['restart_period'] // 2

    FIRST_EPOCH = 0

    # ------determined by sys argv------
    parser = ArgumentParser()
    parser.add_argument(
        'model_name', type=str, nargs=1, metavar='MODEL'
    )
    parser.add_argument(
        '--train', action='store_true',
    )
    parser.add_argument(
        '--test', action='store_true',
    )
    parser.add_argument(
        '--from', dest='epoch', metavar='EPOCH',
        type=int, nargs=1, default=(-1,)
    )
    parser.add_argument(
        '--debug', '-d', dest='num_workers',
        action='store_const', const=0, default=cpu_count()
    )  # number of cpu threads for dataloaders
    ARGS = parser.parse_args()
    DIR_RESULT = cfg.DICT_PATH[ARGS.model_name[0]]
    if not os.path.isdir(DIR_RESULT):
        os.makedirs(DIR_RESULT)
    F_RESULT_PREFIX = os.path.join(DIR_RESULT, f'{ARGS.model_name[0]}_')
    if ARGS.epoch[0] < -1:
        raise ArgumentError

    FIRST_EPOCH = ARGS.epoch[0] + 1
    F_STATE_DICT = f'{DIR_RESULT}{ARGS.epoch[0]}.pt' if FIRST_EPOCH > 0 else ''

    if F_STATE_DICT and not os.path.isfile(F_STATE_DICT):
        raise FileNotFoundError

    # Dataset
    # Training + Validation Set
    dataset_temp = IVDataset('train', n_file=cfg.hp.n_file, norm_class=cfg.NORM_CLASS)
    dataset_train, dataset_valid = IVDataset.split(dataset_temp, (0.7, -1))
    dataset_train.set_needs(**cfg.hp.CHANNELS)
    dataset_valid.set_needs(**cfg.CH_WITH_PHASE)

    # Test Set
    dataset_test = IVDataset('test', n_file=cfg.hp.n_file // 4,
                             random_sample=True, **cfg.CH_WITH_PHASE)
    dataset_test.normalize_on_like(dataset_temp)
    del dataset_temp

    # DataLoader
    loader_train = DataLoader(dataset_train,
                              batch_size=cfg.hp.batch_size,
                              shuffle=True,
                              num_workers=ARGS.num_workers,
                              collate_fn=dataset_train.pad_collate,
                              )
    loader_valid = DataLoader(dataset_valid,
                              batch_size=cfg.hp.batch_size,
                              shuffle=False,
                              num_workers=ARGS.num_workers,
                              collate_fn=dataset_valid.pad_collate,
                              )
    loader_test = DataLoader(dataset_test,
                             batch_size=cfg.hp.batch_size,
                             shuffle=False,
                             num_workers=ARGS.num_workers,
                             collate_fn=dataset_test.pad_collate,
                             )
    if ARGS.train:
        # noinspection PyProtectedMember
        dd.io.save(os.path.join(DIR_RESULT, cfg.F_HPARAMS), dict(cfg.hp._asdict()))

        trainer = Trainer('UNet', DIR_RESULT)
        trainer.train(loader_train, loader_valid, FIRST_EPOCH, F_STATE_DICT)
