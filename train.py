from argparse import ArgumentParser
from glob import glob
from multiprocessing import cpu_count  # noqa: F401
import os
import time
from typing import NamedTuple, Tuple, Sequence, Union, Dict

import deepdish as dd
import numpy as np
import scipy.io as scio

import torch
from torch import nn
from torch.utils.data import DataLoader

from iv_dataset import (delta,
                        IVDataset,
                        norm_iv,
                        )
import mypath
from utils import (arr2str,
                   MultipleOptimizer,
                   MultipleScheduler,
                   print_progress,
                   )

from mlp import MLP  # noqa: F401
from adamwr import CosineLRWithRestarts, AdamW
from unet import UNet  # noqa: F401

# from deeplab_xception import DeepLabv3_plus  # noqa: F401

# ---------manually selected---------
CUDA_DEVICES = list(range(torch.cuda.device_count()))
OUT_CUDA_DEV = 1
NORM_PARTS = ('all',)
NEED_ONLY_ALPHA = (False, True)

MODEL_NAME = 'UNet'
DIR_RESULT = f'./result/{MODEL_NAME}'
# DIR_RESULT = './result/Deeplab'
if not os.path.isdir(DIR_RESULT):
    os.makedirs(DIR_RESULT)
DIR_RESULT = os.path.join(DIR_RESULT, f'{MODEL_NAME}_')

# ------determined by IV files------
XNAME = 'IV_room'
YNAME = 'IV_free'

# ------determined by sys argv------
parser = ArgumentParser()
parser.add_argument('--from', type=int, nargs=1, dest='train_epoch', metavar='EPOCH')
parser.add_argument('--test', type=int, nargs=1, dest='test_epoch', metavar='EPOCH')
parser.add_argument('--debug', '-d', dest='num_workers',
                    action='store_const', const=0, default=cpu_count())
ARGS = parser.parse_args()
N_WORKERS = ARGS.num_workers  # number of cpu threads for dataloaders

FIRST_EPOCH = 0
if ARGS.train_epoch:
    F_STATE_DICT = f'{DIR_RESULT}{ARGS.train_epoch[0]}.pt'
    FIRST_EPOCH = ARGS.train_epoch[0] + 1
elif ARGS.test_epoch:
    F_STATE_DICT = f'{DIR_RESULT}{ARGS.test_epoch[0]}.pt'
else:
    F_STATE_DICT = ''

if F_STATE_DICT and not os.path.isfile(F_STATE_DICT):
    raise FileNotFoundError

f_loss_list = glob(f'{DIR_RESULT}loss_*.mat')
F_LOSS = f_loss_list[-1] if f_loss_list else ''
del f_loss_list

avg_snr_seg_pre = None


class HyperParameters(NamedTuple):
    """
    Hyper Parameters of NN
    """
    n_per_frame: int

    UNet: Tuple

    n_epochs = 310
    batch_size = 4*8
    learning_rate = 5e-4
    n_file = 20*500

    # p = 0.5  # Dropout p

    # lr scheduler
    StepLR = dict(step_size=5, gamma=0.8)

    CosineAnnealingLR = dict(T_max=10,
                             eta_min=0,
                             )

    CosineLRWithRestarts = dict(restart_period=10,
                                t_mult=2,
                                eta_threshold=1000,
                                )

    weight_decay = 1e-8  # Adam weight_decay

    weight_loss = (1, 0.7, 0.5)

    # def for_MLP(self) -> Tuple:
    #     n_input = self.L_cut_x * self.n_per_frame
    #     n_hidden = 17 * self.n_per_frame
    #     n_output = self.n_per_frame
    #     return (n_input, n_hidden, n_output, self.p)


PERIOD_SAVE_STATE = HyperParameters.CosineLRWithRestarts['restart_period']//2

if __name__ == '__main__':
    metadata = dd.io.load(os.path.join(mypath.path('iv_train'), 'metadata.h5'))
    hp = HyperParameters(n_per_frame=metadata['N_freq']*4,
                         UNet=(*[1 if only else 4 for only in NEED_ONLY_ALPHA], 64))
    del metadata

    dd.io.save(f'{DIR_RESULT}hparams.h5', dict(hp._asdict()))

    # Dataset
    # Training + Validation Set
    dataset = IVDataset('train', XNAME, YNAME, n_file=hp.n_file)
    dataset_train, dataset_valid = IVDataset.split(dataset, (0.7, -1))

    # Test Set
    dataset_test = IVDataset('test', XNAME, YNAME, n_file=hp.n_file//4, doNormalize=False)
    dataset_test.normalize_on_like(dataset)
    del dataset

    # DataLoader
    loader_train = DataLoader(dataset_train,
                              batch_size=hp.batch_size,
                              shuffle=True,
                              num_workers=N_WORKERS,
                              collate_fn=dataset_train.pad_collate,
                              )
    loader_valid = DataLoader(dataset_valid,
                              batch_size=hp.batch_size,
                              shuffle=False,
                              num_workers=N_WORKERS,
                              collate_fn=dataset_valid.pad_collate,
                              )
    loader_test = DataLoader(dataset_test,
                             batch_size=hp.batch_size,
                             shuffle=False,
                             num_workers=N_WORKERS,
                             collate_fn=dataset_test.pad_collate,
                             )

    # Model (Using Parallel GPU)
    # model = nn.DataParallel(DeepLabv3_plus(4, 4),
    model = nn.DataParallel(UNet(*hp.UNet),
                            device_ids=CUDA_DEVICES,
                            output_device=OUT_CUDA_DEV).cuda()

    # Loss Function
    criterion = nn.MSELoss(reduction='sum')
    # criterion = nn.BCELoss(reduction='sum')

    # Optimizer
    param1 = [p for m in model.modules() if not isinstance(m, nn.PReLU)
              for p in m.parameters()]
    param2 = [p for m in model.modules() if isinstance(m, nn.PReLU)
              for p in m.parameters()]
    optimizer = MultipleOptimizer(
        # torch.optim.Adam(
        AdamW(
            param1,
            lr=hp.learning_rate,
            weight_decay=hp.weight_decay,
        ) if param1 else None,
        # torch.optim.Adam(
        AdamW(
            param2,
            lr=hp.learning_rate,
        ) if param2 else None,
    )

    # Learning Rates Scheduler
    # scheduler = MultipleScheduler(torch.optim.lr_scheduler.StepLR,
    #                               optimizer, **hp.StepLR)
    # scheduler = MultipleScheduler(torch.optim.lr_scheduler.CosineAnnealingLR,
    #                               optimizer, **hp.CosineAnnealingLR,
    #                               last_epoch=FIRST_EPOCH-1)
    scheduler = MultipleScheduler(CosineLRWithRestarts,
                                  optimizer,
                                  batch_size=hp.batch_size,
                                  epoch_size=len(dataset_train),
                                  last_epoch=FIRST_EPOCH - 1,
                                  **hp.CosineLRWithRestarts)

    # Load State Dict
    if F_STATE_DICT:
        tup = torch.load(F_STATE_DICT)
        model.load_state_dict(tup[0])
        optimizer.load_state_dict(tup[1])


def train():
    global avg_snr_seg_pre
    loss_train = np.zeros(hp.n_epochs)
    loss_valid = np.zeros((hp.n_epochs, len(NORM_PARTS))).squeeze()
    avg_snr_seg_valid = np.zeros((hp.n_epochs, len(NORM_PARTS))).squeeze()
    if F_LOSS:
        dict_loss = scio.loadmat(F_LOSS, squeeze_me=True)
        loss_train[:FIRST_EPOCH] = dict_loss['loss_train'][:FIRST_EPOCH]
        loss_valid[:FIRST_EPOCH] = dict_loss['loss_valid'][:FIRST_EPOCH]
        avg_snr_seg_valid[:FIRST_EPOCH] = dict_loss['avg_snr_seg_valid'][:FIRST_EPOCH]

        avg_snr_seg_pre = np.array(dict_loss.get('avg_snr_seg_pre'))
        del dict_loss

    # Start Training
    for epoch in range(FIRST_EPOCH, hp.n_epochs):
        t_start = time.time()

        print()
        print_progress(0, len(loader_train), f'epoch {epoch:3d}:')
        scheduler.step()
        for i_iter, data in enumerate(loader_train):
            # ==================get data=====================
            x, y = data['x'], data['y']  # B, F, T, C
            T_ys = data['T_ys']

            if NEED_ONLY_ALPHA[0] and x.shape[-1] > 1:
                x = x[..., 3:]
            if NEED_ONLY_ALPHA[1] and y.shape[-1] > 1:
                y = y[..., 3:]

            # B, C, F, T
            x = x.permute(0, -1, -3, -2)
            y = y.permute(0, -1, -3, -2)

            x_cuda = x.cuda(device=0)
            y_cuda = y.cuda(device=OUT_CUDA_DEV)

            # ===================forward=====================
            output = model(x_cuda)[..., :y_cuda.shape[-1]]  # B, C, F, T

            # ============ delta & delta-delta ============
            y_cuda = [y_cuda, None, None]
            output = [output, None, None]
            for i_dyn in range(1, 3):
                y_cuda[i_dyn], output[i_dyn] \
                    = delta(y_cuda[i_dyn - 1], output[i_dyn - 1], axis=-1)

            # ============ calculate loss ==================
            loss = torch.tensor([0.]).cuda(OUT_CUDA_DEV)
            for i_dyn, (y_dyn, out_dyn) in enumerate(zip(y_cuda, output)):
                for T, item_y, item_out in zip(T_ys, y_dyn, out_dyn):
                    loss += (hp.weight_loss[i_dyn]/int(T)
                             *criterion(item_out[:, :, :T - 4*i_dyn],
                                        item_y[:, :, :T - 4*i_dyn]))

            loss_train[epoch] += loss.item()

            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.batch_step()

            # ==================== progress ==================
            loss = (loss/len(T_ys)).item()
            print_progress(i_iter + 1, len(loader_train), f'epoch {epoch:3d}: {loss:.1e}')

        loss_train[epoch] /= len(dataset_train)
        print(f'Training Loss: {loss_train[epoch]:.2e}')

        # ==================Validation=========================
        loss_valid[epoch], avg_snr_seg_valid[epoch] \
            = eval_model(loader=loader_valid,
                         fname=f'{DIR_RESULT}IV_{epoch}.mat',
                         norm_parts=NORM_PARTS)

        print(f'Validation Loss: {loss_valid[epoch]:.2e}\t'
              f'Validation SNRseg (dB): {avg_snr_seg_valid[epoch]:.2e}')

        # ================= save loss & model ======================
        try:
            os.remove(f'{DIR_RESULT}loss_{epoch - 1}.mat')
        except FileNotFoundError:
            pass
        scio.savemat(
            f'{DIR_RESULT}loss_{epoch}.mat',
            dict(loss_train=loss_train[:epoch + 1].squeeze(),
                 loss_valid=loss_valid[:epoch + 1].squeeze(),
                 avg_snr_seg_valid=avg_snr_seg_valid[:epoch + 1].squeeze(),
                 avg_snr_seg_pre=avg_snr_seg_pre,
                 )
        )
        if epoch%PERIOD_SAVE_STATE == PERIOD_SAVE_STATE - 1:
            torch.save(
                (model.state_dict(),
                 optimizer.state_dict(),
                 ),
                f'{DIR_RESULT}{epoch}.pt'
            )

        # Time
        tt = time.strftime('%M min %S sec', time.gmtime(time.time() - t_start))
        print(f'epoch {epoch:3d}: {tt}')

    # Test and print
    loss_test, snr_test_dB \
        = eval_model(fname=f'{DIR_RESULT}IV_test.mat', norm_parts=NORM_PARTS)
    print(f'\n'
          f'Test Loss: {arr2str(loss_test)}\t'
          f'Test SNR (dB): {arr2str(snr_test_dB)}')


def calc_snr_seg(loader: DataLoader = None, norm_parts: Union[str, Sequence[str]] = ('all',)):
    if not loader:
        loader = loader_test

    if type(norm_parts) == str:
        norm_parts = (norm_parts,)

    avg_snr_seg = torch.zeros(len(norm_parts), requires_grad=False).cuda(OUT_CUDA_DEV)
    with torch.no_grad():
        print_progress(0, len(loader), f'{"eval":<9}')
        for i_iter, data in enumerate(loader):
            # =======================get data============================
            x = data['x']
            y = data['y']

            N_frames = data['T_ys'] if 'T_ys' in data else y.shape[-1]

            x_cuda = x.cuda(device=OUT_CUDA_DEV)
            y_cuda = y.cuda(device=OUT_CUDA_DEV)
            x_denorm = loader.dataset.denormalize(x_cuda, 'y')
            y_denorm = loader.dataset.denormalize(y_cuda, 'y')

            if type(N_frames) == np.ndarray:
                snr_seg = torch.zeros(len(norm_parts)).cuda(OUT_CUDA_DEV)
                for i_b, (T, item_x, item_y) in enumerate(zip(N_frames, x_denorm, y_denorm)):
                    norm_y = norm_iv(item_y[:, :T, :], parts=norm_parts)
                    norm_err = norm_iv(item_y[:, :T, :] - item_x[:, :T, :], parts=norm_parts)
                    snr_seg += torch.log10(norm_y/norm_err).mean(dim=1)
                snr_seg *= 10
            else:
                norm_y = norm_iv(y_denorm, parts=norm_parts)
                norm_err = norm_iv(x_denorm - y_denorm, parts=norm_parts)
                snr_seg = 10*torch.log10(norm_y/norm_err).mean(dim=1)

            avg_snr_seg += snr_seg
            print_progress(i_iter + 1, len(loader),
                           f'{"eval":<9}: {snr_seg[-1].item():.2e}')
        avg_snr_seg /= len(loader.dataset)

    return avg_snr_seg.cpu().numpy()


def eval_model(loader: DataLoader = None, fname='',
               norm_parts: Union[str, Sequence[str]] = ('all',)) -> Tuple:
    """
    Evaluate the performance of the model.
    loader: DataLoader to use.
    fname: filename of the result. If None, don't save the result.
    norm_parts: one of 'I', 'a', 'all'
    """
    if not loader:
        loader = loader_test

    if type(norm_parts) == str:
        norm_parts = (norm_parts,)

    saved = False
    avg_loss = torch.zeros(len(norm_parts), requires_grad=False).cuda(OUT_CUDA_DEV)
    avg_snr_seg = torch.zeros(len(norm_parts), requires_grad=False).cuda(OUT_CUDA_DEV)

    print_progress(0, len(loader), 'eval:')
    with torch.no_grad():
        model.eval()

        for i_iter, data in enumerate(loader):
            # =======================get data============================
            x, y = data['x'], data['y']  # B, F, T, C
            T_ys = data['T_ys']

            if NEED_ONLY_ALPHA[0] and x.shape[-1] > 1:
                x = x[..., 3:]
            if NEED_ONLY_ALPHA[1] and y.shape[-1] > 1:
                y = y[..., 3:]

            # B, C, F, T
            x = x.permute(0, -1, -3, -2)

            x_cuda = x.cuda(device=0)
            y_cuda = y.cuda(device=OUT_CUDA_DEV)  # B, F, T, C

            # =========================forward=============================
            output = model(x_cuda)  # B, C, F, T

            # ============Reconstruct & Performance Measure==================
            # B, F, T, C
            output = output.permute(0, -2, -1, -3)[..., :y_cuda.shape[-2], :]

            # B, F, T, C
            y_denorm = loader.dataset.denormalize(y_cuda, 'y')
            out_denorm = loader.dataset.denormalize(output, 'y')

            # delta & delta-delta
            y_cuda = [y_cuda, None, None]
            output = [output, None, None]
            for i_dyn in range(1, 3):
                y_cuda[i_dyn], output[i_dyn] \
                    = delta(y_cuda[i_dyn - 1], output[i_dyn - 1], axis=-2)

            # Loss
            loss = torch.zeros(len(norm_parts)).cuda(OUT_CUDA_DEV)
            for i_dyn, (y_dyn, out_dyn) in enumerate(zip(y_cuda, output)):
                for T, item_y, item_out in zip(T_ys, y_dyn, out_dyn):
                    loss += hp.weight_loss[i_dyn]/int(T)*norm_iv(
                        item_out[:, :T - 4*i_dyn, :] - item_y[:, :T - 4*i_dyn, :],
                        reduced_axis=(-3, -2, -1),
                        parts=norm_parts
                    )

            # snr_seg
            snr_seg = torch.zeros(len(norm_parts)).cuda(OUT_CUDA_DEV)
            for i_b, (T, item_out, item_y) in enumerate(zip(T_ys, out_denorm, y_denorm)):
                # PARTS, T
                norm_y = norm_iv(item_y[:, :T, :], parts=norm_parts)
                norm_err = norm_iv(item_out[:, :T, :] - item_y[:, :T, :], parts=norm_parts)

                snr_seg += torch.log10(norm_y/norm_err).mean(dim=1)
            snr_seg *= 10

            avg_loss += loss
            avg_snr_seg += snr_seg

            # Save IV Result
            if (not saved) and fname:
                # F, T, C
                x_denorm = loader.dataset.denormalize_(data['x'][0], 'x')
                y_denorm = loader.dataset.denormalize(data['y'][0], 'y')

                x_one = x_denorm[:, :data['T_xs'][0], :].numpy()
                y_one = y_denorm[:, :T_ys[0], :].numpy()
                out_one = out_denorm[0, :, :T_ys[0], :].cpu().numpy()

                scio.savemat(fname, dict(IV_free=y_one,
                                         IV_room=x_one,
                                         IV_estimated=out_one,
                                         ))
                saved = True

            loss = loss[-1]/len(T_ys)
            print_progress(i_iter + 1, len(loader), f'{"eval":<9}: {loss:.1e}')

        avg_loss /= len(loader.dataset)
        avg_snr_seg /= len(loader.dataset)

        model.train()
    return avg_loss.cpu().numpy(), avg_snr_seg.cpu().numpy()


def run():
    global avg_snr_seg_pre
    if not F_STATE_DICT or ARGS.train_epoch:
        if not F_LOSS:
            avg_snr_seg_pre = calc_snr_seg(loader_valid, norm_parts=NORM_PARTS)
        train()
    elif ARGS.test_epoch:
        parts = ('I', 'a', 'all')
        avg_snr_seg_pre = calc_snr_seg(norm_parts=parts)
        loss_test, avg_snr_seg_test \
            = eval_model(fname=f'{DIR_RESULT}IV_test.mat', norm_parts=parts)

        print(f'SNRseg of Reverberant Data (dB): {arr2str(avg_snr_seg_pre, n_decimal=4)}\n'
              f'Loss: {arr2str(loss_test, n_decimal=4)}\t'
              f'SNRseg (dB): {arr2str(avg_snr_seg_test, n_decimal=4)}'
              )


if __name__ == '__main__':
    run()
