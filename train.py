from argparse import ArgumentParser
from glob import glob
from multiprocessing import cpu_count  # noqa: F401
import os
import time
from typing import NamedTuple, Tuple, Iterable, Union

import deepdish as dd
import numpy as np
import scipy.io as scio

import torch
from torch import nn
from torch.utils.data import DataLoader

from iv_dataset import (IVDataset,
                        delta,
                        norm_iv)
import mypath
from utils import (arr2str,
                   MultipleOptimizer,
                   MultipleScheduler,
                   print_progress,
                   )

from mlp import MLP  # noqa: F401
from cosine_scheduler import CosineLRWithRestarts
from unet import UNet  # noqa: F401
# from deeplab_xception import DeepLabv3_plus  # noqa: F401

# ---------manually selected---------
CUDA_DEVICES = tuple(range(torch.cuda.device_count()))
OUT_CUDA_DEV = 1
NORM_PARTS = 'all'

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
parser.add_argument('--from',
                    type=int, nargs=1, dest='train_epoch', metavar='EPOCH')
parser.add_argument('--test',
                    type=int, nargs=1, dest='test_epoch', metavar='EPOCH')
parser.add_argument('--debug', '-d', dest='num_workers',
                    action='store_const', const=0, default=cpu_count())
ARGS = parser.parse_args()
N_WORKERS = ARGS.num_workers
if ARGS.train_epoch:
    F_STATE_DICT = f'{DIR_RESULT}{ARGS.train_epoch[0]}.pt'
    FIRST_EPOCH = ARGS.train_epoch[0] + 1
elif ARGS.test_epoch:
    F_STATE_DICT = f'{DIR_RESULT}{ARGS.test_epoch[0]}.pt'
    FIRST_EPOCH = 0
else:
    F_STATE_DICT = ''
    FIRST_EPOCH = 0

f_loss_list = glob(f'{DIR_RESULT}loss_*.mat')
if f_loss_list:
    F_LOSS = f_loss_list[-1]
else:
    F_LOSS = ''
del f_loss_list

if F_STATE_DICT and not os.path.isfile(F_STATE_DICT):
    raise FileNotFoundError

avg_snr_seg_pre = None


class HyperParameters(NamedTuple):
    """
    Hyper Parameters of NN
    """
    N_epochs = 310
    batch_size = 32
    learning_rate = 1e-3
    N_file = 20*500

    n_per_frame: int
    p = 0.5  # Dropout p

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

    # weight_dyn_loss = (1, 0.5, 0.5)
    weight_dyn_loss = (1, 0.7, 0.5)

    # def for_MLP(self) -> Tuple:
    #     n_input = self.L_cut_x * self.n_per_frame
    #     n_hidden = 17 * self.n_per_frame
    #     n_output = self.n_per_frame
    #     return (n_input, n_hidden, n_output, self.p)


if __name__ == '__main__':
    metadata = dd.io.load(os.path.join(mypath.path('iv_train'), 'metadata.h5'))
    hp = HyperParameters(n_per_frame=metadata['N_freq']*4)
    del metadata

    dd.io.save(f'{DIR_RESULT}hparams.h5', dict(hp._asdict()))

    # Dataset
    # Training + Validation Set
    dataset = IVDataset('train', XNAME, YNAME, N_file=hp.N_file)
    dataset_train, dataset_valid = IVDataset.split(dataset, (0.7, -1))

    # Test Set
    dataset_test = IVDataset('test', XNAME, YNAME, N_file=hp.N_file // 4,
                             doNormalize=False
                             )
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
                             collate_fn=dataset_test.pad_collate,
                             num_workers=N_WORKERS,
                             )

    # Model (Using Parallel GPU)
    # model = nn.DataParallel(DeepLabv3_plus(4, 4),
    model = nn.DataParallel(UNet(n_channels=4, n_classes=4),
                            device_ids=CUDA_DEVICES,
                            output_device=OUT_CUDA_DEV).cuda()

    # MSE Loss
    criterion = nn.MSELoss(reduction='sum')
    # criterion = nn.L1Loss(reduction='sum')
    # criterion = nn.KLDivLoss(reduction='sum')

    # Optimizer
    param1 = [p for m in model.modules() if not isinstance(m, nn.PReLU)
              for p in m.parameters()]
    param2 = [p for m in model.modules() if isinstance(m, nn.PReLU)
              for p in m.parameters()]
    optimizer = MultipleOptimizer(
        torch.optim.Adam(param1,
                         lr=hp.learning_rate,
                         weight_decay=hp.weight_decay,
                         ) if param1 else None,
        torch.optim.Adam(param2,
                         lr=hp.learning_rate,
                         ) if param2 else None,
    )

    # Scheduler
    # scheduler = MultipleScheduler(torch.optim.lr_scheduler.StepLR,
    #                               optimizer, **hp.StepLR)
    # scheduler = MultipleScheduler(torch.optim.lr_scheduler.CosineAnnealingLR,
    #                               optimizer, **hp.CosineAnnealingLR,
    #                               last_epoch=FIRST_EPOCH-1)
    scheduler = MultipleScheduler(CosineLRWithRestarts,
                                  optimizer,
                                  batch_size=hp.batch_size,
                                  epoch_size=len(dataset_train),
                                  **hp.CosineLRWithRestarts,
                                  last_epoch=FIRST_EPOCH - 1)

    if F_STATE_DICT:
        tup = torch.load(F_STATE_DICT)
        model.load_state_dict(tup[0])
        optimizer.load_state_dict(tup[1])


def train():
    loss_train = np.zeros(hp.N_epochs)
    loss_valid = np.zeros((hp.N_epochs, len(NORM_PARTS)))
    avg_snr_seg_valid = np.zeros((hp.N_epochs, len(NORM_PARTS)))
    if F_LOSS:
        dict_loss = scio.loadmat(F_LOSS, squeeze_me=True)
        loss_train[:FIRST_EPOCH] = dict_loss['loss_train']
        loss_valid[:FIRST_EPOCH] = dict_loss['loss_valid']
        avg_snr_seg_valid[:FIRST_EPOCH] = dict_loss['snr_seg_valid']
        avg_snr_seg_pre = dict_loss['avg_snr_seg_pre']

    # Start Training
    for epoch in range(FIRST_EPOCH, hp.N_epochs):
        t_start = time.time()

        print()
        print_progress(0, len(loader_train), f'epoch {epoch:3d}:')
        scheduler.step()
        for i_iter, data in enumerate(loader_train):
            # ==================ge data=====================
            x = data['x']
            y = data['y']

            N_frames = data['T_ys'] if 'T_ys' in data else y.shape[-1]
            # size = N_frames.sum()
            x = x.permute(0, -1, -3, -2)
            y = y.permute(0, -1, -3, -2)
            x_cuda = x.cuda(device=0)
            y_cuda = y.cuda(device=OUT_CUDA_DEV)

            # ===================forward=====================
            output = model(x_cuda)[..., :y_cuda.shape[-1]]

            # delta & delta-delta
            y_cuda = [y_cuda, None, None]
            output = [output, None, None]
            for i_dyn in range(1, 3):
                y_cuda[i_dyn], tplz_mat = delta(y_cuda[i_dyn - 1], axis=-1)
                output[i_dyn], _ = delta(output[i_dyn - 1], axis=-1, tplz_mat=tplz_mat)

            loss = torch.tensor([0.]).cuda(OUT_CUDA_DEV)
            if type(N_frames) == np.ndarray:
                for i_dyn, (y_dyn, out_dyn) in enumerate(zip(y_cuda, output)):
                    for T, item_y, item_out in zip(N_frames, y_dyn, out_dyn):
                        loss += (hp.weight_dyn_loss[i_dyn] / int(T)
                                 * criterion(item_out[:, :, :T - 4*i_dyn],
                                             item_y[:, :, :T - 4*i_dyn]))
            else:
                for y_dyn, out_dyn in zip(y_cuda, output):
                    loss += (hp.weight_dyn_loss[i_dyn] / y_cuda.shape[-1]
                             * criterion(out_dyn, y_dyn))

            loss_train[epoch] += loss.item()

            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.batch_step()
            loss = (loss / len(N_frames)).item()
            print_progress(i_iter + 1, len(loader_train),
                           f'epoch {epoch:3d}: {loss:.1e}')

        # ================= show and save result ===============
        loss_train[epoch] /= len(dataset_train)
        print(f'Training Loss: {loss_train[epoch]:.2e}')

        # ==================Validation=========================
        loss_valid[epoch], avg_snr_seg_valid[epoch] \
            = eval_model(loader=loader_valid,
                         fname=f'{DIR_RESULT}IV_{epoch}.mat',
                         norm_parts=NORM_PARTS)

        print(f'Validation Loss: {loss_valid[epoch, -1]:.2e}\t'
              f'Validation SNRseg (dB): {avg_snr_seg_valid[epoch, -1]:.2e}')

        try:
            os.remove(f'{DIR_RESULT}loss_{epoch-1}.mat')
        except FileNotFoundError:
            pass
        scio.savemat(
            f'{DIR_RESULT}loss_{epoch}.mat',
            {'loss_train': loss_train[:epoch + 1].squeeze(),
             'loss_valid': loss_valid[:epoch + 1].squeeze(),
             'avg_snr_seg_valid': avg_snr_seg_valid[:epoch + 1].squeeze(),
             'avg_snr_seg_pre': avg_snr_seg_pre,
             }
        )
        if epoch % 5 == 0:
            torch.save((model.state_dict(), optimizer.state_dict()),
                       f'{DIR_RESULT}{epoch}.pt')

        tt = time.strftime('%M min %S sec', time.gmtime(time.time() - t_start))
        print(f'epoch {epoch:3d}: {tt}')

    # Test and print
    loss_test, snr_test_dB = eval_model(fname=f'{DIR_RESULT}IV_test.mat',
                                        norm_parts=NORM_PARTS)
    print()
    print(f'Test Loss: {arr2str(loss_test)}\t'
          f'Test SNR (dB): {arr2str(snr_test_dB)}')


def calc_snr_seg(loader: DataLoader=None, norm_parts='all'):
    if not loader:
        loader = loader_test

    if type(norm_parts) == str:
        norm_parts = (norm_parts,)

    avg_snr_seg = torch.zeros(len(norm_parts),
                              requires_grad=False).cuda(OUT_CUDA_DEV)
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
                    snr_seg += torch.log10(norm_y / norm_err).mean(dim=1)
                snr_seg *= 10
            else:
                norm_y = norm_iv(y_denorm, parts=norm_parts)
                norm_err = norm_iv(x_denorm - y_denorm, parts=norm_parts)
                snr_seg = 10*torch.log10(norm_y / norm_err).mean(dim=1)

            avg_snr_seg += snr_seg
            print_progress(i_iter + 1, len(loader),
                           f'{"eval":<9}: {snr_seg[-1].item():.2e}')
        avg_snr_seg /= len(loader.dataset)

    return avg_snr_seg.cpu().numpy()


# @static_vars(sum_N_frames=0)
def eval_model(loader: DataLoader=None, fname='',
               norm_parts: Union[str, Iterable[str]]='all') -> Tuple:
    """
    Evaluate the performance of the model.
    loader: DataLoader to use.
    fname: filename of the result. If None, don't save the result.
    """
    if not loader:
        loader = loader_test

    if type(norm_parts) == str:
        norm_parts = (norm_parts,)

    dict_to_save = {}
    avg_loss = torch.zeros(len(norm_parts), requires_grad=False).cuda(OUT_CUDA_DEV)
    avg_snr_seg = torch.zeros(len(norm_parts), requires_grad=False).cuda(OUT_CUDA_DEV)

    print_progress(0, len(loader), 'eval:')
    with torch.no_grad():
        model.eval()

        for i_iter, data in enumerate(loader):
            # =======================get data============================
            x = data['x']
            y = data['y']

            N_frames = data['T_ys'] if 'T_ys' in data else y.shape[-1]

            x = x.permute(0, -1, -3, -2)
            # y = y.permute(-1, -3, -2)
            x_cuda = x.cuda(device=0)
            y_cuda = y.cuda(device=OUT_CUDA_DEV)

            # =========================forward=============================
            output = model(x_cuda)
            # ============Reconstruct & Performance Measure==================
            output = output.permute(0, -2, -1, -3)[..., :y_cuda.shape[-2], :]

            y_denorm = loader.dataset.denormalize(y_cuda, 'y')
            out_denorm = loader.dataset.denormalize(output, 'y')

            # delta & delta-delta
            y_cuda = [y_cuda, None, None]
            output = [output, None, None]
            for i_dyn in range(1, 3):
                y_cuda[i_dyn], tplz_mat = delta(y_cuda[i_dyn - 1], axis=-2)
                output[i_dyn], _ = delta(output[i_dyn - 1], axis=-2, tplz_mat=tplz_mat)

            loss = torch.zeros(len(norm_parts)).cuda(OUT_CUDA_DEV)
            if type(N_frames) == np.ndarray:
                # Loss
                for i_dyn, (y_dyn, out_dyn) in enumerate(zip(y_cuda, output)):
                    for T, item_y, item_out in zip(N_frames, y_dyn, out_dyn):
                        loss += hp.weight_dyn_loss[i_dyn] / int(T) \
                            * norm_iv(item_out[:, :, :T - 4*i_dyn] - item_y[:, :, :T - 4*i_dyn],
                                      reduce_axis=(-3, -2, -1),
                                      parts=norm_parts
                                      )

                # snr_seg
                snr_seg = torch.zeros(len(norm_parts)).cuda(OUT_CUDA_DEV)
                for i_b, (T, item_out, item_y) in enumerate(zip(N_frames, out_denorm, y_denorm)):
                    norm_y = norm_iv(item_y[:, :T, :],
                                     parts=norm_parts)
                    norm_err = norm_iv(item_out[:, :T, :] - item_y[:, :T, :],
                                       parts=norm_parts)
                    snr_seg += torch.log10(norm_y / norm_err).mean(dim=1)
                snr_seg *= 10
            else:
                for y_dyn, out_dyn in zip(y_cuda, output):
                    loss += hp.weight_dyn_loss[i_dyn] / y_dyn.shape[-2] \
                        * norm_iv(out_dyn - y_dyn,
                                  reduce_axis=(-3, -2, -1),
                                  parts=norm_parts
                                  )

                norm_y = norm_iv(y_denorm, parts=norm_parts)
                norm_err = norm_iv(out_denorm - y_denorm, parts=norm_parts)
                snr_seg = 10 * torch.log10(norm_y / norm_err).mean(dim=1)

            avg_loss += loss
            avg_snr_seg += snr_seg

            # Save IV Result
            if fname and not dict_to_save:
                x = x.permute(0, -2, -1, -3)
                x = loader.dataset.denormalize_(x, 'x')
                if type(N_frames) == np.ndarray:
                    x = x[0, :, :data['T_xs'][0], :]
                    y_denorm = y_denorm[0, :, :N_frames[0], :]
                    out_denorm = out_denorm[0, :, :N_frames[0], :]

                dict_to_save = {
                    'IV_free': y_denorm.cpu().numpy(),
                    'IV_room': x.numpy(),
                    'IV_estimated': out_denorm.cpu().numpy(),
                }
                scio.savemat(fname, dict_to_save)

            loss = loss[-1] / len(N_frames)
            print_progress(i_iter + 1, len(loader), f'{"eval":<9}: {loss:.1e}')

        avg_loss /= len(loader.dataset)
        avg_snr_seg /= len(loader.dataset)

        model.train()
    return avg_loss.cpu().numpy(), avg_snr_seg.cpu().numpy()


def run():
    global avg_snr_seg_pre
    if not F_LOSS or ARGS.train_epoch:
        if not F_LOSS:
            avg_snr_seg_pre = calc_snr_seg(loader_valid, norm_parts=NORM_PARTS)
        train()
    elif ARGS.test_epoch:
        norm_parts = ('I', 'a', 'all')
        avg_snr_seg_pre = calc_snr_seg(norm_parts=norm_parts)
        loss_test, snr_seg_test = eval_model(fname=f'{DIR_RESULT}IV_test.mat',
                                             norm_parts=norm_parts)
        print(f'SNRseg of Reverberant Data (dB): {arr2str(avg_snr_seg_pre, n_decimal=4)}'
              f'Loss: {arr2str(loss_test, n_decimal=4)}\t'
              f'SNRseg (dB): {arr2str(snr_seg_test, n_decimal=4)}\n'
              )


if __name__ == '__main__':
    run()
