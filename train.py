import pdb  # noqa: F401

import numpy as np
import scipy.io as scio
import deepdish as dd

import torch
from torch import nn
from torch.utils.data import DataLoader

from argparse import ArgumentParser
import os
import time
from multiprocessing import cpu_count  # noqa: F401

from typing import NamedTuple, Tuple

from iv_dataset import IVDataset, norm_iv
from utils import (arr2str, print_progress,
                   MultipleOptimizer, MultipleScheduler, static_vars)
# from mlp import MLP
from unet import UNet
import mypath

# manually selected
NUM_WORKERS = cpu_count()
# NUM_WORKERS = 0
OUT_CUDA_DEV = 1
NORM_PARTS = ('all',)

DIR_RESULT = './result/UNet'
if not os.path.isdir(DIR_RESULT):
    os.makedirs(DIR_RESULT)
DIR_RESULT = os.path.join(DIR_RESULT, os.path.basename(DIR_RESULT)+'_')

# determined by IV files
XNAME = 'IV_room'
YNAME = 'IV_free'

# determined by sys argv
parser = ArgumentParser()
parser.add_argument('--from',
                    type=int, nargs=1, dest='train_epoch', metavar='EPOCH')
parser.add_argument('--test',
                    type=int, nargs=1, dest='test_epoch', metavar='EPOCH')
ARGS = parser.parse_args()
if ARGS.train_epoch:
    f_model_state = f'{DIR_RESULT}{ARGS.train_epoch[0]}.pt'
elif ARGS.test_epoch:
    f_model_state = f'{DIR_RESULT}{ARGS.test_epoch[0]}.pt'
else:
    f_model_state = None


class HyperParameters(NamedTuple):
    """
    Hyper Parameters of NN
    """
    N_epochs = 1000
    batch_size = 24
    learning_rate = 5e-4
    N_file = 20*700

    n_per_frame: int
    p = 0.5  # Dropout p

    # lr scheduler
    step_size = 10
    gamma = 0.8

    weight_decay = 0  # 1e-8  # Adam weight_decay

    # def for_MLP(self) -> Tuple:
    #     n_input = self.L_cut_x * self.n_per_frame
    #     n_hidden = 17 * self.n_per_frame
    #     n_output = self.n_per_frame
    #     return (n_input, n_hidden, n_output, self.p)


if __name__ == '__main__':
    metadata = dd.io.load(os.path.join(mypath.path('iv_train'), 'metadata.h5'))
    hp = HyperParameters(n_per_frame=metadata['N_freq'] * 4)
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
    dataset_test.normalizeOnLike(dataset)
    del dataset

    # DataLoader
    loader_train = DataLoader(dataset_train,
                              batch_size=hp.batch_size,
                              shuffle=True,
                              num_workers=NUM_WORKERS,
                              collate_fn=dataset_train.pad_collate,
                              )
    loader_valid = DataLoader(dataset_valid,
                              batch_size=hp.batch_size,
                              shuffle=False,
                              num_workers=NUM_WORKERS,
                              collate_fn=dataset_valid.pad_collate,
                              )
    loader_test = DataLoader(dataset_test,
                             batch_size=hp.batch_size,
                             shuffle=False,
                             collate_fn=dataset_test.pad_collate,
                             num_workers=NUM_WORKERS,
                             )

    # Model (Using Parallel GPU)
    model = nn.DataParallel(UNet(n_channels=4, n_classes=4),
                            output_device=OUT_CUDA_DEV).cuda()
    if f_model_state:
        model.load_state_dict(torch.load(f_model_state))

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
    scheduler = MultipleScheduler(torch.optim.lr_scheduler.StepLR,
                                  optimizer,
                                  step_size=hp.step_size,
                                  gamma=hp.gamma)


def train():
    loss_train = np.zeros(hp.N_epochs)
    # loss_valid = np.zeros(hp.N_epochs)
    # snr_seg_valid = np.zeros(hp.N_epochs)
    loss_valid = np.zeros((hp.N_epochs, len(NORM_PARTS)))
    snr_seg_valid = np.zeros((hp.N_epochs, len(NORM_PARTS)))

    # Start Training
    for epoch in range(hp.N_epochs):
        t_start = time.time()

        print()
        print_progress(0, len(loader_train), f'epoch {epoch:3d}:')
        scheduler.step()
        for iteration, data in enumerate(loader_train):
            x = data['x']
            y = data['y']

            N_frames = data['N_frames_x'] \
                if 'N_frames_x' in data else x.shape[-1]
            size = sum(N_frames)

            # ===================forward=====================
            _input = x.cuda(device=0)
            output = model(_input)
            y_cuda = y.cuda(device=OUT_CUDA_DEV)
            if type(N_frames) == list:
                loss = torch.tensor([0.]).cuda(OUT_CUDA_DEV)
                for N_frame, item_out, item_y in zip(N_frames, output, y_cuda):
                    loss += criterion(item_out[:, :, :N_frame],
                                      item_y[:, :, :N_frame])
            else:
                loss = criterion(output, y_cuda)

            loss /= size
            loss_train[epoch] += loss.item()

            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print_progress(iteration+1, len(loader_train),
                           f'epoch {epoch:3d}: {loss.item():.1e}')

        loss_train[epoch] /= len(loader_train)
        print(f'Training Loss: {loss_train[epoch]:.2e}')

        # Validation
        loss_valid[epoch], snr_seg_valid[epoch] \
            = eval_model(loader=loader_valid,
                         FNAME=f'{DIR_RESULT}IV_{epoch}.mat',
                         norm_parts=NORM_PARTS)

        # print loss, snr and save
        print(f'Validation Loss: {loss_valid[epoch, -1]:.2e}\t'
              f'Validation SNRseg (dB): {snr_seg_valid[epoch, -1]:.2e}')

        scio.savemat(f'{DIR_RESULT}loss_{epoch}.mat',
                     {'loss_train': loss_train.squeeze(),
                      'loss_valid': loss_valid.squeeze(),
                      'snr_seg_valid': snr_seg_valid.squeeze(),
                      }
                     )
        torch.save(model.state_dict(), f'{DIR_RESULT}{epoch}.pt')

        print(f'epoch {epoch:3d}: '
              + time.strftime('%M min %S sec',
                              time.gmtime(time.time() - t_start)))
        # Early Stopping
        # if epoch >= 4:
        #     # loss_max = loss_valid[epoch-4:epoch].max()
        #     # loss_min = loss_valid[epoch-4:epoch].min()
        #     loss_slice_all = loss_valid[epoch-4:epoch, :].sum(axis=1)
        #     loss_max = loss_slice_all.max()
        #     loss_min = loss_slice_all.min()
        #     if loss_max - loss_min < 0.01 * loss_max:
        #         print('Early Stopped')
        #         break

    # Test and print
    loss_test, snr_test_dB = eval_model(FNAME=f'{DIR_RESULT}IV_test.mat',
                                        norm_parts=NORM_PARTS)
    print()
    print(f'Test Loss: {arr2str(loss_test)}\t'
          f'Test SNR (dB): {arr2str(snr_test_dB)}')


@static_vars(sum_N_frames=0)
def eval_model(loader: DataLoader=None, FNAME='',
               norm_parts='all') -> Tuple[float, float]:
    """
    Evaluate the performance of the model.
    loader: DataLoader to use.
    FNAME: filename of the result. If None, don't save the result.
    """
    if not loader:
        loader = loader_test

    # avg_loss = 0.
    # avg_snr =  0.
    avg_loss = torch.zeros(len(norm_parts),
                           requires_grad=False).cuda(OUT_CUDA_DEV)
    snr_seg = torch.zeros(len(norm_parts),
                          requires_grad=False).cuda(OUT_CUDA_DEV)
    norm_frames = [None] * len(loader)
    norm_errors = [None] * len(loader)
    if not eval_model.sum_N_frames:
        sum_N_frames = 0
    dict_to_save = {}
    print_progress(0, len(loader), 'eval:')
    with torch.no_grad():
        model.eval()

        for iteration, data in enumerate(loader):
            x = data['x']
            y = data['y']

            N_frames = data['N_frames_x'] \
                if 'N_frames_x' in data else x.shape[-1]

            # =========================forward=============================
            _input = x.cuda(device=0)
            output = model(_input)
            y_cuda = y.cuda(device=OUT_CUDA_DEV)
            # =============================================================
            # Reconstruct & Performance Measure
            y_denorm = loader.dataset.denormalize(y_cuda, 'y')
            out_denorm = loader.dataset.denormalize(output, 'y')

            if type(N_frames) == list:
                loss = torch.tensor([0.]).cuda(OUT_CUDA_DEV)
                for N_frame, item_out, item_y in zip(N_frames, output, y_cuda):
                    norm_normalized = norm_iv(
                        item_out[..., :N_frame] - item_y[..., :N_frame],
                        parts=norm_parts
                    )  # 3 x t
                    loss += norm_normalized.sum(dim=1)

                norm_frames[iteration] = []
                norm_errors[iteration] = []
                for N_frame, item_out, item_y \
                        in zip(data['N_frames_y'], out_denorm, y_denorm):
                    norm_frames[iteration].append(
                        norm_iv(
                            item_y[..., :N_frame],
                            keep_freq_axis=True,
                            parts=norm_parts
                        )
                    )
                    norm_errors[iteration].append(
                        norm_iv(
                            item_out[..., :N_frame] - item_y[..., :N_frame],
                            keep_freq_axis=True,
                            parts=norm_parts
                        )
                    )
                norm_frames[iteration] = torch.cat(norm_frames[iteration],
                                                   dim=-1)
                norm_errors[iteration] = torch.cat(norm_errors[iteration],
                                                   dim=-1)

            else:
                # 3 x t
                norm_normalized = norm_iv(output - y_cuda, parts=norm_parts)
                loss = norm_normalized.sum(dim=1)
                # 3 x f x t
                # if y_denorm.shape[-1] < N_frames:
                #     F.pad(y_denorm, (0, N_frames - y_denorm.shape[-1]))
                norm_frames[iteration] = norm_iv(y_denorm,
                                                 keep_freq_axis=True,
                                                 parts=norm_parts,
                                                 )
                norm_errors[iteration] = norm_iv(out_denorm - y_denorm,
                                                 keep_freq_axis=True,
                                                 parts=norm_parts,
                                                 )

            if not eval_model.sum_N_frames:
                sum_N_frames += sum(N_frames)
            avg_loss += loss

            sum_frames = norm_frames[iteration].sum(dim=-2)
            sum_errors = norm_errors[iteration].sum(dim=-2)
            snr_seg += 10 * torch.log10(sum_frames / sum_errors).sum(dim=1)

            # Save IV Result
            if FNAME and not dict_to_save:
                x = loader.dataset.denormalize_(x, 'x')
                if type(N_frames) == list:
                    x = x[0, :, :, :N_frames[0]]
                    y_denorm = y_denorm[0, :, :, :N_frames[0]]
                    out_denorm = out_denorm[0, :, :, :N_frames[0]]

                dict_to_save = {
                    'IV_free': y_denorm.cpu().numpy(),
                    'IV_room': x.numpy(),
                    'IV_estimated': out_denorm.cpu().numpy(),
                }
                scio.savemat(FNAME, dict_to_save)

            print_progress(iteration+1, len(loader),
                           f'{"eval":<9}: {loss[-1]/sum(N_frames):.1e}'
                           )

        if not eval_model.sum_N_frames:
            eval_model.sum_N_frames = sum_N_frames
        avg_loss /= eval_model.sum_N_frames
        snr_seg /= eval_model.sum_N_frames

        # if FNAME:
        #     norm_frames = np.concatenate(norm_frames, axis=1)
        #     norm_errors = np.concatenate(norm_errors, axis=1)
        #
        #     dict_to_save['norm_frames'] = norm_frames[2, :, :]
        #     dict_to_save['norm_errors'] = norm_errors[2, :, :]
        #     scio.savemat(FNAME, dict_to_save)
        model.train()
    return avg_loss.cpu().numpy(), snr_seg.cpu().numpy()


def run():
    if not f_model_state or ARGS.train_epoch:
        train()
    else:
        loss_test, snr_seg_test \
            = eval_model(FNAME=f'MLP_result_{ARGS.test_epoch}_test.mat')
        print(f'Test Loss: {arr2str(loss_test)}\t'
              f'Test SNRseg (dB): {arr2str(snr_seg_test)}')
        quit()


if __name__ == '__main__':
    run()
