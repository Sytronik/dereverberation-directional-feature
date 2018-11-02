import pdb  # noqa: F401

import numpy as np
import scipy.io as scio
import deepdish as dd

import torch
from torch import nn
from torch.utils.data import DataLoader

import sys
import os
import time
from multiprocessing import cpu_count  # noqa: F401

from typing import NamedTuple, Tuple

from iv_dataset import IVDataset, norm_iv

from utils import arr2str, printProgress, MultipleOptimizer, MultipleScheduler
from mlp import MLP
import mypath

NUM_WORKERS = cpu_count()
# NUM_WORKERS = 0
OUT_CUDA_DEV = 1


class HyperParameters(NamedTuple):
    """
    Hyper Parameters of NN
    """
    N_epochs = 100
    batch_size = 400
    learning_rate = 5e-4
    N_file = 100
    L_cut_x = 19

    n_per_frame: int
    p = 0.5  # Dropout p

    # lr scheduler
    step_size = 1
    gamma = 0.9

    weight_decay = 0  # 1e-8  # Adam weight_decay

    def for_MLP(self) -> Tuple:
        n_input = self.L_cut_x * self.n_per_frame
        n_hidden = 17 * self.n_per_frame
        n_output = self.n_per_frame
        return (n_input, n_hidden, n_output, self.p)


if __name__ == '__main__':
    metadata = dd.io.load(os.path.join(mypath.path('iv_train'), 'metadata.h5'))
    hp = HyperParameters(n_per_frame=metadata['N_freq'] * 4)

    if len(sys.argv) > 2 and sys.argv[1] == 'test':
        str_epoch = sys.argv[2]
        f_model_state = f'MLP_{str_epoch}.pt'
    else:
        f_model_state = None

    XNAME = 'IV_room'
    YNAME = 'IV_free'
    DIR_RESULT = './result/MLP'

    if not os.path.isdir(DIR_RESULT):
        os.makedirs(DIR_RESULT)
    DIR_RESULT = os.path.join(DIR_RESULT, os.path.basename(DIR_RESULT)+'_')
    dd.io.save(f'{DIR_RESULT}hparams.h5', dict(hp._asdict()))

    IVDataset.L_cut_x = hp.L_cut_x

    # Dataset
    # Training + Validation Set
    dataset = IVDataset('train', XNAME, YNAME, N_file=hp.N_file)

    # Test Set
    dataset_test = IVDataset('test', XNAME, YNAME, N_file=hp.N_file // 4,
                             doNormalize=False
                             )
    dataset_test.doNormalize(dataset.normalize)

    loader_test = DataLoader(dataset_test,
                             batch_size=hp.batch_size,
                             shuffle=False,
                             # collate_fn=IVDataset.my_collate,
                             num_workers=NUM_WORKERS,
                             )

    # Model (Using Parallel GPU)
    model = nn.DataParallel(MLP(*hp.for_MLP()),
                            output_device=OUT_CUDA_DEV).cuda()
    if f_model_state:
        model.load_state_dict(torch.load(f_model_state))

    # MSE Loss
    criterion = nn.MSELoss(reduction='sum')


def train():
    # Optimizer
    param1 = [p for m in model.modules() if not isinstance(m, nn.PReLU)
              for p in m.parameters()]
    param2 = [p for m in model.modules() if isinstance(m, nn.PReLU)
              for p in m.parameters()]

    optimizer = MultipleOptimizer(
        torch.optim.Adam(param1,
                         lr=hp.learning_rate,
                         weight_decay=hp.weight_decay,
                         ),
        torch.optim.Adam(param2,
                         lr=hp.learning_rate,
                         ) if param2 else None,
    )
    scheduler = MultipleScheduler(torch.optim.lr_scheduler.StepLR,
                                  optimizer,
                                  step_size=hp.step_size,
                                  gamma=hp.gamma)

    dataset_train, dataset_valid = IVDataset.split(dataset, (0.7, -1))

    loader_train = DataLoader(dataset_train,
                              batch_size=hp.batch_size,
                              shuffle=True,
                              num_workers=NUM_WORKERS,
                              # collate_fn=IVDataset.my_collate,
                              )
    loader_valid = DataLoader(dataset_valid,
                              batch_size=hp.batch_size,
                              shuffle=False,
                              num_workers=NUM_WORKERS,
                              # collate_fn=IVDataset.my_collate,
                              )

    loss_train = np.zeros(hp.N_epochs)
    # loss_valid = np.zeros(hp.N_epochs)
    # snr_seg_valid = np.zeros(hp.N_epochs)
    loss_valid = np.zeros((hp.N_epochs, 3))
    snr_seg_valid = np.zeros((hp.N_epochs, 3))

    # Start Training
    for epoch in range(hp.N_epochs):
        t_start = time.time()

        iteration = 0
        print('')
        printProgress(iteration, len(loader_train), f'epoch {epoch+1:3d}:')
        scheduler.step()
        for data in loader_train:
            iteration += 1
            x_stacked = data['x']
            y_stacked = data['y']

            N_frame = x_stacked.size(0)
            # if epoch == 0:
            #     N_total_frame += N_frame

            _input = x_stacked.view(N_frame, -1).cuda(device=0)
            # ===================forward=====================
            output = model(_input)
            y_cuda = y_stacked.view(N_frame, -1).cuda(device=OUT_CUDA_DEV)
            loss = criterion(output, y_cuda)
            loss_train[epoch] += loss.item()
            loss /= N_frame

            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            printProgress(iteration, len(loader_train),
                          f'epoch {epoch+1:3d}: {loss.item():.1e}')
        loss_train[epoch] /= len(loader_train.dataset)
        print(f'Training Loss: {loss_train[epoch]:.2e}')

        # Validation
        loss_valid[epoch], snr_seg_valid[epoch] \
            = eval_model(loader_valid, FNAME=f'{DIR_RESULT}IV_{epoch}.mat')

        # print loss, snr and save
        print(f'Validation Loss: {loss_valid[epoch, 2]:.2e}\t'
              f'Validation SNRseg (dB): {snr_seg_valid[epoch, 2]:.2e}')

        scio.savemat(f'{DIR_RESULT}loss_{epoch}.mat',
                     {'loss_train': loss_train,
                      'loss_valid': loss_valid,
                      'snr_seg_valid': snr_seg_valid,
                      }
                     )
        torch.save(model.state_dict(), f'{DIR_RESULT}{epoch}.pt')

        print(f'epoch {epoch+1:3d}: '
              + time.strftime('%M min %S sec',
                              time.gmtime(time.time() - t_start)))
        # Early Stopping
        # if epoch >= 4:
        #     # loss_max = loss_valid[epoch-4:epoch+1].max()
        #     # loss_min = loss_valid[epoch-4:epoch+1].min()
        #     loss_slice_all = loss_valid[epoch-4:epoch+1, :].sum(axis=1)
        #     loss_max = loss_slice_all.max()
        #     loss_min = loss_slice_all.min()
        #     if loss_max - loss_min < 0.01 * loss_max:
        #         print('Early Stopped')
        #         break

    # Test and print
    loss_test, snr_test_dB = eval_model(FNAME=f'{DIR_RESULT}IV_test.mat')
    print('')
    print(f'Test Loss: {arr2str(loss_test)}\t'
          f'Test SNR (dB): {arr2str(snr_test_dB)}')


def eval_model(self, loader: DataLoader=None, FNAME='') -> Tuple[float, float]:
    """
    Evaluate the performance of the model.
    loader: DataLoader to use.
    FNAME: filename of the result. If None, don't save the result.
    """
    if not loader:
        loader = loader_test

    # avg_loss = 0.
    # avg_snr =  0.
    avg_loss = torch.zeros(3, requires_grad=False).cuda(OUT_CUDA_DEV)
    snr_seg = torch.zeros(3, requires_grad=False).cuda(OUT_CUDA_DEV)
    iteration = 0
    norm_frames = [None] * len(loader)
    norm_errors = [None] * len(loader)
    dict_to_save = {}
    printProgress(iteration, len(loader), 'eval:')
    with torch.no_grad():
        model.eval()

        for data in loader:
            iteration += 1
            x_stacked_cpu = data['x']
            y_stacked_cpu = data['y']

            N_frame = x_stacked_cpu.size(0)

            _input = x_stacked_cpu.view(N_frame, -1).cuda(device=0)
            # =========================forward=============================
            output = model(_input)

            y_stacked = y_stacked_cpu.cuda(device=OUT_CUDA_DEV)
            # =============================================================
            # Reconstruct & Performance Measure
            out_stacked = output.view(y_stacked.size())

            y_denorm = loader.dataset.denormalize(y_stacked, 'y')
            out_denorm = loader.dataset.denormalize(out_stacked, 'y')

            y_recon = IVDataset.unstack_y(y_denorm)
            out_recon = IVDataset.unstack_y(out_denorm)

            norm_normalized = norm_iv(out_stacked - y_stacked,
                                      parts=('I', 'a', 'all'))  # 3 x t
            loss = norm_normalized.sum(dim=1)
            avg_loss += loss

            # 3 x t x f
            norm_frames[iteration - 1] = norm_iv(y_recon,
                                                 keep_freq_axis=True,
                                                 parts=('I', 'a', 'all'))
            norm_errors[iteration - 1] = norm_iv(out_recon - y_recon,
                                                 keep_freq_axis=True,
                                                 parts=('I', 'a', 'all'))

            sum_frames = norm_frames[iteration - 1].sum(dim=2)
            sum_errors = norm_errors[iteration - 1].sum(dim=2)
            snr_seg += torch.log10(sum_frames / sum_errors).sum(dim=1).mul_(10)

            # Save IV Result
            if FNAME and not dict_to_save:
                N_first = loader.dataset.N_frames[0]
                x_stacked_cpu = loader.dataset.denormalize_(x_stacked_cpu, 'x')
                x_recon_cpu = IVDataset.unstack_x(x_stacked_cpu[:N_first])
                dict_to_save = {
                    'IV_free': y_recon[:, :N_first, :].cpu().numpy(),
                    'IV_room': x_recon_cpu.numpy(),
                    'IV_estimated': out_recon[:, :N_first, :].cpu().numpy(),
                }

            printProgress(iteration, len(loader),
                          f'{"eval":<9}: {loss[2]/N_frame:.1e}'
                          )

        avg_loss /= len(loader.dataset)
        snr_seg /= len(loader.dataset)

        if FNAME:
            # norm_frames = np.concatenate(norm_frames, axis=1)
            # norm_errors = np.concatenate(norm_errors, axis=1)
            #
            # dict_to_save['norm_frames'] = norm_frames[2, :, :]
            # dict_to_save['norm_errors'] = norm_errors[2, :, :]
            scio.savemat(FNAME, dict_to_save)
        model.train()
    return avg_loss.cpu().numpy(), snr_seg.cpu().numpy()


def run():
    if not f_model_state:
        train()
    else:
        loss_test, snr_seg_test \
            = eval_model(FNAME=f'MLP_result_{str_epoch}_test.mat')
        print(f'Test Loss: {arr2str(loss_test)}\t'
              f'Test SNRseg (dB): {arr2str(snr_seg_test)}')
        quit()


if __name__ == '__main__':
    run()
