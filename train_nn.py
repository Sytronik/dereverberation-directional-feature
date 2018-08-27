import pdb  # noqa: F401

import numpy as np
import scipy.io as scio

import torch
from torch import nn
from torch.utils.data import DataLoader

import gc
import os
import time
import multiprocessing as mp

from typing import NamedTuple, Tuple

from iv_dataset import IVDataset, norm_iv


def NDARR_TO_STR(a):
    return np.array2string(a, formatter={'float_kind': lambda x: f'{x:.2e}'})


def printProgress(iteration: int, total: int, prefix='', suffix='',
                  decimals=1, barLength=0):
    """
    Print Progress Bar
    """
    percent = f'{100 * iteration / total:>3.{decimals}f}'
    if barLength == 0:
        barLength = min(os.get_terminal_size().columns, 80) \
            - len(prefix) - len(percent) - len(suffix) - 11

    filledLength = barLength * iteration // total
    bar = '#' * filledLength + '-' * (barLength - filledLength)

    if iteration == 0:
        print('')
    print(f'{prefix} |{bar}| {percent}% {suffix}', end='\r')
    if iteration == total:
        print('')


def print_cuda_tensors():
    """
    Print All CUDA Tensors
    """
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) \
                    or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(type(obj), obj.size(), obj.device)
        finally:
            pass


class MLP(nn.Module):
    """
    Multi-layer Perceptron
    """
    def __init__(self, n_input: int, n_hidden: int, n_output: int):
        super(MLP, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(n_input, n_hidden, bias=False),
            nn.BatchNorm1d(n_hidden, momentum=0.1),
            # nn.ReLU(inplace=True),
            nn.PReLU(num_parameters=1, init=0.25),
        )
        self.layer2 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(n_hidden, n_hidden),
            # nn.BatchNorm1d(n_hidden, momentum=0.1),
            # nn.ReLU(inplace=True),
            nn.PReLU(num_parameters=1, init=0.25),
        )
        self.layer3 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(n_hidden, n_hidden),
            # nn.BatchNorm1d(n_hidden),
            # nn.ReLU(inplace=True),
            nn.PReLU(num_parameters=1, init=0.25),
        )
        self.output = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(n_hidden, n_output),
            # nn.Tanh(),
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.output(x)
        return x


class HyperParameters(NamedTuple):
    """
    Hyper Parameters of NN
    """
    N_epochs = 60
    batch_size = 1900
    learning_rate = 5e-4
    N_file = 10800
    L_cut_x = 13

    n_input: int
    n_hidden: int
    n_output: int

    # scheduler step_size, gamma
    # Adam weight_decay
    # Dropout p

    def for_MLP(self) -> Tuple:
        return (self.n_input, self.n_hidden, self.n_output)


# Global Variables
hparams: HyperParameters


class NNTrainer():
    def __init__(self, DIR_TRAIN: str, DIR_TEST: str,
                 XNAME: str, YNAME: str,
                 N_freq: int, L_frame: int, L_hop: int, f_model_state=''):
        self.DIR_TRAIN = DIR_TRAIN
        self.DIR_TEST = DIR_TEST
        self.XNAME = XNAME
        self.YNAME = YNAME
        self.N_freq = N_freq
        self.L_frame = L_frame
        self.L_hop = L_hop

        global hparams
        IVDataset.L_cut_x = HyperParameters.L_cut_x
        hparams = HyperParameters(n_input=IVDataset.L_cut_x*N_freq*4,
                                  n_hidden=17*N_freq*4,
                                  n_output=N_freq*4,
                                  )

        # Dataset
        # Training + Validation Set
        self.data = IVDataset(self.DIR_TRAIN, self.XNAME, self.YNAME,
                              N_file=hparams.N_file)

        # Test Set
        data_test = IVDataset(DIR_TEST, XNAME, YNAME,
                              N_file=hparams.N_file//4,
                              doNormalize=False
                              )
        data_test.doNormalize(self.data.normalize)

        self.loader_test = DataLoader(data_test,
                                      batch_size=hparams.batch_size,
                                      shuffle=False,
                                      # collate_fn=IVDataset.my_collate,
                                      num_workers=mp.cpu_count(),
                                      )

        # Model (Using Parallel GPU)
        self.model = nn.DataParallel(MLP(*hparams.for_MLP()),
                                     output_device=1,
                                     ).cuda()
        if f_model_state:
            self.model.load_state_dict(torch.load(f_model_state))

        # MSE Loss and Adam Optimizer
        self.criterion = nn.MSELoss(reduction='sum')
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=hparams.learning_rate,
                                          weight_decay=0,
                                          )

    def train(self):
        data_train, data_valid = IVDataset.split(self.data, (0.7, -1))

        loader_train = DataLoader(data_train,
                                  batch_size=hparams.batch_size,
                                  shuffle=True,
                                  num_workers=mp.cpu_count(),
                                  # collate_fn=IVDataset.my_collate,
                                  )
        loader_valid = DataLoader(data_valid,
                                  batch_size=hparams.batch_size,
                                  shuffle=False,
                                  num_workers=mp.cpu_count(),
                                  # collate_fn=IVDataset.my_collate,
                                  )

        loss_train = np.zeros(hparams.N_epochs)
        # loss_valid = np.zeros(hparams.N_epochs)
        # snr_valid_dB = np.zeros(hparams.N_epochs)
        loss_valid = np.zeros((hparams.N_epochs, 3))
        snr_valid_dB = np.zeros((hparams.N_epochs, 3))

        scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                    step_size=1, gamma=0.8)
        for epoch in range(hparams.N_epochs):
            t_start = time.time()

            iteration = 0
            printProgress(iteration, len(loader_train), f'epoch {epoch+1:3d}:')
            scheduler.step()
            for data in loader_train:
                iteration += 1
                x_stacked = data['x_stacked']
                y_stacked = data['y_stacked']

                N_frame = x_stacked.size(0)
                # if epoch == 0:
                #     N_total_frame += N_frame

                _input = x_stacked.view(N_frame, -1).cuda(device=0)
                # ===================forward=====================
                output = self.model(_input)
                y_cuda = y_stacked.view(N_frame, -1).cuda(device=1)
                loss = self.criterion(output, y_cuda)/N_frame
                loss_train[epoch] += loss.data.cpu().item()*N_frame

                # ===================backward====================
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                printProgress(iteration, len(loader_train),
                              f'epoch {epoch+1:3d}: {loss.data.item():.1e}')
            loss_train[epoch] /= len(loader_train.dataset)

            # Validation
            loss_valid[epoch], snr_valid_dB[epoch] \
                = self.eval(loader_valid,
                            FNAME=f'MLP_result_{epoch}.mat')

            # print loss, snr and save
            print(f'Validation Loss: {NDARR_TO_STR(loss_valid[epoch])}\t'
                  f'Validation SNR (dB): {NDARR_TO_STR(snr_valid_dB[epoch])}')

            scio.savemat(f'MLP_loss_{epoch}.mat',
                         {'loss_train': loss_train,
                          'loss_valid': loss_valid,
                          'snr_valid_dB': snr_valid_dB,
                          }
                         )
            torch.save(self.model.state_dict(),
                       f'./MLP_{epoch}.pt')

            print(f'epoch {epoch+1:3d}: '
                  + time.strftime('%M min %S sec',
                                  time.gmtime(time.time()-t_start)))
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
        loss_test, snr_test_dB = self.eval(FNAME='MLP_result_test.mat')
        print('')
        print(f'Test Loss: {NDARR_TO_STR(loss_test)}\t'
              f'Test SNR (dB): {NDARR_TO_STR(snr_test_dB)}')

    def eval(self, loader: DataLoader=None, FNAME='') -> Tuple[float, float]:
        """
        Evaluate the performance of the model.
        loader: DataLoader to use.
        FNAME: filename of the result. If None, don't save the result.
        """
        if not loader:
            loader = self.loader_test

        # avg_loss = 0.
        # avg_snr =  0.
        avg_loss = np.zeros(3)
        avg_snr = np.zeros(3)
        iteration = 0
        # norm_frames = [None]*len(loader)
        # norm_errors = [None]*len(loader)
        dict_to_save = {}
        printProgress(iteration, len(loader), 'eval:')
        with torch.no_grad():
            self.model.eval()

            for data in loader:
                iteration += 1
                x_stacked = data['x_stacked']
                y_stacked_cpu = data['y_stacked']

                N_frame = x_stacked.size(0)

                _input = x_stacked.view(N_frame, -1).cuda(device=0)
                # =========================forward=============================
                output = self.model(_input)

                y_stacked = y_stacked_cpu.cuda(device=1)
                y_vec = y_stacked.view(N_frame, -1)
                loss = self.criterion(output, y_vec)/N_frame
                # =============================================================
                # Reconstruct & Performance Measure
                out_stacked = output.view(y_stacked.size())

                y_np = y_stacked_cpu.numpy()
                out_np = out_stacked.cpu().numpy()
                y_denorm = loader.dataset.denormalize(y_np, 'y')
                out_denorm = loader.dataset.denormalize(out_np, 'y')

                y_recon = IVDataset.unstack_y(y_denorm)
                out_recon = IVDataset.unstack_y(out_denorm)

                # norm_frames[iteration-1] = norm_iv(y_recon)
                # norm_errors[iteration-1] = norm_iv(out_recon-y_recon)
                # norm_frames = norm_iv(y_recon)
                # norm_errors = norm_iv(out_recon-y_recon)
                # avg_snr += (norm_frames / norm_errors).sum()
                norm_errors = norm_iv(out_np - y_np,
                                      parts=('I', 'a', 'all'))
                avg_loss += [a.sum() for a in norm_errors]

                norm_frames = norm_iv(y_recon,
                                      parts=('I', 'a', 'all'))
                norm_errors = norm_iv(out_recon - y_recon,
                                      parts=('I', 'a', 'all'))
                avg_snr += [(a/b).sum()
                            for a, b in zip(norm_frames, norm_errors)]

                # Save IV Result
                if FNAME and not dict_to_save:
                    N_frame = loader.dataset.N_frames[0]
                    x_stacked = loader.dataset.denormalize(x_stacked, 'x')
                    x_recon = IVDataset.unstack_x(x_stacked[:N_frame]).numpy()
                    dict_to_save = {'IV_free': y_recon[:, :N_frame, :],
                                    'IV_room': x_recon,
                                    'IV_estimated': out_recon[:, :N_frame, :],
                                    }

                printProgress(iteration, len(loader),
                              f'{"eval":<9}: {loss.data.item():.1e}')

            # norm_frames = np.concatenate(norm_frames)
            # norm_errors = np.concatenate(norm_errors)
            # avg_loss /= norm_frames.shape[0]
            # avg_snr = (norm_frames / norm_errors).mean()
            avg_loss /= len(loader.dataset)
            avg_snr /= len(loader.dataset)
            avg_snr_dB = 10*np.log10(avg_snr)

            if FNAME:
                # dict_to_save['norm_frames'] = norm_frames
                # dict_to_save['norm_errors'] = norm_errors
                scio.savemat(FNAME, dict_to_save)
            self.model.train()
        return avg_loss, avg_snr_dB
