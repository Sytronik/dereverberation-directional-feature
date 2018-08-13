import pdb  # noqa: F401

import numpy as np
import scipy.io as scio

import torch
from torch import nn
from torch.utils.data import DataLoader

import gc
import sys
import time
import multiprocessing as mp

from typing import NamedTuple, Tuple

from iv_dataset import IVDataset


# Progress Bar
def printProgress(iteration:int, total:int,
                  prefix='', suffix='',
                  decimals=1, barLength=57):
    percent = f'{100 * iteration / total:.{decimals}f}'
    filledLength = barLength * iteration // total
    bar = '#' * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write(f'\r{prefix} |{bar}| {percent}% {suffix}')
    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()


# Print All CUDA Tensors
def print_cuda_tensors():
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) \
                    or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(type(obj), obj.size(), obj.device)
        finally:
            pass


class MLP(nn.Module):
    def __init__(self, n_input:int, n_hidden:int, n_output:int):
        super(MLP, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(n_input, n_hidden, bias=False),
            nn.BatchNorm1d(n_hidden, momentum=0.1),
            # nn.ReLU(inplace=True),
            nn.PReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(n_hidden, n_hidden),
            # nn.BatchNorm1d(n_hidden, momentum=0.1),
            # nn.ReLU(inplace=True),
            nn.PReLU(),
        )
        self.layer3 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(n_hidden, n_hidden),
            # nn.BatchNorm1d(n_hidden),
            # nn.ReLU(inplace=True),
            nn.PReLU(),
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
    N_epochs = 100
    batch_size = 6
    learning_rate = 1e-3
    N_data = 7200
    L_cut_x = 19

    n_input = 0
    n_hidden = 0
    n_output = 0

    def for_MLP(self) -> Tuple:
        return (self.n_input, self.n_hidden, self.n_output)

    # scheduler step_size, gamma
    # Adam weight_decay
    # Dropout p


hparams = HyperParameters()


class NNTrainer():
    def __init__(self, DIR_TRAIN:str, DIR_TEST:str,
                 XNAME:str, YNAME:str,
                 N_freq:int, L_frame:int, L_hop:int, F_MODEL_STATE=''):
        self.DIR_TRAIN = DIR_TRAIN
        self.DIR_TEST = DIR_TEST
        self.XNAME = XNAME
        self.YNAME = YNAME
        self.N_freq = N_freq
        self.L_frame = L_frame
        self.L_hop = L_hop

        IVDataset.L_cut_x = hparams.L_cut_x
        hparams.n_input = hparams.L_cut_x*N_freq*4
        hparams.n_hidden = 17*N_freq*4
        hparams.n_output = N_freq*4

        # Dataset
        data = IVDataset(self.DIR_TRAIN, self.XNAME, self.YNAME,
                         N_data=hparams.N_data)

        data_test = IVDataset(DIR_TEST, XNAME, YNAME,
                              N_data=hparams.N_data//4,
                              normalize=False
                              )
        data_test.do_normalize(data.mean_x, data.mean_y,
                               data.std_x, data.std_y)

        self.loader_test = DataLoader(data_test,
                                      batch_size=1,
                                      shuffle=False,
                                      collate_fn=IVDataset.my_collate,
                                      num_workers=mp.cpu_count(),
                                      )

        # Model (Using Parallel GPU)
        self.model = nn.DataParallel(MLP(*hparams.for_MLP()),
                                     output_device=1,
                                     ).cuda()
        if F_MODEL_STATE:
            self.model.load_state_dict(torch.load(F_MODEL_STATE))

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
                                  collate_fn=IVDataset.my_collate,
                                  num_workers=mp.cpu_count(),
                                  )
        loader_valid = DataLoader(data_valid,
                                  batch_size=1,
                                  shuffle=False,
                                  collate_fn=IVDataset.my_collate,
                                  num_workers=mp.cpu_count(),
                                  )

        loss_train = np.zeros(hparams.N_epochs)
        loss_valid = np.zeros(hparams.N_epochs)
        snr_valid_dB = np.zeros(hparams.N_epochs)

        N_total_frame = 0
        scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                    step_size=1, gamma=0.8)
        for epoch in range(hparams.N_epochs):
            t_start = time.time()

            iteration = 0
            printProgress(iteration, len(loader_train), f'epoch {epoch+1:3d}:')
            scheduler.step()
            N_frame = 0
            for data in loader_train:
                iteration += 1
                x_stacked, y_stacked = data

                N_frame = x_stacked.size(0)
                if epoch == 0:
                    N_total_frame += N_frame

                input = x_stacked.view(N_frame, -1).cuda(device=0)
                # ===================forward=====================
                output = self.model(input)
                y_cuda = y_stacked.view(N_frame, -1).cuda(device=1)
                loss = self.criterion(output, y_cuda)/N_frame
                loss_train[epoch] += loss.data.cpu().item()*N_frame

                # ===================backward====================
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                printProgress(iteration, len(loader_train),
                              f'epoch {epoch+1:3d}: {loss.data.item():.1e}')
            loss_train[epoch] /= N_total_frame
            # ===================log========================
            # print(('epoch [{}/{}]: loss of the last data: {:.2e}')
            #       .format(epoch + 1, hparams.N_epochs,
            #               loss.data.item()/N_frame))

            loss_valid[epoch], snr_valid_dB[epoch] \
                = self.eval(loader_valid,
                            FNAME=f'MLP_result_{epoch}.mat')
            print(f'Validation Loss: {loss_valid[epoch]:.2e}', end='\t')
            print(f'Validation SNR (dB): {snr_valid_dB[epoch]:.2e}')

            scio.savemat(f'MLP_loss_{epoch}.mat',
                         {'loss_train': loss_train,
                          'loss_valid': loss_valid,
                          'snr_valid_dB': snr_valid_dB,
                          }
                         )
            torch.save(self.model.state_dict(),
                       f'./MLP_{epoch}.pt')

            print(time.strftime('%M min %S sec',
                                time.gmtime(time.time()-t_start)))
            # Early Stopping
            if epoch >= 4:
                loss_max = loss_valid[epoch-4:epoch+1].max()
                loss_min = loss_valid[epoch-4:epoch+1].min()
                if loss_max - loss_min < 0.01 * loss_max:
                    print('Early Stopped')
                    break

        loss_test, snr_test_dB = self.eval(FNAME='MLP_result_test.mat')
        print(f'\nTest Loss: {loss_test:.2e}', end='\t')
        print(f'Test SNR (dB): {snr_test_dB:.2e}')

    def eval(self, loader:DataLoader=None, FNAME='') -> Tuple[float, float]:
        if not loader:
            loader = self.loader_test
        avg_loss = 0.
        avg_snr = 0.
        iteration = 0
        N_total_frame = 0
        norm_frames = [None]*len(loader)
        norm_errors = [None]*len(loader)
        dict_to_save = {}
        printProgress(iteration, len(loader), 'eval:')
        with torch.no_grad():
            self.model.eval()

            for data in loader:
                iteration += 1
                x_stacked, y_stacked = data

                N_frame = x_stacked.size(0)
                N_total_frame += N_frame

                input = x_stacked.view(N_frame, -1).cuda(device=0)
                # ===================forward=====================
                output = self.model(input)

                y_stacked = y_stacked.cuda(device=1)
                y_vec = y_stacked.view(N_frame, -1)
                loss = self.criterion(output, y_vec)/N_frame

                output_stacked = output.view(y_stacked.size())

                y_unstack = IVDataset.unstack_y(y_stacked)
                output_unstack = IVDataset.unstack_y(output_stacked)
                y_np = y_unstack.cpu().numpy()
                output_np = output_unstack.cpu().numpy()

                # y_recon = np.arctanh(y_np)
                # output_recon = np.arctanh(output_np)
                y_recon = loader.dataset.std_y*y_np + loader.dataset.mean_y
                output_recon \
                    = loader.dataset.std_y*output_np + loader.dataset.mean_y

                # y_recon = (y_recon*x)
                # output_recon = (output_recon*x)
                norm_frames[iteration-1] = (y_recon**2).sum(axis=(0,-1))
                norm_errors[iteration-1] \
                    = ((output_recon-y_recon)**2).sum(axis=(0,-1))

                if FNAME and not dict_to_save:
                    x_np = IVDataset.unstack_x(
                        loader.dataset.std_x*x_stacked.numpy()
                        + loader.dataset.mean_x
                    )
                    # x_recon = np.arctanh(x_np)
                    # x_recon = loader.dataset.std_x*x_np \
                    #     + loader.dataset.mean_x
                    x_recon = x_np
                    dict_to_save = {'IV_free':y_recon,
                                    'IV_room':x_recon,
                                    'IV_estimated':output_recon,
                                    }

                avg_loss += loss.data.cpu().item()*N_frame
                printProgress(iteration, len(loader),
                              f"{'eval':^9}: {loss.data.item():.1e}")

            avg_loss /= N_total_frame
            norm_frames = np.concatenate(norm_frames)
            norm_errors = np.concatenate(norm_errors)
            avg_snr = (norm_frames / norm_errors).sum()/N_total_frame
            avg_snr_dB = 10*np.log10(avg_snr)
            if np.isnan(avg_snr_dB):
                pdb.set_trace()

            if FNAME:
                dict_to_save['norm_frames'] = norm_frames
                dict_to_save['norm_errors'] = norm_errors
                scio.savemat(FNAME, dict_to_save)
            self.model.train()
        return avg_loss, avg_snr_dB
