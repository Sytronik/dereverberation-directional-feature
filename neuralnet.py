import pdb  # noqa: F401

import numpy as np
import scipy.io as scio
import deepdish as dd

import torch
from torch import nn
from torch.utils.data import DataLoader

import gc
import os
import time
from multiprocessing import cpu_count  # noqa: F401

from typing import NamedTuple, Tuple
from collections import OrderedDict

from iv_dataset import IVDataset, norm_iv


NUM_WORKERS = cpu_count()
# NUM_WORKERS = 0


def array2string(a):
    return np.array2string(a, formatter={'float_kind': lambda x: f'{x:.2e}'})


def printProgress(iteration: int, total: int, prefix='', suffix='',
                  decimals=1, barLength=0):
    """
    Print Progress Bar
    """
    percent = f'{100 * iteration / total:>{decimals+4}.{decimals}f}'
    if barLength == 0:
        barLength = min(os.get_terminal_size().columns, 80) \
            - len(prefix) - len(percent) - len(suffix) - 11

    filledLength = barLength * iteration // total
    bar = '#' * filledLength + '-' * (barLength - filledLength)

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
    def __init__(self, n_input: int, n_hidden: int, n_output: int, p: float):
        super(MLP, self).__init__()

        self.layer1 = nn.Sequential(OrderedDict([
            ('fc', nn.Linear(n_input, n_hidden, bias=False)),
            ('bn', nn.BatchNorm1d(n_hidden, momentum=0.1)),
            # ('act', nn.ReLU(inplace=True)),
            ('act', nn.PReLU(num_parameters=1, init=0.25)),
        ]))
        self.layer2 = nn.Sequential(OrderedDict([
            # ('do', nn.Dropout(p=p)),
            ('fc', nn.Linear(n_hidden, n_hidden)),
            ('bn', nn.BatchNorm1d(n_hidden, momentum=0.1)),
            # ('act', nn.ReLU(inplace=True)),
            ('act', nn.PReLU(num_parameters=1, init=0.25)),
        ]))
        self.layer3 = nn.Sequential(OrderedDict([
            # ('do', nn.Dropout(p=p)),
            ('fc', nn.Linear(n_hidden, n_hidden)),
            ('bn', nn.BatchNorm1d(n_hidden, momentum=0.1)),
            # nn.ReLU(inplace=True),
            ('act', nn.PReLU(num_parameters=1, init=0.25)),
        ]))
        self.output = nn.Sequential(OrderedDict([
            ('do', nn.Dropout(p=p)),
            ('fc', nn.Linear(n_hidden, n_output)),
        ]))

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
    N_epochs = 100
    batch_size = 2000
    learning_rate = 5e-4
    N_file = 100
    L_cut_x = 13

    n_per_frame: int
    p = 0.5  # Dropout p

    # lr scheduler
    step_size = 1
    gamma = 0.9

    weight_decay = 1e-8  # Adam weight_decay

    def for_MLP(self) -> Tuple:
        n_input = self.L_cut_x * self.n_per_frame
        n_hidden = 17 * self.n_per_frame
        n_output = self.n_per_frame
        return (n_input, n_hidden, n_output, self.p)


# Global Variables
hparams: HyperParameters


class MultipleOptimizer(object):
    def __init__(self, *op):
        self.optimizers = op

    def zero_grad(self):
        for op in self.optimizers:
            op.zero_grad()

    def step(self):
        for op in self.optimizers:
            op.step()

    def __len__(self):
        return len(self.optimizers)

    def __getitem__(self, idx: int):
        return self.optimizers[idx]


class MultipleScheduler(object):
    def __init__(self, cls_scheduler: type,
                 optimizers: MultipleOptimizer, **kwargs):
        self.schedulers = []
        for op in optimizers:
            self.schedulers.append(cls_scheduler(op, **kwargs))

    def step(self):
        for sch in self.schedulers:
            sch.step()

    def __len__(self):
        return len(self.schedulers)

    def __getitem__(self, idx: int):
        return self.schedulers[idx]


class NNTrainer():
    global hparams

    def __init__(self, DIR_TRAIN: str, DIR_TEST: str,
                 XNAME: str, YNAME: str, f_model_state=''):
        self.DIR_TRAIN = DIR_TRAIN
        self.DIR_TEST = DIR_TEST
        self.XNAME = XNAME
        self.YNAME = YNAME
        self.DIR_RESULT = './MLP/MLP_'

        if not os.path.isdir(self.DIR_RESULT):
            os.makedirs(self.DIR_RESULT)
        dd.io.save(self.DIR_RESULT+'hparams.h5', dict(hparams._asdict()))

        IVDataset.L_cut_x = hparams.L_cut_x

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
                                      num_workers=NUM_WORKERS,
                                      )

        # Model (Using Parallel GPU)
        self.model = nn.DataParallel(MLP(*hparams.for_MLP()),
                                     output_device=1,
                                     ).cuda()
        if f_model_state:
            self.model.load_state_dict(torch.load(f_model_state))

        # MSE Loss
        self.criterion = nn.MSELoss(reduction='sum')

        # Optimizer
        param1 = [value
                  for key, value in self.model.named_parameters()
                  if 'act' not in key]
        param2 = [value
                  for key, value in self.model.named_parameters()
                  if 'act' in key]
        self.optimizer = MultipleOptimizer(
            torch.optim.Adam(param1,
                             lr=hparams.learning_rate,
                             weight_decay=hparams.weight_decay,
                             ),
            torch.optim.Adam(param2,
                             lr=hparams.learning_rate,
                             ),
        )

    def train(self):
        data_train, data_valid = IVDataset.split(self.data, (0.7, -1))

        loader_train = DataLoader(data_train,
                                  batch_size=hparams.batch_size,
                                  shuffle=True,
                                  num_workers=NUM_WORKERS,
                                  # collate_fn=IVDataset.my_collate,
                                  )
        loader_valid = DataLoader(data_valid,
                                  batch_size=hparams.batch_size,
                                  shuffle=False,
                                  num_workers=NUM_WORKERS,
                                  # collate_fn=IVDataset.my_collate,
                                  )

        loss_train = np.zeros(hparams.N_epochs)
        # loss_valid = np.zeros(hparams.N_epochs)
        # snr_seg_valid = np.zeros(hparams.N_epochs)
        loss_valid = np.zeros((hparams.N_epochs, 3))
        snr_seg_valid = np.zeros((hparams.N_epochs, 3))

        scheduler \
            = MultipleScheduler(torch.optim.lr_scheduler.StepLR,
                                self.optimizer,
                                step_size=hparams.step_size,
                                gamma=hparams.gamma)

        for epoch in range(hparams.N_epochs):
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
            print(f'Training Loss: {loss_train[epoch]:.2e}')

            # Validation
            loss_valid[epoch], snr_seg_valid[epoch] \
                = self.eval(loader_valid,
                            FNAME=f'{self.DIR_RESULT}IV_{epoch}.mat')

            # print loss, snr and save
            print(f'Validation Loss: {loss_valid[epoch, 2]:.2e}\t'
                  f'Validation SNRseg (dB): {snr_seg_valid[epoch, 2]:.2e}')

            scio.savemat(f'{self.DIR_RESULT}loss_{epoch}.mat',
                         {'loss_train': loss_train,
                          'loss_valid': loss_valid,
                          'snr_seg_valid': snr_seg_valid,
                          }
                         )
            torch.save(self.model.state_dict(),
                       f'{self.DIR_RESULT}{epoch}.pt')

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
        loss_test, snr_test_dB \
            = self.eval(FNAME=f'{self.DIR_RESULT}IV_test.mat')
        print('')
        print(f'Test Loss: {array2string(loss_test)}\t'
              f'Test SNR (dB): {array2string(snr_test_dB)}')

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
        snr_seg = np.zeros(3)
        iteration = 0
        norm_frames = [None]*len(loader)
        norm_errors = [None]*len(loader)
        dict_to_save = {}
        printProgress(iteration, len(loader), 'eval:')
        with torch.no_grad():
            self.model.eval()

            for data in loader:
                iteration += 1
                x_stacked = data['x']
                y_stacked_cpu = data['y']

                N_frame = x_stacked.size(0)

                _input = x_stacked.view(N_frame, -1).cuda(device=0)
                # =========================forward=============================
                output = self.model(_input)

                y_stacked = y_stacked_cpu.cuda(device=1)
                # y_vec = y_stacked.view(N_frame, -1)
                # loss = self.criterion(output, y_vec)/N_frame
                # =============================================================
                # Reconstruct & Performance Measure
                out_stacked = output.view(y_stacked.size())

                y_np = y_stacked_cpu.numpy()
                out_np = out_stacked.cpu().numpy()
                y_denorm = loader.dataset.denormalize(y_np, 'y')
                out_denorm = loader.dataset.denormalize(out_np, 'y')

                y_recon = IVDataset.unstack_y(y_denorm)
                out_recon = IVDataset.unstack_y(out_denorm)

                norm_normalized = norm_iv(out_np - y_np,
                                          parts=('I', 'a', 'all'))  # 3 x t
                loss = norm_normalized.sum(axis=1)
                avg_loss += loss

                # 3 x t x f
                norm_frames[iteration-1] = norm_iv(y_recon,
                                                   keep_freq_axis=True,
                                                   parts=('I', 'a', 'all'))
                norm_errors[iteration-1] = norm_iv(out_recon - y_recon,
                                                   keep_freq_axis=True,
                                                   parts=('I', 'a', 'all'))

                sum_frames = norm_frames[iteration-1].sum(axis=2)
                sum_errors = norm_errors[iteration-1].sum(axis=2)
                snr_seg += 10*np.log10(sum_frames/sum_errors).sum(axis=1)

                # Save IV Result
                if FNAME and not dict_to_save:
                    N_first = loader.dataset.N_frames[0]
                    x_stacked = loader.dataset.denormalize(x_stacked, 'x')
                    x_recon = IVDataset.unstack_x(x_stacked[:N_first]).numpy()
                    dict_to_save = {'IV_free': y_recon[:, :N_first, :],
                                    'IV_room': x_recon,
                                    'IV_estimated': out_recon[:, :N_first, :],
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
            self.model.train()
        return avg_loss, snr_seg
