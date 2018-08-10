import pdb  # noqa: F401

import numpy as np
import scipy.io as scio

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

import os
from glob import glob
import gc
import sys
import copy
import time
import multiprocessing as mp


# Progress Bar
def printProgress(iteration, total,
                  prefix='', suffix='',
                  decimals=1, barLength=57):
    formatStr = "{0:." + str(decimals) + "f}"
    percent = formatStr.format(100 * (iteration / float(total)))
    filledLength = int(round(barLength * iteration / float(total)))
    bar = '#' * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percent, '%', suffix))
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


class IVDataset(Dataset):
    L_cut_x = 1
    L_cut_y = 1

    def __init__(self, DIR:str, XNAME:str, YNAME:str,
                 N_data=-1, normalize=True):
        self.DIR = DIR
        self.XNAME = XNAME
        self.YNAME = YNAME
        self.normalize = normalize

        # for file in os.scandir(DIR):
        self.all_files = glob(os.path.join(DIR,'*.npy'))
        if N_data != -1:
            self.all_files = self.all_files[:N_data]
        for file in self.all_files[:]:
            if file.endswith('metadata.npy'):
                self.all_files.remove(file)

        # Calculate summation & no. of total frames (parallel)
        if normalize:
            N_CORES = mp.cpu_count()
            pool = mp.Pool(N_CORES)
            result = pool.map(IVDataset.sum_files,
                              [(file, XNAME, YNAME)
                               for file in self.all_files])

            sum_x = np.sum([res[0] for res in result], axis=0
                           )[np.newaxis,:,:,:]
            N_frame_x = np.sum([res[1] for res in result])
            sum_y = np.sum([res[2] for res in result], axis=0
                           )[:,np.newaxis,:]
            N_frame_y = np.sum([res[3] for res in result])

            # mean
            self.mean_x = sum_x / N_frame_x
            self.mean_y = sum_y / N_frame_y
            # self.mean_x = (sum_x+sum_y) / (N_frame_x + N_frame_y)
            # self.mean_y = (sum_x+sum_y) / (N_frame_x + N_frame_y)

            # Calculate Standard Deviation
            result = pool.map(IVDataset.sum_dev_files,
                              [(file, XNAME, YNAME, self.mean_x, self.mean_y)
                               for file in self.all_files])

            pool.close()

            sum_dev_x = np.sum([res[0] for res in result], axis=0
                               )[np.newaxis,:,:,:]
            sum_dev_y = np.sum([res[1] for res in result], axis=0
                               )[:,np.newaxis,:]

            self.std_x = np.sqrt(sum_dev_x / N_frame_x + 1e-5)
            self.std_y = np.sqrt(sum_dev_y / N_frame_y + 1e-5)
            # self.std_x = np.sqrt((sum_dev_x + sum_dev_y)
            #                      /(N_frame_x + N_frame_y)
            #                      + 1e-5)
            # self.std_y = np.sqrt((sum_dev_x + sum_dev_y)
            #                      /(N_frame_x + N_frame_y)
            #                      + 1e-5)
        else:
            self.mean_x = 0.
            self.mean_y = 0.
            self.std_x = 1.
            self.std_y = 1.

        print('{} data prepared from {}.'.format(len(self),
                                                 os.path.basename(DIR)))

    @classmethod
    def sum_files(cls, tup):
        file, XNAME, YNAME = tup
        try:
            data_dict = np.load(file).item()
            x = data_dict[XNAME]
            y = data_dict[YNAME]
        except:  # noqa: E722
            pdb.set_trace()

        x_stacked = cls.stack_x(x, L_trunc=y.shape[1])

        return (x_stacked.sum(axis=0), x_stacked.shape[0],
                y.sum(axis=1), y.shape[1],
                )

    @classmethod
    def sum_dev_files(cls, tup):
        file, XNAME, YNAME, mean_x, mean_y = tup
        data_dict = np.load(file).item()
        x = data_dict[XNAME]
        y = data_dict[YNAME]

        x_stacked = cls.stack_x(x, L_trunc=y.shape[1])

        return (((x_stacked - mean_x)**2).sum(axis=0),
                ((y - mean_y)**2).sum(axis=1),
                )

    def do_normalize(self, mean_x, mean_y, std_x, std_y):
        self.normalize = True
        self.mean_x = mean_x
        self.mean_y = mean_y
        self.std_x = std_x
        self.std_y = std_y

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        # File Open
        data_dict = np.load(self.all_files[idx]).item()
        x = data_dict[self.XNAME]
        y = data_dict[self.YNAME]

        # Stack & Normalize
        x_stacked = IVDataset.stack_x(x, L_trunc=y.shape[1])
        if self.normalize:
            x_stacked = (x_stacked - self.mean_x)/self.std_x
            y = (y - self.mean_y)/self.std_y
        y_stacked = IVDataset.stack_y(y)

        x_stacked = torch.from_numpy(x_stacked).float()
        y_stacked = torch.from_numpy(y_stacked).float()
        sample = {'x_stacked': x_stacked, 'y_stacked': y_stacked}

        return sample

    # Make groups of the frames of x and stack the groups
    # x_stacked: (time_length) x (N_freq) x (L_cut_x) x (XYZ0 channel)
    @classmethod
    def stack_x(cls, x, L_trunc=0):
        if x.ndim != 3:
            raise Exception('Dimension Mismatch')
        if cls.L_cut_x == 1:
            return x

        L0, L1, L2 = x.shape

        half = int(cls.L_cut_x/2)

        x = np.concatenate((np.zeros((L0, half, L2)),
                            x,
                            np.zeros((L0, half, L2))),
                           axis=1)

        if L_trunc != 0:
            L1 = L_trunc

        return np.stack([x[:, ii - half:ii + half + 1, :]
                         for ii in range(half, half + L1)
                         ])

    @classmethod
    def stack_y(cls, y):
        if y.ndim != 3:
            raise Exception('Dimension Mismatch')

        return y.transpose((1, 0, 2))[:,:,np.newaxis,:]

    @classmethod
    def unstack_x(cls, x):
        if type(x) == torch.Tensor:
            if x.dim() != 4 or x.size(2) <= int(cls.L_cut_x/2):
                raise Exception('Dimension/Size Mismatch')
            x = x[:,:,int(cls.L_cut_x/2),:].squeeze()
            return x.transpose(1, 0)
        else:
            if x.ndim != 4 or x.shape[2] <= int(cls.L_cut_x/2):
                raise Exception('Dimension/Size Mismatch')
            x = x[:,:,int(cls.L_cut_x/2),:].squeeze()
            return x.transpose((1, 0, 2))

    @classmethod
    def unstack_y(cls, y):
        if type(y) == torch.Tensor:
            if y.dim() != 4 or y.size(2) != 1:
                raise Exception('Dimension/Size Mismatch')
            return y.squeeze().transpose(1, 0)
        else:
            if y.ndim != 4 or y.shape[2] != 1:
                raise Exception('Dimension/Size Mismatch')
            return y.squeeze().transpose((1, 0, 2))

    @staticmethod
    def my_collate(batch):
        x_stacked = torch.cat([item['x_stacked'] for item in batch])
        y_stacked = torch.cat([item['y_stacked'] for item in batch])
        return [x_stacked, y_stacked]

    @classmethod
    def split(cls, a, ratio):
        if type(a) != cls:
            raise TypeError
        n_split = len(ratio)
        ratio = np.array(ratio)
        mask = (ratio == -1)
        ratio[np.where(mask)] = 0
        if mask.sum() > 1:
            raise Exception('Only one element of the parameter \'ratio\'' +
                            'can have the value of -1')
        if ratio.sum() >= 1:
            raise Exception('The sum of ratio must be 1')
        if mask.sum() == 1:
            ratio[np.where(mask)] = 1 - ratio.sum()

        idx_data = np.cumsum(np.insert(ratio, 0, 0) * len(a), dtype=int)
        result = [copy.copy(a) for ii in range(n_split)]
        all_f_per = np.random.permutation(a.all_files)
        for ii in range(n_split):
            result[ii].all_files = all_f_per[idx_data[ii]:idx_data[ii + 1]]

        return result


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


class NNTrainer():
    N_epochs = 100
    batch_size = 6
    learning_rate = 1e-3
    N_data = 7200

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

        L_cut_x = 19
        IVDataset.L_cut_x = L_cut_x
        n_input = L_cut_x*N_freq*4
        n_hidden = 17*N_freq*4
        n_output = N_freq*4

        # Test Dataset
        data = IVDataset(self.DIR_TRAIN, self.XNAME, self.YNAME,
                         N_data=NNTrainer.N_data)

        data_test = IVDataset(DIR_TEST, XNAME, YNAME,
                              N_data=int(NNTrainer.N_data/4),
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
        self.model = nn.DataParallel(MLP(n_input, n_hidden, n_output),
                                     output_device=1,
                                     ).cuda()
        if F_MODEL_STATE:
            self.model.load_state_dict(torch.load(F_MODEL_STATE))

        # MSE Loss and Adam Optimizer
        self.criterion = nn.MSELoss(reduction='sum')
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=NNTrainer.learning_rate,
                                          weight_decay=0,
                                          )

    def train(self):
        data_train, data_valid = IVDataset.split(self.data, (0.7, -1))

        loader_train = DataLoader(data_train,
                                  batch_size=NNTrainer.batch_size,
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

        loss_train = np.zeros(NNTrainer.N_epochs)
        loss_valid = np.zeros(NNTrainer.N_epochs)
        snr_valid_dB = np.zeros(NNTrainer.N_epochs)

        N_total_frame = 0
        scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                    step_size=1, gamma=0.8)
        for epoch in range(NNTrainer.N_epochs):
            t_start = time.time()

            iteration = 0
            printProgress(iteration, len(loader_train),
                          'epoch [{}/{}]:'.format(epoch+1, NNTrainer.N_epochs))
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
                              'epoch [{}/{}]:{:.1e}'
                              .format(epoch+1, NNTrainer.N_epochs,
                                      loss.data.item()))
            loss_train[epoch] /= N_total_frame
            # ===================log========================
            # print(('epoch [{}/{}]: loss of the last data: {:.2e}')
            #       .format(epoch + 1, NNTrainer.N_epochs,
            #               loss.data.item()/N_frame))

            loss_valid[epoch], snr_valid_dB[epoch] \
                = self.eval(loader_valid,
                            FNAME='MLP_result_{}.mat'.format(epoch))
            print('Validation Loss: {:.2e}'.format(loss_valid[epoch]),end='\t')
            print('Validation SNR (dB): {:.2e}'.format(snr_valid_dB[epoch]))

            scio.savemat('MLP_loss_{}.mat'.format(epoch),
                         {'loss_train': loss_train,
                          'loss_valid': loss_valid,
                          'snr_valid_dB': snr_valid_dB,
                          }
                         )
            torch.save(self.model.state_dict(),
                       './MLP_{}.pt'.format(epoch))

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
        print('\nTest Loss: {:.2e}'.format(loss_test))
        print('Test SNR (dB): {:.2e}'.format(snr_test_dB))

    def eval(self, loader=None, FNAME=''):
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
                              'eval:{:.1e}'.format(loss.data.item()))
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
