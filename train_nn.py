import pdb  # noqa: F401

import numpy as np

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

import os
import gc
import sys
import copy
import time


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
    def __init__(self, DIR:str, XNAME:str, YNAME:str,
                 L_cut_x, L_cut_y=1, N_data=-1):
        self.DIR = DIR
        self.XNAME = XNAME
        self.YNAME = YNAME

        self.all_files = []
        len = 0
        for file in os.scandir(DIR):
            if file.is_file() and file.name.endswith('.npy'):
                self.all_files.append(file.path)
                len += 1
            if len == N_data:
                break

        self.L_cut_x = L_cut_x
        self.L_cut_y = L_cut_y

        print('{} data prepared from {}.'.format(len, os.path.basename(DIR)))
    #     self._rand_idx = self.shuffle()
    #
    # def shuffle(self):
    #     idx = list(range(self.__len__))
    #     return random.shuffle(idx)

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        # File Open
        try:
            data_dict = np.load(self.all_files[idx]).item()
            x = data_dict[self.XNAME]
            y = data_dict[self.YNAME]
        except:  # noqa: E722
            pdb.set_trace()

        # x[:,:,3] = np.log10(x[:,:,3] + 1)
        # y[:,:,3] = np.log10(y[:,:,3] + 1)
        # xy = np.concatenate([x, y], axis=1)
        # x = (x - xy.min())/(xy.max() - xy.min())
        # y = (y - xy.min())/(xy.max() - xy.min())
        x[:,:,:] = np.tanh(x[:,:,:])
        y[:,:,:] = np.tanh(y[:,:,:])

        N_freq = x.shape[0]
        N_ch = x.shape[-1]

        # # Zero-Padding for the same length of x and y
        # if x.shape[1] < y.shape[1]:
        #     x = np.concatenate(
        #         (x, np.zeros((N_freq, y.shape[1]-x.shape[1], N_ch))),
        #         axis=1
        #     )
        # elif x.shape[1] > y.shape[1]:
        #     y = np.concatenate(
        #         (y, np.zeros((N_freq, x.shape[1]-y.shape[1], N_ch))),
        #         axis=1
        #     )

        # Zero-Padding for grouping of frames
        half = int(self.L_cut_x/2)
        if self.L_cut_x > 1:
            x = np.concatenate(
                (np.zeros((N_freq, half, N_ch)),
                 x,
                 np.zeros((N_freq, half - 1, N_ch))
                 ),
                axis=1
            )

        # if self.L_cut_y > 1:
        #     y=np.concatenate((np.zeros(N_fft, int(self.L_cut_y/2)),
        #                     y,
        #                     np.zeros(N_fft, int(self.L_cut_y/2-1))), axis=1)

        # Make groups of the frames of x and stack the groups
        # x_stacked: (time_length) x (N_freq) x (L_cut_x) x (XYZ0 channel)
        # y: (time_length) x (N_freq) x 1 x (XYZ0 channel)
        length = y.shape[1]

        x_stacked = np.stack([x[:, ii - half:ii + half, :]
                              for ii in range(half, half + length)
                              ])
        y = y.transpose((1, 0, 2)).reshape(length, N_freq, 1, -1)

        x_stacked = torch.tensor(x_stacked[:y.shape[0],:,:,:],
                                 dtype=torch.float)
        y = torch.tensor(y, dtype=torch.float)
        sample = {'x_stacked': x_stacked, 'y': y}

        return sample

    @staticmethod
    def my_collate(batch):
        x_stacked = torch.cat([item['x_stacked'] for item in batch])
        y = torch.cat([item['y'] for item in batch])
        return [x_stacked, y]

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
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(n_hidden, n_hidden,),  # bias=False),
            nn.Dropout(p=0.5),
            # nn.BatchNorm1d(n_hidden, momentum=0.1),
            nn.ReLU(inplace=True),
        )
        # self.layer3 = nn.Sequential(
        #     nn.Dropout(p=0.5),
        #     nn.Linear(n_hidden, n_hidden,),# bias=False),
        #     # nn.BatchNorm1d(n_hidden),
        #     nn.ReLU(True)
        #     # nn.PReLU()
        # )
        self.output = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(n_hidden, n_output),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        # x = self.layer3(x)
        x = self.output(x)
        return x


class NNTrainer():
    N_epochs = 50
    batch_size = 6
    learning_rate = 1e-3
    N_data = 7200

    def __init__(self, DIR_TRAIN:str, DIR_TEST:str,
                 XNAME:str, YNAME:str,
                 N_fft:int, L_frame:int, L_hop:int, F_MODEL_STATE=''):
        self.DIR_TRAIN = DIR_TRAIN
        self.DIR_TEST = DIR_TEST
        self.XNAME = XNAME
        self.YNAME = YNAME
        self.N_fft = N_fft
        self.L_frame = L_frame
        self.L_hop = L_hop

        L_cut_x = int(3840/L_hop)
        self.L_cut_x = L_cut_x
        n_input = int(L_cut_x*N_fft/2*4)
        n_hidden = int(3520/L_hop*N_fft/2*4)
        n_output = int(N_fft/2*4)

        # Test Dataset
        data_test = IVDataset(DIR_TEST, XNAME, YNAME,
                              L_cut_x, N_data=NNTrainer.N_data/4)
        self.loader_test = DataLoader(data_test,
                                      batch_size=1,
                                      shuffle=False,
                                      collate_fn=IVDataset.my_collate)

        # Model (Using Parallel GPU)
        self.model = nn.DataParallel(MLP(n_input, n_hidden, n_output)).cuda()
        if F_MODEL_STATE != '':
            self.model.load_state_dict(torch.load(F_MODEL_STATE))

        # MSE Loss and Adam Optimizer
        self.criterion = nn.MSELoss(size_average=False)
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=NNTrainer.learning_rate,
                                          weight_decay=1e-5)

    def train(self):
        data = IVDataset(self.DIR_TRAIN, self.XNAME, self.YNAME,
                         self.L_cut_x, N_data=NNTrainer.N_data)
        data_train, data_valid = IVDataset.split(data, (0.7, -1))
        del data

        loader_train = DataLoader(data_train,
                                  batch_size=NNTrainer.batch_size,
                                  shuffle=True,
                                  collate_fn=IVDataset.my_collate)

        loader_valid = DataLoader(data_valid,
                                  batch_size=1,
                                  shuffle=False,
                                  collate_fn=IVDataset.my_collate)

        loss_train = np.zeros(NNTrainer.N_epochs)
        loss_valid = np.zeros(NNTrainer.N_epochs)
        error_valid = np.zeros(NNTrainer.N_epochs)
        N_total_frame = 0
        scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                    step_size=2, gamma=0.5)
        for epoch in range(NNTrainer.N_epochs):
            t_start = time.time()

            iteration = 0
            printProgress(iteration, len(loader_train),
                          'epoch [{}/{}]:'.format(epoch+1, NNTrainer.N_epochs))
            scheduler.step()
            y = None
            for data in loader_train:
                iteration += 1
                x_stacked, y = data
                input = x_stacked.view(x_stacked.size(0), -1).cuda()
                del x_stacked
                # ===================forward=====================
                output = self.model(input)
                loss = self.criterion(output, y.view(y.size(0), -1).cuda())
                loss_train[epoch] += loss.data.cpu().item()
                if epoch == 0:
                    N_total_frame += y.size(0)
                # ===================backward====================
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                printProgress(iteration, len(loader_train),
                              'epoch [{}/{}]:{:.1e}'
                              .format(epoch+1, NNTrainer.N_epochs,
                                      loss.data.item()/y.size(0)))
            loss_train[epoch] /= N_total_frame
            # ===================log========================
            print(('epoch [{}/{}]: loss of the last data: {:.2e}')
                  .format(epoch + 1, NNTrainer.N_epochs,
                          loss.data.item()/y.size(0)))

            loss_valid[epoch], error_valid[epoch] = self.eval(loader_valid)
            print('Validation Loss: {:.2e}'.format(loss_valid[epoch]))
            print('Validation Error rate: {:.2e}'.format(error_valid[epoch]))

            np.save('loss_epoch_{}.npy'.format(epoch),
                    {'loss_train': loss_train, 'loss_valid': loss_valid,
                     'error_valid': error_valid})
            torch.save(self.model.state_dict(),
                       './MLP_epoch_{}.pth'.format(epoch))

            print(time.strftime('%M min %S sec',
                                time.gmtime(time.time()-t_start)))
            # Early Stopping
            if epoch >= 2:
                loss_max = loss_valid[epoch-2:epoch+1].max()
                loss_min = loss_valid[epoch-2:epoch+1].min()
                if loss_max - loss_min < 0.1 * loss_valid[epoch]:
                    print('Early Stopped')
                    break
            # if epoch % 10 == 0:
            #     mat = to_mat(output.cpu().data, y.size(2))
            #     os.path.join(DIR_IV, 'MLP_out/')
            #     save_image(mat, './MLP_img/image_{}.png'.format(epoch))
        loss_test, error_test = self.eval()
        print('\nTest Loss: {:.2e}'.format(loss_test))
        print('Test Error rate: {:.2e}'.format(error_test))

    def eval(self, loader=None, save_one=True):
        if not loader:
            loader = self.loader_test
        avg_loss = 0
        avg_error = 0
        iteration = 0
        N_total_frame = 0
        printProgress(iteration, len(loader), 'eval:')
        with torch.no_grad():
            self.model.eval()
            for data in loader:
                iteration += 1
                x_stacked, y = data
                input = x_stacked.view(x_stacked.size(0), -1).cuda()
                del x_stacked
                # ===================forward=====================
                output = self.model(input)

                if save_one:
                    np.save('test_img.npy',
                            {'IV_estimated':
                             output.view(y.size(0), y.size(1), -1)
                                .cpu().numpy().transpose((1, 0, 2)),
                             'IV_free':
                             y.view(y.size(0), y.size(1), -1)
                                .numpy().transpose((1, 0, 2))}
                            )
                    save_one = False

                y = y.view(y.size(0), -1).cuda()
                loss = self.criterion(output, y)

                # output = output.view(y.size())
                error = (((output-y)**2).sum(-1)/(y**2).sum(-1)).sum()
                N_total_frame += y.size(0)

                avg_loss += loss.data.cpu().item()
                avg_error += error.data.cpu().item()
                printProgress(iteration, len(loader),
                              'eval:{:.1e}'.format(loss.data.item()/y.size(0)))
            avg_loss /= N_total_frame
            avg_error /= N_total_frame
            # ===================log========================

            self.model.train()
        return avg_loss, avg_error
