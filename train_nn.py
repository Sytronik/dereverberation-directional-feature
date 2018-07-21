import os
import pdb

import numpy as np

import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image

import gc
import sys


def printProgress (iteration, total,
                   prefix = '', suffix = '',
                   decimals = 1, barLength = 57):
    formatStr = "{0:." + str(decimals) + "f}"
    percent = formatStr.format(100 * (iteration / float(total)))
    filledLength = int(round(barLength * iteration / float(total)))
    bar = '#' * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percent, '%', suffix))
    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()


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
        self.len = 0

        for file in os.scandir(DIR):
            if file.is_file() and file.name.endswith('.npy'):
                self.all_files.append(file.path)
                self.len += 1
            if self.len == N_data:
                break;

        self.L_cut_x = L_cut_x
        self.L_cut_y = L_cut_y

        print('{} data prepared.'.format(len(self.all_files)))
    #     self._rand_idx = self.shuffle()
    #
    # def shuffle(self):
    #     idx = list(range(self.__len__))
    #     return random.shuffle(idx)

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        try:
            data_dict = np.load(self.all_files[idx]).item()
            x = data_dict[self.XNAME]
            y = data_dict[self.YNAME]
        except:
            pdb.set_trace()

        N_freq = x.shape[0]
        channel = x.shape[2]

        if x.shape[1] < y.shape[1]:
            x = np.concatenate(
                (x, np.zeros((N_freq, y.shape[1]-x.shape[1], channel))),
                axis=1
            )
        elif x.shape[1] > y.shape[1]:
            y = np.concatenate(
                (y, np.zeros((N_freq, x.shape[1]-y.shape[1], channel))),
                axis=1
            )

        len = x.shape[1]
        if self.L_cut_x > 1:
            x = np.concatenate(
                (np.zeros((N_freq, int(self.L_cut_x/2), channel)),
                 x,
                 np.zeros((N_freq, int(self.L_cut_x/2-1), channel))
                ),
                axis=1
            )

        # if self.L_cut_y > 1:
        #     y=np.concatenate((np.zeros(N_fft, int(self.L_cut_y/2)),
        #                     y,
        #                     np.zeros(N_fft, int(self.L_cut_y/2-1))), axis=1)

        x_stacked = np.stack([
                x[:, ii-int(self.L_cut_x/2):ii+int(self.L_cut_x/2), :]
                for ii in range(int(self.L_cut_x/2), len+int(self.L_cut_x/2))])
        y = y.transpose((1, 0, 2)).reshape(len, N_freq, 1, -1)

        x_stacked = torch.tensor(x_stacked, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.float)
        sample = {'x_stacked': x_stacked, 'y': y}

        return sample

    @staticmethod
    def my_collate(batch):
        x_stacked = torch.cat([item['x_stacked'] for item in batch])
        y = torch.cat([item['y'] for item in batch])
        return [x_stacked, y]


class MLP(nn.Module):
    def __init__(self, n_input:int, n_hidden:int, n_output:int):
        super(MLP, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.ReLU(True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(True)
        )
        self.output = nn.Sequential(
            nn.Linear(n_hidden, n_output),
            # nn.ReLU(True)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.output(x)
        return x


class NNTrainer():
    num_epochs = 50
    batch_size = 1
    learning_rate = 1e-3

    # @staticmethod
    # def to_mat(x, len):
    #     x = 0.5 * (x + 1)
    #     x = x.clamp(0, 1)
    #     x = x.view(x.size(0), N_fft/2, len, 4)
    #     return x

    def __init__(self, DIR_TRAIN:str, DIR_TEST:str,
                 XNAME:str, YNAME:str,
                 N_fft:int, L_frame:int, L_hop:int):
        self.N_fft = N_fft
        self.L_frame = L_frame
        self.L_hop = L_hop
        self.DIR_TRAIN = DIR_TRAIN
        self.DIR_TEST = DIR_TEST

        L_cut_x = 3200/L_hop
        n_input = int(L_cut_x*N_fft/2*4)
        n_hidden = int(1600/L_hop*N_fft/2*4)
        n_output = int(N_fft/2*4)

        data_train = IVDataset(DIR_TRAIN, XNAME, YNAME, L_cut_x, N_data=36000)
        self.loader_train = DataLoader(data_train,
                                       batch_size=NNTrainer.batch_size,
                                       shuffle=True,
                                       collate_fn=IVDataset.my_collate)

        data_test = IVDataset(DIR_TEST, XNAME, YNAME, L_cut_x, N_data=9000)
        self.loader_test = DataLoader(data_test,
                                       batch_size=1,
                                       shuffle=True,
                                       collate_fn=IVDataset.my_collate)

        self.model = nn.DataParallel(MLP(n_input, n_hidden, n_output),
                                     device_ids=[*range(
                                                      torch.cuda.device_count()
                                                  )]
                                    ).cuda()

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=NNTrainer.learning_rate,
                                          weight_decay=1e-5)

    def train(self):
        loss_train = np.zeros(NNTrainer.num_epochs)
        loss_test = np.zeros(NNTrainer.num_epochs)
        for epoch in range(NNTrainer.num_epochs):
            iteration = 0
            printProgress(iteration, len(self.loader_train),
                          'epoch [{}/{}]: '.format(epoch+1,
                                                   NNTrainer.num_epochs))
            for data in self.loader_train:
                iteration += 1
                x_stacked, y = data
                input = x_stacked.view(x_stacked.size(0), -1).cuda()
                # input = Variable(input).cuda()
                del x_stacked
                # ===================forward=====================
                output = self.model(input)
                loss = self.criterion(output, y.view(y.size(0), -1).cuda())
                loss_train[epoch] += loss.data.cpu().item()
                # ===================backward====================
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                printProgress(iteration, len(self.loader_train),
                              'epoch [{}/{}]: '.format(epoch+1,
                                                       NNTrainer.num_epochs))
            # ===================log========================
            print(('epoch [{}/{}], average training loss: {:e}, '
                  + 'loss of the last data: {:e}')
                  .format(epoch + 1, NNTrainer.num_epochs,
                          loss_train[-1].item()/len(self.loader_train), loss.data.item()))

            loss_test[epoch] = self.test()
            # print('test loss: {:e}'.format(loss_test[epoch]))
            if epoch > 0 \
                    and loss_test[-1] - loss_test[-2] < loss_test[-2]/10.:
                break
            # if epoch % 10 == 0:
            #     mat = to_mat(output.cpu().data, y.size(2))
            #     os.path.join(DIR_IV, 'MLP_out/')
            #     save_image(mat, './MLP_img/image_{}.png'.format(epoch))
        np.save('loss.npy', {'loss_train':loss_train,
                             'loss_test':loss_test})
        torch.save(model.state_dict(), './MLP.pth')

    def test(self):
        with torch.no_grad():
            loss_test = 0
            iteration = 0
            printProgress(iteration, len(self.loader_test), 'test: ')
            for data in self.loader_test:
                iteration += 1
                x_stacked, y = data
                input = x_stacked.view(x_stacked.size(0), -1).cuda()
                # input = Variable(input).cuda()
                del x_stacked
                # ===================forward=====================
                output = self.model(input)
                loss = self.criterion(output, y.view(y.size(0), -1).cuda())
                loss_test += loss.data.cpu().item()
                printProgress(iteration, len(self.loader_test), 'test: ')

            print('loss of the test data: {:e} %'
                  .format(loss_test / len(self.loader_test)))
        return loss_test
