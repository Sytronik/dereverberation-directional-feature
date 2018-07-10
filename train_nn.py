import os
import pdb

import random
import numpy as np
import scipy as sc
import scipy.io as scio

import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image

import gc

def print_cuda_tensors():
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(type(obj), obj.size())
        except:
            continue


class IVDataset(Dataset):
    def __init__(self, DIR:str, FORM:str, X_POSTFIX:str, Y_POSTFIX:str,
                 L_cut_x, L_cut_y=1):
        self.DIR = DIR
        self.FORM = FORM
        self.X_POSTFIX = X_POSTFIX
        self.Y_POSTFIX = Y_POSTFIX
        self.L_cut_x = L_cut_x
        self.L_cut_y = L_cut_y

        n_y = 0
        n_x = 0
        for f in os.scandir(DIR):
            if f.name.endswith(Y_POSTFIX):
                n_y += 1
            elif f.name.endswith(X_POSTFIX):
                n_x += 1

        self.len = np.min((n_y, n_x))
        if self.len==0:
            raise Exception('No data')

        count = 0
        while os.path.isfile(os.path.join(DIR, FORM % (1, count) + Y_POSTFIX)):
            count += 1

        self.N_loc = count
    #     self._rand_idx = self.shuffle()
    #
    # def shuffle(self):
    #     idx = list(range(self.__len__))
    #     return random.shuffle(idx)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        # idx = self._rand_idx[idx]
        i_wavfile = 1 + int(idx / self.N_loc)
        i_loc = idx % self.N_loc
        x_file = os.path.join(self.DIR,
                              self.FORM%(i_wavfile, i_loc)+self.X_POSTFIX)
        y_file = os.path.join(self.DIR,
                              self.FORM%(i_wavfile, i_loc)+self.Y_POSTFIX)

        x = np.load(x_file)
        y = np.load(y_file)

        N_freq = x.shape[0]
        channel = x.shape[2]

        if x.shape[1] < y.shape[1]:
            x=np.concatenate(
                (x, np.zeros((N_freq, y.shape[1]-x.shape[1], channel))),
                axis=1
            )
        elif x.shape[1] > y.shape[1]:
            y=np.concatenate(
                (y, np.zeros((N_freq,x.shape[1]-y.shape[1], channel))),
                axis=1
            )

        len = x.shape[1]
        if self.L_cut_x > 1:
            x=np.concatenate(
                (
                    np.zeros((N_freq, int(self.L_cut_x/2), channel)),
                    x,
                    np.zeros((N_freq, int(self.L_cut_x/2-1), channel))
                ), axis=1
            )

        # if self.L_cut_y > 1:
        #     y=np.concatenate((np.zeros(N_fft, int(self.L_cut_y/2)),
        #                     y,
        #                     np.zeros(N_fft, int(self.L_cut_y/2-1))), axis=1)

        x_stacked = np.stack([ \
                x[:,ii-int(self.L_cut_x/2):ii+int(self.L_cut_x/2),:] \
                for ii in range(int(self.L_cut_x/2), len+int(self.L_cut_x/2)) \
        ])
        y=y.transpose((1,0,2)).reshape(len,N_freq,1,-1);

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
            nn.ReLU(True)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.output(x)
        return x


class NNTrainer():
    num_epochs = 100
    batch_size = 1
    learning_rate = 1e-3

    # @staticmethod
    # def to_mat(x, len):
    #     x = 0.5 * (x + 1)
    #     x = x.clamp(0, 1)
    #     x = x.view(x.size(0), N_fft/2, len, 4)
    #     return x

    def __init__(self, Fs, N_fft, L_frame, L_hop,
                 DIR, DIR_TRAIN, DIR_TEST, FORM_IV, X_POSTFIX, Y_POSTFIX):
        self.Fs = Fs
        self.N_fft = N_fft
        self.L_frame = L_frame
        self.L_hop = L_hop
        self.DIR = DIR
        self.DIR_TRAIN = DIR_TRAIN
        self.DIR_TEST = DIR_TEST
        self.FORM_IV = FORM_IV

        L_cut_x = 8000/L_hop
        n_input = int(L_cut_x*N_fft/2*4)
        n_hidden = int(4000/L_hop*N_fft/2*4)
        n_output = int(N_fft/2*4)

        data_train = IVDataset(DIR_TRAIN, FORM_IV, X_POSTFIX, Y_POSTFIX,
                               L_cut_x)
        self.loader_train = DataLoader(data_train,
                                       batch_size=NNTrainer.batch_size,
                                       shuffle=True,
                                       collate_fn=IVDataset.my_collate)

        self.model = MLP(n_input, n_hidden, n_output).cuda()

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=NNTrainer.learning_rate,
                                     weight_decay=1e-5)

    def train(self):
        for epoch in range(NNTrainer.num_epochs):
            for data in self.loader_train:
                # pdb.set_trace()
                x_stacked, y = data
                input = x_stacked.view(x_stacked.size(0), -1)
                input = Variable(input).cuda()
                del x_stacked
                # ===================forward=====================
                output = self.model(input)
                loss = self.criterion(output, y.view(y.size(0),-1).cuda())
                # ===================backward====================
                self.optimizer.zero_grad()
                loss.backward()
                pdb.set_trace()
                self.optimizer.step()
            # ===================log========================
            print('epoch [{}/{}], loss:{:.4f}'
                  .format(epoch + 1, num_epochs, loss.data[0]))
            if epoch % 10 == 0:
                mat = to_mat(output.cpu().data, y.size(2))
                # os.path.join(DIR_IV, 'MLP_out/')
                # save_image(mat, './MLP_img/image_{}.png'.format(epoch))

        torch.save(model.state_dict(), './sim_MLP.pth')
