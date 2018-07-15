import os
import pdb

import numpy as np

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
    def __init__(self, DIR:str, FORM:str, XNAME:str, YNAME:str,
                 N_wavfile:int, N_LOC:int, L_cut_x, L_cut_y=1):
        self.DIR = DIR
        self.FORM = FORM
        self.XNAME = XNAME
        self.YNAME = YNAME
        while not os.path.isfile(os.path.join(DIR, FORM%(N_wavfile, N_LOC-1))):
            N_wavfile -= 1
        self.len = N_wavfile*N_LOC
        self.N_LOC = N_LOC
        self.L_cut_x = L_cut_x
        self.L_cut_y = L_cut_y

    #     self._rand_idx = self.shuffle()
    #
    # def shuffle(self):
    #     idx = list(range(self.__len__))
    #     return random.shuffle(idx)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        # idx = self._rand_idx[idx]
        i_wavfile = 1 + int(idx / self.N_LOC)
        i_loc = idx % self.N_LOC
        data_dict = np.load(os.path.join(self.DIR, self.FORM%(i_wavfile, i_loc))).item()

        x = data_dict[self.XNAME]
        y = data_dict[self.YNAME]

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

    def __init__(self, DIR:str, DIR_TRAIN:str, DIR_TEST:str, FORM_IV:str,
                 XNAME:str, YNAME:str,
                 Fs, N_fft:int, L_frame:int, L_hop:int,
                 N_wavfile:int, N_LOC:int):
        self.Fs = Fs
        self.N_fft = N_fft
        self.L_frame = L_frame
        self.L_hop = L_hop
        self.DIR = DIR
        self.DIR_TRAIN = DIR_TRAIN
        self.DIR_TEST = DIR_TEST
        self.FORM_IV = FORM_IV

        L_cut_x = 4160/L_hop
        n_input = int(L_cut_x*N_fft/2*4)
        n_hidden = int(3200/L_hop*N_fft/2*4)
        n_output = int(N_fft/2*4)

        data_train = IVDataset(DIR_TRAIN, FORM_IV, XNAME, YNAME,
                               N_wavfile, N_LOC, L_cut_x)
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
                input = x_stacked.view(x_stacked.size(0), -1).cuda()
                # input = Variable(input).cuda()
                del x_stacked
                # ===================forward=====================
                output = self.model(input)
                loss = self.criterion(output, y.view(y.size(0),-1).cuda())
                # ===================backward====================
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                print('.', end='')
            # ===================log========================
            print('epoch [{}/{}], loss:{:.4f}'
                  .format(epoch + 1, NNTrainer.num_epochs, loss.data[0]))
            # if epoch % 10 == 0:
            #     mat = to_mat(output.cpu().data, y.size(2))
                # os.path.join(DIR_IV, 'MLP_out/')
                # save_image(mat, './MLP_img/image_{}.png'.format(epoch))

        torch.save(model.state_dict(), './sim_MLP.pth')

    def test(self):
        with torch.no_grad():
            correct = 0
            total = 0
            # for images, labels in test_loader:
            #     images = images.reshape(-1, 28*28).to(device)
            #     labels = labels.to(device)
            #     outputs = model(images)
            #     _, predicted = torch.max(outputs.data, 1)
            #     total += labels.size(0)
            #     correct += (predicted == labels).sum().item()

            print('Accuracy of the network on the 10000 test images: {} %' \
                  .format(100 * correct / total))
