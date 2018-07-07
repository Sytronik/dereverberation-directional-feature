import os

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.utils import save_image

class IVDataset(Dataset):
    def __init__(self, dir:str, form:str, x_postfix:str, y_postfix:str):
        self.dir = dir
        self.form = form
        self.x_postfix = x_postfix
        self.y_postfix = y_postfix

        n_y = 0
        n_x = 0
        for f in os.scandir(dir):
            if f.name.endswith(y_postfix):
                n_y += 1
            elif f.name.endswith(x_postfix):
                n_x += 1

        self.len = np.min((n_y, n_x))
        if self.len==0:
            raise Exception('No data')

        count = 0
        while os.path.isfile(os.path.join(dir, form % (1, count) + y_postfix)):
            count += 1

        self.Nloc = count

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        i_wavfile = 1 + int(idx / Nloc)
        i_loc = idx % Nloc
        x_file = os.path.join(self.root_dir,
        self.form%(i_wavfile, i_loc)+self.x_postfix)
        y_file = os.path.join(self.root_dir,
        self.form%(i_wavfile, i_loc)+self.y_postfix)

        x = np.load(x_file)
        y = np.load(y_file)

        sample = {'x': x, 'y': y}

        return sample

Metadata = scio.loadmat('Metadata.mat',
                        variable_names = ['fs','Nfft','Lframe','Lhop',
                                            'Nwavfile','Nloc', 'DIR_IV'])
fs = Metadata['fs'].reshape(-1)[0]
Nfft = int(Metadata['Nfft'].reshape(-1)[0])
Lframe = int(Metadata['Lframe'].reshape(-1)[0])
Lhop = int(Metadata['Lhop'].reshape(-1)[0])
Nwavfile = int(Metadata['Nwavfile'].reshape(-1)[0])
Nloc = int(Metadata['Nloc'].reshape(-1)[0])
DIR_IV = Metadata['DIR_IV'].reshape(-1)[0]


def to_mat(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x

num_epochs = 100
batch_size = 72
learning_rate = 1e-3


dataset = IVDataset()
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True), nn.Linear(64, 12), nn.ReLU(True), nn.Linear(12, 3))
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(True),
            nn.Linear(12, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True), nn.Linear(128, 28 * 28), nn.Tanh())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


model = autoencoder().cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=learning_rate, weight_decay=1e-5)

for epoch in range(num_epochs):
    for data in dataloader:
        img, _ = data
        img = img.view(img.size(0), -1)
        img = Variable(img).cuda()
        # ===================forward=====================
        output = model(img)
        loss = criterion(output, img)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # ===================log========================
    print('epoch [{}/{}], loss:{:.4f}'
          .format(epoch + 1, num_epochs, loss.data[0]))
    if epoch % 10 == 0:
        mat = to_mat(output.cpu().data)
        os.path.join(DIR_IV, '')
        save_image(mat, './mlp_img/image_{}.png'.format(epoch))

torch.save(model.state_dict(), './sim_autoencoder.pth')
