from pathlib import Path
from typing import Dict, Sequence, Tuple

import torch
from numpy import ndarray
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torchsummary import summary
from tqdm import tqdm

from hparams import hp
from adamwr import AdamW, CosineLRWithRestarts
from dataset import DirSpecDataset
from models import UNet
from tbXwriter import CustomWriter
from utils import arr2str, print_to_file


class Trainer:
    def __init__(self, path_state_dict=''):
        self.model_name = hp.model_name
        module = eval(hp.model_name)

        self.model = module(**getattr(hp, hp.model_name))
        self.criterion = nn.MSELoss(reduction='none')

        self.__init_device(hp.device, hp.out_device)

        self.writer: CustomWriter = None

        self.optimizer = AdamW(self.model.parameters(),
                               lr=hp.learning_rate,
                               weight_decay=hp.weight_decay,
                               )

        # Load State Dict
        if path_state_dict:
            st_model, st_optim = torch.load(path_state_dict, map_location=self.in_device)
            try:
                if isinstance(self.model, nn.DataParallel):
                    self.model.module.load_state_dict(st_model)
                else:
                    self.model.load_state_dict(st_model)
                self.optimizer.load_state_dict(st_optim)
            except:
                raise Exception('The model is different from the state dict.')

        path_summary = hp.logdir / 'summary.txt'
        if not path_summary.exists():
            print_to_file(
                path_summary,
                summary,
                (self.model, hp.dummy_input_size),
                dict(device=self.str_device[:4])
            )
            # dd.io.save((hp.logdir / hp.hparams_fname).with_suffix('.h5'), asdict(hp))
            with (hp.logdir / 'hparams.txt').open('w') as f:
                f.write(repr(hp))

    def __init_device(self, device, out_device):
        if device == 'cpu':
            self.in_device = torch.device('cpu')
            self.out_device = torch.device('cpu')
            self.str_device = 'cpu'
            return

        # device type: List[int]
        if type(device) == int:
            device = [device]
        elif type(device) == str:
            device = [int(device.replace('cuda:', ''))]
        else:  # sequence of devices
            if type(device[0]) != int:
                device = [int(d.replace('cuda:', '')) for d in device]

        self.in_device = torch.device(f'cuda:{device[0]}')

        if len(device) > 1:
            if type(out_device) == int:
                self.out_device = torch.device(f'cuda:{out_device}')
            else:
                self.out_device = torch.device(out_device)
            self.str_device = ', '.join([f'cuda:{d}' for d in device])

            self.model = nn.DataParallel(self.model,
                                         device_ids=device,
                                         output_device=self.out_device)
        else:
            self.out_device = self.in_device
            self.str_device = str(self.in_device)

        self.model.cuda(self.in_device)
        self.criterion.cuda(self.out_device)

        torch.cuda.set_device(self.in_device)

    def preprocess(self, data: Dict[str, Tensor],
                   dataset: DirSpecDataset) -> Tuple[Tensor, Tensor]:
        # B, F, T, C
        x = data['x']
        y = data['y']

        if self.in_device == torch.device('cpu'):
            x, y = dataset.normalize(x, y)
        else:
            x = x.to(self.in_device, non_blocking=True)
            y = y.to(self.out_device, non_blocking=True)

            x, y = dataset.normalize_(x, y)

        if self.model_name.startswith('UNet'):
            # B, C, F, T
            x = x.permute(0, -1, -3, -2)
            y = y.permute(0, -1, -3, -2)

        return x, y

    @torch.no_grad()
    def postprocess(self, output: Tensor, Ts: ndarray, idx: int,
                    dataset: DirSpecDataset) -> Dict[str, ndarray]:
        one = output[idx, :, :, :Ts[idx]]
        if self.model_name.startswith('UNet'):
            one = one.permute(1, 2, 0)  # F, T, C

        one = dataset.denormalize_(y=one)
        one = one.cpu().numpy()

        return dict(out=one)

    def calc_loss(self, output: Tensor, y: Tensor, T_ys: Sequence[int],
                  dataset: DirSpecDataset) -> Tensor:
        loss_batch = self.criterion(output, y)
        loss = torch.zeros(1, device=loss_batch.device)
        for T, loss_sample in zip(T_ys, loss_batch):
            loss += torch.sum(loss_sample[:, :, :T]) / T

        return loss

    def train(self, loader_train: DataLoader, loader_valid: DataLoader,
              logdir: Path, first_epoch=0):
        # Learning Rates Scheduler
        scheduler = CosineLRWithRestarts(self.optimizer,
                                         batch_size=hp.batch_size,
                                         epoch_size=len(loader_train.dataset),
                                         last_epoch=first_epoch - 1,
                                         **hp.scheduler)
        avg_loss = torch.zeros(1, device=self.out_device)

        self.writer = CustomWriter(str(logdir), group='train', purge_step=first_epoch)

        # self.writer.add_graph(self.model.module.cpu(),
        #                       torch.zeros(1, hp.get_for_UNet()[0], 256, 256),
        #                       True,
        #                       )

        # Start Training
        for epoch in range(first_epoch, hp.n_epochs):

            print()
            scheduler.step()
            pbar = tqdm(loader_train, desc=f'epoch {epoch:3d}', postfix='[]', dynamic_ncols=True)

            for i_iter, data in enumerate(pbar):
                # get data
                x, y = self.preprocess(data, loader_train.dataset)  # B, C, F, T
                T_ys = data['T_ys']

                # forward
                output = self.model(x)[..., :y.shape[-1]]  # B, C, F, T

                loss = self.calc_loss(output, y, T_ys, loader_train.dataset)
                loss_sum = loss.sum()

                # backward
                self.optimizer.zero_grad()
                loss_sum.backward()

                self.optimizer.step()
                scheduler.batch_step()

                # print
                with torch.no_grad():
                    avg_loss += loss
                    loss_np = loss.cpu().numpy() / len(T_ys)
                    pbar.set_postfix_str(arr2str(loss_np, ndigits=1))

            avg_loss /= len(loader_train.dataset)
            tag = 'loss/train'
            self.writer.add_scalar(tag, avg_loss.sum().item(), epoch)

            # Validation
            self.validate(loader_valid, logdir, epoch)

            # save loss & model
            if epoch % hp.period_save_state == hp.period_save_state - 1:
                torch.save(
                    (self.model.module.state_dict(),
                     self.optimizer.state_dict(),
                     ),
                    logdir / f'{epoch}.pt'
                )

    @torch.no_grad()
    def validate(self, loader: DataLoader, logdir: Path, epoch: int):
        """ Evaluate the performance of the model.

        :param loader: DataLoader to use.
        :param logdir: path of the result files.
        :param epoch:
        """

        self.model.eval()

        avg_loss = torch.zeros(1, device=self.out_device)

        pbar = tqdm(loader, desc='validate ', postfix='[0]', dynamic_ncols=True)
        for i_iter, data in enumerate(pbar):
            # get data
            x, y = self.preprocess(data, loader.dataset)  # B, C, F, T
            T_ys = data['T_ys']

            # forward
            output = self.model(x)[..., :y.shape[-1]]

            # loss
            loss = self.calc_loss(output, y, T_ys, loader.dataset)
            avg_loss += loss

            # print
            loss_np = loss.cpu().numpy() / len(T_ys)
            pbar.set_postfix_str(arr2str(loss_np, ndigits=1))

            # write summary
            if i_iter == 0:
                # F, T, C
                if not self.writer.reused_sample:
                    one_sample = DirSpecDataset.decollate_padded(data, 0)
                else:
                    one_sample = dict()

                out_one = self.postprocess(output, T_ys, 0, loader.dataset)

                DirSpecDataset.save_dirspec(
                    logdir / hp.form_result.format(epoch),
                    **self.writer.reused_sample, **out_one
                )

                # Process(
                #     target=write_one,
                #     args=(x_one, y_one, x_ph_one, y_ph_one, out_one, epoch)
                # ).start()
                self.writer.write_one(epoch, **one_sample, **out_one)

        avg_loss /= len(loader.dataset)
        tag = 'loss/valid'
        self.writer.add_scalar(tag, avg_loss.sum().item(), epoch)

        self.model.train()

        return avg_loss

    @torch.no_grad()
    def test(self, loader: DataLoader, logdir: Path):
        group = logdir.name.split('_')[0]

        self.writer = CustomWriter(str(logdir), group=group)

        avg_measure = None
        self.model.eval()

        pbar = tqdm(loader, desc=group, dynamic_ncols=True)
        for i_iter, data in enumerate(pbar):
            # get data
            x, y = self.preprocess(data, loader.dataset)  # B, C, F, T
            T_ys = data['T_ys']

            # forward
            output = self.model(x)  # [..., :y.shape[-1]]

            # write summary
            one_sample = DirSpecDataset.decollate_padded(data, 0)  # F, T, C

            out_one = self.postprocess(output, T_ys, 0, loader.dataset)

            DirSpecDataset.save_dirspec(
                logdir / hp.form_result.format(i_iter),
                **one_sample, **out_one
            )

            measure = self.writer.write_one(i_iter, **out_one, **one_sample)
            if avg_measure is None:
                avg_measure = measure
            else:
                avg_measure += measure

            # print
            str_measure = arr2str(measure).replace('\n', '; ')
            pbar.write(str_measure)

        self.model.train()

        avg_measure /= len(loader.dataset)

        self.writer.add_text(f'{group}/Average Measure/Proposed', str(avg_measure[0]))
        self.writer.add_text(f'{group}/Average Measure/Reverberant', str(avg_measure[1]))
        self.writer.close()  # Explicitly close

        print()
        str_avg_measure = arr2str(avg_measure).replace('\n', '; ')
        print(f'Average: {str_avg_measure}')
