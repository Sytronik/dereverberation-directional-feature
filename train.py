import os
import time
from os.path import join as pathjoin
from typing import Dict, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchsummary import summary

import config as cfg
from adamwr import AdamW, CosineLRWithRestarts
from audio_utils import delta
from iv_dataset import IVDataset
from models import UNet
from normalize import LogInterface as LogModule
from stft import STFT
from tbXwriter import CustomWriter
from utils import (MultipleOptimizer,
                   MultipleScheduler,
                   arr2str,
                   print_progress,
                   print_to_file,
                   )


class TrainerMeta(type):
    def __call__(cls, *args, **kwargs):
        if cls is Trainer:
            if cfg.hp.method == 'mag':
                return MagTrainer(*args, **kwargs)
            elif cfg.hp.method == 'complex':
                return ComplexTrainer(*args, **kwargs)
        else:
            return type.__call__(cls, *args, **kwargs)


class Trainer(metaclass=TrainerMeta):
    __slots__ = ('model', 'model_name',
                 'use_cuda', 'x_device', 'y_device',
                 'criterion', 'name_loss_terms',
                 'writer',
                 )

    def __new__(cls, *args, **kwargs):
        if cfg.hp.method == 'mag':
            return super().__new__(MagTrainer)
        elif cfg.hp.method == 'complex':
            return super().__new__(ComplexTrainer)

    def __init__(self, model_name: str, use_cuda=True):
        # Model (Using Parallel GPU)
        # model = nn.DataParallel(DeepLabv3_plus(4, 4),
        self.model_name = model_name
        self.use_cuda = use_cuda
        if model_name == 'UNet':
            module = UNet
        else:
            raise NotImplementedError
        self.model = nn.DataParallel(module(*cfg.hp.get_for_UNet()),
                                     device_ids=cfg.CUDA_DEVICES,
                                     output_device=cfg.OUT_CUDA_DEV)
        if use_cuda:
            self.model = self.model.cuda()
            self.x_device = torch.device('cuda:0')
            self.y_device = torch.device(f'cuda:{cfg.OUT_CUDA_DEV}')
        else:
            self.model = self.model.module.cpu()
            self.x_device = torch.device('cpu')
            self.y_device = torch.device('cpu')

        self.criterion = cfg.criterion
        self.name_loss_terms = ''

        self.writer: CustomWriter = None

    def _pre(self, data: Dict[str, np.ndarray], dataset: IVDataset) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    def _post_one(self, output: torch.Tensor, Ts: np.ndarray,
                  idx: int, dataset: IVDataset) -> Dict[str, np.ndarray]:
        pass

    def _calc_loss(self, y: torch.Tensor, output: torch.Tensor,
                   T_ys: Sequence[int], dataset) -> torch.Tensor:
        pass

    def train(self, loader_train: DataLoader, loader_valid: DataLoader, dir_result: str,
              first_epoch=0, f_state_dict=''):
        f_prefix = pathjoin(dir_result, f'{self.model_name}_')

        # Optimizer
        param1 = [p for m in self.model.modules() if not isinstance(m, nn.PReLU)
                  for p in m.parameters()]
        param2 = [p for m in self.model.modules() if isinstance(m, nn.PReLU)
                  for p in m.parameters()]

        optimizer = MultipleOptimizer(
            # torch.optim.Adam(
            AdamW(
                param1,
                lr=cfg.hp.learning_rate,
                weight_decay=cfg.hp.weight_decay,
            ) if param1 else None,
            # torch.optim.Adam(
            AdamW(
                param2,
                lr=cfg.hp.learning_rate,
            ) if param2 else None,
        )

        # Load State Dict
        if f_state_dict:
            tup = torch.load(f_state_dict)
            try:
                self.model.module.load_state_dict(tup[0])
                optimizer.load_state_dict(tup[1])
            except:
                raise Exception('The model is different from the state dict.')

        # Learning Rates Scheduler
        scheduler = MultipleScheduler(CosineLRWithRestarts,
                                      optimizer,
                                      batch_size=cfg.hp.batch_size,
                                      epoch_size=len(loader_train.dataset),
                                      last_epoch=first_epoch - 1,
                                      **cfg.hp.CosineLRWithRestarts)

        avg_loss = torch.zeros(cfg.N_LOSS_TERM, device=self.y_device)

        self.writer = CustomWriter(dir_result, purge_step=first_epoch)
        print_to_file(
            pathjoin(dir_result, 'summary'),
            summary,
            (self.model.module, (cfg.hp.get_for_UNet()[0], cfg.N_freq, 256)),
        )
        # self.writer.add_graph(self.model.module.cpu(),
        #                       torch.zeros(1, cfg.hp.get_for_UNet()[0], 256, 256),
        #                       True,
        #                       )

        # Start Training
        for epoch in range(first_epoch, cfg.hp.n_epochs):
            t_start = time.time()

            print()
            print_progress(0, len(loader_train), f'epoch {epoch:3d}:')
            scheduler.step()
            for i_iter, data in enumerate(loader_train):
                # get data
                x, y = self._pre(data, loader_train.dataset)  # B, C, F, T
                T_ys = data['T_ys']

                # forward
                output = self.model(x)[..., :y.shape[-1]]  # B, C, F, T

                loss = self._calc_loss(y, output, T_ys, loader_train.dataset)
                loss_sum = loss.sum()
                avg_loss += loss

                # backward
                optimizer.zero_grad()
                loss_sum.backward()
                if epoch <= 1:
                    gradnorm = nn.utils.clip_grad_norm_(self.model.parameters(), 10**10)
                    self.writer.add_scalar('train/grad', gradnorm,
                                           epoch * len(loader_train) + i_iter)
                    del gradnorm

                optimizer.step()
                scheduler.batch_step()

                # print
                loss_sum = (loss_sum / len(T_ys)).item()
                print_progress(i_iter + 1, len(loader_train), f'epoch {epoch:3d}: {loss_sum:.1e}')

            avg_loss /= len(loader_train.dataset)
            tag = 'loss/train'
            self.writer.add_scalar(tag, avg_loss.sum().item(), epoch)
            if len(self.name_loss_terms) > 1:
                for idx, (n, ll) in enumerate(zip(self.name_loss_terms, avg_loss)):
                    self.writer.add_scalar(f'{tag}/{idx + 1}. {n}', ll.item(), epoch)

            # Validation
            self.validate(loader_valid, f_prefix, epoch)

            # save loss & model
            if epoch % cfg.PERIOD_SAVE_STATE == cfg.PERIOD_SAVE_STATE - 1:
                torch.save(
                    (self.model.module.state_dict(),
                     optimizer.state_dict(),
                     ),
                    f'{f_prefix}{epoch}.pt'
                )

            # Time
            tt = time.strftime('%M min %S sec', time.gmtime(time.time() - t_start))
            print(f'epoch {epoch:3d}: {tt}')

    def validate(self, loader: DataLoader, f_prefix: str, epoch: int):
        """ Evaluate the performance of the model.

        :param loader: DataLoader to use.
        :param f_prefix: path and prefix of the result files.
        :param epoch:
        """

        with torch.no_grad():
            self.model.eval()

            wrote = False if self.writer else True
            avg_loss = torch.zeros(cfg.N_LOSS_TERM, device=self.y_device)

            print_progress(0, len(loader), f'validate : ')
            for i_iter, data in enumerate(loader):
                # get data
                x, y = self._pre(data, loader.dataset)  # B, C, F, T
                T_ys = data['T_ys']

                # forward
                output = self.model(x)[..., :y.shape[-1]]

                # loss
                loss = self._calc_loss(y, output, T_ys, loader.dataset)
                avg_loss += loss

                # print
                loss = loss.cpu().numpy() / len(T_ys)
                print_progress(i_iter + 1, len(loader),
                               f'validate : {arr2str(loss, n_decimal=1)}')

                # write summary
                if not wrote:
                    # F, T, C
                    if not self.writer.one_sample:
                        self.writer.one_sample = IVDataset.decollate_padded(data, 0)

                    out_one = self._post_one(output, T_ys, 0, loader.dataset)

                    IVDataset.save_iv(f'{f_prefix}IV_{epoch}',
                                      **self.writer.one_sample, **out_one)

                    wrote = True

                    # Process(
                    #     target=write_one,
                    #     args=(x_one, y_one, x_ph_one, y_ph_one, out_one, epoch)
                    # ).start()
                    self.writer.write_one(epoch, **out_one, group='train')

            avg_loss /= len(loader.dataset)
            tag = 'loss/valid'
            self.writer.add_scalar(tag, avg_loss.sum().item(), epoch)
            if len(self.name_loss_terms) > 1:
                for idx, (n, ll) in enumerate(zip(self.name_loss_terms, avg_loss)):
                    self.writer.add_scalar(f'{tag}/{idx + 1}. {n}', ll.item(), epoch)

            self.model.train()

        return avg_loss

    def test(self, loader: DataLoader, dir_result: str, f_state_dict: str):
        if self.use_cuda:
            self.model.module.load_state_dict(torch.load(f_state_dict)[0])
        else:
            self.model.load_state_dict(torch.load(f_state_dict)[0])
        self.writer = CustomWriter(dir_result)

        group = os.path.basename(dir_result)

        avg_measure = None
        with torch.no_grad():
            self.model.eval()

            print(' 0.00%:')
            for i_iter, data in enumerate(loader):
                t_start = time.time()
                # get data
                x, y = self._pre(data, loader.dataset)  # B, C, F, T
                T_ys = data['T_ys']

                # forward
                output = self.model(x)[..., :y.shape[-1]]

                # write summary
                # F, T, C
                one_sample = IVDataset.decollate_padded(data, 0)

                out_one = self._post_one(output, T_ys, 0, loader.dataset)

                IVDataset.save_iv(pathjoin(dir_result, f'IV_{i_iter}'),
                                  **one_sample, **out_one)

                measure = self.writer.write_one(i_iter, **out_one, group=group, **one_sample)
                if avg_measure is None:
                    avg_measure = measure
                else:
                    avg_measure += measure

                # print
                str_measure = arr2str(measure).replace('\n', '; ')
                tt = time.strftime('(%Ss)', time.gmtime(time.time() - t_start))
                print(f'{(i_iter + 1) / len(loader) * 100:5.2f}%: {str_measure}\t{tt}')

            self.model.train()

        avg_measure /= len(loader.dataset)

        self.writer.add_text(f'{group}/Average Measure/Proposed', str(avg_measure[0]))
        self.writer.add_text(f'{group}/Average Measure/Reverberant', str(avg_measure[1]))

        print()
        str_avg_measure = arr2str(avg_measure).replace('\n', '; ')
        print(f'Average: {str_avg_measure}')


class MagTrainer(Trainer):
    def _pre(self, data: Dict[str, np.ndarray], dataset: IVDataset) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        # B, F, T, C
        x = data['x']
        y = data['y']

        x = torch.tensor(x, dtype=torch.float32, device=self.x_device)
        y = torch.tensor(y, dtype=torch.float32, device=self.y_device)

        x = dataset.normalize_('x', x)
        y = dataset.normalize_('y', y)

        if self.model_name == 'UNet':
            # B, C, F, T
            x = x.permute(0, -1, -3, -2)
            y = y.permute(0, -1, -3, -2)

        return x, y

    def _post_one(self, output: torch.Tensor, Ts: np.ndarray,
                  idx: int, dataset: IVDataset) -> Dict[str, np.ndarray]:
        if self.model_name == 'UNet':
            one = output[idx, :, :, :Ts[idx]].permute(1, 2, 0)  # F, T, C
        else:
            # one = output[idx, :, :, :Ts[idx]]  # C, F, T
            raise NotImplementedError

        one = dataset.denormalize_('y', one)
        one = one.cpu().numpy()

        return dict(out=one)

    def _calc_loss(self, y: torch.Tensor, output: torch.Tensor,
                   T_ys: Sequence[int], dataset) -> torch.Tensor:
        if not self.name_loss_terms:
            self.name_loss_terms = ('',)
        y = [y, None, None]
        output = [output, None, None]
        for i_dyn in range(1, 3):
            if cfg.hp.weight_loss[i_dyn] > 0:
                y[i_dyn], output[i_dyn] \
                    = delta(y[i_dyn - 1], output[i_dyn - 1], axis=-1)

        # Loss
        loss = torch.zeros(1).cuda(cfg.OUT_CUDA_DEV)
        for i_dyn, (y_dyn, out_dyn) in enumerate(zip(y, output)):
            if cfg.hp.weight_loss[i_dyn] > 0:
                for T, item_y, item_out in zip(T_ys, y_dyn, out_dyn):
                    loss += (cfg.hp.weight_loss[i_dyn] / int(T)
                             * self.criterion(item_out[:, :, :T - 4 * i_dyn],
                                              item_y[:, :, :T - 4 * i_dyn]))
        return loss


class ComplexTrainer(Trainer):
    __slots__ = ('stft_module',
                 )

    def __init__(self, model_name: str, use_cuda=True):
        super().__init__(model_name, use_cuda)
        self.stft_module = STFT(cfg.N_fft, cfg.L_hop).cuda(self.y_device)

    def _pre(self, data: Dict[str, np.ndarray], dataset: IVDataset) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        # B, F, T, C
        x = data['x']
        y = data['y']
        x_phase = data['x_phase']
        y_phase = data['y_phase']

        x = torch.tensor(x, dtype=torch.float32, device=self.x_device)
        y = torch.tensor(y, dtype=torch.float32, device=self.y_device)
        x_phase = torch.tensor(x_phase, dtype=torch.float32, device=self.x_device)
        y_phase = torch.tensor(y_phase, dtype=torch.float32, device=self.y_device)

        x = dataset.normalize_('x', x, x_phase)
        y = dataset.normalize_('y', y, y_phase)

        if self.model_name == 'UNet':
            # B, C, F, T
            x = x.permute(0, -1, -3, -2)
            y = y.permute(0, -1, -3, -2)

        return x, y

    def _post_one(self, output: torch.Tensor, Ts: np.ndarray,
                  idx: int, dataset: IVDataset) -> Dict[str, np.ndarray]:
        if self.model_name == 'UNet':
            one = output[idx, :, :, :Ts[idx]].permute(1, 2, 0)  # F, T, C
        else:
            # one = output[idx, :, :, :Ts[idx]]  # C, F, T
            raise NotImplementedError

        one, one_phase = dataset.denormalize_('y', one)
        one = one.cpu().numpy()
        one_phase = one_phase.cpu().numpy()

        return dict(out=one, out_phase=one_phase)

    def _calc_loss(self, y: torch.Tensor, output: torch.Tensor,
                   T_ys: Sequence[int], dataset) -> torch.Tensor:
        if not self.name_loss_terms:
            self.name_loss_terms = ('mse', 'mse of log mag', 'mse of wave')
        loss = torch.zeros(3).cuda(cfg.OUT_CUDA_DEV)
        for T, item_y, item_out in zip(T_ys, y, output):
            # F, T, C
            mag_out, phase_out \
                = dataset.denormalize('y', item_out[:, :, :T].permute(1, 2, 0))
            mag_y, phase_y \
                = dataset.denormalize('y', item_y[:, :, :T].permute(1, 2, 0))

            if cfg.hp.weight_loss[0] > 0:
                loss[0] += self.criterion(item_out[:, :, :T], item_y[:, :, :T]) / int(T)
            if cfg.hp.weight_loss[1] > 0:
                # F, T, C
                logmag_out = LogModule.log(mag_out)
                logmag_y = LogModule.log(mag_y)

                loss[1] += self.criterion(logmag_out, logmag_y) / int(T)
            if cfg.hp.weight_loss[2] > 0:
                # 1, n
                wav_out = self.stft_module.inverse(mag_out.permute(2, 0, 1),
                                                   phase_out.permute(2, 0, 1))
                wav_y = self.stft_module.inverse(mag_y.permute(2, 0, 1),
                                                 phase_y.permute(2, 0, 1))

                loss[2] += self.criterion(wav_out, wav_y) / int(T)

        for w, ll in zip(cfg.hp.weight_loss, loss):
            ll *= w

        return loss
