from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
import torch
from numpy import ndarray
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torchsummary import summary
from tqdm import tqdm

from hparams import hp
import stft
from adamwr import AdamW, CosineLRWithRestarts
from dirspecgram import DirSpecDataset, XY
from dirspecgram.transform import BPD
from models import UNet, UNetNALU
from tbXwriter import CustomWriter
from utils import (arr2str,
                   MultipleOptimizer,
                   MultipleScheduler,
                   print_to_file,
                   )


class TrainerMeta(type):  # error if try to create a Trainer instance
    def __call__(cls, *args, **kwargs):
        if cls is Trainer:
            raise NotImplementedError
        else:
            return type.__call__(cls, *args, **kwargs)


class Trainer(metaclass=TrainerMeta):
    @classmethod
    def create(cls, *args, **kwargs):
        """ create a proper Trainer

        :param args: args for Trainer.__init__
        :param kwargs: kwargs for Trainer.__init__
        :rtype: Trainer
        """
        if hp.method == 'mag':
            return MagTrainer(*args, **kwargs)
        elif hp.method == 'complex':
            return ComplexTrainer(*args, **kwargs)
        elif hp.method == 'magphase':
            return MagPhaseTrainer(*args, **kwargs)
        elif hp.method == 'magbpd':
            return MagBPDTrainer(*args, **kwargs)

    def __init__(self, path_state_dict: str):
        module = eval(hp.model_name)

        self.model = module(**getattr(hp, model_name))
        self.criterion = nn.MSELoss(reduction='sum')

        self.__init_device(hp.device, hp.out_device)

        self.name_loss_terms: Sequence[str] = None

        self.writer: CustomWriter = None

        self.optimizer = AdamW(self.model.parameters(),
                               lr=hp.learning_rate,
                               weight_decay=hp.weight_decay,
                               )

        # Load State Dict
        if f_state_dict:
            tup = torch.load(f_state_dict)
            try:
                if isinstance(self.model, nn.DataParallel):
                    self.model.module.load_state_dict(state_dict)
                else:
                    self.model.load_state_dict(state_dict)
                self.optimizer.load_state_dict(tup[1])
            except:
                raise Exception('The model is different from the state dict.')

        print_to_file(
            dir_result / 'summary',
            summary,
            (self.model, hp.dummy_input_size),
            dict(device=self.str_device[:4])
        )
        dd.io.save((logdir_train / hp.hparams_fname).with_suffix('.h5'), asdict(hp))
        with (logdir_train / hp.hparams_fname).open('w') as f:
            f.write(repr(asdict(hp)))

    def __init_device(self, device, out_device):
        if device == 'cpu':
            self.in_device = torch.device('cpu')
            self.out_device = torch.device('cpu')
            self.str_device = 'cpu'
            return

        # device type
        if type(device) == int:
            device = [device]
        elif type(device) == str:
            device = [int(device[-1])]
        else:  # sequence of devices
            if type(device[0]) == int:
                device = device
            else:
                device = [int(d[-1]) for d in device]

        # out_device type
        if type(out_device) == int:
            out_device = torch.device(f'cuda:{out_device}')
        else:
            out_device = torch.device(out_device)

        self.in_device = torch.device(f'cuda:{device[0]}')
        self.out_device = out_device

        if len(device) > 1:
            self.model = nn.DataParallel(self.model,
                                         device_ids=device,
                                         output_device=out_device)
            self.str_device = ', '.join([f'cuda:{d}' for d in device])
        else:
            self.str_device = str(self.in_device)

        self.model.cuda(self.in_device)
        self.criterion.cuda(self.out_device)
        if self.sigmoid:
            self.sigmoid.cuda(self.in_device)
        torch.cuda.set_device(self.in_device)

    def _pre(self, data: Dict[str, ndarray], dataset: DirSpecDataset, for_summary=False) \
            -> Tuple[Tensor, Tensor]:
        pass

    def _post_one(self, output: Tensor, Ts: ndarray,
                  idx: int, dataset: DirSpecDataset) -> Dict[str, ndarray]:
        pass

    def _calc_loss(self, y: Tensor, output: Tensor,
                   T_ys: Sequence[int], dataset: DirSpecDataset) -> Tensor:
        pass

    def train(self, loader_train: DataLoader, loader_valid: DataLoader, logdir: Path,
              first_epoch=0, f_state_dict: Path = None):
        # Learning Rates Scheduler
        scheduler = CosineLRWithRestarts(self.optimizer,
                                         batch_size=hp.batch_size,
                                         epoch_size=len(loader_train.dataset),
                                         last_epoch=first_epoch - 1,
                                         **hp.CosineLRWithRestarts)
        avg_loss = torch.zeros(hp.N_LOSS_TERM, device=self.out_device)

        self.writer = CustomWriter(str(logdir), purge_step=first_epoch)

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
                x, y = self._pre(data, loader_train.dataset)  # B, C, F, T
                T_ys = data['T_ys']

                # forward
                output = self.model(x)[..., :y.shape[-1]]  # B, C, F, T

                loss = self._calc_loss(y, output, T_ys, loader_train.dataset)
                loss_sum = loss.sum()

                # backward
                self.optimizer.zero_grad()
                loss_sum.backward()
                if epoch <= 2:
                    gradnorm = nn.utils.clip_grad_norm_(self.model.parameters(), 10**10)
                    self.writer.add_scalar('train/grad', gradnorm,
                                           epoch * len(loader_train) + i_iter)
                    del gradnorm

                self.optimizer.step()
                scheduler.batch_step()

                # print
                with torch.no_grad():
                    avg_loss += loss
                    loss = loss.cpu().numpy() / len(T_ys)
                    pbar.set_postfix_str(arr2str(loss, ndigits=1))

            avg_loss /= len(loader_train.dataset)
            tag = 'loss/train'
            self.writer.add_scalar(tag, avg_loss.sum().item(), epoch)
            if len(self.name_loss_terms) > 1:
                for idx, (n, ll) in enumerate(zip(self.name_loss_terms, avg_loss)):
                    self.writer.add_scalar(f'{tag}/{idx + 1}. {n}', ll.item(), epoch)

            # Validation
            self.validate(loader_valid, logdir, epoch)

            # save loss & model
            if epoch % hp.period_save_state == hp.period_save_state - 1:
                torch.save(
                    (self.model.module.state_dict(),
                     self.optimizer.state_dict(),
                     ),
                    logdir / f'{self.model_name}_{epoch}.pt'
                )

    @torch.no_grad()
    def validate(self, loader: DataLoader, logdir: str, epoch: int):
        """ Evaluate the performance of the model.

        :param loader: DataLoader to use.
        :param logdir: path of the result files.
        :param epoch:
        """

        self.model.eval()

        avg_loss = torch.zeros(hp.n_loss_term, device=self.out_device)

        pbar = tqdm(loader, desc='validate ', postfix='[0]', dynamic_ncols=True)
        for i_iter, data in enumerate(pbar):
            # get data
            x, y = self._pre(data, loader.dataset, for_summary=(i_iter == 0))  # B, C, F, T
            T_ys = data['T_ys']

            # forward
            output = self.model(x)  # [..., :y.shape[-1]]

            # loss
            loss = self._calc_loss(y, output, T_ys, loader.dataset)
            avg_loss += loss

            # print
            loss = loss.cpu().numpy() / len(T_ys)
            pbar.set_postfix_str(arr2str(loss, ndigits=1))

            # write summary
            if i_iter == 0:
                # F, T, C
                if not self.writer.one_sample:
                    self.writer.one_sample = DirSpecDataset.decollate_padded(data, 0)
                    if hp.method == 'magbpd':
                        self.writer.one_sample['x_bpd'] \
                            = (np.pi * x[0, -1:, :, :data['T_xs'][0]]).cpu().numpy()
                        self.writer.one_sample['y_bpd'] \
                            = (np.pi * y[0, -1:, :, :T_ys[0]]).cpu().numpy()
                        if self.model_name.startswith('UNet'):
                            self.writer.one_sample['x_bpd'] \
                                = self.writer.one_sample['x_bpd'].transpose((1, 2, 0))
                            self.writer.one_sample['y_bpd'] \
                                = self.writer.one_sample['y_bpd'].transpose((1, 2, 0))

                out_one = self._post_one(output, T_ys, 0, loader.dataset)

                DirSpecDataset.save_dirspec(
                    logdir / hp.form_result_dirspec.format(epoch),
                    **self.writer.one_sample, **out_one
                )

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

    @torch.no_grad()
    def test(self, loader: DataLoader, logdir: Path, f_state_dict: Path):
        if self.use_cuda:
            self.model.module.load_state_dict(torch.load(f_state_dict)[0])
        else:
            self.model.load_state_dict(torch.load(f_state_dict)[0])
        self.writer = CustomWriter(str(logdir))

        group = logdir.name.split('_')[0]

        avg_measure = None
        self.model.eval()

        pbar = tqdm(loader, desc=group, dynamic_ncols=True)
        for i_iter, data in enumerate(pbar):
            # get data
            x, y = self._pre(data, loader.dataset, for_summary=True)  # B, C, F, T
            T_ys = data['T_ys']

            # forward
            output = self.model(x)  # [..., :y.shape[-1]]

            # write summary
            one_sample = DirSpecDataset.decollate_padded(data, 0)  # F, T, C
            if hp.method == 'magbpd':
                # C, F, T
                one_sample['x_bpd'] = (np.pi * x[0, -1:, :, :data['T_xs'][0]]).cpu().numpy()
                one_sample['y_bpd'] = (np.pi * y[0, -1:, :, :T_ys[0]]).cpu().numpy()
                if self.model_name.startswith('UNet'):
                    # F, T, C
                    one_sample['x_bpd'] = one_sample['x_bpd'].transpose((1, 2, 0))
                    one_sample['y_bpd'] = one_sample['y_bpd'].transpose((1, 2, 0))

            out_one = self._post_one(output, T_ys, 0, loader.dataset)

            DirSpecDataset.save_dirspec(
                logdir / hp.form_result_dirspec.format(i_iter),
                **one_sample, **out_one
            )

            measure = self.writer.write_one(i_iter, **out_one, group=group, **one_sample)
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


class MagTrainer(Trainer):
    def _pre(self, data: Dict[str, ndarray], dataset: DirSpecDataset, for_summary=False) \
            -> Tuple[Tensor, Tensor]:
        # B, F, T, C
        x = data['x']
        y = data['y']

        x = x.to(self.in_device)
        y = y.to(self.out_device)

        x = dataset.preprocess_(XY.x, x)
        y = dataset.preprocess_(XY.y, y)

        if self.model_name.startswith('UNet'):
            # B, C, F, T
            x = x.permute(0, -1, -3, -2)
            y = y.permute(0, -1, -3, -2)

        return x, y

    @torch.no_grad()
    def _post_one(self, output: Tensor, Ts: ndarray,
                  idx: int, dataset: DirSpecDataset) -> Dict[str, ndarray]:
        if self.model_name.startswith('UNet'):
            one = output[idx, :, :, :Ts[idx]].permute(1, 2, 0)  # F, T, C
            # one = output[idx].permute(1, 2, 0)  # warning
        else:
            # one = output[idx, :, :, :Ts[idx]]  # C, F, T
            raise NotImplementedError

        one = dataset.postprocess_(XY.y, one)
        one = one.cpu().numpy()

        return dict(out=one)

    def _calc_loss(self, y: Tensor, output: Tensor,
                   T_ys: Sequence[int], dataset: DirSpecDataset) -> Tensor:
        if not self.name_loss_terms:
            self.name_loss_terms = ('',)
        # y = [y, None, None]
        # output = [output, None, None]
        # for i_dyn in range(1, 3):
        #     if hp.weight_loss[i_dyn] > 0:
        #         y[i_dyn], output[i_dyn] \
        #             = delta(y[i_dyn - 1], output[i_dyn - 1], axis=-1)
        #
        # # Loss
        # loss = torch.zeros(1, device=self.out_device)
        # for i_dyn, (y_dyn, out_dyn) in enumerate(zip(y, output)):
        #     if hp.weight_loss[i_dyn] > 0:
        #         for T, item_y, item_out in zip(T_ys, y_dyn, out_dyn):
        #             loss += (hp.weight_loss[i_dyn] / int(T)
        #                      * self.criterion(item_out[:, :, :T - 4 * i_dyn],
        #                                       item_y[:, :, :T - 4 * i_dyn]))
        # return loss

        loss = torch.zeros(1, device=self.out_device)
        for T, item_y, item_out in zip(T_ys, y, output):
            loss += self.criterion(item_out[:, :, :T], item_y[:, :, :T]) / int(T)

        return loss


class ComplexTrainer(Trainer):
    __slots__ = ('stft_module',
                 )

    def __init__(self, model_name: str, use_cuda=True):
        super().__init__(model_name, use_cuda)
        self.stft_module = stft.get_STFT_module(self.out_device, hp.n_fft, hp.l_hop)

    def _pre(self, data: Dict[str, ndarray], dataset: DirSpecDataset, for_summary=False) \
            -> Tuple[Tensor, Tensor]:
        # B, F, T, C
        x = data['x']
        y = data['y']
        x_phase = data['x_phase']
        y_phase = data['y_phase']

        x = x.to(self.in_device)
        y = y.to(self.out_device)
        x_phase = x_phase.to(self.in_device)
        y_phase = y_phase.to(self.out_device)

        x = dataset.preprocess_(XY.x, x, x_phase)
        y = dataset.preprocess_(XY.y, y, y_phase)

        if self.model_name.startswith('UNet'):
            # B, C, F, T
            x = x.permute(0, -1, -3, -2)
            y = y.permute(0, -1, -3, -2)

        return x, y

    @torch.no_grad()
    def _post_one(self, output: Tensor, Ts: ndarray,
                  idx: int, dataset: DirSpecDataset) -> Dict[str, ndarray]:
        if self.model_name.startswith('UNet'):
            one = output[idx, :, :, :Ts[idx]].permute(1, 2, 0)  # F, T, C
        else:
            # one = output[idx, :, :, :Ts[idx]]  # C, F, T
            raise NotImplementedError

        one, one_phase = dataset.postprocess_(XY.y, one)
        one = one.cpu().numpy()
        one_phase = one_phase.cpu().numpy()

        return dict(out=one, out_phase=one_phase)

    def _calc_loss(self, y: Tensor, output: Tensor,
                   T_ys: Sequence[int], dataset: DirSpecDataset) -> Tensor:
        if not self.name_loss_terms:
            self.name_loss_terms = ('mse', 'mse of log mag', 'mse of wave')
            self.name_loss_terms = self.name_loss_terms[:hp.n_loss_term]
        loss = torch.zeros(hp.n_loss_term, device=self.out_device)
        for T, item_y, item_out in zip(T_ys, y, output):
            if hp.weight_loss[0] > 0:
                loss[0] += self.criterion(item_out[:, :, :T], item_y[:, :, :T]) / int(T)

            if not (hp.weight_loss[1] or hp.weight_loss[2]):
                continue
            # F, T, C
            mag_out, phase_out \
                = dataset.postprocess(XY.y, item_out[:, :, :T].permute(1, 2, 0))
            mag_y, phase_y \
                = dataset.postprocess(XY.y, item_y[:, :, :T].permute(1, 2, 0))

            if hp.weight_loss[1] > 0:
                # F, T, C
                mag_norm_out = dataset.preprocess(XY.y, mag_out, idx=1)
                mag_norm_y = dataset.preprocess(XY.y, mag_y, idx=1)

                loss[1] += self.criterion(mag_norm_out, mag_norm_y) / int(T)
            if len(hp.weight_loss) >= 3 and hp.weight_loss[2] > 0:
                # 1, n
                wav_out = self.stft_module.inverse(mag_out.permute(2, 0, 1),
                                                   phase_out.permute(2, 0, 1))
                wav_y = self.stft_module.inverse(mag_y.permute(2, 0, 1),
                                                 phase_y.permute(2, 0, 1))

                temp = self.criterion(wav_out, wav_y) / int(T)
                if not torch.isnan(temp) and temp < 10**4:
                    loss[2] += temp

        for w, ll in zip(hp.weight_loss, loss):
            ll *= w

        return loss


class MagPhaseTrainer(Trainer):
    def _pre(self, data: Dict[str, ndarray], dataset: DirSpecDataset, for_summary=False) \
            -> Tuple[Tensor, Tensor]:
        # B, F, T, C
        x = data['x']
        y = data['y']
        x_phase = data['x_phase']
        y_phase = data['y_phase']

        x = x.to(self.in_device)
        y = y.to(self.out_device)
        x_phase = x_phase.to(self.in_device)
        y_phase = y_phase.to(self.out_device)

        x = dataset.preprocess_(XY.x, x)
        x_phase /= np.pi
        x = torch.cat((x, x_phase), dim=-1)

        y = dataset.preprocess_(XY.y, y)
        y_phase /= np.pi
        y = torch.cat((y, y_phase), dim=-1)
        # y = dataset.preprocess_(XY.y, y, y_phase, idx=1)

        if self.model_name.startswith('UNet'):
            # B, C, F, T
            x = x.permute(0, -1, -3, -2)
            y = y.permute(0, -1, -3, -2)

        return x, y

    @torch.no_grad()
    def _post_one(self, output: Tensor, Ts: ndarray,
                  idx: int, dataset: DirSpecDataset) -> Dict[str, ndarray]:
        if self.model_name.startswith('UNet'):
            one = output[idx, :, :, :Ts[idx]].permute(1, 2, 0)  # F, T, C
        else:
            # one = output[idx, :, :, :Ts[idx]]  # C, F, T
            raise NotImplementedError

        one_phase = one[..., -1:] * np.pi
        one = dataset.postprocess_(XY.y, one[..., :-1])
        # one, one_phase = dataset.postprocess_(XY.y, one, idx=1)
        one_phase = one_phase.cpu().numpy()
        one = one.cpu().numpy()

        return dict(out=one, out_phase=one_phase)

    def _calc_loss(self, y: Tensor, output: Tensor,
                   T_ys: Sequence[int], dataset: DirSpecDataset) -> Tensor:
        if not self.name_loss_terms:
            self.name_loss_terms = ('mse', 'mse of log mag', 'mse of bpd')
        loss = torch.zeros(len(self.name_loss_terms), device=self.out_device)
        for T, item_y, item_out in zip(T_ys, y, output):
            phase_out = None
            if hp.weight_loss[0] > 0:
                # F, T, C
                mag_out = dataset.postprocess(
                    'y', item_out[-2:-1, :, :T].permute(1, 2, 0)
                )
                mag_y = dataset.postprocess(
                    'y', item_y[-2:-1, :, :T].permute(1, 2, 0)
                )
                phase_out = item_out[-1:, :, :T].permute(1, 2, 0) * np.pi
                phase_y = item_y[-1:, :, :T].permute(1, 2, 0) * np.pi

                out = dataset.preprocess_(XY.y, mag_out, phase_out, idx=1)
                y = dataset.preprocess_(XY.y, mag_y, phase_y, idx=1)

                temp = self.criterion(out, y) / int(T)
                if not torch.isnan(temp) and temp < 10**4:
                    loss[0] += temp

            if hp.weight_loss[1] > 0:
                loss[1] += self.criterion(item_out[:-1, :, :T],
                                          item_y[:-1, :, :T]) / int(T)

            if hp.weight_loss[2] > 0:
                if phase_out is None:
                    phase_out = item_out[-1:, :, :T].permute(1, 2, 0) * np.pi
                    phase_y = item_y[-1:, :, :T].permute(1, 2, 0) * np.pi
                loss[2] += self.criterion(BPD.phase2bpd(phase_out),
                                          BPD.phase2bpd(phase_y)) / int(T)

            # if hp.weight_loss[0] > 0:
            #     loss[0] += self.criterion(item_out[:, :, :T], item_y[:, :, :T]) / int(T)
            #
            # if hp.weight_loss[1] > 0:
            #     # F, T, C
            #     mag_out, _ \
            #         = dataset.postprocess(XY.y, item_out[:, :, :T].permute(1, 2, 0), idx=1)
            #     mag_y, _ \
            #         = dataset.postprocess(XY.y, item_y[:, :, :T].permute(1, 2, 0), idx=1)
            #     mag_norm_out = dataset.preprocess(XY.y, mag_out, idx=0)
            #     mag_norm_y = dataset.preprocess(XY.y, mag_y, idx=0)
            #
            #     loss[1] += self.criterion(mag_norm_out, mag_norm_y) / int(T)

        for w, ll in zip(hp.weight_loss, loss):
            ll *= w

        return loss


class MagBPDTrainer(Trainer):
    __slots__ = ('stft_module',
                 )

    def __init__(self, model_name: str, use_cuda=True):
        super().__init__(model_name, use_cuda)
        self.stft_module = stft.get_STFT_module(self.out_device, hp.n_fft, hp.l_hop)

    def _pre(self, data: Dict[str, ndarray], dataset: DirSpecDataset, for_summary=False) \
            -> Tuple[Tensor, Tensor]:
        # B, F, T, C
        x = data['x']
        y = data['y']
        x_phase = data['x_phase']
        y_phase = data['y_phase']

        x = x.to(self.in_device)
        y = y.to(self.out_device)
        x_phase = x_phase.to(self.in_device)
        y_phase = y_phase.to(self.out_device)

        if for_summary:
            x_trunc = x[0, :, :data['T_ys'][0], -1:].to(self.out_device, copy=True)
            x_phase_trunc = x_phase[0, :, :data['T_ys'][0], :].to(self.out_device, copy=True)

            dataset.set_transformer_var(
                phase_init=x_phase_trunc,
                x_mag=x_trunc,
                along_freq=True,
            )

        x = dataset.preprocess_(XY.x, x, x_phase)
        y = dataset.preprocess_(XY.y, y, y_phase)

        if self.model_name.startswith('UNet'):
            # B, C, F, T
            x = x.permute(0, -1, -3, -2)
            y = y.permute(0, -1, -3, -2)

        return x, y

    @torch.no_grad()
    def _post_one(self, output: Tensor, Ts: ndarray,
                  idx: int, dataset: DirSpecDataset) -> Dict[str, ndarray]:
        if self.model_name.startswith('UNet'):
            one = output[idx, :, :, :Ts[idx]].permute(1, 2, 0)  # F, T, C
        else:
            # one = output[idx, :, :, :Ts[idx]]  # C, F, T
            raise NotImplementedError

        one_bpd = (np.pi * one[:, :, -1:]).cpu().numpy()
        one, one_phase = dataset.postprocess_(XY.y, one)
        one = one.cpu().numpy()
        one_phase = one_phase.cpu().numpy()

        return dict(out=one, out_phase=one_phase, out_bpd=one_bpd)

    def _calc_loss(self, y: Tensor, output: Tensor,
                   T_ys: Sequence[int], dataset: DirSpecDataset) -> Tensor:
        if not self.name_loss_terms:
            self.name_loss_terms = ('',)

        # Loss
        loss = torch.zeros(1, device=self.out_device)
        for T, item_y, item_out in zip(T_ys, y, output):
            loss += self.criterion(item_out[:, :, :T], item_y[:, :, :T]) / int(T)

        return loss
