import atexit
import os
import time
from typing import Tuple, Sequence

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch import nn
from torch.utils.data import DataLoader

import config as cfg
from adamwr import AdamW, CosineLRWithRestarts
from audio_utils import (bnkr_equalize_,
                         delta,
                         draw_spectrogram,
                         Measurement,
                         reconstruct_wave,
                         )
from iv_dataset import IVDataset
from models import UNet
from normalize import LogInterface as LogModule
from utils import (arr2str, MultipleOptimizer, MultipleScheduler, print_progress)


class Trainer:
    __slots__ = ('model', 'model_name',
                 'criterion', 'writer', 'one_sample',
                 'recon_sample', 'measure_x', 'kwargs_fig',
                 )

    def __init__(self, model_name: str):
        # Model (Using Parallel GPU)
        # model = nn.DataParallel(DeepLabv3_plus(4, 4),
        self.model_name = model_name
        if model_name == 'UNet':
            module = UNet
        else:
            raise NotImplementedError
        self.model = nn.DataParallel(module(*cfg.hp.get_for_UNet()),
                                     device_ids=cfg.CUDA_DEVICES,
                                     output_device=cfg.OUT_CUDA_DEV).cuda()

        self.criterion = cfg.criterion

        self.writer: SummaryWriter = None
        self.one_sample = dict()
        self.recon_sample = dict()
        self.measure_x = dict()
        self.kwargs_fig = dict()

    def pre(self, x: np.ndarray, y: np.ndarray,
            dataset: IVDataset) -> Tuple[torch.Tensor, torch.Tensor]:
        # B, F, T, C
        x_cuda = torch.tensor(x, dtype=torch.float32, device=0)
        y_cuda = torch.tensor(y, dtype=torch.float32, device=cfg.OUT_CUDA_DEV)

        x_cuda = dataset.normalize(x_cuda, 'x')
        y_cuda = dataset.normalize(y_cuda, 'y')

        if self.model_name == 'UNet':
            # B, C, F, T
            x_cuda = x_cuda.permute(0, -1, -3, -2)
            y_cuda = y_cuda.permute(0, -1, -3, -2)

        return x_cuda, y_cuda

    def calc_loss(self, y_cuda: torch.Tensor, output: torch.Tensor,
                  T_ys: Sequence[int]):
        y_cuda = [y_cuda, None, None]
        output = [output, None, None]
        for i_dyn in range(1, 3):
            y_cuda[i_dyn], output[i_dyn] \
                = delta(y_cuda[i_dyn - 1], output[i_dyn - 1], axis=-1)

        # Loss
        loss = torch.zeros(1).cuda(cfg.OUT_CUDA_DEV)
        for i_dyn, (y_dyn, out_dyn) in enumerate(zip(y_cuda, output)):
            for T, item_y, item_out in zip(T_ys, y_dyn, out_dyn):
                loss += (cfg.hp.weight_loss[i_dyn] / int(T)
                         * self.criterion(item_out[:, :, :T - 4 * i_dyn],
                                          item_y[:, :, :T - 4 * i_dyn]))
        return loss

    def train(self, loader_train: DataLoader, loader_valid: DataLoader, dir_result,
              first_epoch=0, f_state_dict=''):
        f_prefix = os.path.join(dir_result, f'{self.model_name}_')
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

        # Learning Rates Scheduler
        scheduler = MultipleScheduler(CosineLRWithRestarts,
                                      optimizer,
                                      batch_size=cfg.hp.batch_size,
                                      epoch_size=len(loader_train.dataset),
                                      last_epoch=first_epoch - 1,
                                      **cfg.hp.CosineLRWithRestarts)

        # Load State Dict
        if f_state_dict:
            tup = torch.load(f_state_dict)
            self.model.module.load_state_dict(tup[0])
            optimizer.load_state_dict(tup[1])
        avg_loss = torch.zeros(1, device=cfg.OUT_CUDA_DEV)

        self.writer = SummaryWriter(dir_result, purge_step=first_epoch)
        atexit.register(self.writer.close)

        # Start Training
        for epoch in range(first_epoch, cfg.hp.n_epochs):
            t_start = time.time()

            print()
            print_progress(0, len(loader_train), f'epoch {epoch:3d}:')
            scheduler.step()
            for i_iter, data in enumerate(loader_train):
                # ==================get data=====================
                x, y = data['x'], data['y']  # B, F, T, C
                T_ys = data['T_ys']

                x_cuda, y_cuda = self.pre(x, y, loader_train.dataset)  # B, C, F, T

                # ===================forward=====================
                output = self.model(x_cuda)[..., :y_cuda.shape[-1]]  # B, C, F, T

                loss = self.calc_loss(y_cuda, output, T_ys)
                avg_loss += loss.item()
                # writer.add_histogram()

                # ===================backward====================
                optimizer.zero_grad()
                loss.backward()
                # grads = torch.cat([p.grad.view(-1) for p in self.model.parameters()])
                # self.writer.add_histogram('grad',
                #                           grads, epoch*len(loader_train)+i_iter,
                #                           bins='auto')
                # del grads

                optimizer.step()
                scheduler.batch_step()

                # ==================== progress ==================
                loss = (loss / len(T_ys)).item()
                print_progress(i_iter + 1, len(loader_train), f'epoch {epoch:3d}: {loss:.1e}')

            avg_loss /= len(loader_train.dataset)
            self.writer.add_scalar('train/loss', avg_loss, epoch)

            # ==================Validation=========================
            self.validate(loader_valid, f_prefix, epoch)

            # ================= save loss & model ======================
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

        # Test and print
        # loss_test = validate(model, loader=loader_test, )
        # print(f'\nTest Loss: {arr2str(loss_test, n_decimal=4)}')

    def write_one(self, idx: int, out: np.ndarray, *,
                  group='validation', **kwargs) -> np.ndarray:
        """ write summary about one sample of output(and x and y optionally).

        :param idx:
        :param out:
        :param group:
        :param kwargs: dict(x, y, x_phase, y_phase)
        :return:
        """

        if kwargs:
            one_sample = kwargs
            do_reuse = False
        else:
            one_sample = self.one_sample
            do_reuse = True if self.recon_sample else False

        if do_reuse:
            y = self.recon_sample['y']
            x_phase = self.recon_sample['x_phase']
            y_phase = self.recon_sample['y_phase']
            y_wav = self.recon_sample['y_wav']
            pad_one = self.recon_sample['pad_one']

            snrseg_x = self.measure_x['SNRseg']
            pesq_x = self.measure_x['PESQ']
            stoi_x = self.measure_x['STOI']
        else:
            # F, T, 1
            x = one_sample['x'][..., -1:]
            y = one_sample['y'][..., -1:]

            np.sqrt(x, out=x)
            np.sqrt(y, out=y)

            x, x_phase = bnkr_equalize_(x, one_sample['x_phase'])
            y, y_phase = bnkr_equalize_(y, one_sample['y_phase'])

            snrseg_x = Measurement.calc_snrseg(y, x[:, :y.shape[1], :])

            # T,
            x_wav = reconstruct_wave(x, x_phase)
            y_wav = reconstruct_wave(y, y_phase)

            pesq_x, stoi_x = Measurement.calc_pesq_stoi(y_wav, x_wav[:y_wav.shape[0]])

            pad_one = np.ones(
                (y.shape[0], x.shape[1] - y.shape[1], y.shape[2])
            )
            vmin, vmax = 20 * LogModule.log(np.array((y.min(), y.max())))

            fig_x = draw_spectrogram(x)
            fig_y = draw_spectrogram(np.append(y, y.min() * pad_one, axis=1))

            self.writer.add_figure(f'{group}/1. Anechoic Spectrum', fig_y, idx)
            self.writer.add_figure(f'{group}/2. Reverberant Spectrum', fig_x, idx)

            self.writer.add_audio(
                f'{group}/1. Anechoic Wave', torch.from_numpy(y_wav), idx,
                sample_rate=cfg.Fs
            )
            self.writer.add_audio(
                f'{group}/2. Reverberant Wave', torch.from_numpy(x_wav), idx,
                sample_rate=cfg.Fs
            )

            self.recon_sample = dict(x=x, y=y,
                                     x_phase=x_phase, y_phase=y_phase,
                                     x_wav=x_wav, y_wav=y_wav,
                                     pad_one=pad_one)
            self.measure_x = dict(SNRseg=snrseg_x, PESQ=pesq_x, STOI=stoi_x)
            self.kwargs_fig = dict(vmin=vmin, vmax=vmax)

        out = out[..., -1:]
        np.sqrt(out, out=out)
        out = bnkr_equalize_(out)

        snrseg = Measurement.calc_snrseg(y, out)

        np.maximum(out, 0, out=out)

        out_wav = reconstruct_wave(out, x_phase[:, :out.shape[1], :],
                                   do_griffin_lim=True)
        out_wav_y_ph = reconstruct_wave(out, y_phase)

        pesq, stoi = Measurement.calc_pesq_stoi(y_wav, out_wav)

        out_wav = torch.from_numpy(out_wav)
        out_wav_y_ph = torch.from_numpy(out_wav_y_ph)

        fig_out = draw_spectrogram(np.append(out, y.min() * pad_one, axis=1),
                                   **self.kwargs_fig)

        self.writer.add_scalars(f'{group}/1. SNRseg',
                                dict(Reverberant=snrseg_x, Proposed=snrseg),
                                idx)
        self.writer.add_scalars(f'{group}/2. PESQ',
                                dict(Reverberant=pesq_x, Proposed=pesq),
                                idx)
        self.writer.add_scalars(f'{group}/3. STOI',
                                dict(Reverberant=stoi_x, Proposed=stoi),
                                idx)

        self.writer.add_figure(
            f'{group}/3. Estimated Anechoic Spectrum', fig_out, idx
        )
        self.writer.add_audio(
            f'{group}/3. Estimated Anechoic Wave', out_wav, idx,
            sample_rate=cfg.Fs
        )
        self.writer.add_audio(
            f'{group}/4. Estimated Wave with Anechoic Phase', out_wav_y_ph, idx,
            sample_rate=cfg.Fs
        )
        return np.array(((snrseg, pesq, stoi), (snrseg_x, pesq_x, stoi_x)))

    def validate(self, loader: DataLoader, f_prefix: str, epoch: int):
        """ Evaluate the performance of the model.

        :param loader: DataLoader to use.
        :param f_prefix: path and prefix of the result files.
        :param epoch:
        """

        with torch.no_grad():
            self.model.eval()

            wrote = False if self.writer else True
            avg_loss = torch.zeros(1).cuda(cfg.OUT_CUDA_DEV)

            print_progress(0, len(loader), f'validate : ')
            for i_iter, data in enumerate(loader):
                # =======================get data============================
                x, y = data['x'], data['y']  # B, F, T, C
                T_ys = data['T_ys']

                x_cuda, y_cuda = self.pre(x, y, loader.dataset)  # B, C, F, T

                # =========================forward=============================
                output = self.model(x_cuda)[..., :y_cuda.shape[-1]]

                # ==========================loss===============================
                loss = self.calc_loss(y_cuda, output, T_ys)
                avg_loss += loss

                loss = loss[-1] / len(T_ys)
                print_progress(i_iter + 1, len(loader), f'validate : {loss:.1e}')

                # ======================write summary==========================
                if not wrote:
                    # F, T, C
                    if not self.one_sample:
                        self.one_sample = IVDataset.decollate_padded(data, 0)

                    out_one = output[0, :, :, :T_ys[0]].permute(1, 2, 0)
                    out_one = loader.dataset.denormalize_(out_one, 'y')
                    out_one = out_one.cpu().numpy()

                    IVDataset.save_IV(f'{f_prefix}IV_{epoch}',
                                      **self.one_sample, out=out_one)

                    wrote = True

                    # Process(
                    #     target=write_one,
                    #     args=(x_one, y_one, x_ph_one, y_ph_one, out_one, epoch)
                    # ).start()
                    self.write_one(epoch, out_one)

            avg_loss /= len(loader.dataset)
            if self.writer:
                self.writer.add_scalar('validation/loss', avg_loss.item(), epoch)

            self.model.train()

            return avg_loss.item()

    def test(self, loader: DataLoader, dir_result: str, f_state_dict: str):
        self.model.load_state_dict(torch.load(f_state_dict)[0])
        self.writer = SummaryWriter(dir_result)
        group = os.path.basename(dir_result)
        atexit.register(self.writer.close)

        with torch.no_grad():
            self.model.eval()

            avg_measure = np.zeros((2, 3))
            t_start = time.time()

            print('0%:')
            for i_iter, data in enumerate(loader):
                # =======================get data============================
                x, y = data['x'], data['y']  # B, F, T, C
                T_ys = data['T_ys']

                x_cuda, y_cuda = self.pre(x, y, loader.dataset)  # B, C, F, T

                # =========================forward=============================
                output = self.model(x_cuda)[..., :y_cuda.shape[-1]]

                # ======================write summary==========================
                # F, T, C
                one_sample = IVDataset.decollate_padded(data, 0)

                out_one = output[0, :, :, :T_ys[0]].permute(1, 2, 0)
                out_one = loader.dataset.denormalize_(out_one, 'y')
                out_one = out_one.cpu().numpy()

                IVDataset.save_IV(os.path.join(dir_result, f'IV_{i_iter}'),
                                  **one_sample, out=out_one)

                # Process(
                #     target=write_one,
                #     args=(x_one, y_one, x_ph_one, y_ph_one, out_one, epoch)
                # ).start()
                measure = self.write_one(i_iter, out_one, group=group, **one_sample)
                avg_measure += measure

                print(f'{(i_iter + 1)/len(loader)*100}: {arr2str(measure)}')

            self.model.train()

        avg_measure /= len(loader.dataset)

        self.writer.add_text(f'{group}/1. Average SNRseg/Proposed', str(avg_measure[0, 0]))
        self.writer.add_text(f'{group}/1. Average SNRseg/Reverberant', str(avg_measure[1, 0]))
        self.writer.add_text(f'{group}/2. Average PESQ/Proposed', str(avg_measure[0, 1]))
        self.writer.add_text(f'{group}/2. Average PESQ/Reverberant', str(avg_measure[1, 1]))
        self.writer.add_text(f'{group}/3. Average STOI/Proposed', str(avg_measure[0, 2]))
        self.writer.add_text(f'{group}/3. Average STOI/Reverberant', str(avg_measure[1, 2]))

        print(time.strftime('%H h %M m', time.gmtime(time.time() - t_start)))
        print()
        print(arr2str(avg_measure))
