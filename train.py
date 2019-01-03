import atexit
import os
import time
from typing import Tuple

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch import nn
from torch.utils.data import DataLoader

from adamwr import AdamW, CosineLRWithRestarts
from audio_utils import delta, draw_spectrogram, Measurement, reconstruct_wave, bnkr_equalize_
from iv_dataset import IVDataset
from models import UNet
from utils import (MultipleOptimizer,
                   MultipleScheduler,
                   print_progress,
                   )
import config as cfg


class Trainer:
    def calc_loss(self, y_cuda, output, T_ys):
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

    @staticmethod
    def pre(x: np.ndarray, y: np.ndarray,
            dataset: IVDataset) -> Tuple[torch.Tensor, torch.Tensor]:
        # B, F, T, C
        x_cuda = torch.tensor(x, dtype=torch.float32, device=0)
        y_cuda = torch.tensor(y, dtype=torch.float32, device=cfg.OUT_CUDA_DEV)

        x_cuda = dataset.normalize(x_cuda, 'x')
        y_cuda = dataset.normalize(y_cuda, 'y')

        # B, C, F, T
        x_cuda = x_cuda.permute(0, -1, -3, -2)
        y_cuda = y_cuda.permute(0, -1, -3, -2)

        return x_cuda, y_cuda

    def __init__(self, model_name, dir_result):
        self.writer = None
        # Model (Using Parallel GPU)
        # model = nn.DataParallel(DeepLabv3_plus(4, 4),
        if model_name == 'UNet':
            module = UNet
        else:
            raise NotImplementedError
        self.model = nn.DataParallel(module(*cfg.hp.get_for_UNet()),
                                     device_ids=cfg.CUDA_DEVICES,
                                     output_device=cfg.OUT_CUDA_DEV).cuda()

        self.dir_result = dir_result

        self.f_prefix = os.path.join(dir_result, f'{model_name}_')
        self.criterion = cfg.criterion

    def train(self,
              loader_train: DataLoader, loader_valid: DataLoader,
              first_epoch=0, f_state_dict=''):
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
            self.model.load_state_dict(tup[0])
            optimizer.load_state_dict(tup[1])
        avg_loss = torch.zeros(1, device=cfg.OUT_CUDA_DEV)

        self.writer = SummaryWriter(self.dir_result, purge_step=first_epoch)
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
                optimizer.step()
                scheduler.batch_step()

                # ==================== progress ==================
                loss = (loss / len(T_ys)).item()
                print_progress(i_iter + 1, len(loader_train), f'epoch {epoch:3d}: {loss:.1e}')

            avg_loss /= len(loader_train.dataset)
            self.writer.add_scalar('train/loss', avg_loss, epoch)

            # ==================Validation=========================
            self.validate(loader_valid, epoch)

            # ================= save loss & model ======================
            if epoch % cfg.PERIOD_SAVE_STATE == cfg.PERIOD_SAVE_STATE - 1:
                torch.save(
                    (self.model.state_dict(),
                     optimizer.state_dict(),
                     ),
                    f'{self.f_prefix}{epoch}.pt'
                )

            # Time
            tt = time.strftime('%M min %S sec', time.gmtime(time.time() - t_start))
            print(f'epoch {epoch:3d}: {tt}')

        # Test and print
        # loss_test = validate(model, loader=loader_test, )
        # print(f'\nTest Loss: {arr2str(loss_test, n_decimal=4)}')

    def write_one(self, x_one, y_one, x_ph_one, y_ph_one, out_one, epoch):
        x_one, x_ph_one = bnkr_equalize_(x_one, x_ph_one)
        y_one, y_ph_one = bnkr_equalize_(y_one, y_ph_one)
        out_one = bnkr_equalize_(out_one)

        snrseg = Measurement.calc_snrseg(y_one, out_one)

        np.maximum(out_one, 0, out=out_one)

        x_wav_one, Fs = reconstruct_wave(x_one, x_ph_one)
        y_wav_one, _ = reconstruct_wave(y_one, y_ph_one)
        out_wav_one, _ = reconstruct_wave(out_one, x_ph_one[:, :out_one.shape[1], :],
                                          do_griffin_lim=True)
        out_wav_y_phase, _ = reconstruct_wave(out_one, y_ph_one)

        pesq, stoi = Measurement.calc_pesq_stoi(y_wav_one, out_wav_one)

        x_wav_one = torch.from_numpy(x_wav_one)
        y_wav_one = torch.from_numpy(y_wav_one)
        out_wav_one = torch.from_numpy(out_wav_one)
        out_wav_y_phase = torch.from_numpy(out_wav_y_phase)

        pad_one = np.ones(
            (y_one.shape[0], x_one.shape[1] - y_one.shape[1], y_one.shape[2])
        )
        fig_out = draw_spectrogram(np.append(out_one, out_one.min() * pad_one, axis=1))

        group = 'validation'
        self.writer.add_scalar(f'{group}/SNRseg', snrseg, epoch)
        self.writer.add_scalar(f'{group}/pesq', pesq, epoch)
        self.writer.add_scalar(f'{group}/stoi', stoi, epoch)

        self.writer.add_figure(f'{group}/3. Estimated Anechoic Spectrum', fig_out, epoch)

        self.writer.add_audio(
            f'{group}/3. Estimated Anechoic Wave', out_wav_one, epoch, sample_rate=Fs
        )
        self.writer.add_audio(
            f'{group}/4. Estimated Wave with Anechoic Phase', out_wav_y_phase, epoch,
            sample_rate=Fs
        )

        if epoch == 0:
            fig_x = draw_spectrogram(x_one)
            fig_y = draw_spectrogram(np.append(y_one, y_one.min() * pad_one, axis=1))
            self.writer.add_figure(f'{group}/1. Anechoic Spectrum', fig_y, epoch)
            self.writer.add_figure(f'{group}/2. Reverberant Spectrum', fig_x, epoch)

            self.writer.add_audio(
                f'{group}/1. Anechoic Wave', y_wav_one, epoch, sample_rate=Fs
            )
            self.writer.add_audio(
                f'{group}/2. Reverberant Wave', x_wav_one, epoch, sample_rate=Fs
            )

    def validate(self, loader: DataLoader, epoch):
        """
        Evaluate the performance of the model.
        loader: DataLoader to use.
        fname: filename of the result. If None, don't save the result.
        """

        with torch.no_grad():
            self.model.eval()

            wrote = False if self.writer else True
            avg_loss = torch.zeros(1).cuda(cfg.OUT_CUDA_DEV)

            print_progress(0, len(loader), f'{"validate":<9}: ')
            for i_iter, data in enumerate(loader):
                # =======================get data============================
                x, y = data['x'], data['y']  # B, F, T, C
                T_ys = data['T_ys']

                x_cuda, y_cuda = self.pre(x, y, loader.dataset)  # B, C, F, T

                # =========================forward=============================
                output = self.model(x_cuda)[..., :y_cuda.shape[-1]]

                # ==========================loss================================
                loss = self.calc_loss(y_cuda, output, T_ys)
                avg_loss += loss

                loss = loss[-1] / len(T_ys)
                print_progress(i_iter + 1, len(loader), f'{"validate":<9}: {loss:.1e}')

                # ======================write summary=============================
                if not wrote:
                    # F, T, C
                    one_sample = IVDataset.decollate_padded(data, 0)

                    out_one = output[0, :, :, :T_ys[0]].permute(1, 2, 0)
                    out_one = loader.dataset.denormalize_(out_one, 'y')
                    out_one = out_one.cpu().numpy()

                    IVDataset.save_IV(f'{self.f_prefix}IV_{epoch}',
                                      **one_sample, out=out_one)

                    x_one = one_sample['x'][..., -1:]
                    x_ph_one = one_sample['x_phase'][..., -1:]
                    y_one = one_sample['y'][..., -1:]
                    y_ph_one = one_sample['y_phase'][..., -1:]
                    out_one = out_one[..., -1:]

                    wrote = True

                    # Process(
                    #     target=write_one,
                    #     args=(x_one, y_one, x_ph_one, y_ph_one, out_one, epoch)
                    # ).start()
                    self.write_one(x_one, y_one, x_ph_one, y_ph_one, out_one, epoch)

            avg_loss /= len(loader.dataset)
            if self.writer:
                self.writer.add_scalar('validation/loss', avg_loss.item(), epoch)

            self.model.train()

            return avg_loss.item()

# def run():
#     if ARGS.test_epoch:
#         loss_test = validate(loader=loader_test)
#
#         print(f'Test Loss: {arr2str(loss_test, n_decimal=4)}')
#     else:
#         train()
#
#
# if __name__ == '__main__':
#     run()
#     if writer:
#         writer.close()
