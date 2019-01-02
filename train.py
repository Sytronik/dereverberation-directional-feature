import matlab.engine
import atexit
import os
import time
from argparse import ArgumentParser
from multiprocessing import cpu_count, Process
from typing import NamedTuple, Tuple

import deepdish as dd
import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch import nn
from torch.utils.data import DataLoader

import mypath
from adamwr import AdamW, CosineLRWithRestarts
from audio_utils import delta, draw_spectrogram, Measurement, reconstruct_wave, bnkr_equalize_
from iv_dataset import IVDataset
from models import UNet
from utils import (arr2str,
                   MultipleOptimizer,
                   MultipleScheduler,
                   print_progress,
                   )

# ---------manually selected---------
CUDA_DEVICES = list(range(torch.cuda.device_count()))
OUT_CUDA_DEV = 1
CHANNELS = dict(x='all', y='alpha',
                fname_wav=False,
                )
CH_VALID = dict(**CHANNELS, x_phase=True, y_phase=True)

MODEL_NAME = 'UNet'
DIR_RESULT = mypath.path(MODEL_NAME)
if not os.path.isdir(DIR_RESULT):
    os.makedirs(DIR_RESULT)
F_SUMMARY = DIR_RESULT[:]
DIR_RESULT = os.path.join(DIR_RESULT, f'{MODEL_NAME}_')


class HyperParameters(NamedTuple):
    """
    Hyper Parameters of NN
    """
    n_per_frame: int

    UNet: Tuple

    n_epochs = 310
    batch_size = 4 * 9
    learning_rate = 5e-4
    n_file = 20 * 500

    # p = 0.5  # Dropout p

    # lr scheduler
    StepLR = dict(step_size=5, gamma=0.8)

    CosineAnnealingLR = dict(T_max=10,
                             eta_min=0,
                             )

    CosineLRWithRestarts = dict(restart_period=10,
                                t_mult=2,
                                eta_threshold=1.5,
                                )

    weight_decay = 1e-8  # Adam weight_decay

    weight_loss = (1, 0.7, 0.5)

    # def for_MLP(self) -> Tuple:
    #     n_input = self.L_cut_x * self.n_per_frame
    #     n_hidden = 17 * self.n_per_frame
    #     n_output = self.n_per_frame
    #     return (n_input, n_hidden, n_output, self.p)


metadata = dd.io.load(os.path.join(mypath.path('iv_train'), 'metadata.h5'))
Fs = metadata['Fs']
ch_in = 4 if CHANNELS['x'] == 'all' else 1
ch_out = 4 if CHANNELS['y'] == 'all' else 1
hp = HyperParameters(n_per_frame=metadata['N_freq'] * 4, UNet=(ch_in, ch_out, 32))
PERIOD_SAVE_STATE = hp.CosineLRWithRestarts['restart_period'] // 2
del metadata

FIRST_EPOCH = 0
writer = None

if __name__ == '__main__':
    # ------determined by sys argv------
    parser = ArgumentParser()
    parser.add_argument(
        '--from', type=int, nargs=1, dest='train_epoch', metavar='EPOCH',
        default=(-1,)
    )
    parser.add_argument(
        '--test', type=int, nargs=1, dest='test_epoch', metavar='EPOCH'
    )
    parser.add_argument(
        '--debug', '-d', dest='num_workers', action='store_const', const=0,
        default=cpu_count()
    )
    ARGS = parser.parse_args()
    if ARGS.test_epoch and ARGS.train_epoch[0] > -1:
        raise ValueError
    N_WORKERS = ARGS.num_workers  # number of cpu threads for dataloaders

    if ARGS.test_epoch:
        F_STATE_DICT = f'{DIR_RESULT}{ARGS.test_epoch[0]}.pt'
    else:
        FIRST_EPOCH = ARGS.train_epoch[0] + 1
        F_STATE_DICT = f'{DIR_RESULT}{ARGS.train_epoch[0]}.pt' if FIRST_EPOCH else ''
        writer = SummaryWriter(F_SUMMARY, purge_step=FIRST_EPOCH)
        atexit.register(writer.close)

    if F_STATE_DICT and not os.path.isfile(F_STATE_DICT):
        raise FileNotFoundError

    dd.io.save(f'{DIR_RESULT}hparams.h5', dict(hp._asdict()))

    # Dataset
    # Training + Validation Set
    dataset_temp = IVDataset('train', n_file=hp.n_file, **CHANNELS)
    dataset_train, dataset_valid = IVDataset.split(dataset_temp, (0.7, -1))
    dataset_valid.set_needs(**CH_VALID)

    # Test Set
    dataset_test = IVDataset('test', n_file=hp.n_file // 4,
                             random_sample=True, do_normalize=False, **CHANNELS)
    dataset_test.normalize_on_like(dataset_temp)
    del dataset_temp

    # DataLoader
    loader_train = DataLoader(dataset_train,
                              batch_size=hp.batch_size,
                              shuffle=True,
                              num_workers=N_WORKERS,
                              collate_fn=dataset_train.pad_collate,
                              )
    loader_valid = DataLoader(dataset_valid,
                              batch_size=hp.batch_size,
                              shuffle=False,
                              num_workers=N_WORKERS,
                              collate_fn=dataset_valid.pad_collate,
                              )
    loader_test = DataLoader(dataset_test,
                             batch_size=hp.batch_size,
                             shuffle=False,
                             num_workers=N_WORKERS,
                             collate_fn=dataset_test.pad_collate,
                             )

    # Model (Using Parallel GPU)
    # model = nn.DataParallel(DeepLabv3_plus(4, 4),
    model = nn.DataParallel(UNet(*hp.UNet),
                            device_ids=CUDA_DEVICES,
                            output_device=OUT_CUDA_DEV).cuda()

    # Loss Function
    criterion = nn.MSELoss(reduction='sum')
    # criterion = nn.BCELoss(reduction='sum')

    # Optimizer
    param1 = [p for m in model.modules() if not isinstance(m, nn.PReLU)
              for p in m.parameters()]
    param2 = [p for m in model.modules() if isinstance(m, nn.PReLU)
              for p in m.parameters()]
    optimizer = MultipleOptimizer(
        # torch.optim.Adam(
        AdamW(
            param1,
            lr=hp.learning_rate,
            weight_decay=hp.weight_decay,
        ) if param1 else None,
        # torch.optim.Adam(
        AdamW(
            param2,
            lr=hp.learning_rate,
        ) if param2 else None,
    )

    # Learning Rates Scheduler
    # scheduler = MultipleScheduler(torch.optim.lr_scheduler.StepLR,
    #                               optimizer, **hp.StepLR)
    # scheduler = MultipleScheduler(torch.optim.lr_scheduler.CosineAnnealingLR,
    #                               optimizer, **hp.CosineAnnealingLR,
    #                               last_epoch=FIRST_EPOCH-1)
    scheduler = MultipleScheduler(CosineLRWithRestarts,
                                  optimizer,
                                  batch_size=hp.batch_size,
                                  epoch_size=len(dataset_train),
                                  last_epoch=FIRST_EPOCH - 1,
                                  **hp.CosineLRWithRestarts)

    # Load State Dict
    if F_STATE_DICT:
        tup = torch.load(F_STATE_DICT)
        model.load_state_dict(tup[0])
        optimizer.load_state_dict(tup[1])


def calc_loss(y_cuda, output, T_ys):
    y_cuda = [y_cuda, None, None]
    output = [output, None, None]
    for i_dyn in range(1, 3):
        y_cuda[i_dyn], output[i_dyn] \
            = delta(y_cuda[i_dyn - 1], output[i_dyn - 1], axis=-1)

    # Loss
    loss = torch.zeros(1).cuda(OUT_CUDA_DEV)
    for i_dyn, (y_dyn, out_dyn) in enumerate(zip(y_cuda, output)):
        for T, item_y, item_out in zip(T_ys, y_dyn, out_dyn):
            loss += (hp.weight_loss[i_dyn] / int(T)
                     * criterion(item_out[:, :, :T - 4 * i_dyn],
                                 item_y[:, :, :T - 4 * i_dyn]))
    return loss


def pre(x: np.ndarray, y: np.ndarray,
        dataset: IVDataset) -> Tuple[torch.Tensor, torch.Tensor]:
    # B, F, T, C
    x_cuda = torch.tensor(x, dtype=torch.float32, device=0)
    y_cuda = torch.tensor(y, dtype=torch.float32, device=OUT_CUDA_DEV)

    x_cuda = dataset.normalize(x_cuda, 'x')
    y_cuda = dataset.normalize(y_cuda, 'y')

    # B, C, F, T
    x_cuda = x_cuda.permute(0, -1, -3, -2)
    y_cuda = y_cuda.permute(0, -1, -3, -2)

    return x_cuda, y_cuda


def train():
    avg_loss = torch.zeros(1, device=OUT_CUDA_DEV)

    # Start Training
    for epoch in range(FIRST_EPOCH, hp.n_epochs):
        t_start = time.time()

        print()
        print_progress(0, len(loader_train), f'epoch {epoch:3d}:')
        scheduler.step()
        for i_iter, data in enumerate(loader_train):
            # ==================get data=====================
            x, y = data['x'], data['y']  # B, F, T, C
            T_ys = data['T_ys']

            x_cuda, y_cuda = pre(x, y, loader_train.dataset)  # B, C, F, T

            # ===================forward=====================
            output = model(x_cuda)[..., :y_cuda.shape[-1]]  # B, C, F, T

            loss = calc_loss(y_cuda, output, T_ys)
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

        avg_loss /= len(dataset_train)
        writer.add_scalar('train/loss', avg_loss, epoch)

        # ==================Validation=========================
        validate(epoch)

        # ================= save loss & model ======================
        if epoch % PERIOD_SAVE_STATE == PERIOD_SAVE_STATE - 1:
            torch.save(
                (model.state_dict(),
                 optimizer.state_dict(),
                 ),
                f'{DIR_RESULT}{epoch}.pt'
            )

        # Time
        tt = time.strftime('%M min %S sec', time.gmtime(time.time() - t_start))
        print(f'epoch {epoch:3d}: {tt}')

    # Test and print
    loss_test = validate(f'{DIR_RESULT}IV_test.mat', loader=loader_test)
    print(f'\nTest Loss: {arr2str(loss_test, n_decimal=4)}')


def write_one(x_one, y_one, x_ph_one, y_ph_one, out_one, epoch):
    x_one, x_ph_one = bnkr_equalize_(x_one, x_ph_one)
    y_one, y_ph_one = bnkr_equalize_(y_one, y_ph_one)
    out_one = bnkr_equalize_(out_one)

    snrseg = Measurement.calc_snrseg(y_one, out_one)

    np.maximum(out_one, 0, out=out_one)

    x_wav_one = reconstruct_wave(x_one, x_ph_one)
    y_wav_one = reconstruct_wave(y_one, y_ph_one)
    out_wav_one = reconstruct_wave(out_one, x_ph_one[:, :out_one.shape[1], :],
                                   do_griffin_lim=True)
    out_wav_y_phase = reconstruct_wave(out_one, y_ph_one)

    pesq, stoi = Measurement.calc_pesq_stoi(y_wav_one, out_wav_one)

    x_wav_one = torch.from_numpy(x_wav_one)
    y_wav_one = torch.from_numpy(y_wav_one)
    out_wav_one = torch.from_numpy(out_wav_one)
    out_wav_y_phase = torch.from_numpy(out_wav_y_phase)

    pad_one = np.ones(
        (y_one.shape[0], x_one.shape[1] - y_one.shape[1], y_one.shape[2])
    )

    fig_x = draw_spectrogram(x_one)
    fig_y = draw_spectrogram(np.append(y_one, y_one.min()*pad_one, axis=1))
    fig_out = draw_spectrogram(np.append(out_one, out_one.min()*pad_one, axis=1))

    group = 'validation'
    writer.add_scalar(f'{group}/SNRseg', snrseg, epoch)
    writer.add_scalar(f'{group}/pesq', pesq, epoch)
    writer.add_scalar(f'{group}/stoi', stoi, epoch)

    writer.add_figure(f'{group}/1. Anechoic Spectrum', fig_y, epoch)
    writer.add_figure(f'{group}/2. Reverberant Spectrum', fig_x, epoch)
    writer.add_figure(f'{group}/3. Estimated Anechoic Spectrum', fig_out, epoch)

    writer.add_audio(
        f'{group}/1. Anechoic Wave', y_wav_one, epoch, sample_rate=Fs
    )
    writer.add_audio(
        f'{group}/2. Reverberant Wave', x_wav_one, epoch, sample_rate=Fs
    )
    writer.add_audio(
        f'{group}/3. Estimated Anechoic Wave', out_wav_one, epoch, sample_rate=Fs
    )
    writer.add_audio(
        f'{group}/4. Estimated Wave with Anechoic Phase', out_wav_y_phase, epoch, sample_rate=Fs
    )


def validate(epoch=FIRST_EPOCH, loader: DataLoader = None):
    """
    Evaluate the performance of the model.
    loader: DataLoader to use.
    fname: filename of the result. If None, don't save the result.
    """
    if not loader:
        loader = loader_valid

    with torch.no_grad():
        model.eval()

        wrote = False if writer else True
        avg_loss = torch.zeros(1).cuda(OUT_CUDA_DEV)

        print_progress(0, len(loader), f'{"validate":<9}: ')
        for i_iter, data in enumerate(loader):
            # =======================get data============================
            x, y = data['x'], data['y']  # B, F, T, C
            T_ys = data['T_ys']

            x_cuda, y_cuda = pre(x, y, loader.dataset)  # B, C, F, T

            # =========================forward=============================
            output = model(x_cuda)[..., :y_cuda.shape[-1]]

            # ==========================loss================================
            loss = calc_loss(y_cuda, output, T_ys)
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

                IVDataset.save_IV(f'{DIR_RESULT}IV_{epoch}',
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
                write_one(x_one, y_one, x_ph_one, y_ph_one, out_one, epoch)

        avg_loss /= len(loader.dataset)
        if writer:
            writer.add_scalar('validation/loss', avg_loss.item(), epoch)

        model.train()

        return avg_loss.item()


def run():
    if ARGS.test_epoch:
        loss_test = validate(loader=loader_test)

        print(f'Test Loss: {arr2str(loss_test, n_decimal=4)}')
    else:
        train()


if __name__ == '__main__':
    run()
    if writer:
        writer.close()
