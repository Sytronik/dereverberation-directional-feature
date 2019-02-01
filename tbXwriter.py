import atexit
from os.path import join as pathjoin

import numpy as np
import torch
from tensorboardX import SummaryWriter

import config as cfg
from audio_utils import (bnkr_equalize_,
                         calc_snrseg,
                         calc_using_eval_module,
                         draw_spectrogram,
                         reconstruct_wave,
                         wave_scale_fix,
                         )
from normalize import LogInterface as LogModule


class CustomWriter(SummaryWriter):
    __slots__ = ('one_sample', 'recon_sample', 'measure_x', 'kwargs_fig')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        atexit.register(self.close)
        self.one_sample = dict()
        self.recon_sample = dict()
        self.measure_x = dict()
        self.kwargs_fig = dict()

    def close(self):
        self.export_scalars_to_json(pathjoin(self.log_dir, 'scalars.json'))

    def write_one(self, step: int, group='',
                  out: np.ndarray = None, out_phase: np.ndarray = None,
                  eval_with_y_ph=False, **kwargs) -> np.ndarray:
        """ write summary about one sample of output(and x and y optionally).

        :param step:
        :param group:
        :param out:
        :param out_phase:
        :param eval_with_y_ph:
        :param kwargs: dict(x, y, x_phase, y_phase)

        :return:
        """

        assert out is not None
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
            odict_eval_x = self.measure_x['odict_eval']
        else:
            # F, T, 1
            x = one_sample['x'][..., -1:]
            y = one_sample['y'][..., -1:]

            # x = np.sqrt(x)
            # y = np.sqrt(y)

            if cfg.DO_B_EQ:
                x, x_phase = bnkr_equalize_(x, one_sample['x_phase'][:])
                y, y_phase = bnkr_equalize_(y, one_sample['y_phase'][:])
            else:
                x_phase = one_sample['x_phase'][:]
                y_phase = one_sample['y_phase'][:]

            snrseg_x = calc_snrseg(y, x[:, :y.shape[1], :])

            # T,
            x_wav = reconstruct_wave(x, x_phase)
            y_wav = reconstruct_wave(y, y_phase)

            odict_eval_x = calc_using_eval_module(y_wav, x_wav[:y_wav.shape[0]])

            pad_one = np.ones(
                (y.shape[0], x.shape[1] - y.shape[1], y.shape[2])
            )
            vmin, vmax = 20 * LogModule.log(np.array((y.min(), y.max())))

            fig_x = draw_spectrogram(x)
            fig_y = draw_spectrogram(np.append(y, y.min() * pad_one, axis=1))

            y_wav = wave_scale_fix(y_wav, message='y_wav')
            x_wav = wave_scale_fix(x_wav, message='x_wav')

            self.add_figure(f'{group}/1. Anechoic Spectrum', fig_y, step)
            self.add_figure(f'{group}/2. Reverberant Spectrum', fig_x, step)

            self.add_audio(
                f'{group}/1. Anechoic Wave', torch.from_numpy(y_wav), step,
                sample_rate=cfg.Fs
            )
            self.add_audio(
                f'{group}/2. Reverberant Wave', torch.from_numpy(x_wav), step,
                sample_rate=cfg.Fs
            )

            self.recon_sample = dict(x=x, y=y,
                                     x_phase=x_phase, y_phase=y_phase,
                                     x_wav=x_wav, y_wav=y_wav,
                                     pad_one=pad_one)
            self.measure_x = dict(SNRseg=snrseg_x, odict_eval=odict_eval_x)
            self.kwargs_fig = dict(vmin=vmin, vmax=vmax)

        out = out[..., -1:]
        if np.iscomplexobj(out):
            if cfg.DO_B_EQ:
                out = bnkr_equalize_(out)

            snrseg = calc_snrseg(y, np.abs(out))
            out_wav = reconstruct_wave(out)

            odict_eval = calc_using_eval_module(y_wav, out_wav)
        else:
            np.maximum(out, 0, out=out)
            # np.sqrt(out, out=out)
            if cfg.DO_B_EQ:
                if out_phase is None:
                    out = bnkr_equalize_(out)
                else:
                    out, out_phase = bnkr_equalize_(out, out_phase)

            snrseg = calc_snrseg(y, out)

            if out_phase is None:
                out_wav = reconstruct_wave(out, x_phase[:, :out.shape[1], :],
                                           n_iter=cfg.N_GRIFFIN_LIM)
            else:
                out_wav = reconstruct_wave(out, out_phase)
            out_wav_y_ph = reconstruct_wave(out, y_phase)

            odict_eval = calc_using_eval_module(y_wav,
                                                out_wav_y_ph if eval_with_y_ph else out_wav)
            out_wav_y_ph = torch.from_numpy(
                wave_scale_fix(out_wav_y_ph, message='out_wav_y_ph')
            )
            self.add_audio(
                f'{group}/4. Estimated Wave with Anechoic Phase', out_wav_y_ph, step,
                sample_rate=cfg.Fs
            )

        out_wav = torch.from_numpy(
            wave_scale_fix(out_wav, message='out_wav')
        )

        fig_out = draw_spectrogram(np.append(out, y.min() * pad_one, axis=1),
                                   **self.kwargs_fig)

        self.add_scalars(f'{group}/1. SNRseg',
                         dict(Reverberant=snrseg_x,
                              Proposed=snrseg),
                         step)
        # for idx, key in enumerate(odict_eval.keys()):
        for idx, key in enumerate(odict_eval.keys()):
            self.add_scalars(f'{group}/{2+idx}. {key}',
                             dict(Reverberant=odict_eval_x[key],
                                  Proposed=odict_eval[key]),
                             step)

        self.add_figure(
            f'{group}/3. Estimated Anechoic Spectrum', fig_out, step
        )
        self.add_audio(
            f'{group}/3. Estimated Anechoic Wave', out_wav, step,
            sample_rate=cfg.Fs
        )

        return np.array([[snrseg]+list(odict_eval.values()),
                         [snrseg_x]+list(odict_eval_x.values())])
