from pathlib import Path

import numpy as np
import torch
from numpy import ndarray
from tensorboardX import SummaryWriter

from hparams import hp
from audio_utils import (bnkr_equalize_,
                         calc_snrseg,
                         calc_using_eval_module,
                         draw_spectrogram,
                         reconstruct_wave,
                         EVAL_METRICS,
                         )
from dirspecgram import LogModule


class CustomWriter(SummaryWriter):
    def __init__(self, *args, group='', **kwargs):
        super().__init__(*args, **kwargs)
        self.group = group
        if group == 'valid':
            dict_custom_scalars = dict(loss=['Multiline', ['loss/train',
                                                           'loss/valid']])
        else:
            dict_custom_scalars = dict()

        dict_custom_scalars['1_SNRseg'] = ['Multiline', [f'{group}/1_SNRseg/Reverberant',
                                                         f'{group}/1_SNRseg/Proposed']]

        for i, m in enumerate(EVAL_METRICS):
            dict_custom_scalars[f'{i + 2}_{m}'] = [
                'Multiline', [f'{group}/{i + 2}_{m}/Reverberant',
                              f'{group}/{i + 2}_{m}/Proposed']
            ]

        self.add_custom_scalars({group: dict_custom_scalars})
        self.one_sample = dict()
        self.recon_sample = dict()
        self.measure_x = dict()
        self.kwargs_fig = dict()
        self.y_scale = 1.

    def write_one(self, step: int, group='',
                  out: ndarray = None,
                  out_phase: ndarray = None, out_bpd: ndarray = None,
                  eval_with_y_ph=False, **kwargs: ndarray) -> ndarray:
        """ write summary about one sample of output(and x and y optionally).

        :param step:
        :param group:
        :param out:
        :param out_phase:
        :param out_bpd:
        :param eval_with_y_ph: determine if `out` is evaluated with `y_phase`
        :param kwargs: keywords can be [x, y, x_phase, y_phase]

        :return: evaluation result
        """

        assert out is not None
        if kwargs:
            one_sample = kwargs
            self.one_sample = kwargs
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
            y_scale = self.y_scale
        else:
            # F, T, 1
            x = one_sample['x'][..., -1:] / 2 / np.sqrt(4 * np.pi)  # warning
            y = one_sample['y'][..., -1:]

            if hp.do_bnkr_eq:
                x, x_phase = bnkr_equalize_(x, one_sample['x_phase'].copy())
                y, y_phase = bnkr_equalize_(y, one_sample['y_phase'].copy())
            else:
                x_phase = one_sample['x_phase'].copy()
                y_phase = one_sample['y_phase'].copy()

            snrseg_x = calc_snrseg(y, x[:, :y.shape[1], :])

            # T,
            x_wav = reconstruct_wave(x, x_phase)
            y_wav = reconstruct_wave(y, y_phase)
            y_scale = np.abs(y_wav).max() / 0.707

            odict_eval_x = calc_using_eval_module(y_wav, x_wav[:y_wav.shape[0]])

            pad_one = np.ones(
                (y.shape[0], x.shape[1] - y.shape[1], y.shape[2])
            )
            vmin, vmax = 20 * LogModule.log(np.array((y.min(), y.max())))

            fig_x = draw_spectrogram(x)
            fig_y = draw_spectrogram(np.append(y, y.min() * pad_one, axis=1))

            self.add_figure(f'{group}/1_Anechoic Spectrum', fig_y, step)
            self.add_figure(f'{group}/2_Reverberant Spectrum', fig_x, step)

            self.add_audio(f'{group}/1_Anechoic Wave',
                           torch.from_numpy(y_wav / y_scale),
                           step,
                           sample_rate=hp.fs)
            self.add_audio(f'{group}/2_Reverberant Wave',
                           torch.from_numpy(x_wav / (np.abs(x_wav).max() / 0.707)),
                           step,
                           sample_rate=hp.fs)

            if 'x_bpd' in one_sample:
                fig_x_bpd = draw_spectrogram(
                    one_sample['x_bpd'],
                    to_db=False, vmin=-np.pi, vmax=np.pi
                )
                fig_y_bpd = draw_spectrogram(
                    np.append(one_sample['y_bpd'], 0. * pad_one, axis=1),
                    to_db=False, vmin=-np.pi, vmax=np.pi
                )
                self.add_figure(f'{group}/1_Anechoic BPD', fig_y_bpd, step)
                self.add_figure(f'{group}/2_Reverberant BPD', fig_x_bpd, step)

            self.recon_sample = dict(x=x, y=y,
                                     x_phase=x_phase, y_phase=y_phase,
                                     x_wav=x_wav, y_wav=y_wav,
                                     pad_one=pad_one)
            self.measure_x = dict(SNRseg=snrseg_x, odict_eval=odict_eval_x)
            self.kwargs_fig = dict(vmin=vmin, vmax=vmax)
            self.y_scale = y_scale

        out = out[..., -1:]
        if np.iscomplexobj(out):
            if hp.do_bnkr_eq:
                out = bnkr_equalize_(out)

            snrseg = calc_snrseg(y, np.abs(out))
            out_wav = reconstruct_wave(out)

            odict_eval = calc_using_eval_module(y_wav, out_wav)
        else:
            np.maximum(out, 0, out=out)
            # np.sqrt(out, out=out)
            if hp.DO_B_EQ:
                out, out_phase = bnkr_equalize_(out, out_phase)

            snrseg = calc_snrseg(y, out)

            if out_phase is None:
                out_wav = reconstruct_wave(out, x_phase[:, :out.shape[1], :],
                                           n_iter=hp.n_glim_iter)
            else:
                if hp.use_glim:
                    out_wav = reconstruct_wave(out, out_phase, n_iter=hp.n_glim_iter)
                else:
                    out_wav = reconstruct_wave(out, out_phase)
            out_wav_y_ph = reconstruct_wave(out, y_phase)

            odict_eval = calc_using_eval_module(
                y_wav,
                out_wav_y_ph if eval_with_y_ph else out_wav
            )
            self.add_audio(f'{group}/4_Estimated Wave with Anechoic Phase',
                           torch.from_numpy(out_wav_y_ph / y_scale),
                           step,
                           sample_rate=hp.fs)

        self.add_scalars(f'{group}/1_SNRseg',
                         dict(Reverberant=snrseg_x,
                              Proposed=snrseg),
                         step)
        for idx, key in enumerate(odict_eval.keys()):
            self.add_scalars(f'{group}/{2 + idx}_{key}',
                             dict(Reverberant=odict_eval_x[key],
                                  Proposed=odict_eval[key]),
                             step)

        fig_out = draw_spectrogram(np.append(out, y.min() * pad_one, axis=1),
                                   **self.kwargs_fig)
        self.add_figure(f'{group}/3_Estimated Anechoic Spectrum', fig_out, step)

        self.add_audio(f'{group}/3_Estimated Anechoic Wave',
                       torch.from_numpy(out_wav / y_scale),
                       step,
                       sample_rate=hp.Fs)

        if out_bpd is not None:
            fig_out_bpd = draw_spectrogram(
                np.append(out_bpd, 0. * pad_one, axis=1),
                to_db=False, vmin=-np.pi, vmax=np.pi
            )
            self.add_figure(f'{group}/3_Estimated BPD', fig_out_bpd, step)

        return np.array([[snrseg, *(odict_eval.values())],
                         [snrseg_x, *(odict_eval_x.values())]])
