import numpy as np
import torch
from numpy import ndarray
from tensorboardX import SummaryWriter

from hparams import hp
from audio_utils import (calc_snrseg,
                         calc_using_eval_module,
                         draw_spectrogram,
                         reconstruct_wave,
                         EVAL_METRICS,
                         )
from dataset import LogModule


class CustomWriter(SummaryWriter):
    def __init__(self, *args, group='', **kwargs):
        super().__init__(*args, **kwargs)
        self.group = group
        if group == 'train':
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
        self.reused_sample = dict()
        self.measure_x = dict()
        self.kwargs_fig = dict()
        self.y_scale = 1.

    def write_one(self, step: int,
                  out: ndarray = None,
                  eval_with_y_ph=False, **kwargs: ndarray) -> ndarray:
        """ write summary about one sample of output(and x and y optionally).

        :param step:
        :param out:
        :param eval_with_y_ph: determine if `out` is evaluated with `y_phase`
        :param kwargs: keywords can be [x, y, x_phase, y_phase]

        :return: evaluation result
        """

        assert out is not None
        if kwargs:
            # F, T, 1
            x = kwargs['x'][..., -1:]
            y = kwargs['y'][..., -1:]

            x_phase = kwargs['x_phase']
            y_phase = kwargs['y_phase']

            snrseg_x = calc_snrseg(y, x[:, :y.shape[1], :])

            # T,
            x_wav = reconstruct_wave(x, x_phase)
            y_wav = reconstruct_wave(y, y_phase)
            self.y_scale = np.abs(y_wav).max() / 0.5

            odict_eval_x = calc_using_eval_module(y_wav, x_wav[:y_wav.shape[0]])

            ymin = y[y > 0].min()
            pad_min = np.empty(
                (y.shape[0], x.shape[1] - y.shape[1], y.shape[2])
            )
            pad_min.fill(ymin)
            vmin, vmax = 20 * LogModule.log(np.array((ymin, y.max())))

            fig_x = draw_spectrogram(x)
            fig_y = draw_spectrogram(np.append(y, pad_min, axis=1))

            self.add_figure(f'{self.group}/1_Anechoic Spectrum', fig_y, step)
            self.add_figure(f'{self.group}/2_Reverberant Spectrum', fig_x, step)

            self.add_audio(f'{self.group}/1_Anechoic Wave',
                           torch.from_numpy(y_wav / self.y_scale),
                           step,
                           sample_rate=hp.fs)
            self.add_audio(f'{self.group}/2_Reverberant Wave',
                           torch.from_numpy(x_wav / (np.abs(x_wav).max() / 0.5)),
                           step,
                           sample_rate=hp.fs)

            self.reused_sample = dict(x=x, y=y,
                                      x_phase=x_phase, y_phase=y_phase,
                                      y_wav=y_wav,
                                      pad_min=pad_min)
            self.measure_x = dict(SNRseg=snrseg_x, odict_eval=odict_eval_x)
            self.kwargs_fig = dict(vmin=vmin, vmax=vmax)
        else:
            assert self.reused_sample
            y = self.reused_sample['y']
            x_phase = self.reused_sample['x_phase']
            y_phase = self.reused_sample['y_phase']
            y_wav = self.reused_sample['y_wav']
            pad_min = self.reused_sample['pad_min']

            snrseg_x = self.measure_x['SNRseg']
            odict_eval_x = self.measure_x['odict_eval']

        np.maximum(out, 0, out=out)

        snrseg = calc_snrseg(y, out)

        out_wav = reconstruct_wave(out, x_phase[:, :out.shape[1], :],
                                   n_iter=hp.n_glim_iter)
        out_wav_y_ph = reconstruct_wave(out, y_phase)

        odict_eval = calc_using_eval_module(
            y_wav,
            out_wav_y_ph if eval_with_y_ph else out_wav
        )

        self.add_scalar(f'{self.group}/1_SNRseg/Reverberant', snrseg_x, step)
        self.add_scalar(f'{self.group}/1_SNRseg/Proposed', snrseg, step)
        for i, m in enumerate(odict_eval.keys()):
            j = i + 2
            self.add_scalar(f'{self.group}/{j}_{m}/Reverberant', odict_eval_x[m], step)
            self.add_scalar(f'{self.group}/{j}_{m}/Proposed', odict_eval[m], step)

        fig_out = draw_spectrogram(np.append(out, pad_min, axis=1), **self.kwargs_fig)
        self.add_figure(f'{self.group}/3_Estimated Anechoic Spectrum', fig_out, step)

        self.add_audio(f'{self.group}/3_Estimated Anechoic Wave',
                       torch.from_numpy(out_wav / self.y_scale),
                       step,
                       sample_rate=hp.fs)
        self.add_audio(f'{self.group}/4_Estimated Wave with Anechoic Phase',
                       torch.from_numpy(out_wav_y_ph / self.y_scale),
                       step,
                       sample_rate=hp.fs)

        return np.array([[snrseg, *(odict_eval.values())],
                         [snrseg_x, *(odict_eval_x.values())]])
