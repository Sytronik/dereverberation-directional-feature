# %%
from pathlib import Path

import torch
from captum.attr import IntegratedGradients
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import librosa

from models import UNet
from dataset import DirSpecDataset
from hparams import hp
from audio_utils import vec2hsv
from utils import full_extent

path_root = Path('./backup/result_23_10')

# uncomment below lines before you find the proper color axis.
# after find proper color axis values that can be applied for all plots named `titles`,
# set comment below lines and set the variable be those values
# grad_vec_r_max = 0
# vmin_grad = None
# vmax_grad = None

# SIV
# hp.feature = 'SIV'
path_state_dict = path_root / 'SIV/train/59.pt'
# grad_vec_r_max = 3.11
# vmin_grad = -1.94
# vmax_grad = -vmin_grad

# DV
hp.feature = 'DV'
path_state_dict = path_root / 'DV/train/59.pt'
grad_vec_r_max = 3.13
vmin_grad = -2.37
vmax_grad = -vmin_grad

hp.room_test = 'room1+2+3'
kind = 'unseen'
idx_sample = 1  # % 2 == 1 --> room2

titles = ['max_formant', 'tail_formant', 'max_unvoiced', 'tail_unvoiced']
subtitles = ['input_vec', 'input_mag', 'true_mag', 'ig_vec', 'ig_mag']

t_interval = 25  # no. of time frames to show
xticks = [0, 0.4, 0.8]

fs = 16000
l_hop = 256
n_freq = 257
log_eps_pow = 1e-10
log_eps_grad = 1e-3

cmap = plt.get_cmap('gray')

path_fig = path_root / 'figures'
path_fig.mkdir(exist_ok=True)

# %% hp
hp.init_dependent_vars()

# %% model & dataset
# device = 'cuda:0'  # can't be run on cuda due to out of memory
device = 'cpu'
model = UNet(4, 1, 64, 4).to(device)
state_dict = torch.load(path_state_dict, map_location=device)[0]
model.load_state_dict(state_dict)

# Dataset
dataset_temp = DirSpecDataset('train')
dataset_test = DirSpecDataset(kind,
                              dataset_temp.norm_modules,
                              **hp.channels_w_ph)

# %% retrieve data
data = dataset_test.pad_collate([dataset_test[idx_sample]])

x, y = data['normalized_x'], data['normalized_y']
x, y = x.to(device), y.to(device)
y_denorm = data['y']
y_denorm = y_denorm.permute(0, 3, 1, 2)  # B, C, F, T

x.requires_grad = True

baseline = torch.zeros_like(data['x'])
baseline = dataset_temp.normalize(x=baseline)
baseline = baseline.permute(0, 3, 1, 2)  # B, C, F, T
baseline = baseline.to(device)

# %% set targets
targets = []

# 'max_formant': maximum magnitude bin which is formant
target = np.unravel_index(y_denorm.argmax().item(), y_denorm.shape)[1:]
targets.append(target)

# 'tail_formant': after the formant (silence)
target = list(target)
target[2] += 12
targets.append(tuple(target))

# 'max_unvoiced': unvoiced speech
target = [0, 192, 0]
target[2] = y_denorm[0, 0, 128, :].argmax().item()
targets.append(tuple(target))

# 'tail_unvoiced': after the unvoiced speech (silence)
target[2] += 9
targets.append(tuple(target))

# %% integrated gradients

ig = IntegratedGradients(model)
attributions = []
for target in targets:
    attribution, delta = ig.attribute(
        x,
        baselines=baseline,
        target=target,
        method='gausslegendre',
        return_convergence_delta=True,
    )
    attribution = attribution.detach().numpy()
    attribution = attribution.squeeze().transpose(1, 2, 0)
    attributions.append(attribution)

# %% plot

x_vec_r_max = np.log10(
    (((data['x'][..., :3]**2).sum(-1)**0.5).max() + log_eps_pow) / log_eps_pow
)
x_db = librosa.amplitude_to_db(data['x'][0, ..., -1])
y_db = librosa.amplitude_to_db(data['y'][0, ..., -1])
vmin_db = min(x_db.min(), y_db.min())
vmax_db = max(x_db.max(), y_db.max())

plt.style.use('default')
plt.rc('font', family='Arial', size=18)
kwargs_common = dict(
    origin='lower',
    aspect='auto',
    extent=None,
)

figs = []
all_cbar_axes = []
for i, target in enumerate(targets):
    attribution = attributions[i]
    print(titles[i])

    t_start = max(target[2] - t_interval, 0)
    t_end = target[2] + t_interval
    sl_img = (slice(n_freq), slice(t_start, t_end))

    # gradient of vector (directional feature)
    grad_vec_cart = attribution[sl_img][..., :3]
    print('r_max: ', end='')
    print(np.log10(
        (((grad_vec_cart**2).sum(-1)**0.5).max() + log_eps_grad) / log_eps_grad
    ))
    grad_vec = vec2hsv(grad_vec_cart, log_eps=log_eps_grad, r_max=grad_vec_r_max)

    # gradient of magnitude
    grad_mag = attribution[sl_img][..., 3]
    grad_mag *= np.log10(
        (np.abs(grad_mag) + log_eps_grad) / log_eps_grad
    ) / np.abs(grad_mag)
    print('vmax, vmin: ', end='')
    print((grad_mag.min(), grad_mag.max()))

    # input vector (directional feature), input mag, and output mag
    x_vec = vec2hsv(data['x'][0, ..., :3][sl_img], r_max=x_vec_r_max)
    x_db = librosa.amplitude_to_db(data['x'][0, ..., -1][sl_img])
    y_db = librosa.amplitude_to_db(data['y'][0, ..., -1][sl_img])

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(8.9, 10))
    kwargs_common['extent'] = (0, (t_end - t_start) / fs * l_hop, 0, fs / 2 / 1000)

    axes[0, 0].imshow(x_vec, **kwargs_common)
    axes[0, 1].imshow(x_db,
                      vmin=vmin_db, vmax=vmax_db,
                      cmap=cmap, **kwargs_common)
    axes[0, 2].imshow(y_db,
                      vmin=vmin_db, vmax=vmax_db,
                      cmap=cmap, **kwargs_common)
    axes[1, 0].imshow(grad_vec, **kwargs_common)
    axes[1, 1].imshow(grad_mag,
                      vmin=vmin_grad, vmax=vmax_grad,
                      cmap=cmap, **kwargs_common)
    axes[1, 2].remove()

    patch_x = ((t_end - t_start) / 2 - 2) / fs * l_hop
    patch_y = (target[1] - 5) * fs / 2 / 1000 / n_freq
    cbar_axes = []
    for i, ax in enumerate(axes.reshape(-1)):
        if ax.images:
            cbar = fig.colorbar(ax.images[0], ax=ax)
            ax.add_patch(
                mpatches.Rectangle(
                    [patch_x, patch_y],  # x, y
                    5 / fs * l_hop,  # width
                    11 * fs / 2 / 1000 / n_freq,  # height
                    fill=False, edgecolor='red', linewidth=2,
                )
            )
            ax.set_ylabel('frequency (kHz)')
            ax.set_xlabel('time (sec)')
            ax.set_xticks(xticks)

        if i in (0, 1, 3):
            cbar.ax.set_visible(False)
            cbar_axes.append(None)
        else:
            cbar_axes.append(cbar.ax)

    fig.tight_layout()
    fig.show()
    figs.append(fig)
    all_cbar_axes.append(cbar_axes)

# %%

for i, (title, fig, cbar_axes) in enumerate(zip(titles, figs, all_cbar_axes)):
    # fig.savefig(f'IG_DV_{i}_{title}.png', dpi=600)
    axes = [ax for ax in fig.axes if ax.images]
    for j, (subtitle, ax) in enumerate(zip(subtitles, axes)):
        extent = full_extent(fig, ax, cbar_axes[j])
        fig.savefig(
            path_fig / f'IG-DV-{i}_{title}-{j}_{subtitle}.png', 
            bbox_inches=extent, dpi=300,
        )

# %%
