# %%
import colorsys
from pathlib import Path
from itertools import product

import torch
from captum.attr import IntegratedGradients, Saliency, DeepLift, DeepLiftShap
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import librosa

from models import UNet
from dataset import DirSpecDataset
from hparams import hp


def cart2sph(x, y, z):
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    el = np.arctan2(hxy, z)
    az = np.arctan2(y, x)
    return r, az, el


def vec2hsv(cart, log_eps=1e-10, r_max=0):
    r, az, el = cart2sph(cart[..., 0], cart[..., 1], cart[..., 2])
    az[az < 0] += 2*np.pi
    if log_eps != 0:
        r = np.log10((r+log_eps)/log_eps)
    r /= r.max() if r_max == 0 else r_max
    az /= 2*np.pi
    el /= np.pi

    rgb = np.empty_like(cart)
    for i, j in product(range(rgb.shape[0]), range(rgb.shape[1])):
        rgb[i, j] = colorsys.hsv_to_rgb(az[i, j], el[i, j], r[i, j])

    return rgb


# %% hp
path_state_dict = Path('./backup/result_23_10/SIV_23_10/train/59.pt')
# path_state_dict = Path('./backup/result_23_10/DV_23_10/train/59.pt')
# hp.feature = 'DirAC'
hp.init_dependent_vars()

# %% model & dataset
device = 'cpu'
# device = 'cuda:0'
model = UNet(4, 1, 64, 4).to(device)
state_dict = torch.load(path_state_dict, map_location=device)[0]
model.load_state_dict(state_dict)

dataset_temp = DirSpecDataset('train')
# Test Set
dataset_test = DirSpecDataset('unseen',
                              dataset_temp.norm_modules,
                              **hp.channels_w_ph)
# loader = iter(DataLoader(dataset_test,
#                     batch_size=1,
#                     num_workers=0,
#                     collate_fn=dataset_test.pad_collate,
#                     pin_memory=False,
#                     shuffle=False,
#                     ))

# %% retrieve data
# data = next(loader)
data = dataset_test.pad_collate([dataset_test[1]])
# data2 = next(loader)

x, y = data['normalized_x'], data['normalized_y']
x, y = x.to(device), y.to(device)
y_denorm = data['y']
y_denorm = y_denorm.permute(0, 3, 1, 2)  # B, C, F, T

x.requires_grad = True

baseline = torch.zeros_like(data['x'])
baseline = dataset_temp.normalize(x=baseline)
baseline = baseline.permute(0, 3, 1, 2)

# %% set targets
targets = []
target = np.unravel_index(y_denorm.argmax().item(), y_denorm.shape)[1:]
targets.append(target)
target = list(target)
target[2] += 12
targets.append(tuple(target))
target = [0, 0, 0]
target[1] = 128+64
target[2] = y_denorm[0, 0, 128, :].argmax().item()
targets.append(tuple(target))
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

# # %% saliency
# saliency = Saliency(model)
# attribution = saliency.attribute(x, target=target)

# attribution = attribution.detach().numpy()
# attribution = attribution.squeeze().transpose(1, 2, 0)

# # %% deep lift
# dl = DeepLift(model)
# attribution, delta = dl.attribute(
#     x,
#     # baselines=baseline1,
#     target=target,
#     return_convergence_delta=True,
# )

# attribution = attribution.detach().numpy()
# attribution = attribution.squeeze().transpose(1, 2, 0)

# # %% deep lift shap
# dlshap = DeepLiftShap(model)
# attribution, delta = dlshap.attribute(
#     x,
#     # baselines=baseline1,
#     target=target,
#     return_convergence_delta=True,
# )

# attribution = attribution.detach().numpy()
# attribution = attribution.squeeze().transpose(1, 2, 0)

# %% plot
# cmap = plt.get_cmap('CMRmap')
cmap = None
t_interval = 25
log_eps_grad = 1e-3

# grad_vec_r_max = 3.11  # SIV
# vmin_grad = -1.94  # SIV
# vmax_grad = -vmin_grad

grad_vec_r_max = 3.13  # DV
vmin_grad = -2.37  # DV
vmax_grad = -vmin_grad

# grad_vec_r_max = 0
# vmin_grad = None
# vmax_grad = None

titles = ['max formant', 'tail formant', 'max unvoiced', 'tail unvoiced']

x_vec_r_max = np.log10((((data['x'][..., :3]**2).sum(-1)**0.5).max()+1e-10)/1e-10)
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
for i, target in enumerate(targets):
    attribution = attributions[i]
    print(titles[i])
    t_start = max(target[2]-t_interval, 0)
    t_end = target[2]+t_interval
    # sl_img = tuple(slice(max(i - 30, 0), i + 30) for i in target[1:])
    sl_img = (slice(257), slice(t_start, t_end))
    cart = attribution[sl_img][..., :3]
    print(np.log10((((cart**2).sum(-1)**0.5).max()+log_eps_grad)/log_eps_grad))
    # rgb = cart / np.abs(cart).max()/2 + 0.5

    grad_vec = vec2hsv(cart, log_eps=log_eps_grad, r_max=grad_vec_r_max)

    grad_mag = attribution[sl_img][..., 3]
    grad_mag = grad_mag * np.log10((np.abs(grad_mag)+log_eps_grad)/log_eps_grad)/np.abs(grad_mag)
    print((grad_mag.min(), grad_mag.max()))
    # a /= np.abs(a).max()/2 + 0.5

    x_vec = vec2hsv(data['x'][0, ..., :3][sl_img], r_max=x_vec_r_max)
    x_db = librosa.amplitude_to_db(data['x'][0, ..., -1][sl_img])
    y_db = librosa.amplitude_to_db(data['y'][0, ..., -1][sl_img])

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10, 10))
    kwargs_common['extent'] = (0, (t_end - t_start) * 0.016, 0, 8)

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

    patch_x = ((t_end - t_start) / 2 - 2) * 0.016
    patch_y = (target[1]-5) * 8 / 257
    for i, ax in enumerate(axes.reshape(-1)):
        if ax.images:
            cbar = fig.colorbar(ax.images[0], ax=ax)
            ax.add_patch(
                mpatches.Rectangle(
                    [patch_x, patch_y], 5*0.016, 11*8/257,
                    fill=False, edgecolor='red', linewidth=2,
                )
            )
            ax.set_ylabel('frequency (kHz)')
            ax.set_xlabel('time (sec)')
            ax.set_xticks([0, 0.4, 0.8])
        if i == 0 or i == 3:
            cbar.ax.set_visible(False)

    fig.tight_layout()
    fig.show()
    figs.append(fig)

# %%
for i, (title, fig) in enumerate(zip(titles, figs)):
    fig.savefig(f'SIV_{i}_{title}.png', dpi=600)

# %%
