#%%
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


path_root = Path('./result')
# path_root = Path('./backup/result_23_10')
foldername_methods = {
    'SIV': 'SIV',
    'DV': 'DV',
    'Mulspec32': 'Mulspec32',
    'Mulspec4': 'Mulspec4',
    'Single': 'Single',
}

xticks = [0, 3, 11, 27, 59]  # restart epochs
yticks = np.linspace(6, 16, num=6)
ylim = (6, 16)

tab20 = list(plt.get_cmap('tab20').colors)
tab20c = list(plt.get_cmap('tab20c').colors)
tab20b = list(plt.get_cmap('tab20b').colors)
grey = (0.5, 0.5, 0.5)
cmap = [tab20c[1]] * 2 + [tab20c[5]] * 3 + [tab20c[9], tab20c[13]]
markers = ['o', 'o', 'o', 'o', 'o', 'x', 'x', '.']
linestyles = ['-', '--', '-', '--', ':', '-', '-', ':']

kwargs_legend = dict(ncol=3, columnspacing=1, fontsize='small')

# %% extract loss data from tensorboard
path_methods = {k: path_root / v for k, v in foldername_methods.items()}

list_losses: List[List[float]] = []
for path_method in path_methods.values():
    path = path_method / 'train'
    eventacc = EventAccumulator(
        str(path),
        size_guidance=dict(scalars=0, images=1, audio=1),
    )
    eventacc.Reload()
    _, _, value = zip(*eventacc.Scalars('loss/valid'))
    list_losses.append(value)

array_losses = np.array(list_losses)
epochs = np.arange(len(array_losses[0]))

#%% plot loss
plt.style.use('default')
plt.rc('font', family='Arial', size=18)

fig: plt.Figure
ax: plt.Axes
fig, ax = plt.subplots(figsize=(7, 4))
for i, (label, losses) in enumerate(zip(foldername_methods.keys(), array_losses)):
    ax.plot(epochs, losses, color=cmap[i], label=label, linestyle=linestyles[i])

ax.grid()
ax.set_axisbelow(True)

ax.legend(**kwargs_legend)

ax.set_xticks(xticks)
ax.set_yticks(yticks)
ax.set_ylim(*ylim)
ax.tick_params('x', direction='in')
ax.tick_params('y', direction='in')

ax.set_xlabel('Training Epoch')
ax.set_ylabel('Loss')

fig.tight_layout()

#%% save figure

path_fig = path_root / 'figures'
path_fig.mkdir(exist_ok=True)
fig.savefig(path_fig / f'{",".join(foldername_methods.keys())}-loss_vs_epoch.png', dpi=300)

#%%
