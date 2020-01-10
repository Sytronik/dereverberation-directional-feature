# %% constants
import os
import re
from pathlib import Path
from typing import Dict, List

import librosa
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio
from numpy import ndarray
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from audio_utils import vec2hsv


path_root = Path('./result')
# path_root = Path('./backup/result_23_10')
room_trained = 'room1+2+3'
foldername_methods = {
    'SIV': f'SIV',
    'DV': f'DV',
    'Mulspec32': f'Mulspec32',
    'Mulspec4': f'Mulspec4',
    'Single': f'Single',
}
path_wpes = {
    'WPE-tuned': Path('../wpe/result/wpe (WPE-tuned)/unseen_room4+5+6+7'),
    'WPE': Path('../wpe/result/wpe_3_3_5 (WPE)/unseen_room4+5+6+7'),
}

# uncomment one line only
kind_folders = [
    # 'unseen_59', 'unseen_59_glim20',
    'unseen_room4+5+6+7_59', 'unseen_room4+5+6+7_59_glim20',
    # 'unseen_room9_59',
]

# idx_audio: idx of audio in tensorboard
# draw_spec: True if spectrogram images are needed
# for room1+2+3, idx_audio % 3 == room no. - 1 
# for room4+5+6+7, idx_audio % 4 == room no. - 4 
idx_audio = 27

path_feature = Path('./data')
features = ['SIV', 'DV']

# spectrogram parameters
fs = 16000
n_fft = 512
l_hop = 256
l_win = 512
frames_to_show = slice(20, 65)  # slice(None) for every frame

# ((x, y), width, height)
patches = dict(
    formants=((0.05 + 0.32, 0.05), 0.15, 2.2),
    unvoiced=((0.31 + 0.32, 4), 0.2, 3.9),
)

path_methods = {k: path_root / v for k, v in foldername_methods.items() if v}


# %% get filename of `idx_audio`-th data

# room
match = re.search(r'.*_(room.*)_.*', kind_folders[0])
room = match[1] if match else room_trained
seen_or_unseen = kind_folders[0].split("_")[0].upper()

# get file name of idx_audio-th data
fmetadata = path_feature / f'SIV_{room}/TEST/{seen_or_unseen}/metadata.mat'
fname = scio.loadmat(fmetadata)['list_fname'][idx_audio]

fname = fname.replace('.npz', '')
print(fname)

path_audio = path_root / f'audios/{",".join(foldername_methods.keys())}-{fname}'
path_audio.mkdir(exist_ok=True)
path_fig = path_root / f'figures/{",".join(foldername_methods.keys())}-{fname}'
os.makedirs(path_fig, exist_ok=True)


# %% open dirspec files

dirfeatures: Dict[str, ndarray] = dict()
for feature in features:
    path = path_feature / f'{feature}_{room}/TEST/{seen_or_unseen}/{fname}.npz'
    with np.load(path) as npz:
        dirfeatures[f'dirspec_{feature}_free'] = npz['dirspec_free'][..., :3]
        dirfeatures[f'dirspec_{feature}_room'] = npz['dirspec_room'][..., :3]


# %% get audio from tensorboard

existing_audio_files = list(path_audio.glob('*.wav'))

audios: Dict[str, ndarray] = dict()
for method, path_method in path_methods.items():
    for kind_folder in kind_folders:
        path = path_method / kind_folder
        if not path.exists() \
                or next(path.glob('events.out.tfevents.*'), None) is None:
            print(f'"{path}" does not exists or does not have any tfevent files.')
            continue

        # tbwriter.py writes 4 audios per data
        # : anechoic, reverberant, processed, processed with true phase
        result_key = f'{method}{kind_folder.replace(kind_folders[0], "")}'
        keys = ['anechoic', 'reverberant', result_key, f'{method}_true_ph']

        # If some of the audios were extracted from tensorboard already,
        # just load them and don't open the corresponding tfevent file.
        for key in keys:
            if key in audios:
                continue
            for f in existing_audio_files:
                if re.search(f'[0-9]+_{key}', f.stem) is not None:
                    audios[key], _ = librosa.load(str(f), sr=None)
                    break

        if any(key not in audios for key in keys):
            print(f'loading {", ".join(k for k in keys if k not in audios)} '
                  f'from a tfevent file...')
        else:
            break

        # open tfevent file
        eventacc = EventAccumulator(
            str(path),
            size_guidance=dict(scalars=1, images=1, audio=0),
        )
        eventacc.Reload()
        tags = eventacc.Tags()['audio']

        for tag, key in zip(tags, keys):
            if key in audios:
                continue
            audioevent = eventacc.Audio(tag)[idx_audio]

            # encoded_audio_string is a binary sequence of the .wav format.
            # so it is written in file by 'bw' mode (binary writing mode),
            # and loaded back again by librosa.
            with Path('temp.wav').open('bw') as f:
                f.write(audioevent.encoded_audio_string)
            wav, _ = librosa.load('temp.wav', sr=None)
            audios[key] = wav

if Path('temp.wav').exists():
    os.remove('temp.wav')


# %% get audio from WPE

for title, path in path_wpes.items():
    path = (path / fname).with_suffix('.wav')
    wav, _ = librosa.load(path, sr=None)
    audios[title] = wav


# %% normalize levels and pad zeros to the end

# order change (glim20 and true_ph are last)
for kind_folder in kind_folders:
    for method, path_method in path_methods.items():
        suffix = kind_folder.replace(kind_folders[0], "")
        if suffix and f'{method}{suffix}' in audios:
            backup = audios[f'{method}{suffix}']
            del audios[f'{method}{suffix}']
            audios[f'{method}{suffix}'] = backup

for method, path_method in path_methods.items():
    backup = audios[f'{method}_true_ph']
    del audios[f'{method}_true_ph']
    audios[f'{method}_true_ph'] = backup

maxamps = []
maxlengths = []
for key, audio in audios.items():
    audios[key] /= (audio**2).mean()**0.5
    maxamps.append(np.abs(audio).max())
    maxlengths.append(len(audio))

maxamp = max(maxamps)
maxlength = max(maxlengths)

for key, audio in audios.items():
    audio /= maxamp
    audios[key] = np.pad(audio, (0, maxlength - len(audio)), 'constant')

# %% save audio files

for i, (key, audio) in enumerate(audios.items()):
    librosa.output.write_wav(path_audio / f'{i}_{key}.wav', audio, fs)


# %% draw spectrograms

plt.style.use('default')
plt.rc('font', family='Arial', size=18)

figs: List[plt.Figure] = []

titles = ['anechoic', 'reverberant', *foldername_methods.keys(), *path_wpes.keys()]
specs: List[ndarray] = [
    np.abs(librosa.stft(audio, n_fft, l_hop, l_win))
    for key, audio in audios.items() if key in titles
]

# frames to seconds
if frames_to_show == slice(None):
    sec_to_show = np.array([0., specs[0].shape[1] * l_hop / fs])
else:
    sec_to_show = np.array([frames_to_show.start, frames_to_show.stop]) * l_hop / fs

# trim spectrograms
spec_to_show = [spec[:, frames_to_show] for spec in specs]
maxmag = np.max(spec_to_show[0])
spec_to_show = [spec * maxmag / spec.max() for spec in spec_to_show]  # normalize
spec_to_show = [librosa.amplitude_to_db(spec) for spec in spec_to_show]  # dB

vmax, vmin = np.max(spec_to_show), np.min(spec_to_show)

for i, (spec, title) in enumerate(zip(spec_to_show, titles)):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(spec,
              cmap=plt.get_cmap('gray'),
              vmin=vmin, vmax=vmax,
              extent=(*sec_to_show, 0, fs // 2 // 1000),
              origin='lower', aspect='auto')
    for patchname, patch in patches.items():
        ax.add_patch(
            mpatches.Rectangle(*patch, fill=False, edgecolor='red', linewidth=3.5)
        )
    # ax.set_title(title)
    ax.set_xlabel('time (sec)')
    ax.set_ylabel('frequency (kHz)')
    cbar = fig.colorbar(ax.images[0], ax=ax)
    cbar.set_ticks(np.arange(-50, 40, 10))
    # fig.tight_layout()
    if i >= 2:
        cbar.ax.set_visible(False)

    figs.append(fig)

# %% draw directional features

maxlength = max(v.shape[1] for v in dirfeatures.values())
dirfeatures = {
    k: np.pad(v, ((0, 0), (0, maxlength - v.shape[1]), (0, 0)), 'constant')
    for k, v in dirfeatures.items()
}

fig_dirfeatures = []
for i, (title, vec) in enumerate(dirfeatures.items()):
    fig, ax = plt.subplots(figsize=(6, 5))
    vec = vec[:, frames_to_show]
    r_max = np.log10((((vec**2).sum(-1)**0.5).max() + 1e-10) / 1e-10)
    vec_hsv = vec2hsv(vec, r_max=r_max)
    ax.imshow(vec_hsv,
              extent=(*sec_to_show, 0, fs // 2 // 1000),
              origin='lower', aspect='auto')
    # ax.set_title(title)
    ax.set_xlabel('time (sec)')
    ax.set_ylabel('frequency (kHz)')

    # draw colorbar for the same figure size as spectrogram figures
    cbar = fig.colorbar(ax.images[0], ax=ax)
    cbar.set_ticks([-1, 1])
    cbar.set_ticklabels(['+-40', '+-40'])
    for patchname, patch in patches.items():
        ax.add_patch(
            mpatches.Rectangle(*patch, fill=False, edgecolor='red', linewidth=3.5)
        )

    # fig.tight_layout()
    cbar.ax.set_visible(False)

    fig_dirfeatures.append(fig)


# %% save figs

patchnames = ','.join(patches.keys())
for i, (fig, title) in enumerate(zip(figs, titles)):
    fig.savefig(path_fig / f'{i}_{title}_{patchnames}.png',
                dpi=300, bbox_inches='tight')

for i, (fig, title) in enumerate(zip(fig_dirfeatures, dirfeatures.keys())):
    j = i + len(figs)
    fig.savefig(path_fig / f'{j}_{title}_{patchnames}.png',
                dpi=300, bbox_inches='tight')

# %%
