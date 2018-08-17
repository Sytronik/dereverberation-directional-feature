import os
from glob import glob
import soundfile as sf

path = ('../../De-Reverberation Data/speech/data/lisa/data/timit/raw/TIMIT'
        '/TRAIN')

for _dir, _, _ in os.walk(path):
    files = glob(os.path.join(_dir, '*.WAV'))
    if files:
        for fname in [f for f in files if not f.endswith('_converted.wav')]:
            data, fs = sf.read(fname)
            fname_new = fname.replace('.WAV', '_converted.wav')
            sf.write(fname_new, data, fs)
            print(fname_new)
