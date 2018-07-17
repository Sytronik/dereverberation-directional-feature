import os
from glob import glob
import soundfile as sf

path = './speech/data/lisa/data/timit/raw/TIMIT/TRAIN/'
for dir, _, _ in os.walk(path):
    files = glob(os.path.join(dir, '*.wav'))
    if files is not None:
        for file in [f for f in files if not f.endswith('_converted.wav')]:
            data, fs = sf.read(file)
            newfilename = file.replace('.WAV', '_converted.wav')
            sf.write(newfilename, data, fs)
            print(newfilename)
