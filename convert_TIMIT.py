import os
from glob import glob
import soundfile as sf

for dir,_,_ in os.walk('./speech/data/lisa/data/timit/raw/TIMIT/TRAIN/'):
    files=glob(os.path.join(dir,"*.wav"))
    if files==[]:
        continue
    # for file in files:
    #     if file.endswith("_converted.wav"):
    #         os.remove(file)

    for file in files:
        # print(file)
        if file.endswith('_converted.wav'):
            continue
        data, fs=sf.read(file)
        newfilename=file.replace('.WAV','_converted.wav')
        sf.write(newfilename, data, fs)
        print(newfilename)
