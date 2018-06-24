import pdb
# pdb.set_trace()

import numpy as np
import scipy as sc
import scipy.io as scio

import librosa

import os
import glob
import soundfile as sf

# import pycuda.driver as cuda
# import pycuda.autoinit
# import pycuda.driver as drv
# from pycuda.compiler import SourceModule
# mod=SourceModule("""
# __global__ void
# """)

RIRmat = scio.loadmat('./1_MATLABCode/RIR_Data/RIR.mat', variable_names = 'RIR')
RIR = np.array(RIRmat['RIR'])
RIR = RIR.transpose((2, 0, 1)) #72 x 32 x 48k
N_loc, N_ch = RIR.shape[0:1]
# RIR.astype(np.float32)
# RIR_gpu=cuda.mem_alloc(RIR.nbytes)
# cuda.memcpy_htod(RIR_gpu, RIR)

# speech = []
# speech_convolved = []
N_files = 0;
fs = 48000
for dir, _, _ in os.walk('./speech/data/lisa/data/timit/raw/TIMIT/TRAIN/'):
    files = glob(os.path.join(dir, '*_converted.wav'))
    if files is None:
        continue
    for file in files:
        if fs_original == 0:
            data, fs_original = sf.read(file)
        else:
            data, _ = sf.read(file)
        data = librosa.core.resample(data, fs_original, fs)

        # data = data.astype(np.float32)
        # speech.extend(data)
        length = data.shape[0]
        for i_loc in range(N_loc):
            convolved = np.zeros((length, N_ch))
            for i_ch in range(N_ch):
                convolved[:, i_ch] = sc.signal.fftconvolve(data, RIR[i_loc, i_ch], mode = 'same')
            # convolved=np.asarray(convolved*,dtype=np.int16)
            # speech_convolved.extend(convolved)
            path = './speech/convolved/TRAIN/'+'%4d_%2d.wav'%(N_files, i_loc)
            sf.write(path, convolved, fs)
            print('Saved {}'.format(path))
        N_files += 1

print('Number of data: {}'.format(N_files))
print('Sample Rate: {}'.format(fs))
print('Number of location: {}'.format(N_loc))
print('Number of microphone channel: {}'.format(N_ch))
