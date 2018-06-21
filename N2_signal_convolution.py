import pdb

import numpy as np
import scipy as sc
import scipy.io as scio
import scipy.io.wavfile as sciowav

import librosa

import os
from glob import glob
import soundfile as sf

import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
# mod=SourceModule("""
# __global__ void
# """)

RIR=np.array(scio.loadmat("./1_MATLABCode/RIR_Data/RIR.mat", variable_names="RIR")['RIR'])
RIR=np.transpose(RIR,(2,0,1))
# RIR.astype(np.float32)
# RIR_gpu=cuda.mem_alloc(RIR.nbytes)
# cuda.memcpy_htod(RIR_gpu, RIR)

# speech = []
# speech_convolved = []
fs = 48000
fs_original=0
for dir,_,_ in os.walk("./speech/data/lisa/data/timit/raw/TIMIT/TRAIN/"):
    files=glob(os.path.join(dir,"*_converted.wav"))
    if files==[]:
        continue
    for file in files:
        # print(file)
        if fs_original==0:
            data,fs_original = librosa.load(file)
        else:
            data,_ = librosa.load(file)
        data = librosa.core.resample(data, fs_original, fs)

        # data = data.astype(np.float32)
        # speech.extend(data)
        length = data.shape[0]
        N_loc = RIR.shape[0]
        N_ch = RIR.shape[1]
        # pdb.set_trace()
        for i_loc in range(0,N_loc):
            convolved=np.zeros((length, N_ch))
            for i_ch in range(0,N_ch):
                convolved[:,i_ch]=sc.signal.fftconvolve(data, RIR[i_loc, i_ch], mode='same')
            # convolved=np.asarray(convolved*,dtype=np.int16)
            # speech_convolved.extend(convolved)
            path='./speech/convolved/'+os.path.basename(file).replace('.wav','_{}.wav'.format(i_loc))
            # pdb.set_trace()
            sf.write(path, convolved, fs)
            print(path)


#32 x 48k x 72
