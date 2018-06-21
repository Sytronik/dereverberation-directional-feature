import pdb
import scipy as sc
import scipy.io as scio
import scipy.io.wavfile as sciowav

import librosa

import numpy as np

import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
# mod=SourceModule("""
# __global__ void
# """)

import os
from glob import glob

RIR=np.array(scio.loadmat("./1_MATLABCode/RIR_Data/RIR.mat", variable_names="RIR")['RIR'])
RIR=np.transpose(RIR,(2,0,1))
# RIR.astype(np.float32)
# RIR_gpu=cuda.mem_alloc(RIR.nbytes)
# cuda.memcpy_htod(RIR_gpu, RIR)

speech = []
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
        data=data.astype(np.float32)
        # speech.extend(data)
        # print(file)
        pdb.set_trace()
        length,_ = data.shape
        N_loc,N_ch,_ = RIR.shape
        for i_loc in range(0,N_loc):
            convolved=np.empty([N_ch, length])
            for i_ch in rnage(0,N_ch):
                convolved[i_ch]=scipy.signal.fftconvolve(data, RIR_one_loc[i_ch], mode='same')
            speech_convolved.extend(convolved)

# speech_gpu=cuda.mem_alloc(speech.nbytes)
# cuda.memcpy_htod(speech_gpu, speech)
speech_convolved=[]
for data in speech:
    pdb.set_trace()
    length,_ = data.shape
    N_ch,_ = RIR[1].shape
    for RIR_one_angle in RIR:
        convolved=np.empty([N_ch, length])
        for i_ch in rnage(0,N_ch):
            convolved[i_ch]=scipy.signal.fftconvolve(data, RIR_one_angle[i_ch], mode='same')
        speech_convolved.extend(convolved)




#32 x 48k x 72
