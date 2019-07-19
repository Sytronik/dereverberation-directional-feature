# U-Net-based Speech Dereverberation with directional
feature from spherical microphone array recordings
# create.py

`create.py` performs following procedure:

1. Calculate **anechoic spherical harmonic domain (SHD) signals** from speech sources and spherical Fourier transform basis $\mathbf Y_s$ (`Ys`).
2. Calculate 32-channel spherical microphone array recordings from speech sources, room impulse responses (RIRs), and the modified inverse of rigid sphere modal strength $b^{-1}_n(kr)$ (`bEQf`).
3. Calculate **reverberant SHD signals** from the result of 2.
4. Perform STFT signals.
5. Calculate directional features, one of spatially-averaged intensity vector (**SIV**) and direction vector (**DV**).
6. Save magnitude and phase of the STFT of the 0-th order SHD signals and directional features.

Read docstring of `create.py` for usage.



## main.py

`main.py` is used to train or test DNNs.

Read docstring of `main.py` for usage.



## Model

The DNN model is based on FusionNet (U-Net-like DNN). Refer to `model` directory.



## Evaluation Metrics

Source codes for PESQ, STOI, and fwSegSNR are in `matlab_lib` directory.

Frequency-domain SegSNR is implemented in `audio_utils.py`.