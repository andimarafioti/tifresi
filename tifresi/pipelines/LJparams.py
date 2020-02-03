from tifresi.hparams import HParams
from tifresi.utils import downsample_tf_time
import librosa
import numpy as np

class LJParams(HParams):
    
    # Signal parameters
    sr = 22050 # Sampling frequency of the signal
    M = 2*1024 # Ensure that the signal will be a multiple of M
    
    # STFT parameters
    stft_channels = 1024 # Number of frequency channels
    hop_size = 256 # Hop size
    use_truncated = True # use a truncated Gaussian
    
    stft_dynamic_range_dB = 50 # dynamic range in dB for the STFT
    normalize = True # Normalize STFT
    
    # MEL parameters
    n_mels = 80 # Number of mel frequency band
    fmin = 0 # Minimum frequency for the MEL
    fmax = None # Maximum frequency for the MEL (None -> Nyquist frequency)
    reduction_rate = 2 # Reduction rate for the frequency channel
    mel_dynamic_range_dB = 50 # dynamic range in dB for the MEL
    
    
    mel_basis = librosa.filters.mel(sr=sr, n_fft=stft_channels, n_mels=n_mels, fmin=fmin, fmax=fmax)
    mel_inverse_basis = np.linalg.pinv(mel_basis)

