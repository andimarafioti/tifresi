import numpy as np
import librosa
from tifresi.stft import GaussTF, GaussTruncTF
from tifresi.pipelines.LJparams import LJParams as p
from tifresi.transforms import mel_spectrogram, log_spectrogram
from tifresi.utils import downsample_tf_time, preprocess_signal, load_signal

def compute_mag_mel_from_path(path):
    y, sr = load_signal(path, p.sr)
    y = preprocess_signal(y, p.M)
    return compute_mag_mel(y)

def compute_mag_mel(y):
    '''Compute spectrogram and MEL spectrogram from signal.
    Args:
      y  : signal
    Returns:
      mel: A 2d array of shape (T, n_mels) and dtype of float32.
      mag: A 2d array of shape (T, 1+stft_channels/2) and dtype of float32.
    '''
    if p.use_truncated:
        tfsystem = GaussTruncTF(hop_size=p.hop_size, stft_channels=p.stft_channels)
    else:
        tfsystem = GaussTF(hop_size=p.hop_size, stft_channels=p.stft_channels)
        
    # magnitude spectrogram
    mag = tfsystem.spectrogram(y, normalize=p.normalize)

    # mel spectrogram
    mel = mel_spectrogram(mag, stft_channels=p.stft_channels, n_mels=p.n_mels, fmin=p.fmin, fmax=p.fmax, sr=p.sr)

    # to decibel
    mag = log_spectrogram(mag, dynamic_range_dB=p.stft_dynamic_range_dB)/p.stft_dynamic_range_dB+1
    assert(np.max(mag)<=1)
    assert(np.min(mag)>=0)    
    
    # Reduction rate
    if p.reduction_rate > 1:
        mel = downsample_tf_time(mel, p.reduction_rate)    
    
    mel = log_spectrogram(mel, dynamic_range_dB=p.mel_dynamic_range_dB)/p.mel_dynamic_range_dB+1

    # Float32
    mel = mel.astype(np.float32)
    mag = mag.astype(np.float32) 

    return mel, mag
