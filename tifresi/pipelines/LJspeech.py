import numpy as np
import librosa
from tifresi.stft import GaussTF


def make_spectrograms(y, a, M, n_mels, sr=22050):
    '''Parse the wave file in `fpath` and
    Returns normalized melspectrogram and linear spectrogram.
    Args:
      y  : sound file
    Returns:
      mel: A 2d array of shape (T, n_mels) and dtype of float32.
      mag: A 2d array of shape (T, 1+n_fft/2) and dtype of float32.
    '''

    tfsystem = GaussTF(a=a, M=M)
    linear = tfsystem.dgt(y)

    # magnitude spectrogram
    mag = np.abs(linear)  # (1+n_fft//2, T)
    mag = mag/np.max(mag)

    # mel spectrogram
    mel_basis = librosa.filters.mel(sr, M, n_mels)  # (n_mels, 1+n_fft//2)
    mel = np.dot(mel_basis, mag)  # (n_mels, t)

    # to decibel
    mel = np.log10(np.maximum(1e-5, mel))/2.5+1
    mag = np.log10(np.maximum(1e-5, mag))/2.5+1
    assert(np.max(mag)<=1)
    assert(np.min(mag)>=-1)    

    # Transpose
    mel = mel.astype(np.float32)  # (T, n_mels)
    mag = mag.astype(np.float32)  # (T, 1+n_fft//2)

    return mel, mag
