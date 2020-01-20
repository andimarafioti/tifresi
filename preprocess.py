import librosa
import numpy as np

from stft import GaussTF

__author__ = 'Andres'


def logSpectrogram(time_signal, stft_channels, hop_size,  normalize=True, clipBelow=np.e**-30):
    magSpectrogram = spectrogram(time_signal=time_signal, stft_channels=stft_channels,
                                       hop_size=hop_size, normalize=normalize)

    return np.log(np.clip(magSpectrogram, a_min=clipBelow, a_max=None))


def logMelSpectrograms(spectrogram, stft_channels, n_mels=80, fmin=0, fmax=None, sr=22050, clipBelow=np.e**-30):
    mel_basis = librosa.filters.mel(sr=sr, n_fft=stft_channels, n_mels=n_mels, fmin=fmin, fmax=fmax)
    melSpectrogram = np.dot(mel_basis, spectrogram)
    return np.log(np.clip(melSpectrogram, a_min=clipBelow, a_max=None))


def spectrogram(time_signal, stft_channels, hop_size, normalize=True):
    tfSystem = GaussTF(a=hop_size, M=stft_channels)
    stft = tfSystem.dgt(time_signal)

    magSpectrogram = np.abs(stft)

    if normalize:
        magSpectrogram = magSpectrogram/np.max(magSpectrogram)

    return magSpectrogram