import librosa
import numpy as np
from hparams import HParams as p

__author__ = 'Andres'


def log_spectrogram(spectrogram, dB=p.stft_dB):
    """Compute the log spectrogram representation from a spectrogram."""
    spectrogram = np.abs(spectrogram)  # for safety
    a_min = np.max(spectrogram) / 10 ** (dB / 10)
    return 10 * np.log10(np.clip(spectrogram, a_min=a_min, a_max=None))


def inv_log_spectrogram(log_spec):
    """Inverse the log representation of the spectrogram or mel spectrogram."""
    return 10 ** (log_spec / 10)


def log_mel_spectrograms(spectrogram, stft_channels=p.stft_channels, n_mels=p.n_mels, fmin=p.fmin, fmax=p.fmax, sr=p.sr, dB=p.mel_dB):
    """Compute the log mel spectrogram from a spectrogram."""
    melSpectrogram = mel_spectrogram(spectrogram, stft_channels=stft_channels, n_mels=n_mels, fmin=fmin, fmax=fmax,
                                     sr=sr)
    a_min = np.max(melSpectrogram) / 10 ** (dB / 10)
    return 10 * np.log10(np.clip(melSpectrogram, a_min=a_min, a_max=None))


def mel_spectrogram(spectrogram, stft_channels=p.stft_channels, n_mels=p.n_mels, fmin=p.fmin, fmax=p.fmax, sr=p.sr):
    """Compute the mel spectrogram from a spectrogram."""
    if stft_channels != p.stft_channels or n_mels != p.n_mels or fmin != p.fmin or fmax != p.fmax or sr != p.sr:
        mel_basis = librosa.filters.mel(sr=sr, n_fft=stft_channels, n_mels=n_mels, fmin=fmin, fmax=fmax)
    else:
        mel_basis = p.mel_basis
    return np.dot(mel_basis, spectrogram)
