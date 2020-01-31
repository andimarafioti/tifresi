import librosa
import numpy as np
from tifresi.hparams import HParams as p

__author__ = 'Andres'


def log_spectrogram(spectrogram, dynamic_range_dB=p.stft_dynamic_range_dB):
    """Compute the log spectrogram representation from a spectrogram."""
    spectrogram = np.abs(spectrogram)  # for safety
    minimum_relative_amplitude = np.max(spectrogram) / 10 ** (dynamic_range_dB / 10)
    return 10 * np.log10(np.clip(spectrogram, a_min=minimum_relative_amplitude, a_max=None))


def inv_log_spectrogram(log_spec):
    """Inverse the log representation of the spectrogram or mel spectrogram."""
    return 10 ** (log_spec / 10)


def log_mel_spectrogram(spectrogram, stft_channels=p.stft_channels, n_mels=p.n_mels, fmin=p.fmin, fmax=p.fmax, sr=p.sr,
                        dynamic_range_dB=p.mel_dynamic_range_dB):
    """Compute the log mel spectrogram from a spectrogram."""
    melSpectrogram = mel_spectrogram(spectrogram, stft_channels=stft_channels, n_mels=n_mels, fmin=fmin, fmax=fmax,
                                     sr=sr)
    return log_spectrogram(melSpectrogram, dynamic_range_dB)


def mel_spectrogram(spectrogram, stft_channels=p.stft_channels, n_mels=p.n_mels, fmin=p.fmin, fmax=p.fmax, sr=p.sr):
    """Compute the mel spectrogram from a spectrogram."""
    if stft_channels != p.stft_channels or n_mels != p.n_mels or fmin != p.fmin or fmax != p.fmax or sr != p.sr:
        mel_basis = librosa.filters.mel(sr=sr, n_fft=stft_channels, n_mels=n_mels, fmin=fmin, fmax=fmax)
    else:
        mel_basis = p.mel_basis
    return np.dot(mel_basis, spectrogram)


def pseudo_unmel_spectrogram(mel_spectrogram, stft_channels=p.stft_channels, n_mels=p.n_mels, fmin=p.fmin, fmax=p.fmax, sr=p.sr):
    """Compute the inverse mel spectrogram from a mel spectrogram."""
    if stft_channels != p.stft_channels or n_mels != p.n_mels or fmin != p.fmin or fmax != p.fmax or sr != p.sr:
        mel_basis = librosa.filters.mel(sr=sr, n_fft=stft_channels, n_mels=n_mels, fmin=fmin, fmax=fmax)
        mel_inverse_basis = np.linalg.pinv(mel_basis)
    else:
        mel_inverse_basis = p.mel_inverse_basis
    return np.matmul(mel_inverse_basis, mel_spectrogram)
