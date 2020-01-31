import librosa
import numpy as np
import sys

# sys.path.append('../')
from tifresi.transforms import log_spectrogram, inv_log_spectrogram, log_mel_spectrogram, mel_spectrogram

__author__ = 'Andres'


def test_log_spectrogram():
    x = np.random.rand(1024 * 1024).reshape([1024, 1024])

    log_x = log_spectrogram(x, dynamic_range_dB=80)
    inv_log_x = inv_log_spectrogram(log_x)

    assert (np.linalg.norm(inv_log_x - x) < 1e-7)


def test_log_spectrogram_small_range():
    x = np.random.rand(1024 * 1024).reshape([1024, 1024])

    log_x = log_spectrogram(x, dynamic_range_dB=30)
    inv_log_x = inv_log_spectrogram(log_x)

    assert (np.linalg.norm(inv_log_x - x) < 0.08)


def test_log_mel_spectrogram():
    x = np.random.rand(1024 * 513).reshape([513, 1024])

    x_mel = mel_spectrogram(x)

    log_x = log_mel_spectrogram(x, dynamic_range_dB=80)
    inv_log_x = inv_log_spectrogram(log_x)

    assert (np.linalg.norm(inv_log_x - x_mel) < 1e-7)


def test_log_mel_spectrogram_small_range():
    x = np.random.rand(1024 * 513).reshape([513, 1024])

    x_mel = mel_spectrogram(x)

    log_x = log_mel_spectrogram(x, dynamic_range_dB=30)
    inv_log_x = inv_log_spectrogram(log_x)

    assert (np.linalg.norm(inv_log_x - x_mel) < 0.08)


def test_mel_spectrogram():
    x = np.random.rand(256 * 1025).reshape([1025, 256])

    sr = 28000
    stft_channels = 2048
    n_mels = 40
    fmin = 40
    fmax = 12000
    mel_basis = librosa.filters.mel(sr=sr, n_fft=stft_channels, n_mels=n_mels, fmin=fmin, fmax=fmax)

    x_mel = mel_spectrogram(x, stft_channels=stft_channels, n_mels=n_mels, fmin=fmin, fmax=fmax, sr=sr)
    x_test_mel = np.matmul(mel_basis, x)

    assert (np.linalg.norm(x_test_mel - x_mel) < 1e-20)


if __name__ == "__main__":
    test_log_spectrogram()
    test_log_spectrogram_small_range()
    test_log_mel_spectrogram()
    test_log_mel_spectrogram_small_range()
    test_mel_spectrogram()
