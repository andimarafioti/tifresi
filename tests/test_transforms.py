import numpy as np

from transforms import log_spectrogram, inv_log_spectrogram

__author__ = 'Andres'


def test_log_spectrogram():
    x = np.random.rand(1024 * 1024).reshape([1024, 1024])

    log_x = log_spectrogram(x, dB=80)
    inv_log_x = inv_log_spectrogram(log_x)

    assert (np.linalg.norm(inv_log_x - x) < 1e-7)


def test_log_spectrogram_small_range():
    x = np.random.rand(1024 * 1024).reshape([1024, 1024])

    log_x = log_spectrogram(x, dB=30)
    inv_log_x = inv_log_spectrogram(log_x)

    assert (np.linalg.norm(inv_log_x - x) < 0.08)
