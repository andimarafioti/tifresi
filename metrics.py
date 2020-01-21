import numpy as np

__author__ = 'Andres'


def projection_loss(target_spectrogram, original_spectrogram):
    magnitude_error = np.linalg.norm(np.abs(target_spectrogram) - np.abs(original_spectrogram), 'fro') / \
    np.linalg.norm(np.abs(target_spectrogram), 'fro')
    return 20 * np.log10(1 / magnitude_error)
