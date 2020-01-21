import numpy as np

__author__ = 'Andres'


def projectionLoss(targetSpectrogram, originalSpectrogram):
    magnitudeError = np.linalg.norm(np.abs(targetSpectrogram) - np.abs(originalSpectrogram), 'fro') / \
    np.linalg.norm(np.abs(targetSpectrogram), 'fro')
    return 20 * np.log10(1 / magnitudeError)
