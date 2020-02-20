import numpy as np
from tifresi.transforms import inv_log_spectrogram

__author__ = 'Andres'


def projection_loss(target_spectrogram, original_spectrogram):
    magnitude_error = np.linalg.norm(np.abs(target_spectrogram) - np.abs(original_spectrogram), 'fro') / \
    np.linalg.norm(np.abs(target_spectrogram), 'fro')
    return 20 * np.log10(1 / magnitude_error)


def consistency(log10_spectrogram):
    log_spectrogram = np.log(inv_log_spectrogram(log10_spectrogram))

    ttderiv = log_spectrogram[1:-1, :-2] - 2 * log_spectrogram[1:-1, 1:-1] + log_spectrogram[1:-1, 2:] + np.pi / 4
    ffderiv = log_spectrogram[:-2, 1:-1] - 2 * log_spectrogram[1:-1, 1:-1] + log_spectrogram[2:, 1:-1] + np.pi / 4

    absttderiv = substractMeanAndDivideByStd(np.abs(ttderiv))
    absffderiv = substractMeanAndDivideByStd(np.abs(ffderiv))

    consistencies = np.sum(absttderiv * absffderiv)
    return consistencies


def substractMeanAndDivideByStd(aDistribution):
    unmeaned = aDistribution - np.mean(aDistribution, keepdims=True)
    shiftedtt = unmeaned / np.sqrt(np.sum(np.abs(unmeaned) ** 2, keepdims=True))
    return shiftedtt
