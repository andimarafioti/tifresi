import numpy as np

from modGabPhaseGrad import modgabphasegrad
from pghi import pghi
from stft import GaussTF

__author__ = 'Andres'


def invertSpectrogram(spectrogram, audio_length, stft_channels, hop_size, original_stft_channels=None, original_hop_size=None):
    if original_stft_channels is None:
        original_stft_channels = stft_channels
    if original_hop_size is None:
        original_hop_size = hop_size

    tfr = original_hop_size * original_stft_channels / audio_length
    g_analysis = {'name': 'gauss', 'tfr': tfr}

    tgrad, fgrad = modgabphasegrad('abs', spectrogram, g_analysis, hop_size,
                                   stft_channels)

    logMagSpectrogram = np.log(np.clip(spectrogram.astype(np.float64), a_min=np.exp(-10), a_max=None))
    phase = pghi(logMagSpectrogram, tgrad, fgrad, hop_size, stft_channels, audio_length, tol=10)

    reComplexStft = (np.e ** logMagSpectrogram) * np.exp(1.0j * phase)

    tfSystem = GaussTF(a=original_hop_size, M=original_stft_channels)
    return tfSystem.idgt(reComplexStft, hop_size, stft_channels)
