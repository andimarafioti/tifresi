from ltfatpy import dgtreal, idgtreal
from ltfatpy.gabor.gabdual import gabdual
import numpy as np
from tifresi.hparams import HParams as p

from tifresi.phase.modGabPhaseGrad import modgabphasegrad
from tifresi.phase.pghi import pghi


class GaussTF(object):
    """Time frequency transform object based on a Gauss window.
    
    The Gauss window is necessary to apply the PGHI (Phase gradient heap integration) algorithm.
    """

    def __init__(self, hop_size=p.hop_size, stft_channels=p.stft_channels):
        assert (np.mod(stft_channels, 2) == 0), 'The number of stft channels needs to be even'
        self.hop_size = hop_size
        self.stft_channels = stft_channels

    def dgt(self, x, hop_size=None, stft_channels=None):
        """Compute the DGT of a real signal with a gauss window."""
        if hop_size is None:
            hop_size = self.hop_size
        if stft_channels is None:
            stft_channels = self.stft_channels
        assert (len(x.shape) == 1)
        assert (np.mod(len(x), hop_size) == 0)
        assert (np.mod(stft_channels, 2) == 0), 'The number of stft channels needs to be even'
        assert (np.mod(len(x), stft_channels) == 0)
        g_analysis = self._analysis_window(x)
        return dgtreal(x.astype(np.float64), g_analysis, hop_size, stft_channels)[0]

    def idgt(self, X, hop_size=None, stft_channels=None):
        """Compute the inverse DGT of real signal x with a gauss window."""
        if hop_size is None:
            hop_size = self.hop_size
        if stft_channels is None:
            stft_channels = self.stft_channels
        assert (len(X.shape) == 2)
        assert (np.mod(stft_channels, 2) == 0), 'The number of stft channels needs to be even'
        assert (X.shape[0] == stft_channels // 2 + 1)
        g_synthesis = self._synthesis_window(X, hop_size, stft_channels)
        return idgtreal(X.astype(np.complex128), g_synthesis, hop_size, stft_channels)[0]

    def invert_spectrogram(self, spectrogram, stft_channels=None, hop_size=None):
        """Invert a spectrogram by reconstructing the phase with PGHI."""
        if hop_size is None:
            hop_size = self.hop_size
        if stft_channels is None:
            stft_channels = self.stft_channels

        audio_length = hop_size * spectrogram.shape[1]
        tfr = self.hop_size * self.stft_channels / audio_length
        g_analysis = {'name': 'gauss', 'tfr': tfr}

        tgrad, fgrad = modgabphasegrad('abs', spectrogram, g_analysis, hop_size,
                                       stft_channels)
        phase = pghi(spectrogram, tgrad, fgrad, hop_size, stft_channels, audio_length)

        reComplexStft = spectrogram * np.exp(1.0j * phase)

        return self.idgt(reComplexStft, hop_size, stft_channels)

    def spectrogram(self, time_signal, normalize=p.normalize, **kwargs):
        """Compute the spectrogram of a real signal."""
        stft = self.dgt(time_signal, **kwargs)

        magSpectrogram = np.abs(stft)

        if normalize:
            magSpectrogram = magSpectrogram / np.max(magSpectrogram)
        return magSpectrogram

    def _analysis_window(self, x):
        return {'name': 'gauss', 'tfr': self.hop_size * self.stft_channels / len(x)}

    def _synthesis_window(self, X, hop_size, stft_channels):
        L = hop_size * X.shape[1]
        tfr = self.hop_size * self.stft_channels / L
        g_analysis = {'name': 'gauss', 'tfr': tfr}
        return {'name': ('dual', g_analysis['name']), 'tfr': tfr}

    

class GaussTruncTF(GaussTF):
    """Time frequency transform object based on a Truncated Gauss window.
    """

    def __init__(self, hop_size=p.hop_size, stft_channels=p.stft_channels, min_height=1e-4):
        super().__init__(hop_size, stft_channels)
        self.min_height = min_height

    def _analysis_window(self, x):
        Lgtrue = np.sqrt(-4 * self.hop_size * self.stft_channels * np.log(self.min_height) / np.pi)
        LgLong = np.ceil(Lgtrue / self.stft_channels) * self.stft_channels

        x = (1 / Lgtrue) * np.concatenate([np.arange(.5 * LgLong), np.arange(-.5 * LgLong, 0)])
        g = np.exp(4 * np.log(self.min_height) * (x ** 2))
        g = g / np.linalg.norm(g)
        return g

    def _synthesis_window(self, X, hop_size, stft_channels):
        g_analysis = self._analysis_window(None)
        return gabdual(g_analysis, hop_size, stft_channels, 8*max(stft_channels, self.stft_channels))
