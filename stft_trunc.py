import ltfatpy
from ltfatpy.gabor.gabdual import gabdual
import numpy as np
from hparams import HParams as p

from stft import GaussTF


class GaussTruncTF(GaussTF):
    """Time frequency transform object based on a Truncated Gauss window.
    """

    def __init__(self, hop_size=p.hop_size, stft_channels=p.stft_channels, min_height=1e-4):
        super().__init__(hop_size, stft_channels)
        self.min_height = min_height

    def _analysis_window(self, x):
        Lgtrue = np.sqrt(-4 * self.hop_size * self.stft_channels * np.log(self.min_height) / np.pi)
        LgLong = np.ceil(Lgtrue / self.stft_channels) * self.stft_channels

        x = (1 / Lgtrue) * np.concatenate([np.arange(0, .5 * LgLong - 1), np.arange(-.5 * LgLong, -1)])
        g = np.exp(4 * np.log(self.min_height) * (x ** 2))
        g = g / np.linalg.norm(g)
        return g

    def _synthesis_window(self, X, hop_size):
        g_analysis = self._analysis_window(None)
        return gabdual(g_analysis, self.hop_size, self.stft_channels)
