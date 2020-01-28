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
        Lg = np.round(np.sqrt(- 4 * self.hop_size * self.stft_channels * np.log(self.min_height) / np.pi))

        if np.mod(Lg, 2) == 0:
            x = np.concatenate([np.arange(0, .5 - 1 / Lg, 1 / Lg), np.arange(-.5, -1 / Lg, 1 / Lg)])
        else:
            x = np.concatenate([np.arange(0, .5 - .5 / Lg, 1 / Lg), np.arange(-.5+.5/Lg, -1 / Lg, 1 / Lg)])

        return np.exp(4 * np.log(self.min_height) * (x ** 2))

    def _synthesis_window(self, X, hop_size):
        g_analysis = self._analysis_window(None)
        return gabdual(g_analysis, self.hop_size, self.stft_channels)
