import ltfatpy
import numpy as np
from hparams import HParams as p

from modGabPhaseGrad import modgabphasegrad
from pghi import pghi


class GaussTF(object):
    """Time frequency transform object based on a Gauss window.
    
    The Gauss window is necessary to apply the PGHI (Phase gradient heap integration) algorithm.
    """
    def __init__(self, hop_size=p.hop_size, stft_channels=p.stft_channels):
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
        assert (np.mod(len(x), stft_channels) == 0)
        g_analysis = {'name': 'gauss', 'tfr': self.hop_size * self.stft_channels / len(x)}
        return ltfatpy.dgtreal(x.astype(np.float64), g_analysis, hop_size, stft_channels)[0]

    def idgt(self, X, hop_size=None, stft_channels=None):
        """Compute the inverse DGT of real signal x with a gauss window."""
        if hop_size is None:
            hop_size = self.hop_size
        if stft_channels is None:
            stft_channels = self.M
        assert (len(X.shape) == 2)
        assert (X.shape[0] == stft_channels // 2 + 1)
        L = hop_size * X.shape[1]
        tfr = self.hop_size * self.stft_channels / L
        g_analysis = {'name': 'gauss', 'tfr': tfr}
        g_synthesis = {'name': ('dual', g_analysis['name']), 'tfr': tfr}
        return ltfatpy.idgtreal(X.astype(np.complex128), g_synthesis, hop_size, stft_channels)[0]
    
    
    def invert_spectrogram(self, spectrogram, audio_length=None, stft_channels=None, hop_size=None):
        """Invert a spectrogram by reconstructing the phase with PGHI."""
        if hop_size is None:
            hop_size = self.hop_size
        if stft_channels is None:
            stft_channels = self.stft_channels
        if audio_length is None:
            audio_length = hop_size*spectrogram.shape[1]

        tfr = self.hop_size * self.stft_channels / audio_length
        g_analysis = {'name': 'gauss', 'tfr': tfr}

        tgrad, fgrad = modgabphasegrad('abs', spectrogram, g_analysis, hop_size,
                                       stft_channels)
        a_min = np.exp(-10)*np.max(spectrogram)
        logMagSpectrogram = np.log(np.clip(spectrogram.astype(np.float64), a_min=a_min, a_max=None))
        phase = pghi(logMagSpectrogram, tgrad, fgrad, hop_size, stft_channels, audio_length, tol=10)

        reComplexStft = (np.e ** logMagSpectrogram) * np.exp(1.0j * phase)

        return self.idgt(reComplexStft, hop_size, stft_channels)
    
    def spectrogram(self, time_signal, normalize=p.normalize, **kwargs):
        """Compute the spectrogram of a real signal."""
        stft = self.dgt(time_signal, **kwargs)

        magSpectrogram = np.abs(stft)

        if normalize:
            magSpectrogram = magSpectrogram/np.max(magSpectrogram)
        return magSpectrogram
    
    

