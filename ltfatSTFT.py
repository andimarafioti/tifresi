import numpy as np

from stft import GaussTF

__author__ = 'Andres'


class LTFATStft(object):
    gausstf = GaussTF(a=256, M=1024)

    def oneSidedStft(self, signal, windowLength, hopSize):
        return self.gausstf.dgt(signal, hopSize, windowLength)

    def inverseOneSidedStft(self, signal, windowLength, hopSize):
        return self.gausstf.idgt(signal, hopSize, windowLength)

    def magAndPhaseOneSidedStft(self, signal, windowLength, hopSize):
        stft = self.oneSidedStft(signal, windowLength, hopSize)
        return np.abs(stft), np.angle(stft)

    def log10MagAndPhaseOneSidedStft(self, signal, windowLength, hopSize, clipBelow=1e-14):
        realDGT = self.oneSidedStft(signal, windowLength, hopSize)
        return self.log10MagFromRealDGT(realDGT, clipBelow), np.angle(realDGT)

    def log10MagFromRealDGT(self, realDGT, clipBelow=1e-14):
        return np.log10(np.clip(np.abs(realDGT), a_min=clipBelow, a_max=None))

    def reconstructSignalFromLogged10Spectogram(self, logSpectrogram, phase, windowLength, hopSize):
        reComplexStft = (10 ** logSpectrogram) * np.exp(1.0j * phase)
        return self.inverseOneSidedStft(reComplexStft, windowLength, hopSize)

    def logMagAndPhaseOneSidedStft(self, signal, windowLength, hopSize, clipBelow=np.e**-30, normalize=False):
        realDGT = self.oneSidedStft(signal, windowLength, hopSize)
        spectrogram = self.logMagFromRealDGT(realDGT, clipBelow, normalize)
        return spectrogram, np.angle(realDGT)

    def logMagFromRealDGT(self, realDGT, clipBelow=np.e**-30, normalize=False):
        spectrogram = np.abs(realDGT)
        if normalize:
            spectrogram = spectrogram/np.max(spectrogram)
        return np.log(np.clip(spectrogram, a_min=clipBelow, a_max=None))

    def reconstructSignalFromLoggedSpectogram(self, logSpectrogram, phase, windowLength, hopSize):
        reComplexStft = (np.e ** logSpectrogram) * np.exp(1.0j * phase)
        return self.inverseOneSidedStft(reComplexStft, windowLength, hopSize)
