from utils.modGabPhaseGrad import modgabphasegrad
from utils.numba_pghi import pghi
from utils.ourLTFATStft import LTFATStft

import numpy as np

__author__ = 'Andres'


class SpectrogramInverter(object):
	def __init__(self, fft_size, fft_hop_size, audio_length):
		super().__init__()
		self._fft_window_length = fft_size
		self._fft_hop_size = fft_hop_size
		self._audio_length = audio_length
		self._anStftWrapper = LTFATStft()

	def _magnitudeErr(self, targetSpectrogram, originalSpectrogram):
		return np.linalg.norm(np.abs(targetSpectrogram) - np.abs(originalSpectrogram), 'fro') / \
			   np.linalg.norm(np.abs(targetSpectrogram), 'fro')

	def invertSpectrograms(self, unprocessed_spectrograms):
		reconstructed_audio_signals = np.zeros([unprocessed_spectrograms.shape[0], self._audio_length])

		for index, spectrogram in enumerate(unprocessed_spectrograms):
			reconstructed_audio_signals[index] = self._invertSpectrogram(spectrogram)
		return reconstructed_audio_signals

	def projectionLoss(self, unprocessed_spectrograms):
		reconstructed_audio_signals = self.invertSpectrograms(unprocessed_spectrograms)
		projection_loss = np.zeros([unprocessed_spectrograms.shape[0]])

		for index, spectrogram in enumerate(unprocessed_spectrograms):
			reconstructed_spectrogram, _ = self._anStftWrapper.magAndPhaseOneSidedStft(reconstructed_audio_signals[index],
																						  self._fft_window_length,
																						  self._fft_hop_size)
			projection_loss[index] = 20*np.log10(1/self._magnitudeErr(reconstructed_spectrogram[:-1], spectrogram))
		return projection_loss

	def projectionLossBetween(self, unprocessed_spectrograms, audio_signals):
		projection_loss = np.zeros([unprocessed_spectrograms.shape[0]])

		for index, audio_signal in enumerate(audio_signals):
			reconstructed_spectrogram, _ = self._anStftWrapper.magAndPhaseOneSidedStft(audio_signal,
																						  self._fft_window_length,
																						  self._fft_hop_size)
			projection_loss[index] = 20*np.log10(1/self._magnitudeErr(reconstructed_spectrogram[:-1], unprocessed_spectrograms[index]))
		return projection_loss

	def _invertSpectrogram(self, unprocessed_spectrogram):
		unprocessed_spectrogram = np.concatenate([unprocessed_spectrogram,
												  np.zeros_like(unprocessed_spectrogram)[0:1, :]], axis=0) # Fill last column of freqs with zeros

		tfr = 256 * 1024 / self._audio_length
		g_analysis = {'name': 'gauss', 'tfr': tfr}

		tgrad, fgrad = modgabphasegrad('abs', unprocessed_spectrogram, g_analysis, self._fft_hop_size,
											self._fft_window_length)

		logMagSpectrogram = np.log(np.clip(unprocessed_spectrogram.astype(np.float64), a_min=np.exp(-10), a_max=None))
		phase = pghi(logMagSpectrogram, tgrad, fgrad, self._fft_hop_size,
					 self._fft_window_length,
					 self._audio_length, tol=10)
		return self._anStftWrapper.reconstructSignalFromLoggedSpectogram(logMagSpectrogram, phase,
																   windowLength=self._fft_window_length,
																   hopSize=self._fft_hop_size)
