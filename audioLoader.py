import librosa
import numpy as np
from utils.ourLTFATStft import LTFATStft

__author__ = 'Andres'


class AudioLoader(object):
	def __init__(self, sampling_rate, window_length, hop_size, clipBelow, normalize=True):
		super(AudioLoader, self).__init__()

		self._sampling_rate = sampling_rate
		self._window_length = window_length
		self._hop_size = hop_size
		self._clipBelow = clipBelow
		self._normalize = normalize
		self._anStftWrapper = LTFATStft()

	def hopSize(self):
		return self._hop_size

	def windowLength(self):
		return self._window_length

	def loadSound(self, file_name):
		audio, sr = librosa.load(file_name, sr=self._sampling_rate, dtype=np.float64)
		return audio

	def computeSpectrogram(self, audio):
		audio = audio[:len(audio)-np.mod(len(audio), self._window_length)]
		audio = audio[:len(audio)-np.mod(len(audio), self._hop_size)]

		spectrogram, _ = self._anStftWrapper.logMagAndPhaseOneSidedStft(audio, windowLength=self._window_length,
																		hopSize=self._hop_size,
																		clipBelow=np.e ** self._clipBelow,
																		normalize=self._normalize)
		spectrogram = spectrogram / (-self._clipBelow / 2) + 1
		return spectrogram

	def loadAsSpectrogram(self, file_name):
		audio = self.loadSound(file_name)
		return self.computeSpectrogram(audio)