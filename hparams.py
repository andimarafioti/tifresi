

class HParams(object):
    
    # Signal parameters
    sr = 22050 # Sampling frequency of the signal
    M = 1024 # Ensure that the signal will be a multiple of M
    
    # STFT parameters
    stft_channels = 1024 # Number of frequency channels
    hop_size = 256 # Hop size
    
    stft_dB = 50 # Number of dB for the STFT
    normalize = True # Normalize STFT
    
    # MEL parameters
    n_mels = 80 # Number of mel frequency band
    fmin = 0 # Minimum frequency for the MEL
    fmax = None # Maximum frequency for the MEL (None -> Nyquist frequency)
    
    mel_dB = 50 # Number of dB for the MEL
    
    
    
    
    