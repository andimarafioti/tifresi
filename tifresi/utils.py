import numpy as np
import librosa
from tifresi.hparams import HParams as p



# This function might need another name
def preprocess_signal(y, M=p.M):
    """Trim and cut signal.
    
    The function ensures that the signal length is a multiple of M.
    """
    # Trimming
    y, _ = librosa.effects.trim(y)

    # Preemphasis
    # y = np.append(y[0], y[1:] - 0.97 * y[:-1])

    # Padding
    left_over = np.mod(len(y), M)
    extra = M - left_over
    y = np.pad(y, (0, extra))
    assert (np.mod(len(y), M) == 0)

    return y


def load_signal(fpath, sr=None):
    """Load a signal from path."""
    # Loading sound file
    y, sr = librosa.load(fpath, sr=sr)
    return y, sr


def downsample_tf_time(mel, rr):
    """Downsample a TF representation along the time axis."""
    tmp = np.zeros([mel.shape[0], mel.shape[1] // rr], mel.dtype)
    for i in range(rr):
        tmp += mel[:, i::rr]
    return tmp / rr
