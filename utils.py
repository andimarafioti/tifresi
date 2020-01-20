import numpy as np
import librosa


# This function might need another name
def preprocess_signal(y, M=1024): 
    """Trim and cut signal"""
    # Trimming
    y, _ = librosa.effects.trim(y)
    
    # Preemphasis
    # y = np.append(y[0], y[1:] - 0.97 * y[:-1])
    
    # Padding
    left_over = np.mod(y, M)
    extra = M-left_over if left_over else 0
    y = np.pad(y,(0,extra))
    assert(np.mod(len(y),M)==0)
    
    return y


def load_signal(fpath, sr=None):
    # Loading sound file
    y, sr = librosa.load(fpath, sr=sr)
    return y, sr


def downsample_tf_time(mel, rr):
    tmp = np.zeros([mel.shape[0], mel.shape[1]//rr], tmp.dtype)
    for i in range(rr):
        tmp += mel[:, i::rr]
    return tmp/rr


    