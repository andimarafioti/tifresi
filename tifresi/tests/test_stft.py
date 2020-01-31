import sys

# sys.path.append('../')

import numpy as np

from tifresi.stft import GaussTF, GaussTruncTF


def test_stft_different_length(a = 128, M = 1024, trunc=False):
    L = 128 * 1024
    if trunc:
        tfsystem = GaussTruncTF(a, M)        
    else:
        tfsystem = GaussTF(a, M)

    x = np.random.rand(L) * 2 - 1
    x = x / np.linalg.norm(x)
    x[:8 * M] = 0
    x[-8 * M:] = 0
    x2 = np.pad(x.copy(), L)[L:]
    X = tfsystem.dgt(x)
    xdot = tfsystem.idgt(X)
    X2 = tfsystem.dgt(x2)
    x2dot = tfsystem.idgt(X2)
    if trunc:
        assert (np.linalg.norm(xdot - x) < 1e-10)
        assert (np.linalg.norm(x2dot - x2) < 1e-10)
        assert (np.sum(np.abs(X2[:, :X.shape[1]] - X)) < 1e-6)
    else:
        assert (np.linalg.norm(xdot - x) < 1e-12)
        assert (np.linalg.norm(x2dot - x2) < 1e-12)
        assert (np.sum(np.abs(X2[:, :X.shape[1]] - X)) < 1e-6)


def test_stft_different_hop_size(a = 128, M = 1024, trunc=False):
    hop_size = a
    if trunc:
        tfsystem = GaussTruncTF(hop_size, M)        
    else:
        tfsystem = GaussTF(hop_size, M)
    L = 128 * 1024
    x = np.random.rand(L) * 2 - 1
    x = x / np.linalg.norm(x)
    X128 = tfsystem.dgt(x, hop_size=128)
    X256 = tfsystem.dgt(x, hop_size=256)
    assert (np.sum(np.abs(X256 - X128[:, ::2])) < 1e-12)
    x256dot = tfsystem.idgt(X256, hop_size=256)
    x128dot = tfsystem.idgt(X128, hop_size=128)
    if trunc:
        assert (np.linalg.norm(x128dot - x) < 1e-10)
        assert (np.linalg.norm(x256dot - x) < 1e-10)
    else:
        assert (np.linalg.norm(x128dot - x) < 1e-12)
        assert (np.linalg.norm(x256dot - x) < 1e-12)
        
def test_stft_different_channels(a = 128, M = 1024, trunc=False):
    hop_size = a
    stft_channels = M
    if trunc:
        tfsystem = GaussTruncTF(hop_size, stft_channels)        
    else:
        tfsystem = GaussTF(hop_size, stft_channels)
    L = 128 * 1024
    x = np.random.rand(L) * 2 - 1
    x = x / np.linalg.norm(x)
    X1024 = tfsystem.dgt(x, stft_channels=1024)
    X512 = tfsystem.dgt(x, stft_channels=512)
    assert (np.sum(np.abs(X512 - X1024[::2, :])) < 1e-12)
    x1024dot = tfsystem.idgt(X1024, stft_channels=1024)
    x512dot = tfsystem.idgt(X512, stft_channels=512)
    if trunc:
        assert (np.linalg.norm(x1024dot - x) < 1e-5)
        assert (np.linalg.norm(x512dot - x) < 1e-5)        
    else:
        assert (np.linalg.norm(x1024dot - x) < 1e-12)
        assert (np.linalg.norm(x512dot - x) < 1e-12)

def main():
    combinations = [
        (128, 1024),
        (128, 512),
        (256, 1024)
    ]
    for trunc in [True, False]:
        for a,M in combinations:
            print("Test combination {},{},{}".format(trunc, a,M))
            test_stft_different_length(a,M, trunc)
            test_stft_different_hop_size(a,M, trunc)
            test_stft_different_channels(a,M, trunc)

if __name__ == "__main__":
    main()
