import ltfatpy
import numpy as np


class GaussTF(object):
    def __init__(self, a, M):
        self.a = a
        self.M = M
    def dgt(self, x, a=None, M=None):
        """Compute the DGT of a real signal x with a gauss window."""
        if a is None:
            a = self.a
        if M is None:
            M = self.M
        assert(len(x.shape)==1)
        assert(np.mod(len(x), a)==0)
        assert(np.mod(len(x), M)==0)
        g_analysis = {'name': 'gauss', 'tfr': self.a*self.M/len(x)}
        return ltfatpy.dgtreal(x, g_analysis, a, M)[0]
    def idgt(self, X, a=None, M=None):
        """Compute the inverse DGT for real signal x with a gauss window."""
        if a is None:
            a = self.a
        if M is None:
            M = self.M
        assert(len(X.shape)==2)
        assert(X.shape[0]==M//2+1)
        L = a*X.shape[1]
        tfr = self.a*self.M/L
        g_analysis = {'name': 'gauss', 'tfr': tfr }
        g_synthesis = {'name': ('dual', g_analysis['name']), 'tfr': tfr}
        return ltfatpy.idgtreal(X, g_synthesis, a, M)[0]