import numpy as np


class Welford:
    # based on: https://www.johndcook.com/blog/standard_deviation/
    def __init__(self):
        self._num = 0
        self._oldM = None
        self._newM = None
        self._oldM2 = None
        self._newM2 = None

    def add(self, sample):
        self._num += 1
        if self._num == 1:
            self._oldM = self._newM = sample
            self._oldM2 = self._newM2 = 0.0
        else:
            self._newM = self._oldM + (sample - self._oldM) / self._num
            self._newM2 = self._oldM2 + (sample - self._oldM) * (sample - self._newM)
            self._oldM = self._newM
            self._oldM2 = self._newM2

    def addAll(self, samples):
        for sample in samples:
            self.add(sample)

    @property
    def num(self):
        return self._num

    @property
    def mean(self):
        return self._newM

    @property
    def var(self):
        if self._num > 1:
            return self._newM2 / (self._num - 1)
        else:
            return None

    @property
    def M2(self):
        return self._newM2

    @property
    def std(self):
        if self.var is not None:
            return np.sqrt(self.var)
        else:
            return None

    def merge(self, other):
        num = self.num + other.num
        delta = self.mean - other.mean
        delta2 = delta * delta
        mean = (self.num * self.mean + other.num * other.mean) / num
        M2 = self.M2 + other.M2 + delta2 * (self.num * other.num) / num

        self._num = num
        self._newM = self._oldM = mean
        self._newM2 = self._oldM2 = M2
