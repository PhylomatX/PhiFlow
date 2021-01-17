import random
from phi.flow import *


def random_scene(domain: Domain, pool_prob: float = 0.3, pool_min: int = 3, pool_max: int = 8,
                 block_min: int = 0, block_size_max: int = 20, block_size_min: int = 5, wall_distance: int = 2,
                 block_num_min: int = 1, block_num_max: int = 2, multiple_blocks_prob: float = 0):
    size = int(domain.bounds.upper[0])
    initial_density = domain.grid().values
    pool = random.random()
    if pool < pool_prob:
        pool_height = random.randint(pool_min, pool_max)
        initial_density.native()[:, :pool_height] = 1
    multiple_blocks = random.random()
    block_num = 1
    if pool > pool_prob and multiple_blocks < multiple_blocks_prob:
        block_num = random.randint(block_num_min, block_num_max)
    for i in range(block_num):
        block_ly = random.randint(block_min, size - block_size_min - wall_distance)
        block_lx = random.randint(1, size - block_size_min - wall_distance)
        block_uy = random.randint(block_ly + block_size_min, min(block_ly + block_size_max, size - wall_distance))
        block_ux = random.randint(block_lx + block_size_min, min(block_lx + block_size_max, size - wall_distance))
        initial_density.native()[block_lx:block_ux, block_ly:block_uy] = 1
    # ensure that no block sticks at top
    initial_density.native()[:, size - 1] = 0
    return initial_density


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
