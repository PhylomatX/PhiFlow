import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import random
from typing import Tuple
import tensorflow as tf
from scipy.spatial import KDTree
from phi.flow import *


# --- HELPERS --- ####

def time_diff(input_sequence):
    return input_sequence[1:, ...] - input_sequence[:-1, ...]


def save2sequence(positions, types):
    stypes = types.tobytes()
    spositions = positions.tobytes()

    types_feat = tf.train.Feature(bytes_list=tf.train.BytesList(value=[stypes]))
    types_feats = tf.train.Features(feature=dict(particle_type=types_feat))

    positions_feat = tf.train.Feature(bytes_list=tf.train.BytesList(value=[spositions]))
    positions_featlist = tf.train.FeatureList(feature=[positions_feat])
    positions_featlists = tf.train.FeatureLists(feature_list=dict(position=positions_featlist))

    return tf.train.SequenceExample(context=types_feats, feature_lists=positions_featlists)


# --- GENERATION --- ####

def random_scene(domain: Domain,
                 pool_prob: float = 0.3,
                 pool_height_range: Tuple[int] = (3, 8),
                 block_size_range: Tuple[int] = (2, 20),
                 wall_distance: int = 2,
                 block_num_range: Tuple[int] = (2, 3),
                 multiple_blocks_prob: float = 0.3,
                 obstacle_prob: float = 0.8,
                 obstacle_num_range: Tuple[int] = (1, 5),
                 obstacle_length_range: Tuple[int] = (2, 20),
                 obstacle_rot_range: Tuple[int] = (0, 90),
                 vel_prob: float = 0.3,
                 vel_range: Tuple[float] = (-5, 5)):

    params = locals()
    size = int(domain.bounds.upper[0])
    initial = []

    if random.random() < pool_prob:
        initial.append(Box[:, :random.randint(*pool_height_range)])

    block_num = block_num_range[0]
    if random.random() < multiple_blocks_prob:
        block_num = random.randint(*block_num_range)
    highest_block = 0
    for i in range(block_num):
        block_ly = random.randint(wall_distance, size - block_size_range[0] - wall_distance)
        block_lx = random.randint(wall_distance, size - block_size_range[0] - wall_distance)
        block_uy = random.randint(block_ly + block_size_range[0], min(block_ly + block_size_range[1], size - wall_distance))
        block_ux = random.randint(block_lx + block_size_range[0], min(block_lx + block_size_range[1], size - wall_distance))
        initial.append(Box[block_lx:block_ux, block_ly:block_uy])
        if block_ly > highest_block:
            highest_block = block_ly

    obs_num = 0
    obstacles = []
    if random.random() < obstacle_prob:
        obs_num = random.randint(*obstacle_num_range)
    for obs in range(obs_num):
        valid = False
        while not valid:
            obs_rot = random.randint(*obstacle_rot_range)
            obs_ly = random.randint(0, highest_block)
            obs_lx = random.randint(0, size)
            obs_uy = obs_ly + 1
            obs_ux = obs_lx + random.randint(*obstacle_length_range)
            obs = Box[obs_lx:obs_ux, obs_ly:obs_uy].rotated(math.tensor(obs_rot))
            # prevents distortion of fluids when passing by obstacles
            bounding_box = Box[obs_lx-1:obs_ux+1, obs_ly-1:obs_uy+1].rotated(math.tensor(obs_rot))
            obs_mask = domain.grid(HardGeometryMask(union([bounding_box])))
            valid = max(np.unique(domain.grid(HardGeometryMask(union(initial))).values.numpy() + obs_mask.values.numpy())) == 1
        obstacles.append(obs)

    vel = (0, 0)
    if random.random() < vel_prob:
        vel = (random.uniform(*vel_range), random.uniform(*vel_range))

    return initial, obstacles, vel, params


# --- STATISTICS --- ####

def get_radius_stats(trajectory, radius, step):
    means = []
    maxs = []
    mins = []
    for frame in range(0, trajectory.shape[0], step):
        positions = trajectory[frame]
        tree = KDTree(positions)
        queries = tree.query_ball_point(positions, radius)
        lengths = []
        for query in queries:
            lengths.append(len(query))
        lengths = np.array(lengths)
        means.append(lengths.mean())
        maxs.append(lengths.max())
        mins.append(lengths.min())
    return np.array(means), np.array(maxs), np.array(mins)


class Welford:
    """
    Based on: https://www.johndcook.com/blog/standard_deviation/
    """

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
            return 0

    @property
    def M2(self):
        return self._newM2

    @property
    def std(self):
        return np.sqrt(self.var)

    def merge(self, other):
        num = self.num + other.num
        delta = self.mean - other.mean
        delta2 = delta * delta
        mean = (self.num * self.mean + other.num * other.mean) / num
        M2 = self.M2 + other.M2 + delta2 * (self.num * other.num) / num

        self._num = num
        self._newM = self._oldM = mean
        self._newM2 = self._oldM2 = M2
