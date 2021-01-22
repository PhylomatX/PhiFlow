import os
import random
import pickle as pkl
from phi.flow import *
import tensorflow as tf
from typing import List, Union
from phi.math import Tensor
from utils import Welford, random_scene
from phi.field._point_cloud import _distribute_points

from absl import app
from absl import flags

flags.DEFINE_string("save_path", None, help="Path where record should get saved.")
flags.DEFINE_string("name", None, help="Name of file.")
flags.DEFINE_integer("size", 50, help="Number of examples")
flags.DEFINE_integer("duration", 150, help="Number of examples")
flags.DEFINE_integer("seed", 1, help="Random seed")
FLAGS = flags.FLAGS


def time_diff(input_sequence):
    return input_sequence[1:, ...] - input_sequence[:-1, ...]


def sim2file(domain: Domain, idensity: Tensor, duration: int = 100,
             step_size: Union[int, float] = 0.1, inflow: int = 0,
             obstacles: List[Obstacle] = None, scale: List[float] = None,
             pic: bool = False, tf_example: bool = True, vel: np.ndarray = None,
             particle_num_wf: Welford = None, vel_wf: Welford = None, acc_wf: Welford = None,
             point_density: int = 8):
    # generate points
    initial_points = _distribute_points(idensity, point_density)

    points = PointCloud(Sphere(initial_points, 0), add_overlapping=True)
    if vel is None:
        initial_velocity = math.tensor(np.zeros(initial_points.shape), names=['points', 'vector'])
    else:
        initial_velocity = np.ones(initial_points.shape) * vel
        initial_velocity = math.tensor(initial_velocity, names=['points', 'vector'])
    velocity = PointCloud(points.elements, values=initial_velocity)

    # initialize masks
    density = points.at(domain.grid())
    ones = domain.grid(1, extrapolation=density.extrapolation)
    zeros = domain.grid(0, extrapolation=density.extrapolation)
    sones = domain.sgrid(1, extrapolation=density.extrapolation)
    szeros = domain.sgrid(0, extrapolation=density.extrapolation)

    # define obstacles
    if obstacles is None:
        obstacles = []

    obstacle_mask = domain.grid(HardGeometryMask(union([obstacle.geometry for obstacle in obstacles]))).values
    obstacle_points = _distribute_points(obstacle_mask, 3)
    obstacle_points = PointCloud(Sphere(obstacle_points, 0))

    # define initial state
    state = dict(velocity=velocity, v_field=velocity.at(domain.sgrid()), pressure=domain.grid(0),
                 t=0, domain=domain, ones=ones, zeros=zeros, sones=sones, szeros=szeros,
                 obstacles=obstacles, inflow=inflow, initial_points=initial_points,
                 initial_velocity=initial_velocity, pic=pic)

    point_num = len(initial_points.native()) + len(obstacle_points.elements.center.native())
    positions = np.zeros((duration, point_num, 2), dtype=np.float32)
    types = np.ones(point_num, dtype=np.int64)
    types[len(initial_points.native()):] = 0

    for i in range(duration):
        state = step(dt=step_size, **state)
        velocity = state['velocity']
        positions[i, ...] = np.vstack((velocity.elements.center.numpy(), obstacle_points.elements.center.numpy()))

    if scale is not None:
        upper = domain.bounds.upper[0]
        scale_factor = upper.numpy() / (scale[1] - scale[0])
        positions = (positions / scale_factor) + scale[0]

    vels = time_diff(positions)
    if vel_wf is not None:
        vel_wf.addAll(vels.reshape(-1, 2))
    if acc_wf is not None:
        accs = time_diff(vels)
        acc_wf.addAll(accs.reshape(-1, 2))
    if particle_num_wf is not None:
        particle_num_wf.add(vels.shape[1])

    if tf_example:
        stypes = types.tobytes()
        spositions = positions.tobytes()

        types_feat = tf.train.Feature(bytes_list=tf.train.BytesList(value=[stypes]))
        types_feats = tf.train.Features(feature=dict(particle_type=types_feat))

        positions_feat = tf.train.Feature(bytes_list=tf.train.BytesList(value=[spositions]))
        positions_featlist = tf.train.FeatureList(feature=[positions_feat])
        positions_featlists = tf.train.FeatureLists(feature_list=dict(position=positions_featlist))

        return tf.train.SequenceExample(context=types_feats, feature_lists=positions_featlists), vel_wf, acc_wf, particle_num_wf
    else:
        return positions, types, vel_wf, acc_wf


def step(velocity, v_field, pressure, t, ones, zeros, domain, sones, szeros, obstacles,
         inflow, initial_velocity, initial_points, pic, dt):

    # get liquid masks
    cmask = field.where(PointCloud(velocity.elements).at(domain.grid()), ones, zeros)
    smask = field.where(PointCloud(velocity.elements).at(domain.sgrid()), sones, szeros)
    bcs = liquid.get_bcs(domain, obstacles)

    v_force_field = liquid.apply_gravity(dt, v_field)
    v_div_free_field, pressure = liquid.make_incompressible(v_force_field, bcs, cmask, smask, pressure)
    if pic:
        velocity_values = liquid.map2particle(velocity, v_div_free_field, smask)
    else:
        velocity_values = liquid.map2particle(velocity, v_div_free_field, smask, v_field)
    velocity = advect.advect(velocity, v_div_free_field, dt, bcs=bcs, mode='rk4_extp')
    velocity = PointCloud(velocity.elements, values=velocity_values)
    if t < inflow:
        velocity = liquid.add_inflow(velocity, initial_points, initial_velocity)
    velocity = liquid.respect_boundaries(domain, obstacles, velocity)

    # sample new velocity field
    v_field = velocity.at(domain.sgrid())

    return dict(velocity=velocity, v_field=v_field, pressure=pressure, t=t + 1, ones=ones, zeros=zeros,
                domain=domain, sones=sones, szeros=szeros, obstacles=obstacles, inflow=inflow,
                initial_velocity=initial_velocity, initial_points=initial_points, pic=pic)


def main(_):
    x = 32
    y = 32
    domain = Domain(x=x, y=y, boundaries=CLOSED, bounds=Box[0:x, 0:y])
    dataset_size = FLAGS.size
    scale = [0.1, 0.9]
    sequence_length = FLAGS.duration
    dt = 0.05

    examples = []
    vel_wf = Welford()
    acc_wf = Welford()
    particle_num_wf = Welford()

    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    point_density = 8

    for sim_ix in range(dataset_size):
        point_limit = 900
        exclude = True
        while exclude:
            initial_density, obstacles, vel = random_scene(domain, pool_max=4, block_num_max=4, multiple_blocks_prob=0.4,
                                                           block_size_max=15, block_size_min=1, pool_min=2, obstacle_prob=0.8,
                                                           vel_prob=0.5, vel_range=(-5, 0))
            exclude = np.sum(initial_density.numpy()) * point_density >= point_limit
        example, vel_wf, acc_wf, particle_num_wf = sim2file(domain, initial_density, duration=sequence_length + 1,
                                                            step_size=dt, scale=scale, particle_num_wf=particle_num_wf,
                                                            vel_wf=vel_wf, acc_wf=acc_wf, point_density=point_density,
                                                            obstacles=obstacles, vel=vel)
        examples.append(example)

    metadata = dict(bounds=[scale, scale], sequence_length=sequence_length, default_connectivity_radius=0.015, dim=2, dt=dt,
                    dataset_size=dataset_size, vel_wf=vel_wf, acc_wf=acc_wf, particle_num_wf=particle_num_wf, examples=examples)

    if not os.path.exists(FLAGS.save_path):
        os.makedirs(FLAGS.save_path)
    with open(os.path.join(FLAGS.save_path, FLAGS.name) + '.pkl', 'wb') as f:
        pkl.dump(metadata, f)


if __name__ == '__main__':
    app.run(main)
