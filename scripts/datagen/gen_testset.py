import os
import random
import json
import pickle as pkl
from phi.flow import *
import tensorflow as tf
from typing import List, Union, Tuple
from phi.math import Tensor
from utils import Welford
from phi.physics._effect import Gravity, gravity_tensor
from phi.field._point_cloud import _distribute_points

from absl import app
from absl import flags

flags.DEFINE_string("save_path", None, help="Path where record should get saved.")
flags.DEFINE_string("name", None, help="Name of file.")
flags.DEFINE_integer("size", 50, help="Number of examples")
flags.DEFINE_integer("duration", 200, help="Number of examples")
FLAGS = flags.FLAGS


def time_diff(input_sequence):
    return input_sequence[1:, ...] - input_sequence[:-1, ...]


def sim2file(domain: Domain, idensity: Tensor, duration: int = 100,
             step_size: Union[int, float] = 0.1, inflow: int = 0,
             obstacles: List[Obstacle] = None, scale: List[float] = None,
             pic: bool = False, tf_example: bool = True, vel: np.ndarray = None,
             particle_num_wf: Welford = None, vel_wf: Welford = None, acc_wf: Welford = None):
    # generate points
    initial_points = _distribute_points(idensity, 8)
    points = PointCloud(Sphere(initial_points, 0), add_overlapping=True, bounds=domain.bounds)
    if vel is None:
        initial_velocity = math.tensor(np.zeros(initial_points.shape), names=['points', 'vector'])
    else:
        initial_velocity = np.ones(initial_points.shape) * vel
        initial_velocity = math.tensor(initial_velocity, names=['points', 'vector'])
    velocity = PointCloud(points.elements, values=initial_velocity)

    # initialize masks
    density = points.at(domain.grid())
    sdensity = points.at(domain.sgrid())
    ones = domain.grid(1, extrapolation=density.extrapolation)
    zeros = domain.grid(0, extrapolation=density.extrapolation)
    cmask = field.where(density, ones, zeros)
    sones = domain.sgrid(1, extrapolation=density.extrapolation)
    szeros = domain.sgrid(0, extrapolation=density.extrapolation)
    smask = field.where(sdensity, sones, szeros)

    # define obstacles
    if obstacles is None:
        obstacles = []

    obstacle_mask = domain.grid(HardGeometryMask(union([obstacle.geometry for obstacle in obstacles]))).values
    obstacle_points = _distribute_points(obstacle_mask, 1)
    obstacle_points = PointCloud(Sphere(obstacle_points, 0))

    # define initial state
    state = dict(points=points, velocity=velocity, density=density, v_force_field=domain.sgrid(0),
                 v_change_field=domain.sgrid(0), v_div_free_field=domain.sgrid(0), v_field=velocity.at(domain.sgrid()),
                 pressure=domain.grid(0), divergence=domain.grid(0), smask=smask, cmask=cmask,
                 accessible=domain.grid(0), iter=0, domain=domain, ones=ones, zeros=zeros, sones=sones,
                 szeros=szeros, obstacles=obstacles, inflow=inflow, initial_points=initial_points,
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


def step(points, velocity, v_field, pressure, dt, iter, density, cmask, ones, zeros, domain, sones, szeros, obstacles,
         inflow, initial_velocity, initial_points, pic, **kwargs):
    # get domain
    cmask = field.where(density, ones, zeros)
    smask = field.where(points.at(domain.sgrid()), sones, szeros)
    accessible = domain.grid(1 - HardGeometryMask(union([obstacle.geometry for obstacle in obstacles])))
    accessible_mask = domain.grid(accessible, extrapolation=domain.boundaries.accessible_extrapolation)
    hard_bcs = field.stagger(accessible_mask, math.minimum, accessible_mask.extrapolation)

    # apply forces
    force = dt * gravity_tensor(Gravity(), v_field.shape.spatial.rank)
    v_force_field = (v_field + force)

    # solve pressure
    v_force_field = field.extp_sgrid(v_force_field * smask, 1) * hard_bcs  # conserves falling shapes
    div = field.divergence(v_force_field) * cmask
    laplace = lambda p: field.where(cmask, field.divergence(field.gradient(p, type=StaggeredGrid) * hard_bcs), 1e6 * p)
    converged, pressure, iterations = field.solve(laplace, div, pressure, solve_params=math.LinearSolve(None, 1e-5))
    gradp = field.gradient(pressure, type=type(v_force_field))
    v_div_free_field = v_force_field - gradp

    # update velocities
    if pic:
        v_change_field = domain.sgrid(0)
        v_div_free_field = field.extp_sgrid(v_div_free_field * smask, 1)
        velocity = v_div_free_field.sample_at(points.elements.center)
    else:
        v_change_field = v_div_free_field - v_field
        v_change_field = field.extp_sgrid(v_change_field * smask, 1)  # conserves falling shapes (no hard_bcs here!)
        v_change = v_change_field.sample_at(points.elements.center)
        velocity = velocity.values + v_change

    # advect
    points = advect.advect(points, v_div_free_field, dt, bcs=hard_bcs, mode='rk4_extp')
    velocity = PointCloud(points.elements, values=velocity)

    # add possible inflow
    if iter < inflow:
        new_points = math.tensor(math.concat([points.points, initial_points], dim='points'), names=['points', 'vector'])
        points = PointCloud(Sphere(new_points, 0), add_overlapping=True, bounds=points.bounds)
        new_velocity = math.tensor(math.concat([velocity.values, initial_velocity], dim='points'),
                                   names=['points', 'vector'])
        velocity = PointCloud(points.elements, values=new_velocity)

    # push particles inside obstacles outwards and particles outside domain inwards.
    for obstacle in obstacles:
        shift = obstacle.geometry.shift_points(points.elements.center, shift_amount=0.5)
        points = PointCloud(points.elements.shifted(shift), add_overlapping=True, bounds=points.bounds)
    shift = (~domain.bounds).shift_points(points.elements.center, shift_amount=0.5)
    points = PointCloud(points.elements.shifted(shift), add_overlapping=True, bounds=points.bounds)
    velocity = PointCloud(points.elements, values=velocity.values)

    # get new velocity field
    v_field = velocity.at(domain.sgrid())

    return dict(points=points, velocity=velocity, v_field=v_field, v_force_field=v_force_field,
                v_change_field=v_change_field,
                v_div_free_field=v_div_free_field, density=points.at(domain.grid()), pressure=pressure, divergence=div,
                smask=smask, cmask=cmask, accessible=accessible * 2 + cmask, iter=iter + 1, ones=ones, zeros=zeros,
                domain=domain, sones=sones, szeros=szeros, obstacles=obstacles, inflow=inflow,
                initial_velocity=initial_velocity,
                initial_points=initial_points, pic=pic)


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
    if multiple_blocks < multiple_blocks_prob:
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

    for sim_ix in range(dataset_size):
        initial_density = random_scene(domain, pool_prob=0, block_num_max=1, block_size_max=8, block_size_min=8)
        example, vel_wf, acc_wf, particle_num_wf = sim2file(domain, initial_density, duration=sequence_length + 1,
                                                            step_size=dt, scale=scale, particle_num_wf=particle_num_wf,
                                                            vel_wf=vel_wf, acc_wf=acc_wf)
        examples.append(example)

    metadata = dict(bounds=[scale, scale], sequence_length=sequence_length, default_connectivity_radius=0.015, dim=2, dt=dt,
                    dataset_size=dataset_size, vel_wf=vel_wf, acc_wf=acc_wf, particle_num_wf=particle_num_wf, examples=examples)

    if not os.path.exists(FLAGS.save_path):
        os.makedirs(FLAGS.save_path)
    with open(os.path.join(FLAGS.save_path, FLAGS.name) + '.pkl', 'wb') as f:
        pkl.dump(metadata, f)


if __name__ == '__main__':
    app.run(main)
