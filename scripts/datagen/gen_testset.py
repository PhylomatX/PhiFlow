import os
import random
import pickle as pkl
from phi.flow import *
import tensorflow as tf
from typing import List, Union, Tuple
from phi.math import Tensor
from phi.physics._effect import Gravity, gravity_tensor
from phi.field._point_cloud import _distribute_points


def time_diff(input_sequence):
    return input_sequence[1:, ...] - input_sequence[:-1, ...]


def sim2file(domain: Domain, idensity: Tensor, duration: int = 100, step_size: Union[int, float] = 0.1, inflow: int = 0,
             obstacles: List[Obstacle] = None, scale: List[float] = None, pic: bool = False, tf_example: bool = True, vel: np.ndarray = None):
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
                 pressure=domain.grid(0),  divergence=domain.grid(0), smask=smask, cmask=cmask,
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
        scale_factor = upper.numpy() / (scale[1]-scale[0])
        positions = (positions / scale_factor) + scale[0]

    vels = time_diff(positions)
    accs = time_diff(vels)
    vels_squared = vels * vels
    accs_squared = accs * accs
    vel_sum = vels.sum(axis=0).sum(axis=0)
    acc_sum = accs.sum(axis=0).sum(axis=0)
    vel_sqr_sum = vels_squared.sum(axis=0).sum(axis=0)
    acc_sqr_sum = accs_squared.sum(axis=0).sum(axis=0)
    vel_num = vels.shape[0] * vels.shape[1]
    acc_num = accs.shape[0] * accs.shape[1]

    if tf_example:
        stypes = types.tobytes()
        spositions = positions.tobytes()

        types_feat = tf.train.Feature(bytes_list=tf.train.BytesList(value=[stypes]))
        types_feats = tf.train.Features(feature=dict(particle_type=types_feat))

        positions_feat = tf.train.Feature(bytes_list=tf.train.BytesList(value=[spositions]))
        positions_featlist = tf.train.FeatureList(feature=[positions_feat])
        positions_featlists = tf.train.FeatureLists(feature_list=dict(position=positions_featlist))

        return tf.train.SequenceExample(context=types_feats, feature_lists=positions_featlists), vel_sum, vel_num, acc_sum, acc_num, vel_sqr_sum, acc_sqr_sum
    else:
        return positions, types, vel_sum, vel_num, acc_sum, acc_num, vel_sqr_sum, acc_sqr_sum


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
        new_velocity = math.tensor(math.concat([velocity.values, initial_velocity], dim='points'), names=['points', 'vector'])
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

    return dict(points=points, velocity=velocity, v_field=v_field, v_force_field=v_force_field, v_change_field=v_change_field,
                v_div_free_field=v_div_free_field, density=points.at(domain.grid()), pressure=pressure, divergence=div,
                smask=smask, cmask=cmask, accessible=accessible * 2 + cmask, iter=iter + 1, ones=ones, zeros=zeros,
                domain=domain, sones=sones, szeros=szeros, obstacles=obstacles, inflow=inflow, initial_velocity=initial_velocity,
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
        block_ly = random.randint(block_min, size - block_size_max - wall_distance)
        block_lx = random.randint(1, size - block_size_max - wall_distance)
        block_uy = random.randint(block_ly + block_size_min, block_ly + block_size_max)
        block_ux = random.randint(block_lx + block_size_min, block_lx + block_size_max)
        initial_density.native()[block_lx:block_ux, block_ly:block_uy] = 1
    # ensure that no block sticks at top
    initial_density.native()[:, size-1] = 0
    return initial_density


x = 64
y = 64
domain = Domain(x=x, y=y, boundaries=CLOSED, bounds=Box[0:x, 0:y])
dataset_size = 5
scale = [0.1, 0.9]
sequence_length = 100
dt = 0.1

save_loc = os.path.expanduser('~/Projekte/BA/datasets/GNN_tests/training_set1/')
if not os.path.exists(save_loc):
    os.makedirs(save_loc)

# initial_density = domain.grid().values
#
# initial_density.native()[10:30, 30:50] = 1
# vel = np.array([20, 0])
#
# # initial_density.native()[:, :15] = 1
# # initial_density.native()[20:50, 45:60] = 1
# obstacles = [Obstacle(Box[40:42, 10:60])]
# # obstacles = ()
# random.seed = 1
# np.random.seed(1)
# flip, types = sim2file(domain, initial_density, duration=sequence_length + 1, step_size=dt, scale=None, tf_example=False, vel=vel, obstacles=obstacles)
# pic, _ = sim2file(domain, initial_density, duration=sequence_length + 1, step_size=dt, scale=None, tf_example=False, pic=True, vel=vel, obstacles=obstacles)
#
# sims = dict(flip=flip, pic=pic, types=types)
# with open(save_loc + 'scene_4.pkl', 'wb') as f:
#     pkl.dump(sims, f)

examples = []
tvel_sum = np.array([0, 0])
tvel_sqr_sum = np.array([0, 0])
tacc_sum = np.array([0, 0])
tacc_sqr_sum = np.array([0, 0])
tvel_num = 0
tacc_num = 0

for sim_ix in range(dataset_size):
    initial_density = random_scene(domain)
    example, vel_sum, vel_num, acc_sum, acc_num, vel_sqr_sum, acc_sqr_sum = \
        sim2file(domain, initial_density, duration=sequence_length + 1, step_size=dt, scale=scale)
    examples.append(example)
    tvel_sum += vel_sum
    tacc_sum += acc_sum
    tvel_num += vel_num
    tacc_num += acc_num
    tvel_sqr_sum += vel_sqr_sum
    tacc_sqr_sum += acc_sqr_sum

with tf.io.TFRecordWriter(save_loc + 'train.tfrecord') as writer:
    for example in examples:
        writer.write(example.SerializeToString())

vel_mean = tvel_sum / tvel_num
acc_mean = tacc_sum / tacc_num
vel_std = np.sqrt(tvel_sqr_sum / tvel_num - vel_mean * vel_mean)
acc_std = np.sqrt(tacc_sqr_sum / tacc_num - acc_mean * acc_mean)

metadata = dict(bounds=[scale, scale], sequence_length=sequence_length, default_connectivity_radius=0.015, dim=2, dt=dt,
                dataset_size=dataset_size)
with open(save_loc + 'metadata.pkl', 'wb') as f:
    pkl.dump(metadata, f)

