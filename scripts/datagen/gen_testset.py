import os
import random
from phi.flow import *
import tensorflow as tf
from typing import List, Union
from phi.math import Tensor
from phi.physics._effect import Gravity, gravity_tensor
from phi.field._point_cloud import _distribute_points


def sim2example(domain: Domain, idensity: Tensor, duration: int = 100, step_size: Union[int, float] = 0.1, inflow: int = 0, obstacles: List[Obstacle] = None):
    # generate points
    initial_points = _distribute_points(idensity, 8)
    points = PointCloud(Sphere(initial_points, 0), add_overlapping=True, bounds=domain.bounds)
    initial_velocity = math.tensor(np.zeros(initial_points.shape), names=['points', 'vector'])
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

    # define initial state
    state = dict(points=points, velocity=velocity, density=density, v_force_field=domain.sgrid(0),
                 v_change_field=domain.sgrid(0), v_div_free_field=domain.sgrid(0), v_field=velocity.at(domain.sgrid()),
                 pressure=domain.grid(0),  divergence=domain.grid(0), smask=smask, cmask=cmask,
                 accessible=domain.grid(0), iter=0, domain=domain, ones=ones, zeros=zeros, sones=sones,
                 szeros=szeros, obstacles=obstacles, inflow=inflow, initial_points=initial_points,
                 initial_velocity=initial_velocity)

    positions = np.zeros((duration, len(initial_points.native()), 2))
    types = np.ones(len(initial_points.native()))

    for i in range(duration):
        state = step(dt=step_size, **state)
        velocity = state['velocity']
        positions[i, ...] = velocity.elements.center.native()

    stypes = types.tobytes()
    spositions = positions.tobytes()

    types_feat = tf.train.Feature(bytes_list=tf.train.BytesList(value=[stypes]))
    types_feats = tf.train.Features(feature=dict(particle_type=types_feat))

    positions_feat = tf.train.Feature(bytes_list=tf.train.BytesList(value=[spositions]))
    positions_featlist = tf.train.FeatureList(feature=[positions_feat])
    positions_featlists = tf.train.FeatureLists(feature_list=dict(position=positions_featlist))

    return tf.train.SequenceExample(context=types_feats, feature_lists=positions_featlists)


def step(points, velocity, v_field, pressure, dt, iter, density, cmask, ones, zeros, domain, sones, szeros, obstacles,
         inflow, initial_velocity, initial_points, **kwargs):
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
    v_force_field = field.extp_sgrid(v_force_field * smask, 2) * hard_bcs
    div = field.divergence(v_force_field) * cmask
    laplace = lambda p: field.where(cmask, field.divergence(field.gradient(p, type=StaggeredGrid) * hard_bcs), -4 * p)
    converged, pressure, iterations = field.solve(laplace, div, pressure, solve_params=math.LinearSolve(None, 1e-3))
    gradp = field.gradient(pressure, type=type(v_force_field))
    v_div_free_field = v_force_field - gradp

    # update velocities
    v_change_field = v_div_free_field - v_field
    v_change_field = field.extp_sgrid(v_change_field * smask, 2)
    v_change = v_change_field.sample_at(points.elements.center)
    velocity = velocity.values + v_change

    # advect
    v_div_free_field *= hard_bcs * smask
    points = advect.advect(points, v_div_free_field, dt)
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
        shift = obstacle.geometry.shift_points(points.elements.center)
        points = PointCloud(points.elements.shifted(shift), add_overlapping=True, bounds=points.bounds)
    shift = (~domain.bounds).shift_points(points.elements.center)
    points = PointCloud(points.elements.shifted(shift), add_overlapping=True, bounds=points.bounds)
    velocity = PointCloud(points.elements, values=velocity.values)

    # get new velocity field
    v_field = velocity.at(domain.sgrid())

    return dict(points=points, velocity=velocity, v_field=v_field, v_force_field=v_force_field, v_change_field=v_change_field,
                v_div_free_field=v_div_free_field, density=points.at(domain.grid()), pressure=pressure, divergence=div,
                smask=smask, cmask=cmask, accessible=accessible * 2 + cmask, iter=iter + 1, ones=ones, zeros=zeros,
                domain=domain, sones=sones, szeros=szeros, obstacles=obstacles, inflow=inflow, initial_velocity=initial_velocity,
                initial_points=initial_points)


def random_scene(domain: Domain, pool_prob: float = 0.3, pool_min: int = 3, pool_max: int = 8,
                 block_min: int = 0, block_size_max: int = 20, block_size_min: int = 5, wall_distance: int = 5,
                 block_num_min: int = 1, block_num_max: int = 1):
    size = int(domain.bounds.upper[0])
    initial_density = domain.grid().values
    pool = random.random()
    if pool < pool_prob:
        pool_height = random.randint(pool_min, pool_max)
        initial_density.native()[:, :pool_height] = 1
    block_num = random.randint(block_num_min, block_num_max)
    for i in range(block_num):
        block_ly = random.randint(block_min, size - block_size_min - wall_distance)
        block_lx = random.randint(1, size - block_size_min - wall_distance)
        block_uy = random.randint(block_ly + block_size_min, block_ly + block_size_max - wall_distance)
        block_ux = random.randint(block_lx + block_size_min, block_lx + block_size_max - wall_distance)
        initial_density.native()[block_lx:block_ux, block_ly:block_uy] = 1
    # ensure that no block sticks at top
    initial_density.native()[:, size-1] = 0
    return initial_density


x = 32
y = 32
domain = Domain(x=x, y=y, boundaries=CLOSED, bounds=Box[0:x, 0:y])
steps = 10

save_loc = os.path.expanduser('~/Projekte/BA/datasets/GNN_tests/test_set1/')
if not os.path.exists(save_loc):
    os.makedirs(save_loc)
examples = []

for sim_ix in range(steps):
    initial_density = random_scene(domain)
    examples.append(sim2example(domain, initial_density, duration=101, step_size=0.1))

with tf.io.TFRecordWriter(save_loc + 'train.tfrecord') as writer:
    for example in examples:
        writer.write(example.SerializeToString())
