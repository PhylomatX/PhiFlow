import os
import random
import pickle as pkl
from phi.flow import *
import tensorflow as tf
from typing import List, Union
from phi.math import Tensor
from utils import Welford, random_scene

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
             point_density: int = 8, distribution: str = 'uniform'):
    # generate points
    initial_points = distribute_points(idensity, point_density, dist=distribution)

    if vel is None:
        initial_velocity = math.tensor(np.zeros(initial_points.shape), names=['points', 'vector'])
    else:
        initial_velocity = np.ones(initial_points.shape) * vel
        initial_velocity = math.tensor(initial_velocity, names=['points', 'vector'])
    initial_velocity = PointCloud(Sphere(initial_points, 0), values=initial_velocity)

    # define obstacles
    if obstacles is None:
        obstacles = []
    obstacle_mask = domain.grid(HardGeometryMask(union([obstacle.geometry for obstacle in obstacles]))).values
    obstacle_points = distribute_points(obstacle_mask, 3)
    obstacle_points = PointCloud(Sphere(obstacle_points, 0))

    # define initial state
    state = dict(velocity=initial_velocity, v_field=initial_velocity.at(domain.sgrid()), pressure=domain.grid(0),
                 t=0, domain=domain, obstacles=obstacles, inflow=inflow, initial_velocity=initial_velocity, pic=pic,
                 c_occupied=PointCloud(initial_velocity.elements, values=1) >> domain.grid())

    point_num = len(initial_points.native()) + len(obstacle_points.elements.center.native())
    positions = np.zeros((duration, point_num, 2), dtype=np.float32)
    types = np.ones(point_num, dtype=np.int64)
    types[len(initial_points.native()):] = 0

    for i in range(duration):
        velocity = state['velocity']
        positions[i, ...] = np.vstack((velocity.elements.center.numpy(), obstacle_points.elements.center.numpy()))
        state = step(dt=step_size, **state)

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

        return tf.train.SequenceExample(context=types_feats,
                                        feature_lists=positions_featlists), vel_wf, acc_wf, particle_num_wf
    else:
        return positions, types, vel_wf, acc_wf


def step(velocity, v_field, t, domain, obstacles, inflow, initial_velocity, pic, dt, **kwargs):
    # get liquid masks
    points = PointCloud(velocity.elements, values=1)
    cmask = points >> domain.grid()
    smask = points >> domain.sgrid()
    bcs = flip.get_accessible_mask(domain, obstacles)

    v_force_field = v_field + dt * gravity_tensor(Gravity(), v_field.shape.spatial.rank)
    v_div_free_field, pressure = flip.make_incompressible(v_force_field, bcs, cmask, smask, domain.grid(0))
    if pic:
        velocity = flip.map_velocity_to_particles(velocity, v_div_free_field, smask)
    else:
        velocity = flip.map_velocity_to_particles(velocity, v_div_free_field, smask, v_field)
    velocity = advect.advect(velocity, v_div_free_field, dt, occupied=smask, valid=bcs, mode='rk4')
    if t < inflow:
        velocity = velocity & initial_velocity
    velocity = flip.respect_boundaries(velocity, domain, obstacles, offset=0.5)

    # sample new velocity field
    v_field = velocity >> domain.sgrid()

    return dict(velocity=velocity, v_field=v_field, pressure=pressure, t=t + 1, domain=domain,
                obstacles=obstacles, inflow=inflow, initial_velocity=initial_velocity, pic=pic,
                c_occupied=cmask)


def main(_):
    size = 64
    domain = Domain(x=size, y=size, boundaries=CLOSED, bounds=Box[0:size, 0:size])
    dataset_size = FLAGS.size
    scale = [0.1, 0.9]
    sequence_length = FLAGS.duration
    dt = 0.1
    point_density = 8
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    dataset = False

    if not os.path.exists(FLAGS.save_path):
        os.makedirs(FLAGS.save_path)

    if dataset:
        examples = []
        vel_wf = Welford()
        acc_wf = Welford()
        particle_num_wf = Welford()

        for sim_ix in range(dataset_size):
            point_limit = 900
            exclude = True
            while exclude:
                initial_density, obstacles, vel = random_scene(domain, pool_max=4, block_num_max=4,
                                                               multiple_blocks_prob=0.4,
                                                               block_size_max=15, block_size_min=1, pool_min=2,
                                                               obstacle_prob=0.8,
                                                               vel_prob=0.5, vel_range=(-5, 5))
                exclude = np.sum(initial_density.numpy()) * point_density >= point_limit
            example, vel_wf, acc_wf, particle_num_wf = sim2file(domain, initial_density, duration=sequence_length + 1,
                                                                step_size=dt, scale=scale,
                                                                particle_num_wf=particle_num_wf,
                                                                vel_wf=vel_wf, acc_wf=acc_wf,
                                                                point_density=point_density,
                                                                obstacles=obstacles, vel=vel)
            examples.append(example)

        metadata = dict(bounds=[scale, scale], sequence_length=sequence_length, default_connectivity_radius=0.015,
                        dim=2, dt=dt,
                        dataset_size=dataset_size, vel_wf=vel_wf, acc_wf=acc_wf, particle_num_wf=particle_num_wf,
                        examples=examples)
        with open(os.path.join(FLAGS.save_path, FLAGS.name) + '.pkl', 'wb') as f:
            pkl.dump(metadata, f)
    else:
        scale = [0, size]
        point_mask = domain.grid(HardGeometryMask(union([Box[10:30, 30:40]])))
        obs = None
        positions_flip, types, _, _ = sim2file(domain, point_mask.values, duration=sequence_length + 1, step_size=dt,
                                               scale=scale, point_density=point_density, tf_example=False,
                                               obstacles=obs)
        positions_pic, types, _, _ = sim2file(domain, point_mask.values, duration=sequence_length + 1, step_size=dt,
                                              scale=scale, point_density=point_density, tf_example=False, obstacles=obs,
                                              pic=True)
        with open(os.path.join(FLAGS.save_path, FLAGS.name) + '.pkl', 'wb') as f:
            pkl.dump(dict(flip=positions_flip, pic=positions_pic, types=types, size=size), f)


if __name__ == '__main__':
    app.run(main)
