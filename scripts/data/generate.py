from scripts.data.utils import *
from absl import app
from absl import flags
import pickle as pkl

flags.DEFINE_integer("size", 32, help="")
flags.DEFINE_float("dt", 0.05, help="")
flags.DEFINE_integer('seed', 1, help="")
flags.DEFINE_integer('sim_num', 100, help="")
flags.DEFINE_bool('flip', True, help="")
flags.DEFINE_bool('view', False, help="")
flags.DEFINE_bool('obs_center', True, help="")
flags.DEFINE_integer('particle_limit', 1500, help="")
flags.DEFINE_integer('duration', 500, help="")
flags.DEFINE_list('scale', [0.1, 0.9], help="")
flags.DEFINE_bool('record', True, help="")
flags.DEFINE_string('path', None, help="")
flags.DEFINE_integer('dim', 2, help="")
flags.DEFINE_string('name', None, help="")

FLAGS = flags.FLAGS


def main(_):
    # --- DEFINE DATASET SPECS --- ####
    DOMAIN = Domain(x=FLAGS.size, y=FLAGS.size, boundaries=CLOSED, bounds=Box[0:FLAGS.size, 0:FLAGS.size])
    GRAVITY = math.tensor([0, -9.81])
    SCENE = None
    SCENE_PARAMS = None
    RADIUS_SPECS = dict(radii=[0.015, 0.03], step=50, freq=5)

    # --- PREPARE FOR GENERATION --- ####
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    VEL_STATS = Welford()
    ACC_STATS = Welford()
    PART_STATS = Welford()
    RADIUS_STATS = [None] * len(RADIUS_SPECS['radii'])
    RADIUS_STATS_NUM = 0
    HIGHEST = 0
    LOWEST = FLAGS.particle_limit

    EXAMPLES = []

    if not os.path.exists(FLAGS.path):
        os.makedirs(FLAGS.path)

    def step(points, velocity, accessible, obs_points, obs, **kwargs):
        div_free_velocity, pressure, _, _, occupied = flip.make_incompressible(velocity + FLAGS.dt * GRAVITY, DOMAIN, accessible, points)
        if FLAGS.flip:
            points = flip.map_velocity_to_particles(points, div_free_velocity, occupied, previous_velocity_grid=velocity * accessible)
        else:
            points = flip.map_velocity_to_particles(points, div_free_velocity, occupied)
        points = advect.runge_kutta_4(points, div_free_velocity, FLAGS.dt, accessible=accessible, occupied=occupied)
        points = flip.respect_boundaries(points, DOMAIN, obs)
        return dict(points=points, accessible=accessible, pressure=pressure, scene=points & obs_points, obs_points=obs_points,
                    obs=obs, velocity=points >> DOMAIN.staggered_grid(), div_free_velocity=div_free_velocity, occupied=occupied)

    for i in range(FLAGS.sim_num):
        valid = False
        while not valid:
            if SCENE is None:
                initial, obs, vel, params = random_scene(DOMAIN)
                if i == 0:
                    SCENE_PARAMS = params
            else:
                initial, obs, vel = SCENE
            obs_points = DOMAIN.distribute_points(obs, color='#000000', center=FLAGS.obs_center, points_per_cell=1) * (0, 0)
            ACCESSIBLE = DOMAIN.accessible_mask(obs, type=StaggeredGrid)
            particles = DOMAIN.distribute_points(union(initial)) * vel
            scene = particles & obs_points
            valid = SCENE is not None or scene.elements.center.shape[0] < FLAGS.particle_limit

        print(f"Generating scene {i} with {particles.elements.center.shape[0]} particles and {obs_points.elements.center.shape[0]} obstacles.")

        point_num = scene.elements.center.shape[0]
        trajectory = np.zeros((FLAGS.duration + 1, point_num, 2), dtype=np.float32)  # dtype is important for deserializing
        types = np.ones(point_num, dtype=np.int64)  # dtype is important for deserializing
        types[particles.elements.center.shape[0]:] = 0

        state = dict(points=particles, accessible=ACCESSIBLE, pressure=DOMAIN.grid(0), scene=particles & obs_points, obs_points=obs_points,
                     obs=obs, velocity=particles >> DOMAIN.staggered_grid(), div_free_velocity=DOMAIN.staggered_grid(0), occupied=DOMAIN.staggered_grid(0))

        if FLAGS.view:
            app = App()
            app.set_state(state, step_function=step, show=['scene', 'pressure', 'velocity', 'div_free_velocity', 'occupied', 'accessible'])
            show(app, display='scene')
            continue
        else:
            for frame in range(FLAGS.duration + 1):
                state = step(**state)
                trajectory[frame] = state['scene'].elements.center.numpy()

        if FLAGS.scale is not None:
            upper = DOMAIN.bounds.upper[0]
            SCALE_factor = upper.numpy() / (FLAGS.scale[1] - FLAGS.scale[0])
            trajectory = (trajectory / SCALE_factor) + FLAGS.scale[0]

        vels = time_diff(trajectory[:, :particles.elements.center.shape[0], :])
        VEL_STATS.addAll(vels.reshape(-1, 2))
        accs = time_diff(vels)
        ACC_STATS.addAll(accs.reshape(-1, 2))
        PART_STATS.add(point_num)
        if point_num > HIGHEST:
            HIGHEST = point_num
        if point_num < LOWEST:
            LOWEST = point_num
        if i % RADIUS_SPECS['freq'] == 0:
            print("Radius analysis")
            RADIUS_STATS_NUM += 1
            for ix, radius in enumerate(RADIUS_SPECS['radii']):
                means, maxs, mins = get_radius_stats(trajectory, radius, RADIUS_SPECS['step'])
                if RADIUS_STATS[ix] is None:
                    RADIUS_STATS[ix] = means
                else:
                    RADIUS_STATS[ix] += means
            print("Finished radius analysis")

        if FLAGS.record:
            EXAMPLES.append(save2sequence(trajectory, types))
        else:
            if FLAGS.scale is not None:
                size = FLAGS.scale
            else:
                size = FLAGS.size
            if FLAGS.flip:
                rollout = dict(flip=trajectory, size=size, types=types)
            else:
                rollout = dict(pic=trajectory, size=size, types=types)
            with open(os.path.join(FLAGS.path, f'sim_{i}.pkl'), 'wb') as f:
                pkl.dump(rollout, f)

    if FLAGS.record:
        radius_stats = list(map(lambda x: x / RADIUS_STATS_NUM, RADIUS_STATS))
        SCENE_PARAMS.pop('domain')
        metadata = dict(bounds=[FLAGS.scale, FLAGS.scale],
                        sequence_length=FLAGS.duration,
                        dim=FLAGS.dim,
                        dt=FLAGS.dt,
                        dataset_size=FLAGS.sim_num,
                        vel_wf=VEL_STATS,
                        acc_wf=ACC_STATS,
                        particle_num_wf=PART_STATS,
                        examples=EXAMPLES,
                        scene=SCENE_PARAMS,
                        particle_num_range=(LOWEST, HIGHEST),
                        radius_stats=radius_stats)
    
        with open(os.path.join(FLAGS.path, FLAGS.name) + '.pkl', 'wb') as f:
            pkl.dump(metadata, f)


if __name__ == "__main__":
    app.run(main)
