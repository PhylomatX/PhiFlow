from scripts.data.utils import *
import pickle as pkl

# --- DEFINE DATASET SPECS --- ####
SIZE = 32
DOMAIN = Domain(x=SIZE, y=SIZE, boundaries=CLOSED, bounds=Box[0:SIZE, 0:SIZE])
GRAVITY = math.tensor([0, -9.81])
DT = 0.05
SEED = 1
SIM_NUM = 5
FLIP = True
VIEW = False
OBS_CENTER = True
PARTICLE_LIMIT = 1500
DURATION = 300
SCALE = None  # [0.1, 0.9]
RECORD = False
SAVE_PATH = '/home/john/Projekte/BA/Data/datasets/test/'
SCENE = None
SCENE_PARAMS = None
DIM = 2
NAME = ''
RADIUS_SPECS = dict(radii=[], step=50, freq=1)

# --- PREPARE FOR GENERATION --- ####
random.seed(SEED)
np.random.seed(SEED)
VEL_STATS = Welford()
ACC_STATS = Welford()
PART_STATS = Welford()
RADIUS_STATS = [None] * len(RADIUS_SPECS['radii'])
RADIUS_STATS_NUM = 0
HIGHEST = 0
LOWEST = PARTICLE_LIMIT

EXAMPLES = []

if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)


def step(points, accessible, obs_points, obs, **kwargs):
    velocity = points >> DOMAIN.staggered_grid()
    div_free_velocity, pressure, _, _, occupied = flip.make_incompressible(velocity + DT * GRAVITY, DOMAIN, accessible, points)
    if FLIP:
        points = flip.map_velocity_to_particles(points, div_free_velocity, occupied, previous_velocity_grid=velocity)
    else:
        points = flip.map_velocity_to_particles(points, div_free_velocity, occupied)
    points = advect.runge_kutta_4(points, div_free_velocity, DT, accessible=accessible, occupied=occupied)
    points = flip.respect_boundaries(points, DOMAIN, obs)
    return dict(points=points, accessible=accessible, pressure=pressure, scene=points & obs_points, obs_points=obs_points, obs=obs)


for i in range(SIM_NUM):
    valid = False
    while not valid:
        if SCENE is None:
            initial, obs, vel, params = random_scene(DOMAIN)
            if i == 0:
                SCENE_PARAMS = params
        else:
            initial, obs, vel = SCENE
        obs_points = DOMAIN.distribute_points(obs, color='#000000', center=OBS_CENTER, points_per_cell=1) * (0, 0)
        ACCESSIBLE = DOMAIN.accessible_mask(obs, type=StaggeredGrid)
        particles = DOMAIN.distribute_points(union(initial)) * vel
        scene = particles & obs_points
        valid = SCENE is not None or scene.elements.center.shape[0] < PARTICLE_LIMIT

    print(f"Generating scene {i} with {particles.elements.center.shape[0]} particles and {obs_points.elements.center.shape[0]} obstacles.")

    point_num = scene.elements.center.shape[0]
    trajectory = np.zeros((DURATION, point_num, 2))
    types = np.ones(point_num, dtype=np.int64)
    types[particles.elements.center.shape[0]:] = 0

    state = dict(points=particles, accessible=ACCESSIBLE, pressure=DOMAIN.grid(0), scene=particles & obs_points, obs_points=obs_points, obs=obs)

    if VIEW:
        app = App()
        app.set_state(state, step_function=step, show=['scene', 'pressure'])
        show(app)
        continue
    else:
        for frame in range(DURATION):
            state = step(**state)
            trajectory[frame] = state['scene'].elements.center.numpy()
            
    if SCALE is not None:
        upper = DOMAIN.bounds.upper[0]
        SCALE_factor = upper.numpy() / (SCALE[1] - SCALE[0])
        trajectory = (trajectory / SCALE_factor) + SCALE[0]

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

    if RECORD:
        EXAMPLES.append(save2sequence(trajectory, types))
    else:
        if SCALE is not None:
            size = SCALE
        else:
            size = SIZE
        if FLIP:
            rollout = dict(flip=trajectory, size=size, types=types)
        else:
            rollout = dict(pic=trajectory, size=size, types=types)
        with open(os.path.join(SAVE_PATH, f'sim_{i}.pkl'), 'wb') as f:
            pkl.dump(rollout, f)

if RECORD:
    radius_stats = list(map(lambda x: x / RADIUS_STATS_NUM, RADIUS_STATS))
    metadata = dict(bounds=[SCALE, SCALE],
                    sequence_length=DURATION,
                    dim=DIM,
                    dt=DT,
                    dataset_size=SIM_NUM,
                    vel_wf=VEL_STATS,
                    acc_wf=ACC_STATS,
                    particle_num_wf=PART_STATS,
                    examples=EXAMPLES,
                    scene=SCENE_PARAMS,
                    particle_num_range=(LOWEST, HIGHEST),
                    radius_stats=radius_stats)

    with open(os.path.join(SAVE_PATH, NAME) + '.pkl', 'wb') as f:
        pkl.dump(metadata, f)
