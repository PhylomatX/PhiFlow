""" FLIP simulation for liquids

A liquid block collides with a rotated obstacle and falls into a liquid pool.
"""

import random
from phi.flow import *
from phi.geom._union import Union
import pickle as pkl

SIZE = 64
DOMAIN = Domain(x=SIZE, y=SIZE, boundaries=CLOSED, bounds=Box[0:SIZE, 0:SIZE])
GRAVITY = math.tensor([0, -9.81])
DT = 0.05
OBSTACLE = Union([Box[:, 0:1], Box[0:1, :], Box[63:, :]])
OBSTACLE = ()
_OBSTACLE_POINTS = DOMAIN.distribute_points(OBSTACLE, color='#000000', center=True) * (0, 0)
ACCESSIBLE = DOMAIN.accessible_mask(OBSTACLE, type=StaggeredGrid)
particles = DOMAIN.distribute_points(union(Box[26:32, 40:45])) * (0, 0)

random.seed(1)
np.random.seed(1)

state = dict(parts=particles, pressure=DOMAIN.grid(0), scene=particles & _OBSTACLE_POINTS)


def step(parts, **kwargs):
    velocity = parts >> DOMAIN.staggered_grid()
    div_free_velocity, pressure, _, _, occupied = flip.make_incompressible(velocity + DT * GRAVITY, DOMAIN, ACCESSIBLE, parts)
    parts = flip.map_velocity_to_particles(parts, div_free_velocity, occupied, previous_velocity_grid=velocity)
    parts = advect.runge_kutta_4(parts, div_free_velocity, DT, accessible=ACCESSIBLE, occupied=occupied)
    parts = flip.respect_boundaries(parts, DOMAIN, [])
    return dict(parts=parts, pressure=pressure, scene=parts & _OBSTACLE_POINTS)



# app = App()
# app.set_state(state, step_function=step, show=['scene', 'pressure'])
# show(app)

duration = 500
trajectory = np.zeros((duration, particles.elements.center.shape[0] + _OBSTACLE_POINTS.elements.center.shape[0], 2))
# trajectory = np.zeros((duration, particles.elements.center.shape[0], 2))
types = np.ones(len(trajectory[0]))
types[particles.elements.center.shape[0]:] = 0

for i in range(duration):
    state = step(**state)
    trajectory[i] = (state['parts'] & _OBSTACLE_POINTS).elements.center.numpy()
    # trajectory[i] = state['parts'].elements.center.numpy()

rollout = dict(flip=trajectory, size=SIZE, types=types)
with open('/home/john/Projekte/BA/Data/obstacle_boundaries_test/05_noobs_norespect_activeextp.pkl', 'wb') as f:
    pkl.dump(rollout, f)
