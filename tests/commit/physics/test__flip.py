from unittest import TestCase

from phi.flow import *
from phi.geom._union import Union


def step(particles, domain, dt, accessible):
    velocity = particles >> domain.staggered_grid()
    div_free_velocity, pressure, _, _, occupied = \
        flip.make_incompressible(velocity + dt * math.tensor([0, -9.81]), domain, accessible, particles)
    particles = flip.map_velocity_to_particles(particles, div_free_velocity, occupied, previous_velocity_grid=velocity)
    particles = advect.runge_kutta_4(particles, div_free_velocity, dt, accessible=accessible, occupied=occupied)
    particles = flip.respect_boundaries(particles, domain, [])
    return dict(particles=particles, domain=domain, dt=dt, accessible=accessible)


class FluidTest(TestCase):

    def test_pool(self):
        SIZE = 32
        DOMAIN = Domain(x=SIZE, y=SIZE, boundaries=CLOSED, bounds=Box[0:SIZE, 0:SIZE])
        DT = 0.05
        ACCESSIBLE = DOMAIN.accessible_mask([], type=StaggeredGrid)
        PARTICLES = DOMAIN.distribute_points(union(Box[:, :10])) * (0, 0)

        state = dict(particles=PARTICLES, domain=DOMAIN, dt=DT, accessible=ACCESSIBLE)
        for i in range(100):
            state = step(**state)

        occupied_start = PARTICLES.with_(values=1) >> DOMAIN.grid()
        occupied_end = state['particles'].with_(values=1) >> DOMAIN.grid()
        math.assert_close(occupied_start.values, occupied_end.values)
        math.assert_close(PARTICLES.elements.center, state['particles'].elements.center, abs_tolerance=1e-5)

    def test_falling_shape(self):
        DOMAIN = Domain(x=32, y=128, boundaries=CLOSED, bounds=Box[0:32, 0:128])
        DT = 0.05
        ACCESSIBLE = DOMAIN.accessible_mask([], type=StaggeredGrid)
        PARTICLES = DOMAIN.distribute_points(union(Box[12:20, 110:120]), center=True) * (0, 0)
        extent = math.max(PARTICLES.elements.center, dim='points') - math.min(PARTICLES.elements.center, dim='points')
        state = dict(particles=PARTICLES, domain=DOMAIN, dt=DT, accessible=ACCESSIBLE)
        for i in range(90):
            state = step(**state)
            curr_extent = math.max(state['particles'].elements.center, dim='points') - \
                          math.min(state['particles'].elements.center, dim='points')
            math.assert_close(curr_extent, extent)

    def test_symmetry(self):
        SIZE = 64
        MID = SIZE / 2
        DOMAIN = Domain(x=SIZE, y=SIZE, boundaries=CLOSED, bounds=Box[0:SIZE, 0:SIZE])
        DT = 0.05
        OBSTACLE = Union([Box[20:30, 10:12].rotated(math.tensor(20)), Box[34:44, 10:12].rotated(math.tensor(-20))])
        ACCESSIBLE = DOMAIN.accessible_mask(OBSTACLE, type=StaggeredGrid)
        x_low = 26
        x_high = 38
        y_low = 40
        y_high = 50
        PARTICLES = DOMAIN.distribute_points(union(Box[x_low:x_high, y_low:y_high]), center=True) * (0, 0)

        x_num = int((x_high - x_low) / 2)
        y_num = y_high - y_low
        particles_per_cell = 8
        total = x_num * y_num

        state = dict(particles=PARTICLES, domain=DOMAIN, dt=DT, accessible=ACCESSIBLE)
        for i in range(100):
            state = step(**state)
            particles = state['particles'].elements.center.numpy()
            left = particles[particles[:, 0] < MID]
            right = particles[particles[:, 0] > MID]
            assert len(left) == len(right)
            mirrored = right.copy()
            mirrored[:, 0] = 2 * MID - right[:, 0]
            smirrored = np.zeros_like(mirrored)
            for p in range(particles_per_cell):
                for b in range(x_num):
                    smirrored[p * total + b * y_num:p * total + (b + 1) * y_num] = \
                        mirrored[(p + 1) * total - (b + 1) * y_num:(p + 1) * total - b * y_num]
            mse = np.square(smirrored - left).mean()
            if i < 45:
                assert mse == 0  # block was falling until this frame
            elif i < 80:
                assert mse <= 1e-7  # error increases exponentially after block and obstacles collide
            else:
                assert mse <= 1e-3
