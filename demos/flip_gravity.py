from phi.flow import *
from phi.physics._effect import Gravity, gravity_tensor
from phi.field._point_cloud import _distribute_points

# define world
size = (64, 64)
domain = Domain(size, boundaries=CLOSED, bounds=Box(0, (64, 64)))

# define initial liquid
initial_density = domain.grid().values
initial_density.numpy()[28:36, 28:36] = 1
initial_points = _distribute_points(initial_density, 8)
points = PointCloud(Sphere(initial_points, 0), add_overlapping=True)
initial_velocity = math.tensor(np.zeros(initial_points.shape), names=['points', 'vector'])
velocity = PointCloud(points.elements, values=initial_velocity)
active_mask = PointCloud(points.elements)

# define initial state
state = dict(points=points, velocity=velocity, density=points.at(domain.grid()),
             velocity_field=velocity.at(domain.sgrid()), mask=active_mask.at(domain.grid()))
gravity = Gravity()


def step(points, velocity, velocity_field, mask, density, dt):
    # add forces
    force = dt * gravity_tensor(gravity, velocity_field.rank)
    velocity_field += force

    points = advect.advect(points, velocity_field, dt)

    velocity = PointCloud(points.elements, values=velocity_field.sample_at(points.elements.center))
    velocity_field = velocity.at(domain.sgrid())
    return dict(points=points, velocity=velocity, velocity_field=velocity_field, density=points.at(domain.grid()),
                mask=points.at(domain.grid()))


# for i in range(5):
#     state = step(dt=1, **state)


app = App()
app.set_state(state, step_function=step, dt=0.1, show=['density', 'velocity_field'])
show(app, display=('density', 'velocity_field'))
