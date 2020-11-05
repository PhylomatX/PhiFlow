from phi.flow import *
from phi.field._point_cloud import _distribute_points

# define world
size = (64, 64)
domain = Domain(size, boundaries=CLOSED, bounds=Box(0, (64, 64)))
buoyancy_factor = (0, 0.1)

# define initial liquid
inflow = domain.grid(Sphere(center=(32, 10), radius=3))
initial_points = _distribute_points(inflow.values, 2)
points = PointCloud(Sphere(initial_points, 0), add_overlapping=True)
initial_velocity = math.tensor(np.ones(initial_points.shape), names=['points', 'vector']) * buoyancy_factor
velocity = PointCloud(points.elements, values=initial_velocity)

# define initial state
state = dict(points=points, velocity=velocity, density=points.at(domain.grid()),
             velocity_field=velocity.at(domain.sgrid()), velocity_change_field=domain.sgrid(0),
             div_free_velocity_field=domain.sgrid(0), pressure=domain.grid(0), divergence=domain.grid(0))


def step(points, velocity, velocity_field, dt, **kwargs):
    div_free_velocity_field, pressure, _, divergence = fluid.make_incompressible(velocity_field, domain)
    velocity_change_field = div_free_velocity_field - velocity_field
    velocity_change = velocity_change_field.sample_at(points.elements.center)
    points = advect.advect(points, div_free_velocity_field, dt)
    velocity = velocity.values + velocity_change

    # inflow
    new_points = math.tensor(math.concat([points.points, initial_points], dim='points'), names=['points', 'vector'])
    points = PointCloud(Sphere(new_points, 0), add_overlapping=True)
    new_velocity = math.tensor(math.concat([velocity, initial_velocity], dim='points'), names=['points', 'vector'])
    velocity = PointCloud(points.elements, values=new_velocity)

    velocity_field = velocity.at(domain.sgrid())
    return dict(points=points, velocity=velocity, velocity_field=velocity_field, density=points.at(domain.grid()),
                velocity_change_field=velocity_change_field, div_free_velocity_field=div_free_velocity_field,
                pressure=pressure, divergence=divergence)


# for i in range(5):
#     state = step(**state, dt=1)

app = App()
app.set_state(state, step_function=step, dt=1, show=['density', 'velocity_field', 'velocity_change_field',
                                                     'div_free_velocity_field', 'pressure', 'divergence'])
show(app, display=('density', 'velocity_field', 'velocity_change_field', 'div_free_velocity_field', 'pressure',
                   'divergence'))