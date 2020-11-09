from phi.flow import *
from phi.physics._effect import Gravity, gravity_tensor
from phi.field._point_cloud import _distribute_points

# define world
size = (64, 64)
domain = Domain(size, boundaries=CLOSED, bounds=Box(0, (64, 64)))

# define initial liquid
initial_density = domain.grid().values
# initial_density.numpy()[28:36, 28:36] = 1
initial_density.native()[size[-1] * 2 // 8: size[-1] * 6 // 8, size[-2] * 6 // 8: size[-2] * 8 // 8 - 1] = 1
initial_density.native()[size[-1] * 0 // 8: size[-1] * 8 // 8, size[-2] * 0 // 8: size[-2] * 2 // 8] = 1
initial_points = _distribute_points(initial_density, 8)
points = PointCloud(Sphere(initial_points, 0), add_overlapping=True)
initial_velocity = math.tensor(np.zeros(initial_points.shape), names=['points', 'vector'])
velocity = PointCloud(points.elements, values=initial_velocity)
active_mask = PointCloud(points.elements)

# define initial state
state = dict(points=points, velocity=velocity, density=points.at(domain.grid()), v_force_field=domain.sgrid(0), v_change_field=domain.sgrid(0),
             v_div_free_field=domain.sgrid(0), v_field=velocity.at(domain.sgrid()), mpoints=active_mask, pressure=domain.grid(0),
             divergence=domain.grid(0), smask=domain.sgrid(0), cmask=domain.grid(0))


def step(points, velocity, v_field, mpoints, pressure, dt, **kwargs):
    v_field = velocity.at(domain.sgrid())

    # get domain
    cmask = mpoints.at(domain.grid())
    smask = mpoints.at(domain.sgrid())
    accessible_mask = domain.grid(1, extrapolation=domain.boundaries.accessible_extrapolation)
    hard_bcs = field.stagger(accessible_mask, math.minimum, accessible_mask.extrapolation)

    # apply forces
    force = dt * gravity_tensor(Gravity(), v_field.rank)
    v_force_field = (v_field + force)

    # solve pressure
    v_force_field *= hard_bcs
    div = field.divergence(v_force_field)
    laplace = lambda pressure: field.divergence(field.gradient(pressure, type=StaggeredGrid) * domain.sgrid(1)) * cmask - 4 * (1 - cmask) * pressure
    converged, pressure, iterations = field.solve(laplace, div, pressure, solve_params=math.LinearSolve(None, 1e-3))
    gradp = field.gradient(pressure, type=type(v_force_field))
    v_div_free_field = v_force_field - gradp

    # update velocities
    v_change_field = v_div_free_field - v_field
    v_change_field = field.extp_sgrid(v_change_field * smask, 10)
    v_change = v_change_field.sample_at(points.elements.center)
    velocity = velocity.values + v_change

    # advect
    v_div_free_field = field.extp_sgrid(v_div_free_field * smask, 10)
    v_div_free_field *= hard_bcs

    import ipdb
    ipdb.set_trace()

    points = advect.advect(points, v_div_free_field, dt)
    mpoints = advect.advect(mpoints, v_div_free_field, dt)

    velocity = PointCloud(points.elements, values=velocity)
    return dict(points=points, velocity=velocity, v_field=v_field, v_force_field=v_force_field, v_change_field=v_change_field,
                v_div_free_field=v_div_free_field, density=points.at(domain.grid()), mpoints=mpoints, pressure=pressure,
                divergence=div, smask=smask, cmask=cmask)


for i in range(50):
    state = step(dt=1, **state)


# app = App()
# app.set_state(state, step_function=step, dt=0.1, show=['density', 'v_field', 'v_force_field', 'v_change_field', 'v_div_free_field', 'pressure', 'divergence', 'cmask', 'smask'])
# show(app, display=('density', 'v_field', 'v_force_field', 'v_change_field', 'v_div_free_field', 'pressure', 'divergence', 'cmask', 'smask'), port=8052)
