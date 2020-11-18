from phi.flow import *
import time
from phi.physics._effect import Gravity, gravity_tensor
from phi.field._point_cloud import _distribute_points
import matplotlib.pyplot as plt

# define world
# size = (100, 400)
size = (64, 64)
domain = Domain(size, boundaries=CLOSED, bounds=Box(0, size))
initial_density = domain.grid().values

# block falls into pool
# initial_density.native()[size[-1] * 2 // 8: size[-1] * 6 // 8, size[-2] * 6 // 8: size[-2] * 7 // 8 - 1] = 1
# initial_density.native()[size[-1] * 0 // 8: size[-1] * 8 // 8, size[-2] * 0 // 8: size[-2] * 2 // 8] = 1

# large block falls to bottom
# initial_density.native()[1:size[-1]-1, size[1]-30:size[1]-1] = 1

# TODO: Inflow to bottom violates borders
# initial_density.native()[28:32, 50:55] = 1
# initial_density.native()[size[-1] * 0 // 8: size[-1] * 8 // 8, size[-2] * 0 // 8: size[-2] * 2 // 8] = 1

# multiple small blocks falling into pool
# initial_density.native()[1:10, 50:60] = 1
# initial_density.native()[30:40, 50:60] = 1
# initial_density.native()[50:60, 50:60] = 1
# initial_density.native()[size[-1] * 0 // 8: size[-1] * 8 // 8, size[-2] * 0 // 8: size[-2] * 2 // 8] = 1

# multiple small blocks falling to bottom
# initial_density.native()[1:10, 50:60] = 1
# initial_density.native()[30:40, 50:60] = 1
# initial_density.native()[50:60, 50:60] = 1

# tower on the left
# initial_density.native()[0:10, 0:60] = 1

# towers on both sides
# initial_density.native()[0:10, 0:60] = 1
# initial_density.native()[54:64, 0:60] = 1

# tower in the middle
# initial_density.native()[28:36, 0:60] = 1

# multiple small blocks in different heights
# initial_density.native()[1:10, 20:30] = 1
# initial_density.native()[1:10, 40:50] = 1
# initial_density.native()[30:40, 20:30] = 1
# initial_density.native()[50:60, 20:30] = 1
# initial_density.native()[50:60, 50:60] = 1

# TODO: block which includes border region keeps sticking
# initial_density.native()[0:size[-1], size[1]-30:size[1]] = 1

# block falls into pool from large height => change domain
# initial_density.native()[40:60, 370:390] = 1
# initial_density.native()[size[-1] * 0 // 8: size[-1] * 8 // 8, size[-2] * 0 // 8: size[-2] * 2 // 8] = 1


initial_points = _distribute_points(initial_density, 8)
points = PointCloud(Sphere(initial_points, 0), add_overlapping=True)
initial_velocity = math.tensor(np.zeros(initial_points.shape), names=['points', 'vector'])
velocity = PointCloud(points.elements, values=initial_velocity)
density = points.at(domain.grid())

inflow = 0

ones = domain.grid(1, extrapolation=density.extrapolation)
zeros = domain.grid(0, extrapolation=density.extrapolation)
cmask = field.where(density, ones, zeros)


# define initial state
state = dict(points=points, velocity=velocity, density=density, v_force_field=domain.sgrid(0), v_change_field=domain.sgrid(0),
             v_div_free_field=domain.sgrid(0), v_field=velocity.at(domain.sgrid()), pressure=domain.grid(0),
             divergence=domain.grid(0), smask=field.stagger(cmask, math.minimum, cmask.extrapolation), cmask=cmask, iter=0)


def step(points, velocity, v_field, pressure, dt, iter, density, cmask, **kwargs):
    # get domain
    cmask = field.where(density, ones, zeros)
    smask = field.stagger(cmask, math.minimum, cmask.extrapolation)
    accessible_mask = domain.grid(1, extrapolation=domain.boundaries.accessible_extrapolation)
    hard_bcs = field.stagger(accessible_mask, math.minimum, accessible_mask.extrapolation)

    # apply forces
    force = dt * gravity_tensor(Gravity(), v_field.shape.spatial.rank)
    v_force_field = (v_field + force)

    # solve pressure
    v_force_field = field.extp_sgrid(v_force_field * smask, 2) * hard_bcs
    div = field.divergence(v_force_field) * cmask
    # TODO: Understand why -4 in pressure equation is necessary / Understand why multiplying div with cmask helps with +1 case
    laplace = lambda p: field.where(cmask, field.divergence(field.gradient(p, type=StaggeredGrid) * domain.sgrid(1)), -4 * (1 - cmask) * p)
    converged, pressure, iterations = field.solve(laplace, div, pressure, solve_params=math.LinearSolve(None, 1e-3))
    gradp = field.gradient(pressure, type=type(v_force_field))
    v_div_free_field = v_force_field - gradp

    # update velocities
    v_change_field = v_div_free_field - v_field
    v_change_field = field.extp_sgrid(v_change_field * smask, 2)
    v_change = v_change_field.sample_at(points.elements.center)
    velocity = velocity.values + v_change

    # advect
    v_div_free_field = field.extp_sgrid(v_div_free_field * smask, 2)
    v_div_free_field *= hard_bcs
    points = advect.advect(points, v_div_free_field, dt, simple=True)
    velocity = PointCloud(points.elements, values=velocity)

    # add inflow
    if iter < inflow:
        new_points = math.tensor(math.concat([points.points, initial_points], dim='points'), names=['points', 'vector'])
        points = PointCloud(Sphere(new_points, 0), add_overlapping=True)
        new_velocity = math.tensor(math.concat([velocity.values, initial_velocity], dim='points'), names=['points', 'vector'])
        velocity = PointCloud(points.elements, values=new_velocity)

    # get new velocity field
    v_field = velocity.at(domain.sgrid())

    return dict(points=points, velocity=velocity, v_field=v_field, v_force_field=v_force_field, v_change_field=v_change_field,
                v_div_free_field=v_div_free_field, density=points.at(domain.grid()), pressure=pressure,
                divergence=div, smask=smask, cmask=cmask, iter=iter+1)


# for i in range(50):
#     state = step(dt=1, **state)


app = App()
app.set_state(state, step_function=step, dt=0.1, show=['density', 'v_field', 'v_force_field', 'v_change_field', 'v_div_free_field', 'pressure', 'divergence', 'cmask', 'smask'])
show(app, display=('density', 'v_field', 'v_force_field', 'v_change_field', 'v_div_free_field', 'pressure', 'divergence', 'cmask', 'smask'), port=8052)
