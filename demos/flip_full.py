from phi.flow import *
import time
from phi.physics._effect import Gravity, gravity_tensor
from phi.field._point_cloud import _distribute_points
import matplotlib.pyplot as plt

# define world
# size = (100, 400)
size = (64, 64)
domain = Domain(size, boundaries=CLOSED, bounds=Box(0, size))

# define initial liquid
initial_density = domain.grid().values

# block falls into pool
initial_density.native()[size[-1] * 2 // 8: size[-1] * 6 // 8, size[-2] * 6 // 8: size[-2] * 7 // 8 - 1] = 1
initial_density.native()[size[-1] * 0 // 8: size[-1] * 8 // 8, size[-2] * 0 // 8: size[-2] * 2 // 8] = 1

# large block falls to bottom
# initial_density.native()[1:size[-1]-1, size[1]-30:size[1]-1] = 1

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
active_mask = PointCloud(points.elements)


# define initial state
state = dict(points=points, velocity=velocity, density=points.at(domain.grid()), v_force_field=domain.sgrid(0), v_change_field=domain.sgrid(0),
             v_div_free_field=domain.sgrid(0), v_field=velocity.at(domain.sgrid()), mpoints=active_mask, pressure=domain.grid(0),
             divergence=domain.grid(0), smask=domain.sgrid(0), cmask=domain.grid(0), iter=0)


def plot_sgrid(sgrid: StaggeredGrid, name: str, iter: int):
    if iter > 0:
        print(name)
        plt.imshow(np.concatenate([sgrid.staggered_tensor().native()[..., 0], sgrid.staggered_tensor().native()[..., 1]], axis=1))
        plt.show()


def plot_cgrid(cgrid: CenteredGrid, name: str, iter: int):
    if iter > 0:
        print(name)
        plt.imshow(cgrid.values.numpy())
        plt.show()


def step(points, velocity, v_field, mpoints, pressure, dt, iter, **kwargs):
    # sample particle velocities on grid
    v_field = velocity.at(domain.sgrid())

    # get domain
    cmask = mpoints.at(domain.grid())
    smask = mpoints.at(domain.sgrid())
    accessible_mask = domain.grid(1, extrapolation=domain.boundaries.accessible_extrapolation)
    hard_bcs = field.stagger(accessible_mask, math.minimum, accessible_mask.extrapolation)

    # apply forces
    force = dt * gravity_tensor(Gravity(), v_field.shape.spatial.rank)
    v_force_field = (v_field + force)

    # solve pressure
    v_force_field = field.extp_sgrid(v_force_field * smask, 2) * hard_bcs
    plot_sgrid(v_force_field, 'force_extp', iter)
    div = field.divergence(v_force_field) * cmask
    # TODO: Understand why -4 in pressure equation is necessary / Understand why multiplying div with cmask helps with +1 case
    # laplace = lambda p: field.where(cmask, field.divergence(field.gradient(p, type=StaggeredGrid) * domain.sgrid(1)), -4 * (1 - cmask) * p)
    laplace = lambda p: field.divergence(field.gradient(p, type=StaggeredGrid) * hard_bcs) * cmask - 4 * (1 - cmask) * p
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
    mpoints = advect.advect(mpoints, v_div_free_field, dt, simple=True)
    velocity = PointCloud(points.elements, values=velocity)

    return dict(points=points, velocity=velocity, v_field=v_field, v_force_field=v_force_field, v_change_field=v_change_field,
                v_div_free_field=v_div_free_field, density=points.at(domain.grid()), mpoints=mpoints, pressure=pressure,
                divergence=div, smask=smask, cmask=cmask, iter=0)


# for i in range(50):
#     state = step(dt=1, **state)


app = App()
app.set_state(state, step_function=step, dt=0.1, show=['density', 'v_field', 'v_force_field', 'v_change_field', 'v_div_free_field', 'pressure', 'divergence', 'cmask', 'smask'])
show(app, display=('density', 'v_field', 'v_force_field', 'v_change_field', 'v_div_free_field', 'pressure', 'divergence', 'cmask', 'smask'), port=8052)
