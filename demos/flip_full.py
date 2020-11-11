from phi.flow import *
import time
from phi.physics._effect import Gravity, gravity_tensor
from phi.field._point_cloud import _distribute_points
import matplotlib.pyplot as plt

# define world
size = (64, 64)
domain = Domain(size, boundaries=CLOSED, bounds=Box(0, (64, 64)))

# define initial liquid
initial_density = domain.grid().values
# initial_density.numpy()[28:36, 28:36] = 1
initial_density.native()[size[-1] * 2 // 8: size[-1] * 6 // 8, size[-2] * 6 // 8: size[-2] * 7 // 8 - 1] = 1
initial_density.native()[size[-1] * 0 // 8: size[-1] * 8 // 8, size[-2] * 0 // 8: size[-2] * 2 // 8] = 1
# initial_density.numpy()[28:36, 28:36] = 1
# initial_density.native()[15:45, 180:190] = 1
# initial_density.native()[:, 0:20] = 1
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
    v_field = velocity.at(domain.sgrid())

    plot_sgrid(v_field, 'v', iter)

    # get domain
    cmask = mpoints.at(domain.grid())
    smask = mpoints.at(domain.sgrid())
    accessible_mask = domain.grid(1, extrapolation=domain.boundaries.accessible_extrapolation)
    hard_bcs = field.stagger(accessible_mask, math.minimum, accessible_mask.extrapolation)

    # extp_mask = domain.grid(extrapolation=cmask.extrapolation)
    # extp_mask.values.native()[slice(1, -1), slice(1, -1)] = cmask.values.native()[slice(1, -1), slice(1, -1)]

    # apply forces
    force = dt * gravity_tensor(Gravity(), v_field.rank)
    v_force_field = (v_field + force)

    plot_sgrid(v_force_field, 'force', iter)

    # solve pressure

    v_force_field = field.extp_sgrid(v_force_field * smask, 2) * hard_bcs
    plot_sgrid(v_force_field, 'force_extp', iter)
    div = field.divergence(v_force_field) * cmask

    plot_cgrid(div, 'div', iter)
    plot_cgrid(cmask, 'active', iter)

    laplace = lambda pressure: field.divergence(field.gradient(pressure, type=StaggeredGrid) * domain.sgrid(1)) * cmask + (1 - cmask) * pressure
    converged, pressure, iterations = field.solve(laplace, div, pressure, solve_params=math.LinearSolve(None, 1e-3))

    plot_cgrid(pressure, 'pressure', iter)

    gradp = field.gradient(pressure, type=type(v_force_field))
    v_div_free_field = v_force_field - gradp

    plot_sgrid(v_div_free_field, 'div_free', iter)

    # update velocities
    v_change_field = v_div_free_field - v_field

    plot_sgrid(v_change_field, 'change', iter)

    v_change_field = field.extp_sgrid(v_change_field * smask, 2)
    v_change = v_change_field.sample_at(points.elements.center)
    velocity = velocity.values + v_change

    # advect
    v_div_free_field = field.extp_sgrid(v_div_free_field * smask, 2)
    v_div_free_field *= hard_bcs

    points = advect.advect(points, v_div_free_field, dt)
    mpoints = advect.advect(mpoints, v_div_free_field, dt)

    velocity = PointCloud(points.elements, values=velocity)
    return dict(points=points, velocity=velocity, v_field=v_field, v_force_field=v_force_field, v_change_field=v_change_field,
                v_div_free_field=v_div_free_field, density=points.at(domain.grid()), mpoints=mpoints, pressure=pressure,
                divergence=div, smask=smask, cmask=cmask, iter=0)


# for i in range(50):
#     state = step(dt=1, **state)


app = App()
app.set_state(state, step_function=step, dt=0.1, show=['density', 'v_field', 'v_force_field', 'v_change_field', 'v_div_free_field', 'pressure', 'divergence', 'cmask', 'smask'])
show(app, display=('density', 'v_field', 'v_force_field', 'v_change_field', 'v_div_free_field', 'pressure', 'divergence', 'cmask', 'smask'), port=8052)
