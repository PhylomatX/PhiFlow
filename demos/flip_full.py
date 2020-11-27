from phi.flow import *
from phi.physics._effect import Gravity, gravity_tensor
from phi.field._point_cloud import _distribute_points

# define world
# size = (30, 400)
x = 64
y = 64
domain = Domain(x=x, y=y, boundaries=CLOSED, bounds=Box[0:x, 0:y])
initial_density = domain.grid().values
inflow = 0

# block falls into pool
initial_density.native()[15:50, 45:55] = 1
initial_density.native()[:, :15] = 1

# large block falls to bottom
# initial_density.native()[1:size[-1]-1, size[1]-30:size[1]-1] = 1

# small block (or inflow) to bottom
# initial_density.native()[20:50, 55:62] = 8

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
# initial_density.native()[30:36, 0:60] = 1

# multiple small blocks in different heights
# initial_density.native()[1:10, 20:30] = 1
# initial_density.native()[1:10, 40:50] = 1
# initial_density.native()[30:40, 20:30] = 1
# initial_density.native()[50:60, 20:30] = 1
# initial_density.native()[50:60, 50:60] = 1

# TODO: block which includes border region keeps sticking
# initial_density.native()[0:size[-1], size[1]-30:size[1]] = 1

# block falls into pool from large height => change domain
# initial_density.native()[10:20, 370:390] = 1
# initial_density.native()[size[-1] * 0 // 8: size[-1] * 8 // 8, size[-2] * 0 // 8: size[-2] * 2 // 8] = 1


# generate points
initial_points = _distribute_points(initial_density, 8)
# bounds = Box(-5, (70, 70))
points = PointCloud(Sphere(initial_points, 0), add_overlapping=True, bounds=domain.bounds)
initial_velocity = math.tensor(np.zeros(initial_points.shape), names=['points', 'vector'])
velocity = PointCloud(points.elements, values=initial_velocity)

# initialize masks
density = points.at(domain.grid())
sdensity = points.at(domain.sgrid())
ones = domain.grid(1, extrapolation=density.extrapolation)
zeros = domain.grid(0, extrapolation=density.extrapolation)
cmask = field.where(density, ones, zeros)
sones = domain.sgrid(1, extrapolation=density.extrapolation)
szeros = domain.sgrid(0, extrapolation=density.extrapolation)
smask = field.where(sdensity, sones, szeros)

# define obstacles
# obstacles = [Obstacle(Box[30:50, 30:40])]
# obstacles = [Obstacle(Box[30:50, 30:40].rotated(20))]
obstacles = ()

# define initial state
state = dict(points=points, velocity=velocity, density=density, v_force_field=domain.sgrid(0), v_change_field=domain.sgrid(0),
             v_div_free_field=domain.sgrid(0), v_field=velocity.at(domain.sgrid()), pressure=domain.grid(0),
             divergence=domain.grid(0), smask=smask, cmask=cmask, accessible=domain.grid(0), iter=0)


def step(points, velocity, v_field, pressure, dt, iter, density, cmask, **kwargs):
    # get domain
    cmask = field.where(density, ones, zeros)
    smask = field.where(points.at(domain.sgrid()), sones, szeros)
    accessible = domain.grid(1 - HardGeometryMask(union([obstacle.geometry for obstacle in obstacles])))
    accessible_mask = domain.grid(accessible, extrapolation=domain.boundaries.accessible_extrapolation)
    hard_bcs = field.stagger(accessible_mask, math.minimum, accessible_mask.extrapolation)

    # apply forces
    force = dt * gravity_tensor(Gravity(), v_field.shape.spatial.rank)
    v_force_field = (v_field + force)

    # solve pressure
    # this solves the distortion issue of a falling block by extrapolating the velocity field so that divergence and pressure lay outside of active mask
    v_force_field = field.extp_sgrid(v_force_field * smask, 2) * hard_bcs
    div = field.divergence(v_force_field) * cmask
    # TODO: Understand why -4 in pressure equation is necessary / Understand why multiplying div with cmask helps with +1 case
    laplace = lambda p: field.where(cmask, field.divergence(field.gradient(p, type=StaggeredGrid) * hard_bcs), -4 * p)
    converged, pressure, iterations = field.solve(laplace, div, pressure, solve_params=math.LinearSolve(None, 1e-3))
    gradp = field.gradient(pressure, type=type(v_force_field))
    v_div_free_field = v_force_field - gradp

    # update velocities
    v_change_field = v_div_free_field - v_field
    v_change_field = field.extp_sgrid(v_change_field * smask, 2)
    v_change = v_change_field.sample_at(points.elements.center)
    velocity = velocity.values + v_change

    # advect
    v_div_free_field *= hard_bcs * smask
    points = advect.advect(points, v_div_free_field, dt)
    velocity = PointCloud(points.elements, values=velocity)

    # add possible inflow
    if iter < inflow:
        new_points = math.tensor(math.concat([points.points, initial_points], dim='points'), names=['points', 'vector'])
        points = PointCloud(Sphere(new_points, 0), add_overlapping=True, bounds=points.bounds)
        new_velocity = math.tensor(math.concat([velocity.values, initial_velocity], dim='points'), names=['points', 'vector'])
        velocity = PointCloud(points.elements, values=new_velocity)

    # check if particles are inside obstacles
    for obstacle in obstacles:
        shift = obstacle.geometry.shift_outward(points.elements.center)
        points = PointCloud(points.elements.shifted(shift), add_overlapping=True, bounds=points.bounds)
    velocity = PointCloud(points.elements, values=velocity.values)

    # get new velocity field
    v_field = velocity.at(domain.sgrid())

    return dict(points=points, velocity=velocity, v_field=v_field, v_force_field=v_force_field, v_change_field=v_change_field,
                v_div_free_field=v_div_free_field, density=points.at(domain.grid()), pressure=pressure,
                divergence=div, smask=smask, cmask=cmask, accessible=accessible * 2 + cmask, iter=iter+1)


# for i in range(500):
#     state = step(dt=0.1, **state)

app = App()
app.set_state(state, step_function=step, dt=0.1, show=['density', 'accessible', 'points', 'v_field', 'v_force_field', 'v_change_field', 'v_div_free_field', 'pressure', 'divergence', 'cmask', 'smask'])
show(app, display=('density', 'accessible', 'points', 'v_field', 'v_force_field', 'v_change_field', 'v_div_free_field', 'pressure', 'divergence', 'cmask', 'smask'), port=8052)
