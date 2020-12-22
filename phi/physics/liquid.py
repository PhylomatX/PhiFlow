from phi import math, field
from phi.field import StaggeredGrid, HardGeometryMask, PointCloud
from phi.geom import union, Sphere
from ._effect import Gravity, gravity_tensor


def apply_gravity(dt, v_field):
    force = dt * gravity_tensor(Gravity(), v_field.shape.spatial.rank)
    return v_field + force


def get_bcs(domain, obstacles):
    accessible = domain.grid(1 - HardGeometryMask(union([obstacle.geometry for obstacle in obstacles])))
    accessible_mask = domain.grid(accessible, extrapolation=domain.boundaries.accessible_extrapolation)
    return field.stagger(accessible_mask, math.minimum, accessible_mask.extrapolation)


def make_incompressible(v_field, bcs, cmask, smask, pressure):
    v_force_field = field.extp_sgrid(v_field * smask, 1) * bcs  # conserves falling shapes
    div = field.divergence(v_force_field) * cmask  # cmask prevents falling shape from collapsing

    def laplace(p):
        # TODO: prefactor of pressure should not have any effect, but it has
        return field.where(cmask, field.divergence(field.gradient(p, type=StaggeredGrid) * bcs), 1e6 * p)

    converged, pressure, iterations = field.solve(laplace, div, pressure, solve_params=math.LinearSolve(None, 1e-5))
    gradp = field.gradient(pressure, type=type(v_force_field))
    return v_force_field - gradp


def map2particle(v_particle, current_v_field, smask, orig_v_field=None):
    if orig_v_field is not None:
        # FLIP
        v_change_field = current_v_field - orig_v_field
        v_change_field = field.extp_sgrid(v_change_field * smask, 1)  # conserves falling shapes (no hard_bcs here!)
        v_change = v_change_field.sample_at(v_particle.elements.center)
        return v_particle.values + v_change
    else:
        # PIC
        v_div_free_field = field.extp_sgrid(current_v_field * smask, 1)
        return v_div_free_field.sample_at(v_particle.elements.center)


def add_inflow(particles, inflow_points, inflow_values):
    new_points = math.tensor(math.concat([particles.points, inflow_points], dim='points'), names=['points', 'vector'])
    new_values = math.tensor(math.concat([particles.values, inflow_values], dim='points'), names=['points', 'vector'])
    return PointCloud(Sphere(new_points, 0), add_overlapping=particles.add_overlapping, bounds=particles.bounds,
                      values=new_values)


def respect_boundaries(domain, obstacles, particles):
    points = particles.elements
    for obstacle in obstacles:
        shift = obstacle.geometry.shift_points(points.center, shift_amount=0.5)
        points = particles.elements.shifted(shift)
    shift = (~domain.bounds).shift_points(points.center)
    return PointCloud(particles.elements.shifted(shift), add_overlapping=particles.add_overlapping,
                      bounds=particles.bounds, values=particles.values)
