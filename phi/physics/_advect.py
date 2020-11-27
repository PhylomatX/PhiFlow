"""
Container for different advection schemes for grids and particles.

Examples:

* semi_lagrangian (grid)
* mac_cormack (grid)
* runge_kutta_4 (particle)
* euler (particle)
"""

from phi import math
from phi.field import SampledField, ConstantField, StaggeredGrid, CenteredGrid, Grid, Field, PointCloud


def advect(field: Field, velocity: Field, dt, simple: bool = False):
    """
    Advect `field` along the `velocity` vectors using the default advection method.
    :param field: any built-in Field
    :param velocity: any Field
    :param dt: time increment
    :param simple: flag for using simple advection schemes
    :return: Advected field of same type as `field`
    """
    if isinstance(field, PointCloud):
        if isinstance(velocity, PointCloud) and velocity.elements == field.elements:
            return points(field, velocity, dt)
        if simple:
            return euler(field, velocity, dt=dt)
        return runge_kutta_4(field, velocity, dt=dt)
    if isinstance(field, ConstantField):
        return field
    if isinstance(field, (CenteredGrid, StaggeredGrid)):
        return semi_lagrangian(field, velocity, dt=dt)
    raise NotImplementedError(field)


def semi_lagrangian(field: Grid, velocity: Field, dt) -> Grid:
    """
    Semi-Lagrangian advection with simple backward lookup.

    :param field: Field to be advected
    :param velocity: vector field, need not be compatible with with `field`.
    :param dt: time increment
    :return: Field compatible with input field
    """
    v = velocity.sample_in(field.elements)
    x = field.points - v * dt
    interpolated = field.sample_at(x, reduce_channels=x.shape.non_channel.without(field.shape).names)
    return field._with(interpolated)


def mac_cormack(field: CenteredGrid, velocity: Field, dt, correction_strength=1.0) -> CenteredGrid:
    """
    MacCormack advection uses a forward and backward lookup to determine the first-order error of semi-Lagrangian advection.
    It then uses that error estimate to correct the field values.
    To avoid overshoots, the resulting value is bounded by the neighbouring grid cells of the backward lookup.

    :param correction_strength: the estimated error is multiplied by this factor before being applied. The case correction_strength=0 equals semi-lagrangian advection. Set lower than 1.0 to avoid oscillations.
    :param field: Field to be advected
    :param velocity: vector field, need not be compatible with `field`.
    :param dt: time increment
    :return: Field compatible with input field
    """
    x0 = field.points
    v = velocity.sample_in(field.elements)
    x_bwd = x0 - v * dt
    x_fwd = x0 + v * dt
    field_semi_la = field._with(field.sample_at(x_bwd.values, reduce_channels='not yet implemented'))  # semi-Lagrangian advection
    field_inv_semi_la = field._with(field_semi_la.sample_at(x_fwd.values, reduce_channels='not yet implemented'))  # inverse semi-Lagrangian advection
    new_field = field_semi_la + correction_strength * 0.5 * (field - field_inv_semi_la)
    field_clamped = math.clip(new_field, *field.general_sample_at(x_bwd.values, 'minmax'))  # Address overshoots
    return field_clamped


def euler(field: PointCloud, velocity: Field, dt):
    """
    Advection via a single Euler step.
    :param field: PointCloud with any number of components
    :param velocity: velocity: Vector field
    :param dt: time increment
    :return: SampledField with same data as `field` but advected points
    """
    assert isinstance(field, PointCloud), 'Euler advection works for PointCloud only.'
    assert isinstance(velocity, Field), 'Velocity for Euler advection must have Field type.'
    points = field.elements
    vel = velocity.sample_in(points)
    new_points = points.shifted(dt * vel)
    return PointCloud(new_points, field.values, field.extrapolation, add_overlapping=field._add_overlapping)


def runge_kutta_4(field: PointCloud, velocity: Field, dt):
    """
    Lagrangian advection of particles.
    :param field: SampledField with any number of components
    :type field: SampledField
    :param velocity: Vector field
    :type velocity: Field
    :param dt: time increment
    :return: SampledField with same data as `field` but advected points
    """
    assert isinstance(field, SampledField)
    assert isinstance(velocity, Field)
    points = field.elements
    # --- Sample velocity at intermediate points ---
    vel_k1 = velocity.sample_in(points)
    vel_k2 = velocity.sample_in(points.shifted(0.5 * dt * vel_k1))
    vel_k3 = velocity.sample_in(points.shifted(0.5 * dt * vel_k2))
    vel_k4 = velocity.sample_in(points.shifted(dt * vel_k3))
    # --- Combine points with RK4 scheme ---
    vel = (1/6.) * (vel_k1 + 2 * (vel_k2 + vel_k3) + vel_k4)
    new_points = points.shifted(dt * vel)
    return PointCloud(new_points, field.values, field.extrapolation, add_overlapping=field._add_overlapping)


def points(field: PointCloud, velocity: PointCloud, dt):
    assert field.elements == velocity.elements
    new_points = field.elements.shifted(dt * velocity.values)
    return PointCloud(new_points, field.values, field.extrapolation, add_overlapping=field._add_overlapping)
