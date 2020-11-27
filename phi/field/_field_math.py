import warnings
from functools import wraps

import numpy as np
from phi import math
from phi.geom import Box, Geometry
from . import StaggeredGrid, ConstantField, HardGeometryMask
from ._field import Field, SampledField
from ._grid import CenteredGrid, Grid
from ..math import tensor


def laplace(field: Grid, axes=None):
    result = field._op1(lambda tensor: math.laplace(tensor, dx=field.dx, padding=field.extrapolation, dims=axes))
    return result


def gradient(field: CenteredGrid, type: type = CenteredGrid):
    if type == CenteredGrid:
        values = math.gradient(field.values, field.dx.vector.as_channel(name='gradient'), difference='central', padding=field.extrapolation)
        return CenteredGrid(values, field.bounds, field.extrapolation.gradient())
    elif type == StaggeredGrid:
        return stagger(field, lambda lower, upper: (upper - lower) / field.dx, field.extrapolation.gradient())
    raise NotImplementedError(f"{type(field)} not supported. Only CenteredGrid and StaggeredGrid allowed.")


def shift(grid: CenteredGrid, offsets: tuple, stack_dim='shift'):
    """ Wraps :func:`math.shift` for CenteredGrid. """
    data = math.shift(grid.values, offsets, padding=grid.extrapolation, stack_dim=stack_dim)
    return [CenteredGrid(data[i], grid.box, grid.extrapolation) for i in range(len(offsets))]


def stagger(field: CenteredGrid, face_function: callable, extrapolation: math.extrapolation.Extrapolation):
    all_lower = []
    all_upper = []
    for dim in field.shape.spatial.names:
        all_upper.append(math.pad(field.values, {dim: (0, 1)}, field.extrapolation))
        all_lower.append(math.pad(field.values, {dim: (1, 0)}, field.extrapolation))
    all_upper = math.channel_stack(all_upper, 'vector')
    all_lower = math.channel_stack(all_lower, 'vector')
    values = face_function(all_lower, all_upper)
    return StaggeredGrid(values, field.bounds, extrapolation)


def divergence(field: Grid):
    if isinstance(field, StaggeredGrid):
        components = []
        for i, dim in enumerate(field.shape.spatial.names):
            div_dim = math.gradient(field.values.vector[i], dx=field.dx[i], difference='forward', padding=None, dims=[dim]).gradient[0]
            components.append(div_dim)
        data = math.sum(components, 0)
        return CenteredGrid(data, field.box, field.extrapolation.gradient())
    else:
        raise NotImplementedError(f"{type(field)} not supported. Only StaggeredGrid allowed.")


def diffuse(field: Field, diffusivity, dt, substeps=1):
    """
    Simulate a finite-time diffusion process of the form dF/dt = α · ΔF on a given `Field` F with diffusion coefficient α.

    If `field` is periodic (set via `extrapolation='periodic'`), diffusion may be simulated in Fourier space.
    Otherwise, finite differencing is used to approximate the

    :param field: CenteredGrid, StaggeredGrid or ConstantField
    :param diffusivity: diffusion amount = diffusivity * dt
    :param dt: diffusion amount = diffusivity * dt
    :param substeps: number of iterations to use
    :return: Field of same type as `field`
    :rtype: Field
    """
    if isinstance(field, ConstantField):
        return field
    assert isinstance(field, Grid), "Cannot diffuse field of type '%s'" % type(field)
    amount = diffusivity * dt
    if field.extrapolation == 'periodic' and not isinstance(amount, Field):
        fft_laplace = -(2 * np.pi) ** 2 * squared(fftfreq(field))
        diffuse_kernel = math.exp(fft_laplace * amount)
        return real(ifft(fft(field) * diffuse_kernel))
    else:
        if isinstance(amount, Field):
            amount = amount.at(field)
        for i in range(substeps):
            field += amount / substeps * laplace(field)
        return field


def solve(function, y: Grid, x0: Grid, solve_params: math.Solve, callback=None):
    if callback is not None:
        def field_callback(x):
            x = x0._with(x)
            callback(x)
    else:
        field_callback = None

    data_function = expose_tensors(function, x0)
    converged, x, iterations = math.solve(data_function, y.values, x0.values, solve_params, field_callback)
    return converged, x0._with(x), iterations


def expose_tensors(field_function, *proto_fields):
    @wraps(field_function)
    def wrapper(*field_data):
        fields = [proto._with(data) for data, proto in zip(field_data, proto_fields)]
        return field_function(*fields).values
    return wrapper


def data_bounds(field: SampledField):
    data = field.points
    min_vec = math.min(data, axis=data.shape.spatial.names)
    max_vec = math.max(data, axis=data.shape.spatial.names)
    return Box(min_vec, max_vec)


def mean(field: Grid):
    return math.mean(field.values, field.shape.spatial)


def normalize(field: SampledField, norm: SampledField, epsilon=1e-5):
    data = math.normalize_to(field.values, norm.values, epsilon)
    return field._with(data)


def pad(grid: Grid, widths: int or tuple or list or dict):
    if isinstance(widths, int):
        widths = {axis: (widths, widths) for axis in grid.shape.spatial.names}
    elif isinstance(widths, (tuple, list)):
        widths = {axis: (width if isinstance(width, (tuple, list)) else (width, width)) for axis, width in zip(grid.shape.spatial.names, widths)}
    else:
        assert isinstance(widths, dict)
    widths_list = [widths[axis] for axis in grid.shape.spatial.names]
    if isinstance(grid, Grid):
        data = math.pad(grid.values, widths, grid.extrapolation)
        w_lower = tensor([w[0] for w in widths_list])
        w_upper = tensor([w[1] for w in widths_list])
        box = Box(grid.box.lower - w_lower * grid.dx, grid.box.upper + w_upper * grid.dx)
        return type(grid)(data, box, grid.extrapolation)
    raise NotImplementedError(f"{type(grid)} not supported. Only Grid instances allowed.")


def divergence_free(vector_field: Grid, solve_params: math.LinearSolve = math.LinearSolve(None, 1e-5)):
    """
    Returns the divergence-free part of the given vector field.
    The boundary conditions are taken from `vector_field`.

    This function solves for a scalar potential with an iterative solver.

    :param vector_field: vector grid
    :param solve_params:
    :return: divergence-free vector field, scalar potential, number of iterations performed, divergence
    """
    div = divergence(vector_field)
    div -= mean(div)
    pressure_extrapolation = vector_field.extrapolation  # periodic -> periodic, closed -> boundary, open -> zero
    pressure_guess = CenteredGrid.sample(0, vector_field.resolution, vector_field.box, extrapolation=pressure_extrapolation)
    converged, potential, iterations = solve(laplace, div, pressure_guess, solve_params)
    gradp = gradient(potential, type=StaggeredGrid)
    vector_field -= gradp
    return vector_field, potential, iterations, div


def squared(field: Field):
    raise NotImplementedError()


def real(field: Field):
    raise NotImplementedError()


def imag(field: Field):
    raise NotImplementedError()


def fftfreq(grid: Grid):
    raise NotImplementedError()


def fft(grid: Grid):
    raise NotImplementedError()


def ifft(grid: Grid):
    raise NotImplementedError()


def staggered_curl_2d(grid, pad_width=(1, 2)):
    assert isinstance(grid, CenteredGrid)
    kernel = math.zeros((3, 3, 1, 2))
    kernel[1, :, 0, 0] = [0, 1, -1]  # y-component: - dz/dx
    kernel[:, 1, 0, 1] = [0, -1, 1]  # x-component: dz/dy
    scalar_potential = grid.padded([pad_width, pad_width]).values
    vector_field = math.conv(scalar_potential, kernel, padding='valid')
    return StaggeredGrid(vector_field, bounds=grid.box)


def where(mask: Field or Geometry, field_true: Field, field_false: Field):
    if isinstance(mask, Geometry):
        mask = HardGeometryMask(mask)
    elif isinstance(mask, SampledField):
        field_true = field_true.at(mask)
        field_false = field_false.at(mask)
    elif isinstance(field_true, SampledField):
        mask = mask.at(field_true)
        field_false = field_false.at(field_true)
    elif isinstance(field_false, SampledField):
        mask = mask.at(field_true)
        field_true = field_true.at(mask)
    else:
        raise NotImplementedError('At least one argument must be a SampledField')
    values = math.divide_no_nan(mask.values, mask.values) * field_true.values + (1 - mask.values) * field_false.values
    return field_true._with(values)
