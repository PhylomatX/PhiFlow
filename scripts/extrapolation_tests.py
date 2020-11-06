from phi.flow import *
from phi.math._tensors import Tensor
from functools import partial


def extrapolate(values: Tensor, size: int = 1) -> Tensor:
    if size == 0:
        return values
    # extrapolation vertically and horizontally
    values_l, values_r = math.shift(values, (-1, 1))
    where = partial(math.where, value_true=1, value_false=0)
    mask = math.sum(where(values_l) + where(values_r), axis='shift')
    extp = math.divide_no_nan(math.sum(values_l + values_r, axis='shift'), mask)
    # extrapolate diagonally
    values_ll, values_lr = math.shift(values_l.shift[0], (-1, 1), axes='y')
    values_rl, values_rr = math.shift(values_r.shift[0], (-1, 1), axes='y')
    mask = where(values_ll) + where(values_lr) + where(values_rl) + where(values_rr)
    extp_diag = math.divide_no_nan(values_ll + values_lr + values_rl + values_rr, mask).unstack('shift')[0]
    # prioritize results from vertical and horizontal shifting over diagonal shifting
    extp = math.where(extp, extp, extp_diag)
    return math.where(values, values, extrapolate(extp, size=size-1))


def extp_sgrid(sgrid: StaggeredGrid, size: int = 1) -> StaggeredGrid:
    tensors = []
    for cgrid in sgrid.unstack('vector'):
        tensors.append(extrapolate(cgrid.values, size=size))
    return StaggeredGrid(math.channel_stack(tensors, 'vector'), sgrid.box, sgrid.extrapolation)


# domain = Domain((3, 3), boundaries=CLOSED, bounds=Box(0, (3, 3)))
# positions = math.batch_stack([math.tensor([(1, 1)])], axis='points')

domain = Domain((4, 4), boundaries=CLOSED, bounds=Box(0, (4, 4)))
positions = math.tensor([(1, 1), (1, 2), (2, 1), (2, 2)], names='points,:')
cloud = PointCloud(Sphere(positions, 1))
sgrid = cloud.at(domain.sgrid(0))
cgrid = cloud.at(domain.grid(0))

extp = extp_sgrid(sgrid, 1)

app = App()
app.add_field('cgrid', cgrid)
app.add_field('sgrid', sgrid)
app.add_field('extp', extp)
show(app)
