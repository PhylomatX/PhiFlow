from phi.flow import *
from functools import partial


def extp_cgrid(cgrid: CenteredGrid) -> CenteredGrid:
    # extrapolation vertically and horizontally
    cgrid_l, cgrid_r = math.shift(cgrid.values, (-1, 1))
    where = partial(math.where, value_true=1, value_false=0)
    mask = math.sum(where(cgrid_l) + where(cgrid_r), axis='shift')
    extp = math.divide_no_nan(math.sum(cgrid_l + cgrid_r, axis='shift'), mask)
    # extrapolate diagonally
    cgrid_ll, cgrid_lr = math.shift(cgrid_l.shift[0], (-1, 1), axes='y')
    cgrid_rl, cgrid_rr = math.shift(cgrid_r.shift[0], (-1, 1), axes='y')
    mask = where(cgrid_ll) + where(cgrid_lr) + where(cgrid_rl) + where(cgrid_rr)
    extp_diag = math.divide_no_nan(cgrid_ll + cgrid_lr + cgrid_rl + cgrid_rr, mask).unstack('shift')[0]
    # prioritize results from vertical and horizontal shifting over diagonal shifting
    extp = math.where(extp, extp, extp_diag)
    return CenteredGrid(math.where(cgrid.values, cgrid.values, extp), bounds=cgrid.bounds, extrapolation=cgrid.extrapolation)


# domain = Domain((3, 3), boundaries=CLOSED, bounds=Box(0, (3, 3)))
# positions = math.batch_stack([math.tensor([(1, 1)])], axis='points')

domain = Domain((8, 8), boundaries=CLOSED, bounds=Box(0, (8, 8)))
positions = math.tensor([(3, 3), (3, 4), (4, 3)], names='points,:')

points = PointCloud(Sphere(positions, 1), values=[1, -1])

cgrid = points.at(domain.grid())

extp = extp_cgrid(cgrid)

print(cgrid.values.numpy())
print('\n')
print(extp.values.numpy())

# app = App()
# app.add_field('cgrid', cgrid)
# app.add_field('sgrid', sgrid)
# show(app)
