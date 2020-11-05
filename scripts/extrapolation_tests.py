from phi.flow import *

# domain = Domain((3, 3), boundaries=CLOSED, bounds=Box(0, (3, 3)))
# positions = math.batch_stack([math.tensor([(1, 1)])], axis='points')

domain = Domain((4, 4), boundaries=CLOSED, bounds=Box(0, (4, 4)))
positions = math.tensor([(1, 1), (1, 2), (2, 2), (2, 1)], names='points,:')

points = PointCloud(Sphere(positions, 1)) * [-1, 1]

cgrid = points.at(domain.grid())
sgrid = points.at(domain.sgrid())

cgrid_shifted = math.shift(cgrid.values, (-1, 0))

app = App()
app.add_field('cgrid', cgrid)
app.add_field('sgrid', sgrid)
app.add_field('cgrid_shifted', cgrid_shifted)
show(app)
