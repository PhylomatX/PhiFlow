from phi.flow import *

domain = Domain((4, 4), boundaries=CLOSED, bounds=Box(0, (4, 4)))
positions = math.tensor([(1, 1), (1, 2), (2, 1), (2, 2)], names='points,:')
cloud = PointCloud(Sphere(positions, 1))
sgrid = cloud.at(domain.sgrid(0))
cgrid = cloud.at(domain.grid(0))

extp = field.extp_sgrid(sgrid, 1)

app = App()
app.add_field('cgrid', cgrid)
app.add_field('sgrid', sgrid)
app.add_field('extp', extp)
show(app)
