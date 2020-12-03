from phi.flow import *

domain = Domain((4, 4), boundaries=CLOSED, bounds=Box(0, (4, 4)))
positions = math.tensor([(1, 1), (1, 2), (2, 1)], names='points,:')
cloud = PointCloud(Sphere(positions, 1))
sgrid = cloud.at(domain.sgrid(0))
cgrid = cloud.at(domain.grid(0))
cgrid.values.native()[1, 1] = 0.000005
cgrid.values.native()[1, 2] = -3

extpc = field.extp_cgrid(cgrid, 1)
extps = field.extp_sgrid(sgrid, 1)

app = App()
app.add_field('cgrid', cgrid)
app.add_field('sgrid', sgrid)
app.add_field('extpc', extpc)
app.add_field('extps', extps)
show(app)
