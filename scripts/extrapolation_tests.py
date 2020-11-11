from phi.flow import *

# domain = Domain((4, 4), boundaries=CLOSED, bounds=Box(0, (4, 4)))
# positions = math.tensor([(1, 1), (1, 2), (2, 1), (2, 2)], names='points,:')
# cloud = PointCloud(Sphere(positions, 1))
# sgrid = cloud.at(domain.sgrid(0))
#
# extp = field.extp_sgrid(sgrid, 1)
#
# app = App()
# app.add_field('sgrid', sgrid)
# app.add_field('extp', extp)
# show(app)


domain = Domain((8, 8), boundaries=CLOSED, bounds=Box(0, (8, 8)))
positions = math.tensor([(1, 1), (1, 2), (4, 4), (4, 5), (5, 4)], names='points,:')
cloud = PointCloud(Sphere(positions, 1))
cgrid = cloud.at(domain.grid(0))
cgrid.values.native()[1, 1] = 2
cgrid.values.native()[4, 5] = -2
sgrid = cloud.at(domain.sgrid(0))

extp_s = field.extp_sgrid(sgrid)
extp_c = field.extp_cgrid(cgrid, 1)

app = App()
app.add_field('sgrid', sgrid)
app.add_field('cgrid', cgrid)
app.add_field('extp_c', extp_c)
app.add_field('extp_s', extp_s)
show(app)

cgrid = cloud.at(domain.grid(0))