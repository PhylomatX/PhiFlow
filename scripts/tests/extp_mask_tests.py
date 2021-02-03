from phi.flow import *

x = 4
y = 4
size = [x, y]
domain = Domain(x=x, y=y, boundaries=CLOSED, bounds=Box[0:x, 0:y])
positions = math.tensor([(1.6, 1.6)], names=['points', 'vector'])
cloud = PointCloud(Sphere(positions, 1), values=1)
mask = cloud.at(domain.grid(0))
cgrid = domain.grid(0)
cgrid.values.native()[1, 1] = 2

extpc = field.extp_cgrid(cgrid, mask.values, 1)

# extpc = field.extp_cgrid(cgrid, 1)
# extps = field.extp_sgrid(sgrid, 1)

app = App()
app.add_field('cgrid', cgrid)
app.add_field('mask', mask)
app.add_field('extpc', extpc)
# app.add_field('extps', extps)
show(app)
