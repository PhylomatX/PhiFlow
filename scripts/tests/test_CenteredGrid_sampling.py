from phi.flow import *
import numpy as np

size = (2, 2)
domain = Domain(size, boundaries=CLOSED, bounds=Box(0, size))

positions = math.tensor([[0.6, 0.5], [-5, -5]], names='points,:')
points = PointCloud(Sphere(positions, 0))

cmask = points.at(domain.grid())

x_field = np.array([[1, 4], [2, 5], [3, 6]])
y_field = np.array([[1, 2, 3], [4, 5, 6]])

v_field = domain.sgrid(0)
v_field.x.values.native()[:] = x_field
v_field.y.values.native()[:] = y_field

velocity = v_field.sample_at(points.elements.center)
print(velocity)

app = App()
app.set_state(dict(v_field=v_field, cmask=cmask), dt=0.1, show=['cmask', 'v_field'])
show(app, display=['cmask', 'v_field'])
