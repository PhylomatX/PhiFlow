from phi.flow import *
from phi.field._point_cloud import _distribute_points

size = (64, 64)
domain = Domain(size, boundaries=CLOSED, bounds=Box(0, (64, 64)))

# define initial liquid
density1 = domain.grid().values
density1.numpy()[28:36, 28:36] = 1
density2 = domain.grid().values
density2.numpy()[8:16, 8:16] = 1
field = math.batch_stack([density1, density2])

initial_points = _distribute_points(field, 8)
points = PointCloud(Sphere(initial_points, 0), add_overlapping=True)

density = points.at(domain.grid())

state = dict(density=density)

app = App()
app.set_state(state, dt=0.1, show=['density'])
show(app, display='density', port=8052)
