from phi.flow import *
import numpy as np

x = 30
y = 30
domain = Domain(x=x, y=y, boundaries=CLOSED, bounds=Box((0, 0), (x, y)))
obstacle = Obstacle(Box[10:20, 10:15].rotated(40))

positions = math.tensor([[2, 3], [25, 20]], names=['points', 'vector'])
bounds = Box((-20, -20), (x+20, y+20))
points = PointCloud(Sphere(positions, 0), add_overlapping=True, bounds=bounds)

accessible = domain.grid(1 - HardGeometryMask(union(obstacle.geometry)))
state = dict(points=points, accessible=accessible * 2 + points.at(domain.grid()), outward=False)


def step(points: PointCloud, outward: bool, **kwargs):
    # shift = (~domain.bounds).shift_points(points.elements.center, outward=outward)
    shift = obstacle.geometry.shift_points(points.elements.center, outward=outward, shift_amount=1)
    points = PointCloud(points.elements.shifted(shift), add_overlapping=True, bounds=points.bounds)
    accessible = domain.grid(1 - HardGeometryMask(union(obstacle.geometry)))
    return dict(points=points, accessible=accessible * 2 + points.at(domain.grid()), outward=not outward)


app = App()
app.set_state(state, step_function=step, dt=0.1, show=['points', 'accessible'])
show(app)
