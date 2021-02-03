from phi.flow import *

""" Multiple events. """

# define world
size = (100, 50)
domain = Domain(size, boundaries=CLOSED, bounds=Box(0, size))

position_list = []
velocity_list = []

x = 10

# linear collision
p = [[x, 10], [x, size[1]-10]]
v = [[0, 5], [0, -5]]
position_list += p
velocity_list += v
x += 10

# oblique collision
distance = 10
p = [[x, 10], [x+distance, size[1]-10]]
v = [[1, 5], [-2, -5]]
position_list += p
velocity_list += v
x += distance + 10

# target collision
p = [[x, 10], [x, size[1]-10]]
v = [[0, 0], [0, -5]]
position_list += p
velocity_list += v
x += 10

# diagonal collision
distance = 20
p = [[x, 10], [x+distance, 10]]
v = [[5, 2], [-5, 2]]
position_list += p
velocity_list += v
x += distance + 10

# block collision
y = size[1] - 10
p = [[x, y], [x+1, y], [x, y-1], [x+1, y-1], *[[i, 10] for i in range(x-2, x+5)]]
v = [*[[0, -5] for i in range(4)], *[[0, 0] for i in range(7)]]
position_list += p
velocity_list += v
x += 10


positions = math.tensor(position_list, names='points,:')
points = PointCloud(Sphere(positions, 0))
velocities = math.tensor(velocity_list, names=['points', 'vector'])
velocities = PointCloud(Sphere(positions, 0), values=velocities)

extrapolation = 2

state = dict(points=points, velocities=velocities, v_field=velocities.at(domain.sgrid()), smask=points.at(domain.sgrid()), cmask=points.at(domain.grid()),
             extp_v_field=field.extp_sgrid(velocities.at(domain.sgrid()) * points.at(domain.sgrid()), extrapolation))


def step(points, velocities, v_field, smask, cmask, extp_v_field, dt):
    points = advect.advect(points, extp_v_field, dt)
    velocities = PointCloud(points.elements, values=velocities.values)
    v_field = velocities.at(domain.sgrid())
    return dict(points=points, velocities=velocities, v_field=v_field, smask=points.at(domain.sgrid()), cmask=points.at(domain.grid()),
                extp_v_field=field.extp_sgrid(v_field * points.at(domain.sgrid()), extrapolation))


# for i in range(5):
#     step(**state, dt=0.1)


app = App()
app.set_state(state, step_function=step, dt=0.1, show=['extp_v_field', 'cmask', 'smask', 'v_field'])
show(app, display=['extp_v_field', 'cmask', 'smask', 'v_field'], port=8052)

