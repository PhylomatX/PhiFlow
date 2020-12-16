from phi.flow import *

if __name__ == '__main__':
    size = [2, 2]
    domain = Domain(x=size[0], y=size[1], boundaries=[CLOSED, OPEN], bounds=Box[0:size[0], 0:size[1]])

    positions = math.tensor([(1, 1)], names=['points', 'vector'])
    cloud = PointCloud(Sphere(positions, 0))

    sampled = cloud.at(domain.sgrid())
    sampled - domain.sgrid(0)
