from phi.flow import *
import matplotlib.pyplot as plt

identifier = ''

size = (64, 64)
domain = Domain(size, boundaries=CLOSED, bounds=Box(0, (64, 64)))

divergence_data = np.load(f'/home/john/Projekte/BA/tmp/divergence{identifier}.npy')
active_mask_data = np.load(f'/home/john/Projekte/BA/tmp/active{identifier}.npy')

divergence = domain.grid(0)
divergence.values.native()[:] = divergence_data
active_mask = domain.grid(0, extrapolation=domain.boundaries.active_extrapolation)
active_mask.values.native()[:] = active_mask_data

plt.imshow(divergence.values.numpy() + 100 * active_mask.values.numpy())
plt.show()

plt.imshow(divergence.values.numpy())
plt.show()

plt.imshow(active_mask.values.numpy())
plt.show()

laplace = lambda pressure: field.divergence(field.gradient(pressure, type=StaggeredGrid) * domain.sgrid(1)) * active_mask - 4 * (1 - active_mask) * pressure
converged, pressure, iterations = field.solve(laplace, divergence, domain.grid(0), solve_params=math.LinearSolve(None, 1e-3))

plt.imshow(pressure.values.numpy())
plt.show()
