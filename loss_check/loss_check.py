import numpy as np
import pylab as plt
import tensorflow as tf

np.random.seed(1)

def generate_2d_point_cloud(method, n_dots, show=False):
    if method == 'circle':
        r = 1
        theta = np.random.uniform(0, 2 * np.pi, (n_dots, 1))
        points = np.hstack((r * np.cos(theta), r * np.sin(theta)))
        print points
    else:
        raise Exception('unsupported method: ' + method)

    if show:
        plt.figure()
        plt.plot(points[:, 0], points[:, 1], '.')
        plt.show()

    return points


# Parameters
n_dots = 4

# Get points to estimate
points = generate_2d_point_cloud('circle', n_dots, 1)


