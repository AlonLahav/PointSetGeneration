import numpy as np
import pylab as plt
import imageio
import tensorflow as tf
import tensorflow.contrib.eager as tfe


# TODO:
# . Share with Ayellet
# . Polygon to points loss
# . EMD Loss - implement


# Parameters
n_dots = 40
n_dots_to_train = 40
n_dots_to_add = 10
n_iters = 50
learning_rate = .4
radiuses = [1, 2, 4]
video_output = 1

# Some initializations
np.random.seed(1)
tfe.enable_eager_execution()
if video_output:
    video = imageio.get_writer('points.mp4', fps=20)


def figure_2_np_array(fig):
    fig.add_subplot(111)
    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data


def generate_2d_point_cloud(method, n_dots, rs=[1], show=False):
    if method == 'circle':
        r = np.random.choice(rs, size=n_dots)[:, np.newaxis]
        theta = np.random.uniform(0, 2 * np.pi, (n_dots, 1))
        points = np.hstack((r * np.cos(theta), r * np.sin(theta)))
    else:
        raise Exception('unsupported method: ' + method)

    if show:
        plt.figure()
        plt.plot(points[:, 0], points[:, 1], '.')
        plt.show()

    return points


# Loss
def charmer_distance(points, vr_points):
    loss = 0
    n = 0
    for point in points:
        n += 1
        loss += tf.reduce_min((tf.reduce_sum((point - vr_points) ** 2, axis=1)))
    for p in range(vr_points.shape[0]):
        n += 1
        loss += tf.reduce_min((tf.reduce_sum((vr_points[p] - points) ** 2, axis=1)))
    return loss / n

# Gradient calculation
def grad(points, vr_points):
    with tfe.GradientTape() as tape:
        loss_value = charmer_distance(points, vr_points)
    return tape.gradient(loss_value, [vr_points])


# Get points to estimate
points = generate_2d_point_cloud('circle', n_dots, rs=radiuses, show=0)

if video_output:
    fig = plt.figure(1)
    plt.plot(points[:, 0], points[:, 1], 'bo')
    h_last = None

curr_est_points = np.zeros((0, 2))
while curr_est_points.shape[0] < n_dots_to_train:
    # Variable to train
    tmp = np.random.uniform(-4, 4, (n_dots_to_add, 2)) / 100
    tmp = np.vstack((curr_est_points, tmp))
    vr_points = tfe.Variable(tmp, dtype=tf.float32)

    print vr_points.shape

    for iter in range(n_iters):
        dPoints, = grad(points, vr_points)
        vr_points.assign_sub(dPoints * learning_rate)
        curr_est_points = np.array(vr_points.value())
        if video_output:
            plt.title((curr_est_points.shape[0], iter))
            if h_last != None:
                h_last.remove()
            h_last, = plt.plot(curr_est_points[:, 0], curr_est_points[:, 1], 'r.')
            plt.axis((-4, 4, -4, 4))
            img = figure_2_np_array(fig)
            video.append_data(img)

if video_output:
    video.close()
