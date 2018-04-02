import numpy as np
import pylab as plt
import imageio
import tensorflow as tf
import tensorflow.contrib.eager as tfe


# TODO:
# . Share with Ayellet
# . Add one point at a time
# . Polygon to points loss
# . EMD Loss - implement


# Parameters
n_dots = 100
n_iters = 100
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


# Get points to estimate
points = generate_2d_point_cloud('circle', n_dots, rs=radiuses, show=0)

# Variable to train
tmp = generate_2d_point_cloud('circle', n_dots, rs=[.2], show=0)
tmp = np.random.normal(0, 2, (n_dots, 2))
vr_points = tfe.Variable(tmp, dtype=tf.float32)

# Loss
def loss_l2(points, vr_points):
    loss = tf.reduce_mean((points - vr_points) ** 2)
    return loss

def charmer_distance(points, vr_points):
    loss = 0
    for point in points:
        loss += tf.reduce_min(tf.reduce_sum((point - vr_points) ** 2, axis=1))
    for p in range(vr_points.shape[0]):
        loss += tf.reduce_min(tf.reduce_sum((vr_points[p] - points) ** 2, axis=1))
    return loss / 10

# Gradient calculation
def grad(points, vr_points):
    with tfe.GradientTape() as tape:
        loss_value = charmer_distance(points, vr_points)
        #loss_value = loss_l2(points, vr_points)
    return tape.gradient(loss_value, [vr_points])

if video_output:
    fig = plt.figure(1)
    plt.plot(points[:, 0], points[:, 1], 'bo')
    h_last = None
all_est_points = []
for iter in range(n_iters):
    dPoints, = grad(points, vr_points)
    vr_points.assign_sub(dPoints * learning_rate)
    curr_est_points = np.array(vr_points.value())
    all_est_points.append(curr_est_points)
    if video_output:
        plt.title(iter)
        if h_last != None:
            h_last.remove()
        h_last, = plt.plot(curr_est_points[:, 0], curr_est_points[:, 1], 'r.')
        img = figure_2_np_array(fig)
        video.append_data(img)

all_est_points = np.array(all_est_points)

plt.figure()
plt.plot(points[:, 0], points[:, 1], 'o')
if 0:
    for n in range(n_dots):
        this_pnt = all_est_points[:, n]
        plt.plot(this_pnt[:, 0], this_pnt[:, 1], '.')
else:
    plt.plot(all_est_points[-1, :, 0], all_est_points[-1, :, 1], '*')

if video_output:
    video.close()
