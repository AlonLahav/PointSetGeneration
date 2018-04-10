import time
from time import gmtime, strftime
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
n_dots = 100
n_dots_to_train = n_dots
n_dots_to_add = n_dots_to_train
n_iters = 150
learning_rate = .1 * n_dots_to_train
radiuses = [1, 2, 4]
video_output = 1

modified_loss = 1

# Some initializations
timestr = strftime("%Y-%m-%d_%H:%M", gmtime())
np.random.seed(1)
tfe.enable_eager_execution()
if video_output:
    video = imageio.get_writer('output_tmp/points_' + timestr + '.mp4', fps=20)


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
    for p in range(points.shape[0]):
        n += 1
        loss += 1 * tf.reduce_min((tf.reduce_sum((points[p] - vr_points) ** 2, axis=1)))
    for p in range(vr_points.shape[0]):
        n += 1
        loss += 1 * tf.reduce_min((tf.reduce_sum((vr_points[p] - points) ** 2, axis=1)))
    loss /= n
    return loss , loss

def charmer_distance_mod(points, vr_points):
    loss = 0
    n = 0
    vr_points_tmp = vr_points
    for p in range(points.shape[0]):
        n += 1
        gt_pt_to_all = tf.reduce_sum((points[p] - vr_points_tmp) ** 2, axis=1)
        idx = tf.argmin(gt_pt_to_all)
        loss += tf.gather(gt_pt_to_all, [idx])
        vr_points_tmp = tf.boolean_mask(vr_points_tmp, 1-tf.one_hot(idx, vr_points_tmp.shape[0]))
    points_tmp = points
    for p in range(vr_points.shape[0]):
        n += 1
        vr_pt_to_all = tf.reduce_sum((vr_points[p] - points_tmp) ** 2, axis=1)
        idx = tf.argmin(vr_pt_to_all)
        loss += tf.gather(vr_pt_to_all, [idx])
        points_tmp = tf.boolean_mask(points_tmp, 1 - tf.one_hot(idx, points_tmp.shape[0]))
    loss /= n
    loss_no_rglrz = charmer_distance(points, vr_points)[0]
    return loss , loss_no_rglrz

# Gradient calculation
def grad(points, vr_points):
    with tfe.GradientTape() as tape:
        if modified_loss:
            loss_value, loss_no_rglrz = charmer_distance_mod(points, vr_points)
        else:
            loss_value, loss_no_rglrz = charmer_distance(points, vr_points)
    loss_grad, = tape.gradient(loss_value, [vr_points])
    return loss_grad, loss_value, loss_no_rglrz


# Get points to estimate
points_np = generate_2d_point_cloud('circle', n_dots, rs=radiuses, show=0)
print(points_np.shape)
points = tfe.Variable(points_np, dtype=tf.float32)

if video_output:
    fig = plt.figure(1)
    plt.plot(points_np[:, 0], points_np[:, 1], 'bo')
h_last = None

tb = time.time()
curr_est_points = np.zeros((0, 2))
all_est_pnts = []
loss_all = []
iter_all = []
pure_loss_all = []
i = 0
while curr_est_points.shape[0] < n_dots_to_train:
    # Variable to train
    tmp = np.random.uniform(-4, 4, (n_dots_to_add, 2))
    tmp = np.vstack((curr_est_points, tmp))
    vr_points = tfe.Variable(tmp, dtype=tf.float32)

    print vr_points.shape

    for iter in range(n_iters):
        dPoints,l, loss_no_rglrz = grad(points, vr_points)
        loss_all.append(l)
        iter_all.append(i)
        pure_loss_all.append(loss_no_rglrz)
        i += 1
        vr_points.assign_sub(dPoints * learning_rate)
        curr_est_points = np.array(vr_points.value())
        all_est_pnts.append(curr_est_points.copy())
        if video_output:
            plt.title((curr_est_points.shape[0], iter))
            if h_last != None:
                h_last.remove()
            h_last, = plt.plot(curr_est_points[:, 0], curr_est_points[:, 1], 'r.')
            plt.axis((-4, 4, -4, 4))
            img = figure_2_np_array(fig)
            video.append_data(img)

if video_output:
    plt.close()
    video.close()

runtime = time.time() - tb
print('time: ' + str(runtime))

fig = plt.figure()
plt.subplot(1, 2, 1)
plt.plot(iter_all, loss_all)
plt.plot(iter_all, pure_loss_all)
plt.legend(('trained loss', 'charmer distance'))
plt.title('Loss calculation')
plt.xlabel('Iteration')
plt.subplot(1, 2, 2)
plt.plot(points_np[:, 0], points_np[:, 1], 'bo')
plt.plot(curr_est_points[:, 0], curr_est_points[:, 1], 'r*')
all_est_pnts = np.array(all_est_pnts)
for n in range(all_est_pnts.shape[1]):
    plt.plot(all_est_pnts[:, n, 0], all_est_pnts[:, n, 1], '-*', alpha=0.2)
plt.axis((-4, 4, -4, 4))
plt.title('Points')
if modified_loss:
    loss_str = 'loss: modified charmer distance'
else:
    loss_str = 'loss: charmer distance'
plt.suptitle(loss_str + '\nLast charmer distance: ' + str(float(pure_loss_all[-1])) + '\nRuntime: ' + str(runtime))
fig.set_size_inches(14, 7)
plt.savefig('output_tmp/' + timestr)
plt.show()
