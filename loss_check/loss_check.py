import numpy as np
import pylab as plt
import tensorflow as tf
from tensorflow.python.framework import ops

np.random.seed(1)

def generate_2d_point_cloud(method, n_dots, show=False):
    if method == 'circle':
        r = 1
        theta = np.random.uniform(0, 2 * np.pi, (n_dots, 1))
        points = np.hstack((r * np.cos(theta), r * np.sin(theta)))
    else:
        raise Exception('unsupported method: ' + method)

    if show:
        plt.figure()
        plt.plot(points[:, 0], points[:, 1], '.')
        plt.show()

    return points

# Define custom py_func which takes also a grad op as argument:
def py_func(func, inp, Tout, stateful=True, name=None, grad=None):

    # Need to generate a unique name to avoid duplicates:
    rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))

    tf.RegisterGradient(rnd_name)(grad)  # see _MySquareGrad for grad example
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": rnd_name, "PyFuncStateless": rnd_name}):
        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)
# -------------------------------------------------------------


# Def custom square function using np.square instead of tf.square:
def mysquare(x, name=None):
    # Loss function:
    def _MySquare(x):
        return x ** 2

    # Actual gradient:
    def _MySquareGrad(op, grad):
        x = op.inputs[0]
        return grad * 2 * x  # add a "small" error just to see the difference:

    with ops.name_scope(name, "Mysquare", [x]) as name:
        sqr_x = py_func(_MySquare,
                        [x],
                        [tf.float32],
                        name=name,
                        grad=_MySquareGrad)  # <-- here's the call to the gradient
        return sqr_x[0]

# Def custom charmer distance
def charmer_distance(s1, s2, name=None):
    def _distance(s1, s2):
        return np.sum(np.abs(s1 - s2))
    def _distance_grad(op, grad):
        s1 = op.inputs[0]
        s2 = op.inputs[1]
        print s1.shape
        print grad.shape
        return grad * 2 * s1  # add a "small" error just to see the difference:

    with ops.name_scope(name, "CharmerDistance", [s1, s2]) as name:
        charmer_distance_op = py_func(_distance,
                        [s1, s2],
                        [tf.float32],
                        name=name,
                        grad=_distance_grad)
        return charmer_distance_op[0]


# Parameters
n_dots = 6

# Get points to estimate
points = generate_2d_point_cloud('circle', n_dots, show=0)

gt_points = tf.placeholder(tf.float32, shape=[n_dots, 2], name='gt-points')
vr_points = tf.Variable(np.zeros((n_dots, 2)), dtype=tf.float32)

d = gt_points - vr_points
my_sqt = mysquare(d)
#loss = 2 * tf.reduce_mean(my_sqt)
#loss = 2 * tf.reduce_mean((gt_points - vr_points) ** 2)
loss = charmer_distance(gt_points, vr_points)

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

sess = tf.InteractiveSession()

tf.global_variables_initializer().run()

all_est_points = []
for _ in range(10):
    est_pts, _ = sess.run([vr_points, train_step], feed_dict={gt_points: points})
    all_est_points.append(est_pts)

all_est_points = np.array(all_est_points)

plt.figure()
plt.plot(points[:, 0], points[:, 1], 'o')
for n in range(n_dots):
    this_pnt = all_est_points[:, n]
    plt.plot(this_pnt[:, 0], this_pnt[:, 1], '.')

plt.show()
