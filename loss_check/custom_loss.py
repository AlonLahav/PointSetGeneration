import tensorflow as tf
from tensorflow.python.framework import ops
import numpy as np


# Define custom py_func which takes also a grad op as argument:
def py_func(func, inp, Tout, stateful=True, name=None, grad=None):

    # Need to generate a unique name to avoid duplicates:
    rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))

    tf.RegisterGradient(rnd_name)(grad)  # see _MySquareGrad for grad example
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": rnd_name, "PyFuncStateless": rnd_name}):
        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)


# Def custom square function using np.square instead of tf.square:
def mysquare(x, name=None):
    with ops.name_scope(name, "Mysquare", [x]) as name:
        sqr_x = py_func(_MySquare,
                        [x],
                        [tf.float32],
                        name=name,
                        grad=_MySquareGrad)  # <-- here's the call to the gradient
        return sqr_x[0]

def _MySquare(x):
    return x ** 2

# Actual gradient:
def _MySquareGrad(op, grad):
    x = op.inputs[0]
    return grad * 20 * x  # add a "small" error just to see the difference:


with tf.Session() as sess:
    x = tf.constant([1., 2.])
    y = mysquare(x)
    y_new = mysquare(x)
    tf.global_variables_initializer().run()

    print(x.eval(), y.eval(), y_new.eval(), tf.gradients(y, x)[0].eval(), tf.gradients(y_new, x)[0].eval())
