import tensorflow as tf
from tutorial_3 import function_3 as f3
x = tf.Variable(tf.range(-8, 8, 0.1), dtype=tf.float32)
y = tf.nn.relu(x)
# xyplot(x, y, 'relu')

with tf.GradientTape() as t:
    t.watch(x)
    y = y = tf.nn.relu(x)
dy_dx = t.gradient(y, x)
# f3.xyplot(x, dy_dx, 'grad of relu')

y = tf.nn.sigmoid(x)
# f3.xyplot(x, y, 'sigmoid')

with tf.GradientTape() as t:
    t.watch(x)
    y = y = tf.nn.sigmoid(x)
dy_dx = t.gradient(y, x)
# f3.xyplot(x, dy_dx, 'grad of sigmoid')

y = tf.nn.tanh(x)
# f3.xyplot(x, y, 'tanh')

with tf.GradientTape() as t:
    t.watch(x)
    y=y = tf.nn.tanh(x)
dy_dx = t.gradient(y, x)
f3.xyplot(x, dy_dx, 'grad of tanh')