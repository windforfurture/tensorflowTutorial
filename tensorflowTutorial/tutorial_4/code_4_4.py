import tensorflow as tf
import numpy as np

X = tf.random.uniform((2, 20))


class CenteredLayer(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, inputs):
        return inputs - tf.reduce_mean(inputs)


layer = CenteredLayer()
layer(np.array([1, 2, 3, 4, 5]))

net = tf.keras.models.Sequential()
net.add(tf.keras.layers.Flatten())
net.add(tf.keras.layers.Dense(20))
net.add(CenteredLayer())

Y = net(X)
tf.reduce_mean(Y)


class myDense(tf.keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.units = units

    def build(self, input_shape):  # 这里 input_shape 是第一次运行call()时参数inputs的形状
        self.w = self.add_weight(name='w',
                                 shape=[input_shape[-1], self.units], initializer=tf.random_normal_initializer())
        self.b = self.add_weight(name='b',
                                 shape=[self.units], initializer=tf.zeros_initializer())

    def call(self, inputs):
        y_pred = tf.matmul(inputs, self.w) + self.b
        return y_pred


net = tf.keras.models.Sequential()
net.add(myDense(8))
net.add(myDense(1))

print(net(X))
