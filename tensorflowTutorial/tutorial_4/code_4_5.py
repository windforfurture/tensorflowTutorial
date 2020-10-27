import tensorflow as tf
import numpy as np

x = tf.ones(3)
np.save('x.npy', x)
x2 = np.load('x.npy')

y = tf.zeros(4)
np.save('xy.npy',[x,y])
x2, y2 = np.load('xy.npy', allow_pickle=True)

my_dict = {'x': x, 'y': y}
np.save('my_dict.npy', my_dict)
my_dict_2 = np.load('my_dict.npy', allow_pickle=True)

X = tf.random.normal((2,20))


class MLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten()    # Flatten层将除第一维（batch_size）以外的维度展平
        self.dense1 = tf.keras.layers.Dense(units=256, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=10)

    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.dense1(x)
        output = self.dense2(x)
        return output


net = MLP()
Y = net(X)
net.save_weights("4.5saved_model.h5")

net2 = MLP()
net2(X)
net2.load_weights("4.5saved_model.h5")
Y2 = net2(X)
print(Y2 == Y)