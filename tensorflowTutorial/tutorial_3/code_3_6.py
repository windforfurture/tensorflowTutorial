import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import fashion_mnist
from tutorial_3 import function_3 as f3

batch_size=256
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = tf.cast(x_train, tf.float32) / 255  # 在进行矩阵相乘时需要float型，故强制类型转换为float型
x_test = tf.cast(x_test, tf.float32) / 255  # 在进行矩阵相乘时需要float型，故强制类型转换为float型
train_iter = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
test_iter = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

num_inputs = 784
num_outputs = 10
W = tf.Variable(tf.random.normal(shape=(num_inputs, num_outputs), mean=0, stddev=0.01, dtype=tf.float32))
b = tf.Variable(tf.zeros(num_outputs, dtype=tf.float32))

X = tf.random.normal(shape=(2, 5))
X_prob = f3.softmax(X)


def net(x):
    logits = tf.matmul(tf.reshape(x, shape=(-1, W.shape[0])), W) + b
    return f3.softmax(logits)


y_hat = np.array([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
y = np.array([0, 2], dtype='int32')

num_epochs, lr = 5, 0.1
trainer = tf.keras.optimizers.SGD(lr)
f3.train_ch3(net, train_iter, test_iter, f3.cross_entropy, num_epochs, batch_size, [W, b], lr)

X, y = iter(test_iter).next()

true_labels = f3.get_fashion_mnist_labels(y.numpy())
pred_labels = f3.get_fashion_mnist_labels(tf.argmax(f3.net(X), axis=1).numpy())
titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]

f3.show_fashion_mnist(X[0:9], titles[0:9])


