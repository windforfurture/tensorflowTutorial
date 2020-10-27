import tensorflow as tf
from tensorflow import keras
import numpy as np
import time
import sys
import matplotlib.pyplot as plt
from tutorial_3 import function_3 as f3

from tensorflow.keras.datasets import fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
feature, label = x_train[0], y_train[0]

# X, y = [], []
# for i in range(10):
#     X.append(x_train[i])
#     y.append(y_train[i])
# f35.show_fashion_mnist(X, f35.get_fashion_mnist_labels(y))

batch_size = 256
if sys.platform.startswith('win'):
    num_workers = 0  # 0表示不用额外的进程来加速读取数据
else:
    num_workers = 4
train_iter = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)

start = time.time()
for X, y in train_iter:
    continue
print('%.2f sec' % (time.time() - start))