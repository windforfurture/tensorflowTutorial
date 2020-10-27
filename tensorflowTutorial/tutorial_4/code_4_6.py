import tensorflow as tf
import numpy as np
from tensorflow.python.client import device_lib

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
cpus = tf.config.experimental.list_physical_devices(device_type='CPU')
print("可用的GPU：", gpus, "\n可用的CPU：", cpus)

print(device_lib.list_local_devices())
with tf.device('GPU:0'):
    a = tf.constant([1,2,3],dtype=tf.float32)
    b = tf.random.uniform((3,))
    print(tf.exp(a + b) * 2)