import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as f
from tutorial_6 import function_6 as f6
import numpy as np
import sys

sys.path.append("..")

(corpus_indices, char_to_idx, idx_to_char, vocab_size) = f6.load_data_jay_lyrics()

tf.one_hot(np.array([0, 2]), vocab_size)


def to_onehot(X, size):  # 本函数已保存在d2lzh_tensorflow2包中方便以后使用
    # X shape: (batch), output shape: (batch, n_class)
    return [tf.one_hot(x, size, dtype=tf.float32) for x in X.T]


X = np.arange(10).reshape((2, 5))
inputs = to_onehot(X, vocab_size)
print(len(inputs), inputs[0].shape)

num_inputs, num_hiddens, num_outputs = vocab_size, 256, vocab_size


def get_params():
    def _one(shape):
        return tf.Variable(tf.random.normal(shape=shape, stddev=0.01, mean=0, dtype=tf.float32))

    # 隐藏层参数
    W_xh = _one((num_inputs, num_hiddens))
    W_hh = _one((num_hiddens, num_hiddens))
    b_h = tf.Variable(tf.zeros(num_hiddens), dtype=tf.float32)
    # 输出层参数
    W_hq = _one((num_hiddens, num_outputs))
    b_q = tf.Variable(tf.zeros(num_outputs), dtype=tf.float32)
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    return params


def init_rnn_state(batch_size, num_hiddens):
    return (tf.zeros(shape=(batch_size, num_hiddens)),)


def rnn(inputs, state, params):
    # inputs和outputs皆为num_steps个形状为(batch_size, vocab_size)的矩阵
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        X = tf.reshape(X, [-1, W_xh.shape[0]])
        H = tf.tanh(tf.matmul(X, W_xh) + tf.matmul(H, W_hh) + b_h)
        Y = tf.matmul(H, W_hq) + b_q
        outputs.append(Y)
    return outputs, (H,)


state = init_rnn_state(X.shape[0], num_hiddens)
inputs = to_onehot(X, vocab_size)
params = get_params()
outputs, state_new = rnn(inputs, state, params)

num_epochs, num_steps, batch_size, lr, clipping_theta = 250, 35, 32, 1e2, 1e-2
pred_period, pred_len, prefixes = 50, 50, ['分开', '不分开']
f6.train_and_predict_rnn(rnn, get_params, init_rnn_state, num_hiddens,
                         vocab_size, corpus_indices, idx_to_char,
                         char_to_idx, True, num_epochs, num_steps, lr,
                         clipping_theta, batch_size, pred_period, pred_len,
                         prefixes)
