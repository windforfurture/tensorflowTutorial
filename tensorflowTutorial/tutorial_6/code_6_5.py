import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as f
from tutorial_6 import function_6 as f6
import numpy as np
import sys

sys.path.append("..")

(corpus_indices, char_to_idx, idx_to_char, vocab_size) = f6.load_data_jay_lyrics()

num_hiddens = 256
cell = keras.layers.SimpleRNNCell(num_hiddens, kernel_initializer='glorot_uniform')
rnn_layer = keras.layers.RNN(cell, time_major=True, return_sequences=True, return_state=True)

batch_size = 2
state = cell.get_initial_state(batch_size=batch_size, dtype=tf.float32)

num_steps = 35
X = tf.random.uniform(shape=(num_steps, batch_size, vocab_size))
Y, state_new = rnn_layer(X, state)

model = f6.RNNModel(rnn_layer, vocab_size)

num_epochs, batch_size, lr, clipping_theta = 250, 32, 1e2, 1e-2
pred_period, pred_len, prefixes = 50, 50, ['分开', '不分开']
f6.train_and_predict_rnn_keras(model, num_hiddens, vocab_size,
                               corpus_indices, idx_to_char, char_to_idx,
                               num_epochs, num_steps, lr, clipping_theta,
                               batch_size, pred_period, pred_len, prefixes)
