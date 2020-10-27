import tensorflow as tf
from tensorflow.keras import layers, models, initializers, optimizers, regularizers
import numpy as np
import matplotlib.pyplot as plt
from tutorial_3 import function_3 as f3

n_train, n_test, num_inputs = 20, 100, 200
true_w, true_b = tf.ones((num_inputs, 1)) * 0.01, 0.05

features = tf.random.normal(shape=(n_train + n_test, num_inputs))
labels = tf.keras.backend.dot(features, true_w) + true_b
labels += tf.random.normal(mean=0.01, shape=labels.shape)
train_features, test_features = features[:n_train, :], features[n_train:, :]
train_labels, test_labels = labels[:n_train], labels[n_train:]


def init_params():
    w = tf.Variable(tf.random.normal(mean=1, shape=(num_inputs, 1)))
    b = tf.Variable(tf.zeros(shape=(1,)))
    return [w, b]


def l2_penalty(w):
    return tf.reduce_sum((w ** 2)) / 2


batch_size, num_epochs, lr = 1, 100, 0.003
net, loss = f3.linreg, f3.squared_loss
optimizer = tf.keras.optimizers.SGD()
train_iter = tf.data.Dataset.from_tensor_slices(
    (train_features, train_labels)).batch(batch_size).shuffle(batch_size)


def fit_and_plot(lambd):
    w, b = init_params()
    train_ls, test_ls = [], []
    for _ in range(num_epochs):
        for X, y in train_iter:
            with tf.GradientTape(persistent=True) as tape:
                # 添加了L2范数惩罚项
                l = loss(net(X, w, b), y) + lambd * l2_penalty(w)
            grads = tape.gradient(l, [w, b])
            f3.sgd([w, b], lr, batch_size, grads)
        train_ls.append(tf.reduce_mean(loss(net(train_features, w, b),
                                            train_labels)).numpy())
        test_ls.append(tf.reduce_mean(loss(net(test_features, w, b),
                                           test_labels)).numpy())
    f3.semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'loss',
                range(1, num_epochs + 1), test_ls, ['train', 'test'])
    print('L2 norm of w:', tf.norm(w).numpy())


# fit_and_plot(lambd=0)
# fit_and_plot(lambd=3)


def fit_and_plot_tf2(wd, lr=1e-3):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(1,
                                    kernel_regularizer=regularizers.l2(wd),
                                    bias_regularizer=None))
    model.compile(optimizer=tf.keras.optimizers.SGD(lr=lr),
                  loss=tf.keras.losses.MeanSquaredError())
    history = model.fit(train_features, train_labels, epochs=100, batch_size=1,
                        validation_data=(test_features, test_labels),
                        validation_freq=1, verbose=0)
    train_ls = history.history['loss']
    test_ls = history.history['val_loss']
    f3.semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'loss',
                range(1, num_epochs + 1), test_ls, ['train', 'test'])
    print('L2 norm of w:', tf.norm(model.get_weights()[0]).numpy())


# fit_and_plot_tf2(0, lr)
# fit_and_plot_tf2(3, lr)