import tensorflow as tf
from tutorial_3 import function_3 as f3

num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = tf.random.normal((num_examples, num_inputs), stddev=1)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += tf.random.normal(labels.shape, stddev=0.01)

w = tf.Variable(tf.random.normal((num_inputs, 1), stddev=0.01))
b = tf.Variable(tf.zeros((1,)))

batch_size = 50

lr = 0.03
num_epochs = 20
net = f3.linreg
loss = f3.squared_loss
for epoch in range(num_epochs):
    for x, y in f3.data_iter(batch_size, features, labels):
        with tf.GradientTape() as t:
            t.watch([w, b])
            l = tf.reduce_sum(loss(net(x, w, b), y))
        grads = t.gradient(l, [w, b])
        f32.sgd([w, b], lr, batch_size, grads)
    train_l = loss(net(features, w, b), labels)
    print('epoch %d, loss %f' % (epoch + 1, tf.reduce_mean(train_l)))


print(true_w, w)
print(true_b, b)