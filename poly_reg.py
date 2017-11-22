# Following code counts polynomial regression for different powers of factors
# and then plot graph, that shows relation between polynom power and loss
# in similar conditions

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# Create data
n_observations = 100
xs = np.linspace(-3, 3, n_observations)
ys = np.sin(xs) + np.random.uniform(-0.5, 0.5, n_observations)

### Params
powers = [i for i in range(1,6)]
n_epochs = 200
learning_rate = 0.01
ridge = False

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

power_loss = []

for power in powers:
    Y_pred = tf.Variable(tf.random_normal([1]), name='bias')
    for pow_i in range(1, power):
        W = tf.Variable(tf.random_normal([1]), name='weight_%d' % pow_i)
        Y_pred = tf.add(tf.multiply(tf.pow(X, pow_i), W), Y_pred)

    loss = tf.reduce_sum(tf.pow(Y_pred - Y, 2)) / (n_observations - 1)
    if ridge:
        loss = tf.add(loss, tf.mul(1e-6, tf.global_norm([W])))

    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch_i in range(n_epochs):
            for x, y in zip(xs, ys):
                sess.run(optimizer, feed_dict={X:x, Y:y})
        
        total_loss = training_cost = sess.run(
                loss, feed_dict={X: xs, Y: ys})
        print("Power: ", power, " loss: ", total_loss)
        power_loss.append(total_loss)

plt.figure()
plt.plot(powers, power_loss)
plt.show()