---
layout: post/
title: tensorflow实现梯度下降各种方法
date: 2019-08-03 11:14:07
tags: tensorflow
categories: Deep learning
---
# 不使用tensorflow任何梯度下降方法
```js
# -*- coding: utf8 -*-
import tensorflow as tf
# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./MNIST_data", one_hot=True)

# Parameters
learning_rate = 0.01
training_epochs = 10
batch_size = 100
display_step = 1

# Parameters
learning_rate = 0.01
training_epochs = 10
batch_size = 100
display_step = 1

# tf Graph Input
x = tf.placeholder(tf.float32, [None, 784]) # mnist data image of shape 28*28=784
y = tf.placeholder(tf.float32, [None, 10]) # 0-9 digits recognition => 10 classes

# Set model weights
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# Construct model
pred = tf.nn.softmax(tf.matmul(x, W)+b) # Softmax

# Minimize error using cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))

W_grad =  - tf.matmul ( tf.transpose(x) , y - pred)
b_grad = - tf.reduce_mean( tf.matmul(tf.transpose(x), y - pred), reduction_indices=0)

new_W = W.assign(W - learning_rate * W_grad)
new_b = b.assign(b - learning_rate * b_grad)

init = tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples / batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # Fit training using batch data
            _, _, c = sess.run([new_W, new_b, cost], feed_dict={x: batch_xs, y: batch_ys})

            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if (epoch + 1) % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))

    # test
    acc=tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred,1),tf.argmax(y,1)),tf.float32))
    print('test acc',acc.eval({x: mnist.test.images, y: mnist.test.labels}))

    print("Optimization Finished!")
```
# 使用tf.gradients实现梯度下降
```js
# 使用随机梯度下降
vars=tf.trainable_variables()
vars_grad=tf.gradients(loss_op,vars)
vars_new=[]
for i in range(len(vars)):
    vars_new.append(vars[i].assign(vars[i]-learning_rate*vars_grad[i])) # 权重更新
sess.run(vars_new, feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.8})
```

minist:
```js
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./MNIST_data", one_hot=True)
# Parameters
learning_rate = 0.01
training_epochs = 10
batch_size = 100
display_step = 1

# Parameters
learning_rate = 0.01
training_epochs = 10
batch_size = 100
display_step = 1

# tf Graph Input
x = tf.placeholder(tf.float32, [None, 784]) # mnist data image of shape 28*28=784
y = tf.placeholder(tf.float32, [None, 10]) # 0-9 digits recognition => 10 classes

# Set model weights
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# Construct model
pred = tf.nn.softmax(tf.matmul(x, W)+b) # Softmax

# Minimize error using cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))

# W_grad =  - tf.matmul ( tf.transpose(x) , y - pred)
# b_grad = - tf.reduce_mean( tf.matmul(tf.transpose(x), y - pred), reduction_indices=0)
W_grad, b_grad=tf.gradients(cost,[W,b])

new_W = W.assign(W - learning_rate * W_grad)
new_b = b.assign(b - learning_rate * b_grad)

init = tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples / batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # Fit training using batch data
            _, _, c = sess.run([new_W, new_b, cost], feed_dict={x: batch_xs, y: batch_ys})

            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if (epoch + 1) % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))

    # test
    acc=tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred,1),tf.argmax(y,1)),tf.float32))
    print('test acc',acc.eval({x: mnist.test.images, y: mnist.test.labels}))

    print("Optimization Finished!")
```
# 使用tensorflow内置优化器
#### minimize
```js
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./MNIST_data", one_hot=True)

# Parameters
learning_rate = 0.01
training_epochs = 10
batch_size = 100
display_step = 1

# Parameters
learning_rate = 0.01
training_epochs = 10
batch_size = 100
display_step = 1

# tf Graph Input
x = tf.placeholder(tf.float32, [None, 784]) # mnist data image of shape 28*28=784
y = tf.placeholder(tf.float32, [None, 10]) # 0-9 digits recognition => 10 classes

# Set model weights
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# Construct model
pred = tf.nn.softmax(tf.matmul(x, W)+b) # Softmax

# Minimize error using cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))

# W_grad =  - tf.matmul ( tf.transpose(x) , y - pred)
# b_grad = - tf.reduce_mean( tf.matmul(tf.transpose(x), y - pred), reduction_indices=0)
# W_grad, b_grad=tf.gradients(cost,[W,b])
#
# new_W = W.assign(W - learning_rate * W_grad)
# new_b = b.assign(b - learning_rate * b_grad)
train_op=tf.train.AdamOptimizer().minimize(cost)

init = tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples / batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # Fit training using batch data
            # _, _, c = sess.run([new_W, new_b, cost], feed_dict={x: batch_xs, y: batch_ys})

            _,c=sess.run([train_op,cost],feed_dict={x: batch_xs, y: batch_ys})

            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if (epoch + 1) % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))

    # test
    acc=tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred,1),tf.argmax(y,1)),tf.float32))
    print('test acc',acc.eval({x: mnist.test.images, y: mnist.test.labels}))

    print("Optimization Finished!")
```
#### compute_gradients与apply_gradients
```js
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./MNIST_data", one_hot=True)

# Parameters
learning_rate = 0.01
training_epochs = 10
batch_size = 100
display_step = 1

# Parameters
learning_rate = 0.01
training_epochs = 10
batch_size = 100
display_step = 1

# tf Graph Input
x = tf.placeholder(tf.float32, [None, 784]) # mnist data image of shape 28*28=784
y = tf.placeholder(tf.float32, [None, 10]) # 0-9 digits recognition => 10 classes

with tf.variable_scope('D'):
    # Set model weights
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))

    # Construct model
    pred = tf.nn.softmax(tf.matmul(x, W)+b) # Softmax

# Minimize error using cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))

# W_grad =  - tf.matmul ( tf.transpose(x) , y - pred)
# b_grad = - tf.reduce_mean( tf.matmul(tf.transpose(x), y - pred), reduction_indices=0)
# W_grad, b_grad=tf.gradients(cost,[W,b])
#
# new_W = W.assign(W - learning_rate * W_grad)
# new_b = b.assign(b - learning_rate * b_grad)
# train_op=tf.train.AdamOptimizer().minimize(cost)
# optimizer=tf.train.AdamOptimizer()
# gradients=optimizer.compute_gradients(cost)
# clipped_gradients = [(tf.clip_by_value(_[0], -1, 1), _[1]) for _ in gradients] # _[0] 对应dw ,_[1]对应db
# train_op = optimizer.apply_gradients(clipped_gradients)
# 或
# train_op = optimizer.apply_gradients(gradients)

tvars = tf.trainable_variables()
d_params = [v for v in tvars if v.name.startswith('D/')]
trainerD = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5)
d_grads = trainerD.compute_gradients(cost, d_params)#Only update the weights for the discriminator network.
train_op = trainerD.apply_gradients(d_grads)

init = tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples / batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # Fit training using batch data
            # _, _, c = sess.run([new_W, new_b, cost], feed_dict={x: batch_xs, y: batch_ys})

            _,c=sess.run([train_op,cost],feed_dict={x: batch_xs, y: batch_ys})

            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if (epoch + 1) % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))

    # test
    acc=tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred,1),tf.argmax(y,1)),tf.float32))
    print('test acc',acc.eval({x: mnist.test.images, y: mnist.test.labels}))

    print("Optimization Finished!")
```
