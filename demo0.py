# coding:utf-8
# 简单的线性回归

import tensorflow as tf
import numpy as np

# 训练数据
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3

# 初始化线性模型参数
Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
biases = tf.Variable(tf.zeros([1]))

# 线性模型计算结果
y = Weights * x_data + biases

# 误差
loss = tf.reduce_mean(tf.square(y - y_data))

# 梯度下降
train = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

# 初始化变量
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for step in range(201):
        sess.run(train)
        if step % 20 == 0:
            print(step, sess.run(Weights), sess.run(biases))
