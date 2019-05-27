#coding:utf-8
# 简单前向传播

import tensorflow as tf

# 声明 w1 w2 两个变量，并赋予正态分布随机值
w1 = tf.Variable(tf.random_normal((2,3), stddev=1, seed=1), name="w1")
w2 = tf.Variable(tf.random_normal((3,1), stddev=1, seed=1), name="w2")

# 特征向量占位符（3x2 矩阵）
x = tf.placeholder(tf.float32, shape=(3,2), name="input")

#输出
y1 = tf.matmul(x,w1)
y = tf.matmul(y1,w2)

#变量初始化
init = tf.initialize_all_variables()

with tf.Session() as sess:
	sess.run(init)
	print(sess.run(y, feed_dict={x: [[0.7,0.9], [0.1,0.4], [0.5,0.8]]}))

