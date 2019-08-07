# coding:utf-8
# tf基本运算

import tensorflow as tf

v1 = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
v2 = tf.constant([[1.0, 2.0], [3.0, 4.0]])
v3 = tf.constant([[5.0, 6.0], [7.0, 8.0]])

# tf.clip_by_value 将一个张量中的数值限制在一个范围之内
clip = tf.clip_by_value(v1, 2.5, 4.5)

# tf.log 对张量中所有元素依次求对数
log = tf.log(v1)

# 两个矩阵通过 * 相乘是元素之间的相乘
ele_mul = v2 * v3

# tf.matmul 矩阵相乘
mat_mul = tf.matmul(v2, v3)

mean_1 = tf.reduce_mean(v1)
mean_2 = tf.reduce_mean(v1, 0)
mean_3 = tf.reduce_mean(v1, 1)

with tf.Session() as sess:
    print(sess.run(mean_1))
    print(sess.run(mean_2))
    print(sess.run(mean_3))
    print(sess.run(log))
    print(sess.run(clip))
    print(sess.run(ele_mul))
    print(sess.run(mat_mul))
