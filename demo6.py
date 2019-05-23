#coding:utf-8
# 简单神经网络样例

import tensorflow as tf
import numpy as np

# 定义训练数据 batch 大小（小批量梯度下降）
batch_size = 8

# 定义神经网络的参数
w1 = tf.Variable(tf.random_normal([2,3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3,1], stddev=1, seed=1))

# 定义训练数据占位符
# 在shape的一个纬度上使用 None 可以方便使用不同的 batch 大小
x_i = tf.placeholder(tf.float32, shape=(None,2), name="x-input")
y_i = tf.placeholder(tf.float32, shape=(None,1), name="y-input")

# 定义神经网络前向传播过程
y_1 = tf.matmul(x_i, w1)
y_r = tf.matmul(y_1, w2)# y_result

# 归一化运算结果
y_r = tf.sigmoid(y_r)

# 计算真实值和预测值的交叉熵（这是分类问题中的一个常用损失函数）
cross_entropy = -tf.reduce_mean(y_i * tf.log(tf.clip_by_value(y_r, 1e-10, 1.0))
                                + (1-y_i) * tf.log(tf.clip_by_value(1-y_r, 1e-10, 1.0)))

# 定义优化方法
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)




