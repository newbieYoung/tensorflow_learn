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

# 变量初始化
init_op = tf.global_variables_initializer()

# 生成模拟数据集
rdm = np.random.RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size,2)

# 0表示负样本、1表示正样本
Y = []
for [x1,x2] in X :
    Y.append([int(x1+x2 < 1)])

with tf.Session() as sess :

    sess.run(init_op)

    # 在训练前神经网络参数的值
    print(sess.run(w1))
    print(sess.run(w2))

    # 设定训练的轮数
    STEPS = 5000
    for i in range(STEPS) :
        # 每次选取 batch_size 个样本进行训练
        start = ( i * batch_size ) % dataset_size
        end = min(start+batch_size, dataset_size)

        # 通过选取的样本训练神经网络并更新参数
        sess.run(train_step, feed_dict={x_i:X[start:end], y_i:Y[start:end]})

        # 每隔一段时间计算在所有数据上的交叉熵并输出
        if i % 1000 == 0 :
            total_cross_entropy = sess.run(cross_entropy, feed_dict={x_i:X,y_i:Y})
            print('After %d training steps, cross entropy on all data is %g' % (i, total_cross_entropy))

    # 训练之后神经网络参数的值
    print(sess.run(w1))
    print(sess.run(w2))






