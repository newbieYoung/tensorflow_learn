# coding:utf-8
# 滑动平均模型

# Vt = decay * V(t-1) + (1-decay) * Qt
# decay = min(decay, (1+num/10+num))

import tensorflow as tf

# 定义一个变量用于计算滑动平均值
v1 = tf.Variable(0, dtype=tf.float32)

# 定义step用于模拟神经网络中的迭代轮数
step = tf.Variable(0, trainable=False)

# 定义滑动平均类并设置衰减率为0.99
ema = tf.train.ExponentialMovingAverage(0.99, step)

# 定义滑动平均更新操作，这里需要给定一个列表，每次执行该操作，列表中的变量都会被更新
maintain_averages_op = ema.apply([v1])

with tf.Session() as sess:
    # 初始化所有变量
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    # 刚开始变量 v1 和其滑动平均值 均为 0
    print(sess.run([v1, ema.average(v1)]))

    # 更新变量 v1 的值为 5
    sess.run(tf.assign(v1, 5))
    # 衰减率 decay = min(0.99, (1+0)/(10+0)) = 0.1
    # 变量 v1 的滑动平均值为 0.1*0 + (1-0.1)*5 = 4.5
    sess.run(maintain_averages_op)
    print(sess.run([v1, ema.average(v1)]))

    # 更新变量 v1 的值为 10
    sess.run(tf.assign(v1, 10))
    # 更新变量 step 值为 10000
    sess.run(tf.assign(step, 10000))
    # 衰减率 decay = min(0.99, (1+10000)/(10+10000)) = 0.99
    # 变量 v1 的滑动平均值为 0.99*4.5 + 0.01*10 = 4.555
    sess.run(maintain_averages_op)
    print(sess.run([v1, ema.average(v1)]))

    # 以此类推