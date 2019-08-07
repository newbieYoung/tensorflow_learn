#coding:utf-8
# tensorflow 加载保存的模型

import tensorflow as tf

# 使用和保存模型代码中一样的方式来声明变量
v1 = tf.Variable(tf.constant(1.0, shape=[1]), name='v1')
v2 = tf.Variable(tf.constant(1.0, shape=[1]), name='v2')
result = v1 + v2

saver = tf.train.Saver()

with tf.Session() as sess:
    # 加载已经保存的模型，并通过已经保存的模型中的变量的值来计算加法
    saver.restore(sess, './model/demo11.ckpt')
    print(sess.run(result))

# 这段加载模型的代码基本和保存模型的代码是一样的，在加载模型的代码中也是先定义了 tensorflow 计算图上的所有运算
# 并声明了一个 tf.train.Saver 类；
# 两段代码唯一的不同是在加载模型的代码中没有运行变量的初始化过程，而是将变量的值通过已经保存的模型加载进来，如果不希望重复定义图上的运算，
# 也可以直接加载已经持久化的图，具体见 demo13。