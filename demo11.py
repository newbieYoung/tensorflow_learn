#coding:utf-8
# tensorflow 模型持久化

import tensorflow as tf

# 声明两个变量并计算它们的和
v1 = tf.Variable(tf.constant(1.0, shape=[1]), name='v1')
v2 = tf.Variable(tf.constant(1.0, shape=[1]), name='v2')
result = v1 + v2
print(result) #输出张量

init_op = tf.global_variables_initializer()
# 声明 tf.train.Saver 类用来保存模型
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init_op)
    print(sess.run(result))
    # 将模型保存到 ./model/demo11.ckpt 文件
    saver.save(sess, './model/demo11.ckpt')