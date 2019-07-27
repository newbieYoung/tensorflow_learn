# coding:utf-8
# 基本程序结构

import tensorflow as tf

v_1 = tf.constant([1, 2, 3, 4], name='input1')
v_2 = tf.constant([2, 3, 4, 5], name='input2')

v_add = tf.add_n([v_1, v_2], name='add')

# 生成一个写日志的 writer，并将当前的计算图写入日志
writer = tf.summary.FileWriter('./log/demo4', tf.get_default_graph())
writer.close()

with tf.Session() as sess:
    print(sess.run(v_add))
