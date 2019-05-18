#coding:utf-8
# 基本程序结构

import tensorflow as tf

v_1 = tf.constant([1,2,3,4])
v_2 = tf.constant([2,3,4,5])

v_add = tf.add(v_1,v_2)

with tf.Session() as sess:
	print(sess.run(v_add))