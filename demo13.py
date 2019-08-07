#coding:utf-8
# tensorflow 直接加载持久化的图

import tensorflow as tf

saver = tf.train.import_meta_graph('./model/demo11.ckpt.meta')

with tf.Session() as sess:
    saver.restore(sess, './model/demo11.ckpt')
    # 通过张量的名称来获取张量
    print(sess.run(tf.get_default_graph().get_tensor_by_name('add:0')))

