#coding:utf-8
# tensorflow 变量管理

import tensorflow as tf

# 在命名空间 foo 内创建变量 v
with tf.variable_scope('foo'):
    v = tf.get_variable('v', [1], initializer=tf.constant_initializer(1.0))

# 因为在命名空间 foo 中已经存在变量 v，所以再次创建时会报错
# Variable foo/v already exists, disallowed.
#with tf.variable_scope('foo'):
#    v = tf.get_variable('v', [1])

# 在使用命名空间时，将参数 reuse 设置为 True ，这样 tf.get_variable 函数将直接获取已经声明的变量
with tf.variable_scope('foo', reuse=True):
    v1 = tf.get_variable('v',[1])
    print(v1 == v) # 输出为 True，代表 v,v1 是相同的 tensorflow 中的变量

# 将参数 reuse 设置为 True时，tf.variable_scope 将只能获得已经创建过的变量，因为在命名空间 bar 中还没有创建变量 v，所以以下代码将会报错
# Variable bar/v does not exist, or was not created with tf.get_variable()
with tf.variable_scope('bar', reuse=True):
    v = tf.get_variable('v', [1])
