#coding:utf-8
# 变量

import tensorflow as tf

state = tf.Variable(0,name='counter')
one = tf.constant(1)

new_value = tf.add(state, one)
print(new_value)
# 张量
# Tensor("Add:0", shape=(), dtype=int32)

update = tf.assign(state, new_value)# 赋值 state = new_value

#init = tf.initialize_all_variables()# 变量初始化

with tf.Session() as sess:

	# 可以使用 initialize_all_variables 初始化全部变量
	#sess.run(init)

	# 或者单独初始化
	sess.run(state.initializer)

	for _ in range(3):
		sess.run(update)
		print(sess.run(state))

