#coding:utf-8
# 会话

import tensorflow as tf

mat1 = tf.constant([[3,3]])
mat2 = tf.constant([[2],
					[2]])

product = tf.matmul(mat1,mat2)

# method 1
sess = tf.Session()
result = sess.run(product)
print(result)
sess.close()

# method 2
with tf.Session() as sess:
	result2 = sess.run(product)
	print(result2)
