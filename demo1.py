# coding:utf-8
# 会话拥有并管理程序运行时的所有资源

import tensorflow as tf

mat1 = tf.constant([[3, 3]])
mat2 = tf.constant([[2],
                    [2]])

product = tf.matmul(mat1, mat2)

# 模式1
# 当程序因为异常退出时，关闭会话的函数可能就不会被执行从而导致资源泄漏
# sess = tf.Session() # 创建会话
# print(sess.run(product)) # 执行运算
# sess.close() # 关闭会话

# 模式2
# 为了解决异常退出时资源释放的问题，可以使用Python的上下文管理器
# with tf.Session() as sess:
# 	print(sess.run(product))
# 不需要在再调用关闭会话函数了，当上下文退出时会话关闭和资源释放也自动完成了

# eval取值
# with tf.Session() as sess:
# 	print(product.eval())
# with tf.Session() as sess:
# 	print(product.eval(session=sess))
