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
#with tf.variable_scope('bar', reuse=True):
#    v = tf.get_variable('v', [1])

# tensorflow 中 tf.variable_scope 函数是可以嵌套的，也就意味着命名空间也可以嵌套，当子命名空间不指定 reuse 参数时，这时 reuse 参数的
# 取值会和外层命名空间的 reuse 参数保持一致。
print('    ---    ')
with tf.variable_scope('root'):
    # 可以通过 tf.get_variable_scope 函数来获取当前命名空间中的 reuse 参数的取值
    print(tf.get_variable_scope().reuse)

    with tf.variable_scope('scope1'):
        print(tf.get_variable_scope().reuse)

    with tf.variable_scope('scope2', reuse=True):
        print(tf.get_variable_scope().reuse)

        with tf.variable_scope('scope2'):
            print(tf.get_variable_scope().reuse)

    print(tf.get_variable_scope().reuse)

print('    ---    ')
# 在命名空间中创建的变量名称都会带上这个命名空间名作为前缀
n1 = tf.get_variable('n', [1])
print(n1.name)

with tf.variable_scope('name1'):
    with tf.variable_scope('name2'):
        n2 = tf.get_variable('n', [1])
        print(n2.name)

    n3 = tf.get_variable('n', [1])
    print(n3.name)

# 创建一个名称为空的命名空间，并设置参数 reuse=True，然后直接通过带有命名空间前缀的完整变量名来获取其它命名空间下的变量
with tf.variable_scope('', reuse=True):
    n4 = tf.get_variable('name1/n', [1])
    print(n4 == n3)

    n5 = tf.get_variable('name1/name2/n', [1])
    print(n5 == n2)

    n6 = tf.get_variable('n', [1])
    print(n6 == n1)



