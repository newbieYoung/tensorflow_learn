#coding:utf-8
# mnist 数字识别 单层神经网络 + 隐藏层

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 载入 MNIST 数据集， 如果指定地址下没有已经下载好的数据，那么 Tensorflow 会自动下载
mnist = input_data.read_data_sets('/Users/young/Documents/MNIST_data/', one_hot=True)

# 输出 MNIST 数据集信息
#print('training data size : ', mnist.train.num_examples)
#print('testing data size : ', mnist.test.num_examples)
#print('example traning data :', mnist.train.images[0]) # 数字图片像素数据
#print('example traning data :', mnist.train.labels[0]) # 数字图片类别数据
print('--- mnist data ready! ---')

# MNIST 数据集相关的常数
INPUT_NODE = 784 # 输入层节点数（28 * 28 共 784 个像素）
OUTPUT_NODE = 10 # 输出层节点数（类别数目，因为要区分 0-9 这10个数字，因此这里的输出层节点数为10）

# 配置神经网络的参数
LAYER1_NODE = 500 # 隐藏层节点数
BATCH_SIZE = 100 # 单次训练数据量（小批量）
TRAINING_STEPS = 5000 # 训练轮数
LEARNING_RATE_BASE = 0.8 # 基础学习速率

# 单层神经网络模型
def train_model():

    with tf.name_scope('input'):
        # 输入
        x_i = tf.placeholder(tf.float32, shape=(None,INPUT_NODE), name='x-input')
        y_i = tf.placeholder(tf.float32, shape=(None,OUTPUT_NODE), name='y-input')

    with tf.name_scope('layer1'):
        # 隐藏层参数
        w1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))  # truncated_normal 正态分布产生函数
        b1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))

        y_1 = tf.nn.relu(tf.matmul(x_i, w1)) + b1  # relu 激活函数去线性化

    with tf.name_scope('layer2'):
        # 权重值 和 偏置量
        W = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1))  # truncated_normal 正态分布产生函数
        b = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))

        # 输出
        y = tf.matmul(y_1,W) + b

    with tf.name_scope('loss_function'):
        # 损失函数
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_i, 1))
        cross_entropy_mean = tf.reduce_mean(cross_entropy)

    with tf.name_scope('train_step'):
        # 优化方法
        train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE_BASE).minimize(cross_entropy_mean)

        # 模型评估
        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_i,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

        # 变量初始化
        init_op = tf.global_variables_initializer()

        with tf.Session() as sess :
            sess.run(init_op)

            # 设定训练的轮数
            for i in range(TRAINING_STEPS) :

                # 每次选取 batch_size 个样本进行训练
                start = (i * BATCH_SIZE) % mnist.train.num_examples
                end = min(start + BATCH_SIZE, mnist.train.num_examples)

                # 通过选取的样本训练神经网络并更新参数
                sess.run(train_step, feed_dict={x_i: mnist.train.images[start:end], y_i: mnist.train.labels[start:end]})

                # 每隔一段时间计算在所有训练数据上的交叉熵并输出
                # if i % 200 == 0 :
                #    total_cross_entropy = sess.run(cross_entropy, feed_dict={x_i:mnist.train.images,y_i:mnist.train.labels})
                #    print('After %d training steps, cross entropy on all data is %g' % (i, total_cross_entropy))

            # 正确率
            print(sess.run(accuracy, feed_dict={x_i: mnist.test.images, y_i: mnist.test.labels}))

    #将当前的计算图写入日志
    writer = tf.summary.FileWriter('./log/demo8-1', tf.get_default_graph())
    writer.close()

train_model()# 正确率 0.92左右






