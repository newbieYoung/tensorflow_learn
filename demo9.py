#coding:utf-8
# mnist 数字识别 神经网络模型 加入一层隐藏层

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
LAYER1_NODE = 500 # 隐藏层节点数（这里只加入一层有 500 个节点的隐藏层）
BATCH_SIZE = 100 # 单次训练数据量（小批量）
TRAINING_STEPS = 1000 # 训练轮数
LEARNING_RATE_BASE = 0.005 # 基础学习速率

# 多层神经网络模型
def train_model():
    # 输入
    x_i = tf.placeholder(tf.float32, shape=(None,INPUT_NODE), name='x-input')
    y_i = tf.placeholder(tf.float32, shape=(None,OUTPUT_NODE), name='y-input')

    # 隐藏层参数
    # w1 = tf.Variable(tf.zeros([INPUT_NODE, LAYER1_NODE]))
    # b1 = tf.Variable(tf.zeros([LAYER1_NODE]))
    w1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1)) # truncated_normal 正态分布产生函数
    b1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))

    # 输出层参数
    # W = tf.Variable(tf.zeros([LAYER1_NODE, OUTPUT_NODE]))
    # b = tf.Variable(tf.zeros([OUTPUT_NODE]))
    W = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
    b = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))

    # 隐藏层前向传播结果
    y_1 = tf.nn.relu(tf.matmul(x_i, w1)) + b1 # relu 激活函数去线性化

    # 输出层前向传播结果
    # y = tf.nn.softmax(tf.matmul(y_1, W) + b) # softmax 将神经网络向前传播得到的结果转换为概率分布
    y = tf.matmul(y_1, W) + b

    # 损失函数
    # cross_entropy = -tf.reduce_sum(y_i * tf.log(y)) # 交叉熵
    # cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_i, logits=y)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(y_i, 1), logits=y)

    # 优化方法
    train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE_BASE).minimize(cross_entropy)

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

train_model()# 正确率 0.9125






