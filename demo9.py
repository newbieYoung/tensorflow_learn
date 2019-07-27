# coding:utf-8
# mnist 数字识别 神经网络模型 （隐藏层 + 学习速率指数衰减 + L2正则化 + 滑动平均模型）

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 载入 MNIST 数据集， 如果指定地址下没有已经下载好的数据，那么 Tensorflow 会自动下载
mnist = input_data.read_data_sets('/Users/young/Documents/MNIST_data/', one_hot=True)

# 输出 MNIST 数据集信息
# print('training data size : ', mnist.train.num_examples)
# print('testing data size : ', mnist.test.num_examples)
# print('example traning data :', mnist.train.images[0]) # 数字图片像素数据
# print('example traning data :', mnist.train.labels[0]) # 数字图片类别数据
print('--- mnist data ready! ---')

# MNIST 数据集相关的常数
INPUT_NODE = 784  # 输入层节点数（28 * 28 共 784 个像素）
OUTPUT_NODE = 10  # 输出层节点数（类别数目，因为要区分 0-9 这10个数字，因此这里的输出层节点数为10）

# 配置神经网络的参数
LAYER1_NODE = 500  # 隐藏层节点数（这里只加入一层有 500 个节点的隐藏层）
BATCH_SIZE = 100  # 单次训练数据量（小批量）
TRAINING_STEPS = 5000  # 训练轮数
LEARNING_RATE_BASE = 0.8  # 基础学习速率
LEARNING_RATE_DECAY = 0.99  # 学习率的衰减率
REGULARIZATION_RATE = 0.0001  # 模型复杂度的正则化项在损失函数中的系数
MOVING_AVERAGE_DECAY = 0.99  # 滑动平均衰减率


# 给定神经网络的输入和所有相关参数，计算神经网络的前向传播结果
def inference(input_tensor, avg_class, weight1, biases1, weight2, biases2):
    if avg_class == None:
        # 隐藏层前向传播结果
        y_1 = tf.nn.relu(tf.matmul(input_tensor, weight1)) + biases1  # relu 激活函数去线性化
        # 输出层前向传播结果
        y = tf.matmul(y_1, weight2) + biases2
    else:
        y_1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weight1)) + avg_class.average(biases1))
        y = tf.matmul(y_1, avg_class.average(weight2)) + avg_class.average(biases2)
    return y


# 多层神经网络模型
def train_model(reg, decay, average):
    # 输入
    x_i = tf.placeholder(tf.float32, shape=(None, INPUT_NODE), name='x-input')
    y_i = tf.placeholder(tf.float32, shape=(None, OUTPUT_NODE), name='y-input')

    # 隐藏层参数
    w1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))  # truncated_normal 正态分布产生函数
    b1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))

    # 输出层参数
    W = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
    b = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))

    # 训练轮数变量，这个变量不需要计算滑动平均值，所以这里指定这个变量为不可训练变量（trainable=False）
    # 在使用 Tensorflow 训练神经网络时，一般会将代表训练轮数的变量指定为不可训练的参数
    global_step = tf.Variable(0, trainable=False)

    # 神经网络前向传播
    y = inference(x_i, None, w1, b1, W, b)

    # 加入 滑动平均模型
    if average:
        # 根据滑动平均衰减率和训练轮数，初始化滑动平均类（训练轮数可以加快训练早期变量的更新速度）
        variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)

        # 在所有代表神经网络参数的变量上使用滑动平均模型，其它辅助变量则不需要
        # tf.trainable_variables 返回的就是图上集合 GraphKeys.TRAINABLE_VARIABLES 中的元素，这个集合的元素就是所有没有指定（trainable=False）的参数
        variable_averages_op = variable_averages.apply(tf.trainable_variables())
        y_average = inference(x_i, variable_averages, w1, b1, W, b)

        # 滑动平均不会改变变量本身的值，而是会维护一个影子变量来记录其滑动平均值；
        # 因此在使用滑动平均模型时也需要计算非滑动平均模型的神经网络前向传播结果，并使用该结果计算损失函数；
        # 但是计算正确率时需要使用滑动平均模型的神经网络前向传播结果。
    else:
        y_average = y

    # 损失函数
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_i, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    # 加入 L2 正则化
    if reg:
        regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)  # L2正则化损失函数
        regularization = regularizer(w1) + regularizer(W)  # 计算模型的L2正则化损失
        loss = cross_entropy_mean + regularization  # 总损失等于交叉熵损失和正则化损失
    else:
        loss = cross_entropy_mean

    # 加入 学习速率指数衰减
    if decay:
        learning_rate = tf.train.exponential_decay(
            LEARNING_RATE_BASE,  # 基础学习率
            global_step,  # 当前迭代轮数
            mnist.train.num_examples / BATCH_SIZE,  # 过完所有训练数据的迭代轮数
            LEARNING_RATE_DECAY  # 学习率衰减速度
        )
    else:
        learning_rate = LEARNING_RATE_BASE

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # 加入 滑动平均模型
    if average:
        # 在训练神经网络模型时，每过一遍数据既需要通过反向传播来更新神经网络中的参数，还需要更新每个参数的滑动平均值；
        # 为了一次完成多个操作，Tensorflow 提供了 tf.control_dependencies 和 tf.group 两种机制（相互等价）
        train_op = tf.group(train_step, variable_averages_op)
        # with tf.control_dependencies([train_step, variable_averages_op]):
        # train_op = tf.no_op(name='train')
    else:
        train_op = train_step

    # 模型评估
    correct_prediction = tf.equal(tf.argmax(y_average, 1), tf.argmax(y_i, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # 变量初始化
    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init_op)

        # 验证数据，一般在神经网络训练过程中会通过验证数据来大致判断停止条件和评判训练的结果
        validate_feed = {
            x_i: mnist.validation.images,
            y_i: mnist.validation.labels
        }

        # 测试数据，在真实的应用中，这部分数据在训练时时不可见的，这个数据只是作为模型优劣的最后评判标准
        test_feed = {
            x_i: mnist.test.images,
            y_i: mnist.test.labels
        }

        # 设定训练的轮数
        for i in range(TRAINING_STEPS):

            # 每次选取 batch_size 个样本进行训练
            xs, ys = mnist.train.next_batch(BATCH_SIZE)

            # 通过选取的样本训练神经网络并更新参数
            sess.run(train_op, feed_dict={x_i: xs, y_i: ys})

            # 每隔一段时间计算在验证数据上的交叉熵并输出
            # 当验证数据比较大时，需要将其划分为更小的 batch ，否则会导致计算时间过长甚至发生内存溢出
            if i % 1000 == 0:
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                print('After %d training steps, validation accuracy is %g' % (i, validate_acc))

        # 正确率
        print(sess.run(accuracy, feed_dict=test_feed))


# train_model(False, False, False)# 正确率 0.9787
train_model(True, True, True)
