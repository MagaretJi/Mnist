# coding: utf-8
import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.examples.tutorials.mnist import input_data
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0' # 指定要使用的显卡的编号0-3
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True # 允许显存自适应增长
tf_config.gpu_options.per_process_gpu_memory_fraction = 1.0 # 最多使用30%显存，看情况设置
sess = tf.Session(config=tf_config)

weights={
    'wc1':tf.Variable(tf.truncated_normal([11,11,1,96],stddev=0.1)),
    'wc2':tf.Variable(tf.truncated_normal([5,5,96,256],stddev=0.1)),
    'wc3': tf.Variable(tf.random_normal([3, 3, 256, 384],stddev=0.1)),
    'wc4': tf.Variable(tf.random_normal([3, 3, 384, 384],stddev=0.1)),
    'wc5': tf.Variable(tf.random_normal([3, 3, 384, 256],stddev=0.1)),
    'wd1': tf.Variable(tf.random_normal([4*4*256,4096],stddev=0.1)),
    'wd2': tf.Variable(tf.random_normal([4096,4096],stddev=0.1)),
    'out': tf.Variable(tf.random_normal([4096,10],stddev=0.1)),
}
biases={
    'bc1': tf.Variable(tf.random_normal([96])),
    'bc2': tf.Variable(tf.random_normal([256])),
    'bc3': tf.Variable(tf.random_normal([384])),
    'bc4': tf.Variable(tf.random_normal([384])),
    'bc5': tf.Variable(tf.random_normal([256])),
    'bd1': tf.Variable(tf.random_normal([4096])),
    'bd2': tf.Variable(tf.random_normal([4096])),
    'out': tf.Variable(tf.random_normal([10])),
}


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

def norm(x,lsize=4):
    return tf.nn.lrn(x,lsize,bias=1.0,alpha=0.001/9.0,beta=0.75)

if __name__ == '__main__':
    # 读入数据
    mnist = input_data.read_data_sets('/home/margaret/Mnist/Data', one_hot=True)
    # x为训练图像的占位符、y_为训练图像标签的占位符
    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])
    print(mnist.train.labels)
    keep_prob = tf.placeholder(tf.float32)

    # 将单张图片从784维向量重新还原为28x28的矩阵图片
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    # 第一层卷积层
    W_conv1 =weights['wc1']
    b_conv1 =biases['bc1']
    conv1=conv2d(x_image, W_conv1) + b_conv1
    h_pool1 = max_pool_2x2(conv1)
    norm1=norm(h_pool1,lsize=4)

    # 第二层卷积层
    W_conv2 =weights['wc2']
    b_conv2 =biases['bc2']
    conv2 = conv2d(norm1,W_conv2) + b_conv2
    h_pool2 = max_pool_2x2(conv2)
    norm2=norm(h_pool2,lsize=4)

    #第三层卷积
    W_conv3 =weights['wc3']
    b_conv3 =biases['bc3']
    conv3 = conv2d(norm2, W_conv3) + b_conv3
    h_pool3 = max_pool_2x2(conv3)
    norm3=norm(h_pool3,lsize=4)

    #第四层卷积
    W_conv4 =weights['wc4']
    b_conv4 =biases['bc4']
    conv4 = conv2d(norm3, W_conv4) + b_conv4

    #第五层卷积
    W_conv5 =weights['wc5']
    b_conv5 =biases['bc5']
    conv5 = conv2d(conv4, W_conv5) + b_conv5
    h_pool5 = max_pool_2x2(conv5)
    norm5=norm(h_pool5,lsize=4)

    #全连接层1
    conv5_shape = norm5.get_shape().as_list()
    nodes1 = conv5_shape[1] * conv5_shape[2] * conv5_shape[3]  # 向量的长度为矩阵的长宽及深度的乘积
    h_pool1_flat = slim.flatten(norm5)
    W_fc1 = tf.get_variable('fc1_weights', shape=[nodes1, 4096], dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer(False))
    b_fc1 = biases['bd1']
    #h_pool1_flat = tf.reshape(norm5, [-1,weights['wd1'].get_shape().as_list()[0]])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool1_flat, W_fc1) + b_fc1)
    # 使用Dropout，keep_prob是一个占位符，训练时为0.5，测试时为1

    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    #全连接层2
    conv6_shape = h_fc1_drop.get_shape().as_list()
    nodes2 = conv6_shape[1]
    h_pool2_flat = slim.flatten(h_fc1_drop)
    W_fc2 = tf.get_variable('fc2_weights', shape=[nodes2, 4096], dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer(False))
    b_fc2 = biases['bd2']
    h_fc2 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc2) + b_fc2)
    # 使用Dropout，keep_prob是一个占位符，训练时为0.5，测试时为1
    h_fc1_drop = tf.nn.dropout(h_fc2, keep_prob)

    # 把1024维的向量转换成10维，对应10个类别
    W_out = weights['out']
    b_out = biases['out']
    y_conv = tf.matmul(h_fc1_drop, W_out) + b_out

    # 我们不采用先Softmax再计算交叉熵的方法，而是直接用tf.nn.softmax_cross_entropy_with_logits直接计算
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    # 同样定义train_step
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    # 定义测试的准确率
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # 创建Session和变量初始化
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    # 训练20000步
    for i in range(20000):
        batch = mnist.train.next_batch(50)
        # 每100步报告一次在验证集上的准确度
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
            print("step %d, training accuracy %g" % (i, train_accuracy))
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    # 训练结束后报告在测试集上的准确度
    print("test accuracy %g" % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))