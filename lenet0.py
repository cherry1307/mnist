

import tensorflow as tf
import os
from tensorflow.examples.tutorials.mnist import input_data#导入数据集


mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
sess = tf.InteractiveSession()


"""构建计算图"""

#1.正向传播
x = tf.placeholder("float", shape=[None, 784])  # 原始输入  28*28*1
y_ = tf.placeholder("float", shape=[None, 10])  # 目标值 10个类别

# 我们定义两个函数用于初始化
def weight_variable(shape):
    # 去掉过大偏离点的正太分布,stddev是正态分布的标准偏差
    initial = tf.truncated_normal(shape=shape, stddev=0.1)#随机初始化
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='VALID')

"""第一层卷积"""
W_conv1 = weight_variable([5, 5, 1, 6])
b_conv1 = bias_variable([6])
x_image = tf.reshape(x, [-1, 28, 28, 1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1)+ b_conv1)
'''第一层池化'''
h_pool1 = max_pool_2x2(h_conv1)

"""第二层卷积"""
W_conv2 = weight_variable([5, 5, 6, 12])
b_conv2 = bias_variable([12])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
'''第二层池化'''
h_pool2 = max_pool_2x2(h_conv2)

"""全连接层"""
W_fc1 = weight_variable([4*4*12, 192])
b_fc1 = bias_variable([192])
h_pool2_flat = tf.reshape(h_pool2, [-1, 4*4*12])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
"""使用Dropout减少过拟合"""
# 使用placeholder占位符来表示神经元的输出在dropout中保持不变的概率
# 在训练的过程中启用dropout，在测试过程中关闭dropout
keep_prob = tf.placeholder("float")#选择保留比例
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

"""输出层"""
W_fc2 = weight_variable([192, 10])
b_fc2 = bias_variable([10])

"""模型预测输出"""
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

#2.反向传播

"""交叉熵损失"""
cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
#cross_entropy = cross_entropy + tf.add_n(tf.get_collection('losses'))
# 模型训练,使用AdamOptimizer来做梯度最速下降,自适应的调整学习率
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
# 正确预测,得到True或False的List
correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y_conv, 1))
# 将布尔值转化成浮点数，取平均值作为精确度
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

BATCH_SIZE = 50
#MODEL_SAVE_PATH = 'model'
#MODEL_NAME = 'mnist_model'
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(2000):
        # 每次取50个样本进行训练
        batch = mnist.train.next_batch(BATCH_SIZE)

        if i%100 == 0:

            train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1],keep_prob: 1.0})  # 模型中间不使用dropout
            print("step %d, training accuracy %g" % (i, train_accuracy))
            #saver.save(sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME),global_step=global_step)#保存模型
        train_step.run(feed_dict={x:batch[0], y_:batch[1], keep_prob: 0.5})
    print("test accuracy %g" % accuracy.eval(feed_dict={
                x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))








