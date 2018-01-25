# -*- coding: utf-8 -*-
# author: itimor

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import os


def main(mnist, isTrain):
    # 定义输入变量
    x = tf.placeholder(tf.float32, [None, 784])
    # 定义参数
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    # 定义激励函数
    y = tf.nn.softmax(tf.matmul(x, W) + b)
    # 定义输出变量
    y_ = tf.placeholder(tf.float32, [None, 10])
    # 定义成本函数
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    # 定义优化函数
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    # 初始化变量
    init = tf.global_variables_initializer()
    # 定义会话
    sess = tf.Session()
    with tf.Session() as sess:
        # 运行初始化
        sess.run(init)
        # 定义模型保存对象
        saver = tf.train.Saver()
        # 创建模型保存目录
        model_dir = "saver/mnist_graph"
        model_name = "ckpt"
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)

        tf.add_to_collection('x', x)
        tf.add_to_collection('y', y)
        if isTrain:
            # 循环训练1000次
            for i in range(1000):
                batch_xs, batch_ys = mnist.train.next_batch(100)
                sess.run(train_step, feed_dict={x: batch_xs, y_:batch_ys})
            print("训练完成！")
            # 保存模型
            saver.save(sess, os.path.join(model_dir, model_name))
            print("保存模型成功！")
        else:
            # 恢复模型
            new_saver = tf.train.import_meta_graph('saver/mnist_graph/ckpt.meta')
            new_saver.restore(sess, os.path.join(model_dir, model_name))
            x = tf.get_collection('x')[0]
            y = tf.get_collection('y')[0]

            print("恢复模型成功！")
            # 取出一个测试图片
            idx = 60
            img = mnist.test.images[idx]
            # 根据模型计算结果
            ret = sess.run(y, feed_dict={x: img.reshape(1, 784)})
            print("计算模型结果成功！")
            # 显示测试结果
            print("预测结果:%d" % (ret.argmax()))
            print("实际结果:%d" % (mnist.test.labels[idx].argmax()))


if __name__ == '__main__':
    # 导入mnist数据库
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    # isTrain为True时表示训练，为False时表示测试
    isTrain = False
    main(mnist,isTrain)