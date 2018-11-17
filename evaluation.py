from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gc as gc
import time
import numpy as np
import tensorflow as tf
import gensim
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import collections
import cv2

import mnist_reader

#X_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')
img_size = int(len(X_test))
batch_size = 20
batch_len = int(img_size/batch_size)
test_data = np.array(X_test).reshape(batch_len,batch_size,-1)
test_label = np.array(y_test).reshape(batch_len,batch_size)

def weight(shape):
    initial = tf.truncated_normal(shape,stddev = 0.1)
    return tf.Variable(initial)

def bias(shape):
    initial = tf.constant(0.1,shape = shape)
    return tf.Variable(initial)

def conv_layer(ipt, flt):
    return tf.nn.conv2d(ipt,flt,strides = [1,1,1,1],padding='SAME')

def pooling(ipt):
    return tf.nn.max_pool(ipt,ksize = [1,2,2,1],strides = [1,2,2,1],padding = 'SAME')

"""------CNN------"""
ipt_imgs = tf.placeholder(tf.float32,[batch_size,784])
y_label = tf.placeholder(tf.int32,[batch_size])

input_images = tf.reshape(ipt_imgs,[batch_size,28,28,1])

w1 = weight([3,3,1,32])
b1 = bias([32])
conv1 = tf.nn.relu(conv_layer(input_images,w1)+b1)
pool1 = pooling(conv1)

w2 = weight([3,3,32,64])
b2 = bias([64])
conv2 = tf.nn.relu(conv_layer(pool1,w2)+b2)
pool2 = pooling(conv2)

wfc = weight([7*7*64,1024])
bfc = bias([1024])
out_fc = tf.nn.relu(tf.matmul(tf.reshape(pool2,[batch_size,7*7*64]),wfc)+bfc)

w_y = weight([1024,10])
b_y = bias([10])
y_conv = tf.matmul(out_fc,w_y)+b_y

loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y_label,logits = y_conv))
optimizor = tf.train.AdamOptimizer(1e-4).minimize(loss)

y_pre = tf.argmax(y_conv,1)

init = tf.global_variables_initializer()
saver = tf.train.Saver()
"""---evaluation---"""

with tf.Session(config=tf.ConfigProto(gpu_options=(tf.GPUOptions(per_process_gpu_memory_fraction=0.7)))) as sess:

    saver.restore(sess, "./model/model.ckpt-9")
    correct_num = 0
    for i in range(batch_len):
        feed_cnn = {ipt_imgs: test_data[i], y_label: test_label[i]}
        Y_PRE = sess.run([y_pre],feed_dict=feed_cnn)
        for j in range(batch_size):
            if Y_PRE[0][j] == test_label[i][j]:
                correct_num+=1
    result = correct_num/10000
    print("model accuracy on test dataset is: "+str(result))
            
                
            
            

