from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gc as gc
import time
import numpy as np
import tensorflow as tf
import gensim
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import collections
import cv2
from tensorflow.contrib.rnn import LSTMStateTuple

import vgg19
import utils

"""------------------------------set parameters------------------------------"""
batch_size = 50
image_dim = 512
hidden_size = 1024
word_dim = 512
vocab_size = 30000
keep_prob = 1
t_step = 16
lr = 1e-5
max_grad_norm = 5.0

"""------------------------------load annotations file-----------------------"""
dataDir='coco'
dataType='train2017'
annFile='{}/annotations/captions_{}.json'.format(dataDir,dataType)
#get annotations id list (total 591753  591750)
coco = COCO(annFile)
annIds = coco.getAnnIds()
anns = coco.loadAnns(annIds)[0:batch_size]# get annotations
data = np.array(anns).reshape([1, batch_size])

"""-------------------------------build vocabulary---------------------------"""
f = open("vocab.txt","r")
fr = f.read().replace(".", " .").replace(","," ,").replace(":"," :").split()
counter = collections.Counter(fr)
count_pairs = sorted(counter.items(), key = lambda x: (-x[1],x[0]))
words, values = list(zip(*count_pairs))
words = words[0:vocab_size]
word_to_id = dict(zip(words,range(len(words))))# list[word:id]

"""-------------------------------build graph--------------------------------"""
with tf.variable_scope("adam"):
    with tf.variable_scope("ATT_LSTM",reuse=tf.AUTO_REUSE):
        #build a word embedding
        embedding = tf.get_variable("embedding", [vocab_size, word_dim], initializer = tf.random_uniform_initializer(minval=-1.0, maxval=1.0))
        # MLP for generate h0 and c0
        wh = tf.get_variable('weight_hidden0', [image_dim, hidden_size], initializer = tf.contrib.layers.xavier_initializer())
        bh = tf.Variable(tf.zeros([hidden_size]), 'bias_hidden0')

        wc = tf.get_variable('weight_cell0', [image_dim, hidden_size], initializer = tf.contrib.layers.xavier_initializer())
        bc = tf.Variable(tf.zeros([hidden_size]), 'bias_cell0')
        #MLP for every region j, calculate the score e
        w_h2i = tf.get_variable('weight_hidden_2_image', [hidden_size, image_dim], initializer = tf.contrib.layers.xavier_initializer())
        b_h2i = tf.Variable(tf.zeros([image_dim]), 'bias_hidden_2_image')
        
        w_att = tf.get_variable('weight_att', [image_dim, 1], initializer = tf.contrib.layers.xavier_initializer())

        L0 = tf.get_variable('weight_L0', [word_dim, vocab_size], initializer = tf.contrib.layers.xavier_initializer())
        b_L0 = tf.Variable(tf.zeros([vocab_size]), 'bias_L0')
        
        Lh = tf.get_variable('weight_Lh', [hidden_size, word_dim], initializer = tf.contrib.layers.xavier_initializer())
        b_Lh = tf.Variable(tf.zeros([word_dim]), 'bias_Lh')
        
        Lz = tf.get_variable('weight_Lz', [image_dim, word_dim], initializer = tf.contrib.layers.xavier_initializer())
        b_Lz = tf.Variable(tf.zeros([word_dim]), 'bias_Lz')
        

        # set single lstm_cell, return state = (cell, hidden)
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, forget_bias = 0.0, state_is_tuple = True)
        #dropout dealing
        lstm = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=keep_prob)

        """-------------------get the feature map from CNN-------------------"""
        aL = tf.placeholder("float",[batch_size, 196, image_dim])
        
        """-----------------------get the targets----------------------------"""
        targets = tf.placeholder(tf.int32, [batch_size,t_step])
        
        """----calculate every [batch,hidden_size] h0 & c0 & weight_a--------"""

        a_ave = tf.reduce_mean(aL, 1)# [batch_size, image_dim]

        h0 = tf.nn.tanh(tf.matmul(a_ave, wh)+bh) #[batch_size, hidden_size]
        c0 = tf.nn.tanh(tf.matmul(a_ave, wc)+bc) #[batch_size, hidden_size]
        
        state = LSTMStateTuple(c0, h0)
        
        """--------------attention LSTM model processing---------------------"""
        outputs = []
        pre_sentence = []
        logit_pre = []
        loss = 0.0        
        for i in range(t_step):#each time step
            if i == 0:              
                word_v_last= tf.zeros([batch_size, word_dim], dtype = tf.float32)
                
                h_shape = tf.transpose([h0],[1,0,2]) # 1*batch_size*hidden_size - batch_size*1*hidden_size                                 
                h_ti = tf.tile(h_shape, multiples = [1, 196, 1]) # batch_size* 196 *hidden_size
                h2i = tf.reshape(tf.matmul(tf.reshape(h_ti,[-1,hidden_size]),w_h2i),[batch_size,196,image_dim])
                e = tf.reshape(tf.matmul(tf.reshape(tf.nn.relu(aL+h2i+b_h2i),[-1,image_dim]),w_att),[batch_size, 196])
                a = tf.nn.softmax(e)# [batch_size,196] weights for instance i
                context = tf.reduce_sum(aL * tf.expand_dims(a,2), 1)# [batch_size, image_dim] context vector                       
                z_t = tf.concat([context,word_v_last],1) # input vector [batch_size, image_dim+word_dim]

            if i > 0:
                tf.get_variable_scope().reuse_variables()

                word_v_last= tf.nn.embedding_lookup(embedding,next_words)
                
                h_shape = tf.transpose([h_last],[1,0,2]) # 1*batch_size*hidden_size - batch_size*1*hidden_size                                 
                h_ti = tf.tile(h_shape, multiples = [1, 196, 1]) # batch_size* 196 *hidden_size
                h2i = tf.reshape(tf.matmul(tf.reshape(h_ti,[-1,hidden_size]),w_h2i),[batch_size,196,image_dim])
                e = tf.reshape(tf.matmul(tf.reshape(tf.nn.relu(aL+h2i+b_h2i),[-1,image_dim]),w_att),[batch_size, 196])
                a = tf.nn.softmax(e)# [batch_size,196] weights for instance i
                context = tf.reduce_sum(aL * tf.expand_dims(a,2), 1)# [batch_size, image_dim] context vector                       
                z_t = tf.concat([context,word_v_last],1) # input vector [batch_size, image_dim+word_dim]

                 
            (hidden, state) = lstm(z_t, state)
            outputs.append(hidden)
            h_last = hidden # [batch_size,hidden_size]
            
            """--------------calculate the next word-------------------------"""
            
            h_n = hidden
            input_n = z_t[:, 0:image_dim]
            w_n = word_v_last
            arg1 = tf.matmul(h_n, Lh)+b_Lh
            arg2 = tf.matmul(input_n, Lz)+b_Lz
            prob = tf.matmul(tf.nn.tanh(arg1+arg2+w_n), L0)+b_L0 #[batch_size, vocab_size]
            next_words = tf.argmax(prob, 1) #[batch_size]
            loss += tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = targets[:, i],logits = prob))
            pre_sentence.append(next_words) #[t_step,batch_size] word id tensor                  
            logit_pre.append(prob)
        batch_loss = loss/batch_size 
        logit_pre = tf.concat([logit_pre], 0)# [t_step, batch_size,  vocab_size]
        outputs = tf.concat(outputs, 0)
        
        logits = tf.transpose(logit_pre[:,:,:], [1,0,2]) #[batch_size, t_step, vocab_size]

        
        """--------------------calculate and apply gradients optimize------------"""
        
        tvars = tf.trainable_variables()
        optimizer = tf.train.AdamOptimizer(lr)
        grads = tf.gradients(batch_loss, tvars)
        clipped_grads, _ = tf.clip_by_global_norm(grads, max_grad_norm)

    update = optimizer.apply_gradients(zip(clipped_grads, tvars))

    saver = tf.train.Saver()
    # tf.add_to_collection('pred_network', pre_sentence)

           
with tf.variable_scope("VGG"):
    images = tf.placeholder("float", [batch_size, 224, 224, 3])
    vgg = vgg19.Vgg19()
    vgg.build(images)

with tf.Session(config=tf.ConfigProto(gpu_options=(tf.GPUOptions(per_process_gpu_memory_fraction=0.7)))) as sess:
          """-------------------load a image with the name---------------------"""
          
          saver.restore(sess, "./model/model.ckpt-2")
          cur_batch_i = []
          for j in range(batch_size):
              cur_ann = data[0,j]#current annotation
              img_id = cur_ann['image_id']
              img = coco.loadImgs(img_id)[0]
              img_nm = img['file_name']#get image name
              v_img = utils.load_image("./coco/train2017/"+img_nm).reshape((1,224,224,3))
              cur_batch_i.append(v_img)
          images_input = np.concatenate(cur_batch_i, 0)

          """--------------------feature map processing--------------------"""

          feed_dict = {images: images_input}
          a = sess.run(vgg.conv5_4, feed_dict=feed_dict)
          aL_input = a.reshape([batch_size, 196, image_dim]) # wrap 2-D map to 1-D list
            
          """--------------------------session lstm------------------------"""
      
          feed_lstm = {aL: aL_input}
          prediction_logits = sess.run(logits, feed_dict=feed_lstm)

          """------------------------produce sentence----------------------"""
          predict_words = tf.argmax(tf.nn.softmax(prediction_logits), 2) #tensor : [batch_size, t_step]
          words_id = predict_words.eval() # change tensor to narray [batch,t_step]
          for j in range(batch_size):
              img_id = data[0,j]['image_id']
              img = coco.loadImgs(img_id)[0]
              img_nm = img['file_name']
              sentence_id =words_id[j]
              sentence = []
              for t in range(t_step):
                  cur_id = sentence_id[t]
                  cur_word = words[cur_id]
                  if cur_id != 1:
                      sentence.append(cur_word)
              sentence.append(".")
              for w in sentence:
                  print(w, end = " ")
              print("/"+str(img_nm)+" : "+data[0,j]['caption'])
                      
                  

      



