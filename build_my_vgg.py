# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 15:30:50 2017

@author: zmj
"""

import tensorflow as tf
import skimage
import os
import numpy as np
import matplotlib.pyplot as plt
import skimage.io
import skimage.transform
import logging
import numpy as np
import time
logging.getLogger().setLevel(logging.INFO)
VGG_MEAN = [103.939, 116.779, 123.68]

class Model():
    def __init__(self,epoch,batch_size,cate_size,size,momentum_rate,learning_rate,l2_rate,file_path,check_dir,sess):
        self.epoch=epoch
        self.batch_size=batch_size
        self.label_size=cate_size
        self.img_size=size
        self.lr=learning_rate
        self.momentum=momentum_rate
        self.l2_params=l2_rate
        self.file_path=file_path
        self.sess=sess
        self.counter=0
        self.check_dir=check_dir
        temp=self.img_size/32*self.img_size/32*256
        self.w={
          'conv1_1w':tf.get_variable('w1',[3,3,3,64],initializer=tf.contrib.layers.xavier_initializer_conv2d()),
          'conv1_2w':tf.get_variable('w2',[3,3,64,64],initializer=tf.contrib.layers.xavier_initializer_conv2d()),
          'conv2_1w':tf.get_variable('w3',[3,3,64,128],initializer=tf.contrib.layers.xavier_initializer_conv2d()),
          'conv2_2w':tf.get_variable('w4',[3,3,128,128],initializer=tf.contrib.layers.xavier_initializer_conv2d()),
          'conv3_1w':tf.get_variable('w5',[3,3,128,256],initializer=tf.contrib.layers.xavier_initializer_conv2d()),
          'conv3_2w':tf.get_variable('w6',[3,3,256,256],initializer=tf.contrib.layers.xavier_initializer_conv2d()),
          'conv3_3w':tf.get_variable('w7',[3,3,256,256],initializer=tf.contrib.layers.xavier_initializer_conv2d()),
          'conv4_1w':tf.get_variable('w8',[3,3,256,512],initializer=tf.contrib.layers.xavier_initializer_conv2d()),
          'conv4_2w':tf.get_variable('w9',[3,3,512,512],initializer=tf.contrib.layers.xavier_initializer_conv2d()),
          'conv4_3w':tf.get_variable('w10',[3,3,512,512],initializer=tf.contrib.layers.xavier_initializer_conv2d()),
          'conv5_1w':tf.get_variable('w11',[3,3,512,256],initializer=tf.contrib.layers.xavier_initializer_conv2d()),
          'conv5_2w':tf.get_variable('w12',[3,3,256,256],initializer=tf.contrib.layers.xavier_initializer_conv2d()),
          'conv5_3w':tf.get_variable('w13',[3,3,256,256],initializer=tf.contrib.layers.xavier_initializer_conv2d()),
          'fc6w':tf.Variable(tf.random_normal([int(temp),4096])),
          'fc7w':tf.Variable(tf.random_normal([4096,1000])),
          'fc8w':tf.Variable(tf.random_normal([1000,2])),
          }
        self.w_names=[
          'conv1_1w',
          'conv1_2w',
          'conv2_1w',
          'conv2_2w',
          'conv3_1w',
          'conv3_2w',
          'conv3_3w',
          'conv4_1w',
          'conv4_2w',
          'conv4_3w',
          'conv5_1w',
          'conv5_2w',
          'conv5_3w',
          'fc6w',
          'fc7w',
          'fc8w']     
        self.b={
          'conv1_1b':tf.Variable(tf.zeros([64])),
          'conv1_2b':tf.Variable(tf.zeros([64])),
          'conv2_1b':tf.Variable(tf.zeros([128])),
          'conv2_2b':tf.Variable(tf.zeros([128])),
          'conv3_1b':tf.Variable(tf.zeros([256])),
          'conv3_2b':tf.Variable(tf.zeros([256])),
          'conv3_3b':tf.Variable(tf.zeros([256])),
          'conv4_1b':tf.Variable(tf.zeros([512])),
          'conv4_2b':tf.Variable(tf.zeros([512])),
          'conv4_3b':tf.Variable(tf.zeros([512])),
          'conv5_1b':tf.Variable(tf.zeros([256])),
          'conv5_2b':tf.Variable(tf.zeros([256])),
          'conv5_3b':tf.Variable(tf.zeros([256])),
          'fc6b':tf.Variable(tf.zeros([4096])),
          'fc7b':tf.Variable(tf.zeros([1000])),
          'fc8b':tf.Variable(tf.zeros([2])),
          }
        self.build(True)
    def _max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
            padding='SAME', name=name)

    def _conv_layer(self, bottom, name):
        with tf.variable_scope(name) as scope:
            filt = self.w[name+"w"]
            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.b[name+"b"]
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            return relu

    def _fc_layer(self, bottom, name):
        with tf.variable_scope(name) as scope:
            shape = bottom.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                 dim *= d
            x = tf.reshape(bottom, [-1, dim])

            weights = self.w[name+"w"]
            biases = self.b[name+"b"]

            # Fully connected layer. Note that the '+' operation automatically
            # broadcasts the biases.
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    # Input should be an rgb image [batch, height, width, 3]
    # values scaled [0, 1]
    def build(self, train=False):
        self.rgb=tf.placeholder(tf.float32,[None,self.img_size,self.img_size,3])
        self.y=tf.placeholder(tf.float32,[None,self.label_size])
        rgb_scaled = self.rgb * 255.0

        # Convert RGB to BGR
        red, green, blue = tf.split(3, 3, rgb_scaled)
        assert red.get_shape().as_list()[1:] == [224, 224, 1]
        assert green.get_shape().as_list()[1:] == [224, 224, 1]
        assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        bgr = tf.concat(3, [
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ])
        assert bgr.get_shape().as_list()[1:] == [224, 224, 3]

        self.relu1_1 = self._conv_layer(bgr, "conv1_1")
        self.relu1_2 = self._conv_layer(self.relu1_1, "conv1_2")
        self.pool1 = self._max_pool(self.relu1_2, 'pool1')

        self.relu2_1 = self._conv_layer(self.pool1, "conv2_1")
        self.relu2_2 = self._conv_layer(self.relu2_1, "conv2_2")
        self.pool2 = self._max_pool(self.relu2_2, 'pool2')

        self.relu3_1 = self._conv_layer(self.pool2, "conv3_1")
        self.relu3_2 = self._conv_layer(self.relu3_1, "conv3_2")
        self.relu3_3 = self._conv_layer(self.relu3_2, "conv3_3")
        self.pool3 = self._max_pool(self.relu3_3, 'pool3')

        self.relu4_1 = self._conv_layer(self.pool3, "conv4_1")
        self.relu4_2 = self._conv_layer(self.relu4_1, "conv4_2")
        self.relu4_3 = self._conv_layer(self.relu4_2, "conv4_3")
        self.pool4 = self._max_pool(self.relu4_3, 'pool4')

        self.relu5_1 = self._conv_layer(self.pool4, "conv5_1")
        self.relu5_2 = self._conv_layer(self.relu5_1, "conv5_2")
        self.relu5_3 = self._conv_layer(self.relu5_2, "conv5_3")
        self.pool5 = self._max_pool(self.relu5_3, 'pool5')

        self.fc6 = self._fc_layer(self.pool5, "fc6")
        assert self.fc6.get_shape().as_list()[1:] == [4096]

        self.relu6 = tf.nn.relu(self.fc6)
        if train:
            self.relu6 = tf.nn.dropout(self.relu6, 0.5)

        self.fc7 = self._fc_layer(self.relu6, "fc7")
        self.relu7 = tf.nn.relu(self.fc7)
        if train:
            self.relu7 = tf.nn.dropout(self.relu7, 0.5)

        self.fc8 = self._fc_layer(self.relu7, "fc8")
        self.prob = tf.nn.softmax(self.fc8, name="prob")
        
        l2_loss=0
        for i in range(16):
            print(self.w_names[i])
            print(self.w[self.w_names[i]])
            l2_loss+=self.l2_params*tf.nn.l2_loss(self.w[self.w_names[i]])
        self.loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y,logits=self.fc8))

    def load_image(self,path):
      # load image
      img = skimage.io.imread(path)
      img = img/ 255.0
      assert (0 <= img).all() and (img <= 1.0).all()
      # we crop image from center
      short_edge = min(img.shape[:2])
      yy = int((img.shape[0] - short_edge) / 2)
      xx = int((img.shape[1] - short_edge) / 2)
      crop_img = img[yy : yy + short_edge, xx : xx + short_edge]
      # resize to 224, 224
      resized_img = skimage.transform.resize(crop_img, (224, 224))
#      plt.imshow(resized_img)
#      plt.show()
      return resized_img  
    def load_imgs(self,paths):
        X=[]
        Y=[]
        
        for path in paths:
            X.append(self.load_image(path))
            if path[:3]=="cat":##cat label:[1,0],dog label:[0,1]
                Y.append([1,0])
            else:
                Y.append([0,1])
        return X,Y
        
    def train(self):
        optimizer=tf.train.MomentumOptimizer(self.lr,self.momentum).minimize(self.loss)
        try:
            self.sess.run(tf.global_variables_initializer())
        except:
            self.sess.run(tf.initialize_all_variables())
        all_files=[]
        for file_name in os.listdir(file_path):
            all_files.append(os.path.join(file_path,file_name)) 
        #all_idx=[i for i in range(len(all_files))]
        np.random.shuffle(all_files)
        batch_num=int(len(all_files)/self.batch_size)
        logging.info("Batch size:"+str(self.batch_size))
        logging.info("Batch num:"+str(batch_num))
        saver=tf.train.Saver()
        for i in range(self.epoch):
            l_all=[]
            for j in range(batch_num):
                X,Y=self.load_imgs(all_files[self.counter:min(self.counter+self.batch_size,len(all_files))])
                _,l=self.sess.run([optimizer,self.loss],feed_dict={self.rgb:X,self.y:Y})
                l_all.append(l)
                if j%100==0:
                    logging.info("Loss:"+str(np.mean(l_all)))
                self.counter+=self.batch_size
            saver.save(self.sess,self.check_dir+"model.ckpt",global_step=i+1)
            logging.info("Avg Loss:"+str(np.mean(l_all)))
            self.counter=0

cate=2
size=224
momentum_rate=0.9
learning_rate=0.01
l2_rate=0.0005
batch_size=16
epoch=1
file_path=u"model//Kaggle猫狗大战 540M//train//train"
check_dir="model//vgg_cat_dog_checkpoints//"
##强制使用cpu
#config = tf.ConfigProto(
#        device_count = {'GPU': 0}
#    )
#sess = tf.Session(config=config)
sess=tf.Session()
vgg=Model(epoch,batch_size,cate,size,momentum_rate,learning_rate,l2_rate,file_path,check_dir,sess)
start=time.time()
vgg.train()
logging.info("Time cost:"+str(time.time()-start))
