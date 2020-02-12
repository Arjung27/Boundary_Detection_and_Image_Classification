"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Homework 0: Alohomora: Phase 2 Starter Code


Author(s):
Nitin J. Sanket (nitinsan@terpmail.umd.edu)
PhD Candidate in Computer Science,
University of Maryland, College Park
"""

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import sys
import numpy as np
# Don't generate pyc codes
sys.dont_write_bytecode = True

def random_flip(image):
    return tf.image.random_flip_left_right(image, seed=0)

def random_brightness(image):
    return tf.image.random_brightness(image, 1.0, seed=0)

def random_flip_up_down(image):
    return tf.image.random_flip_up_down(image, seed=0)

def image_rot90(image):
    return tf.image.rot90(image, k=1)

def CIFAR10Model(Img, ImageSize, MiniBatchSize):
    """
    Inputs: 
    Img is a MiniBatch of the current image
    ImageSize - Size of the Image
    Outputs:
    prLogits - logits output of the network
    prSoftMax - softmax output of the network
    """

    #############################
    # Fill your network here!
    #############################
    net = Img;
    net = tf.layers.conv2d(net, 32, kernel_size=[5,5], padding="SAME", name='conv_layer1')
    net = tf.layers.batch_normalization(inputs = net,axis = -1, training=True, center=True, scale=True, name='layer_bn1')
    net = tf.nn.relu(net, name='relu_layer1')
    net = tf.layers.conv2d(net, 64, kernel_size=[5,5], padding="SAME", name='conv_layer2')
    net = tf.layers.batch_normalization(inputs = net,axis = -1, training=True, center=True, scale=True, name='layer_bn2')
    net = tf.nn.relu(net, name='relu_layer2')
    # net = tf.nn.dropout(net, keep_prob=0.6)
    net = tf.layers.max_pooling2d(net, strides=2, pool_size=2, padding="SAME", name='max_pool1')
    net = tf.layers.conv2d(net, 32, kernel_size=[5,5], padding="SAME", name='conv_layer3')
    net = tf.layers.batch_normalization(inputs = net,axis = -1, training=True, center=True, scale=True, name='layer_bn3')
    net = tf.nn.relu(net, name='relu_layer3')
    # net = tf.nn.dropout(net, keep_prob=0.6)
    net = tf.layers.max_pooling2d(net, strides=2, pool_size=2, padding="SAME", name='max_pool2')
    # net = tf.reshape(net, [-1,1])
    net = tf.layers.flatten(net)
    net = tf.layers.dense(net, units=128, activation=tf.nn.relu, name='fully_connected1')
    # net = tf.nn.dropout(net, keep_prob=0.6)
    net = tf.layers.dense(net, units=256, activation=tf.nn.relu, name='fully_connected2')
    # net = tf.nn.dropout(net, keep_prob=0.6)
    net = tf.layers.dense(net, units=10, activation=None, name='last_layer')
    prLogits = net
    prSoftMax = tf.nn.softmax(logits=prLogits)

    return prLogits, prSoftMax

def concatenation(nodes):
    return tf.concat(nodes,axis=3)

def denseBlock(Img, number_layers, layer_name):
    with tf.name_scope(layer_name):
        layers = list()
        layers.append(Img)
        net = tf.layers.batch_normalization(inputs=Img, axis=-1, center=True, \
                                            scale=True, training=True, name=layer_name+'bn'+str(0))
        net = tf.nn.relu(net, name=layer_name+'_relu'+str(0))
        net = tf.layers.conv2d(net, filters=96, kernel_size=[1,1], padding='same', name=layer_name+'conv_layer'+str(1))
        net = tf.layers.batch_normalization(inputs =Img, axis=-1, center=True, \
                                            scale=True, training=True, name=layer_name+'bn'+str(1))
        net = tf.nn.relu(net, name=layer_name+'_relu'+str(1))
        # net = tf.nn.dropout(net, keep_prob=0.6)
        net = tf.layers.conv2d(net, filters=24, kernel_size=[3,3], padding='same', name=layer_name+'conv_layer'+str(2))
        layers.append(net)
        count = 2
        for i in range(number_layers-1):
            with tf.name_scope(layer_name+'_'+str(i)):
                net = concatenation(layers)
                net = tf.layers.batch_normalization(inputs=net, axis=-1, center=True, \
                                                scale=True, training=True, name =layer_name+'bn'+str(count))
                net = tf.nn.relu(net, name=layer_name+'_relu'+str(count))
                # net = tf.nn.dropout(net, keep_prob=0.6)
                count+=1
                net = tf.layers.conv2d(net, filters=96, kernel_size=[1,1], padding='same', name=layer_name+'conv_layer'+str(count))
                net = tf.layers.batch_normalization(inputs=net, axis=-1, center=True, \
                                                scale=True, training=True, name =layer_name+'bn'+str(count))
                net = tf.nn.relu(net, name=layer_name+'_relu'+str(count))
                # net = tf.nn.dropout(net, keep_prob=0.6)
                count+=1
                net = tf.layers.conv2d(net, filters=24, kernel_size=[3,3], padding='same', name=layer_name+'conv_layer'+str(count))

            layers.append(net)

        net = concatenation(net)
        return net


def DenseNet(Img, ImageSize, MiniBatchSize):
    """
    Inputs:
    Img is a MiniBatch of the current image
    ImageSize - Size of the Image
    Outputs:
    prLogits - logits output of the network
    prSoftMax - softmax output of the network
    """
    #Define Filter parameters for the first convolution layer block
    net = Img
    net = tf.layers.conv2d(net, filters=48, strides=2, kernel_size=[7,7], padding='same', name='conv_layer1')
    net = denseBlock(net, number_layers=4, layer_name='dense_block1')
    with tf.name_scope('outside_dense_block1'):
        net = tf.layers.batch_normalization(inputs = net,axis = -1, center = True, \
                                                scale = True, training=True, name='batch_norm1')
        net = tf.nn.relu(net, name='relu0')
        num_filters = net.shape[-1]
        net = tf.layers.conv2d(net, filters=12, kernel_size=[1,1], padding='same', name='conv_layer2')
        net = tf.layers.batch_normalization(inputs = net,axis = -1, center = True, \
                                                scale = True, training=True, name='batch_norm2')
        net = tf.nn.relu(net, name='relu2')
        # net = tf.nn.dropout(net, keep_prob=0.6)
        net = tf.layers.average_pooling2d(net, pool_size=[2,2], strides=2, padding='valid')

    net = denseBlock(net, number_layers=8, layer_name='dense_block2')
    with tf.name_scope('outside_dense_block2'):
        net = tf.layers.batch_normalization(inputs = net,axis = -1, center = True, \
                                                scale = True, training=True, name='batch_norm3')
        net = tf.nn.relu(net, name='relu3')
        num_filters = net.shape[-1]
        net = tf.layers.conv2d(net, filters=12, kernel_size=[1,1], padding='same', name='conv_layer4')
        net = tf.layers.batch_normalization(inputs = net,axis = -1, center = True, \
                                                scale = True, training=True, name='batch_norm4')
        net = tf.nn.relu(net, name='relu4')
        # net = tf.nn.dropout(net, keep_prob=0.6)
        net = tf.layers.average_pooling2d(net, pool_size=[2,2], strides=2, padding='valid')
    net = tf.layers.flatten(net)
    # net = tf.layers.dense(net, units=128, activation=tf.nn.relu, name='fully_connected1')
    net = tf.layers.dense(net, units=256, activation=tf.nn.relu, name='fully_connected2')
    net = tf.layers.dense(net, units=10, activation=None, name='last_layer')
    prLogits = net
    prSoftMax = tf.nn.softmax(logits=prLogits)

    return prLogits, prSoftMax

def resnet_block(net_input, filters, count, flag):
    net = tf.layers.conv2d(net_input, use_bias=False, filters=filters, kernel_size=[3,3], strides=1, padding='same', name='conv_layer'+str(count))
    net = tf.layers.batch_normalization(inputs=net, axis=-1, center=True, scale=True, training=True, name='batch_norm'+str(count))
    net = tf.nn.relu(net, name='relu'+str(count))
    net = tf.nn.dropout(net, keep_prob=0.6)
    net = tf.layers.conv2d(net, use_bias=False, filters=filters, kernel_size=[3,3], strides=1, padding='same', name='conv_layer'+str(count+1))
    net = tf.layers.batch_normalization(inputs=net, axis=-1, center=True, scale=True, training=True, name='batch_norm'+str(count+1))
    if flag == 1:
        net = tf.math.add(net, net_input)
    net = tf.nn.relu(net, name='relu'+str(count+1))
    net = tf.nn.dropout(net, keep_prob=0.6)

    return net

def ResNet(Img, ImageSize, MiniBatchSize):
    """
    Inputs: 
    Img is a MiniBatch of the current image
    ImageSize - Size of the Image
    Outputs:
    prLogits - logits output of the network
    prSoftMax - softmax output of the network
    """

    #############################
    # Fill your network here!
    #############################
    net = Img
    net = tf.layers.conv2d(net, use_bias=False, filters=64, kernel_size=[7,7], padding='same', name='conv_layer1')
    net = tf.layers.batch_normalization(inputs=net, axis=-1, center=True, scale=True, training=True, name='batch_norm1')
    net = tf.layers.max_pooling2d(net, strides=2, pool_size=2, padding="VALID", name='max_pool1')
    filters = 64
    count = 2
    flag = 1
    for i in range(11):
        if (i%4 == 0) and (i > 0):
            filters = filters*2
            flag = 0
        net = resnet_block(net, filters, count, flag)
        flag = 1
        count+=2
    net = tf.layers.average_pooling2d(net, pool_size=[2,2], strides=2, padding='valid')
    net = tf.layers.flatten(net)
    net = tf.layers.dense(net, units=1000, activation=tf.nn.relu, name='fully_connected1')
    net = tf.layers.dense(net, units=10, activation=None, name='last_layer')
    prLogits = net
    prSoftMax = tf.nn.softmax(logits=prLogits)

    return prLogits, prSoftMax

def residual_layer(Img, number_filters, layer_number):
    res_block = 1
    for i in range(res_block):
        layers = list()
        Img_ = Img
        with tf.name_scope('split'+str(layer_number)+str(i)):
            for j in range(5):
                with tf.name_scope('split'+str(layer_number)+str(i)+str(j)):
                    net_ = tf.layers.conv2d(Img_, filters=32, kernel_size=[1,1], strides=1, name='conv1' + 'split'+str(layer_number)+str(i)+str(j))
                    net_ = tf.layers.batch_normalization(inputs=net_, axis=-1, center=True, \
                                                scale=True, training=True, name='batch_norm1'+ 'split'+str(layer_number)+str(i)+str(j))
                    net_ = tf.nn.relu(net_, name='relu1'+ 'split'+str(layer_number)+str(i)+str(j))
                    net_ = tf.nn.dropout(net_, keep_prob=0.6)
                    net_ = tf.layers.conv2d(net_, filters=32, kernel_size=[1,1], strides=1, name='conv2' + 'split'+str(layer_number)+str(i)+str(j+1))
                    net_ = tf.layers.batch_normalization(inputs=net_, axis=-1, center=True, \
                                                scale=True, training=True, name='batch_norm2'+ 'split'+str(layer_number)+str(i)+str(j+1))
                    net_ = tf.nn.relu(net_, name='relu2'+ 'split'+str(layer_number)+str(i)+str(j+1))
                    net_ = tf.nn.dropout(net_, keep_prob=0.6)
                    Img_ = net_
                layers.append(net_)
        net_ = tf.concat(layers, axis=3)
        with tf.name_scope('transition'+str(layer_number)+str(i)):
            net_ = tf.layers.conv2d(net_, filters=16, kernel_size=[1,1], strides=1, name='conv' + 'transition'+str(layer_number)+str(i))
            net_ = tf.layers.batch_normalization(inputs=net_, axis=-1, center=True, \
                                                scale=True, training=True, name='batch_norm1'+ 'transition'+str(layer_number)+str(i))
    net_ = tf.nn.relu(net_ + Img)
    net_ = tf.nn.dropout(net_, keep_prob=0.6)
    return net_

def ResNext(Img, ImageSize, MiniBatchSize):
    """
    Inputs: 
    Img is a MiniBatch of the current image
    ImageSize - Size of the Image
    Outputs:
    prLogits - logits output of the network
    prSoftMax - softmax output of the network
    """

    #############################
    # Fill your network here!
    #############################
    net = Img
    net = tf.layers.conv2d(net, filters=16, name='conv_layer1', padding='same', kernel_size=[3,3], strides=1)
    net = tf.layers.batch_normalization(inputs=net, axis=-1, center=True, \
                                            scale=True, name='batch_norm1')
    net = tf.nn.relu(net, name='relu1')
    net = residual_layer(net, 32, 1)
    net = residual_layer(net, 128, 2)
    net = tf.layers.average_pooling2d(net, pool_size=[2,2], strides=2, padding='valid')
    net = tf.layers.flatten(net)
    net = tf.layers.dense(net, units=10, activation=None, name='last_layer')
    prLogits = net
    prSoftMax = tf.nn.softmax(logits=prLogits)

    return prLogits, prSoftMax
