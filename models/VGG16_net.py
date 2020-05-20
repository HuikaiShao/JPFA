import tensorflow as tf

import sys

sys.path.append("..")
from models import vgg16
    
#---------------------------------------------------------------------------------------------------------------
def Map(inputs, scope='map',is_training=True, reuse=False):
    
    vgg = vgg16.Vgg16()
    vgg.build(inputs)
    net = vgg.pool4

    with tf.variable_scope(scope,reuse = reuse) as scope:
        net = conv(net,[3, 3, 512, 512],[1, 1, 1, 1],is_training,reuse,padding='SAME',scope='conv1')
        
        net = conv(net,[3, 3, 512, 512],[1, 1, 1, 1],is_training,reuse,padding='SAME',scope='conv2')
        
        net = conv(net,[3, 3, 512, 512],[1, 1, 1, 1],is_training,reuse,padding='SAME',scope='conv3')

        net = tf.nn.max_pool(net,[1,2,2,1],[1,2,2,1],'SAME')
    return net
#---------------------------------------------------------------------------------------------------------------
def FC(inputs, scope='FC', is_training=False, reuse=False):
    net = inputs
    net_shape = net.get_shape().as_list()
    shape = net_shape[1]*net_shape[2]*net_shape[3]
    net = tf.reshape(net,[net_shape[0],shape])
    with tf.variable_scope(scope,reuse = reuse) as scope:
        net = FullyConnected(net,4096,Training = is_training,Reuse = reuse,scope='fc1')
        net = FullyConnected(net,4096,Training = is_training,Reuse = reuse,scope='fc2')
        net = FullyConnected(net,2048,Training = is_training,Reuse = reuse,scope='fc3')
        net = FullyConnected(net,128,last=True,Training = is_training,Reuse = reuse,scope='fc4')
    return net
#---------------------------------------------------------------------------------------------------------------
def conv(input,kernel_shape,strides,Training = True,Reuse = False,alpha=0.2,padding='SAME',scope='con'):
    net =input
    with tf.variable_scope(scope,reuse = Reuse) as scope:
        weight1 = tf.get_variable('weight',kernel_shape,tf.float32,tf.glorot_uniform_initializer())
        bias1 = tf.get_variable('bias',kernel_shape[3],tf.float32,tf.zeros_initializer())
        conv1 = tf.nn.conv2d(input=net, filter=weight1, strides=strides, padding=padding)
        if Training:
            mean, variance = tf.nn.moments(conv1, [0, 1, 2])
            net = tf.nn.batch_normalization(conv1, mean, variance, bias1, None, 1e-5)
            net = tf.nn.relu(net)              
        else:
            net = tf.nn.bias_add(conv1,bias1)
            net = tf.nn.relu(net)            
    return net

regularizer = tf.contrib.layers.l2_regularizer(0.0005)
def FullyConnected(input_tensor,output_dim,last=False, Training = True,Reuse = False,alpha=0.2,scope=''):
    net = input_tensor
    with tf.variable_scope(scope,reuse = Reuse) as scope:
        num_batch, input_dim = net.get_shape()
        weight = tf.get_variable('weight',[input_dim.value, output_dim],tf.float32,initializer=tf.truncated_normal_initializer(stddev=0.1))
        bias = tf.get_variable('bias',[output_dim],tf.float32,tf.zeros_initializer())
        if last:
            net = tf.matmul(net,weight) + bias
            net = tf.nn.tanh(net)            
        else:
            net = tf.matmul(net, weight)# + bias
            mean, variance = tf.nn.moments(net, [0, 1])
            net = tf.nn.batch_normalization(net, mean, variance, bias, None, 1e-5)
            net = tf.nn.relu(net) 
            
        if Training: 
            net = tf.nn.dropout(net,0.5)
            tf.add_to_collection('losses',regularizer(weight))

    return net

