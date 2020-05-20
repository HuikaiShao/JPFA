import glob
import os.path
import random
import numpy as np
import tensorflow as tf

import sys

sys.path.append("..")
import models.VGG16_net as model
import loss.M_loss as M_loss
import loss.H_loss as h_loss
import data_loader.data as data

batch_size = 30
omega_size = 20
capacity=1000+3*batch_size
results = []

def main():
    tf.reset_default_graph()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"       
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.65) 
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    logs_train_dir = './save_model/model.ckpt'      
    global_step = tf.Variable(0,trainable = False,name = "global_step")
    leaning_rate = tf.train.exponential_decay(0.001,global_step,100,0.96,staircase = False)
    leaning_rate1 = tf.train.exponential_decay(0.001,global_step,100,0.96,staircase = False)
    opt = tf.train.RMSPropOptimizer(0.0001,0.9)
    #---------------------------------------------------------------------------------------------------------------
    path_source = './SF.txt' #image path of source dataset       
    source_data, source_labels = data.read_image(path_source,10)
    source_size=len(source_data)
    source_batch,source_label_batch = data.get_batch(source_data,source_labels,source_size,batch_size,capacity,True)
    path_fake = './SF_GAN_LF.txt'#image path of fake dataset    
    fake_data, fake_labels = data.read_image(path_fake,10)
    fake_size=len(fake_data)
    fake_batch,fake_label_batch = data.get_batch(fake_data,fake_labels,fake_size,batch_size,capacity,True) 
    path_target = './LF.txt' #image path of target dataset     
    target_data, target_labels = data.read_image(path_target,10)
    target_size=len(target_data)
    target_batch,_ = data.get_batch(target_data,target_labels,target_size,batch_size,capacity,True)
    #---------------------------------------------------------------------------------------------------------------
    batch_source = tf.concat([source_batch, target_batch], 0)
    Map_source = model.Map(batch_source,'source_map', False, False)
    feature_source = model.FC(Map_source, 'source_ft',False,False) #source feature extractor
    source_feature, target_source_feature = tf.split(feature_source,[batch_size,batch_size],axis = 0) 

    batch_fake = tf.concat([fake_batch, target_batch], 0)
    Map_fake = model.Map(batch_fake,'fake_map', False, False)
    feature_fake = model.FC(Map_fake, 'fake_ft',False,False)  #fake feature extractor
    fake_feature, target_fake_feature = tf.split(feature_fake,[batch_size,batch_size],axis = 0)   
    #---------------------------------------------------------------------------------------------------------------
    source_sign = tf.sign(source_feature)
    source_hashing_loss = h_loss.hash_loss(source_feature,source_label_batch,batch_size,omega_size)
    source_Q_loss = tf.reduce_mean(tf.pow(tf.subtract(source_feature, source_sign), 2.0))
    source_loss = source_hashing_loss + 0.5 * source_Q_loss  # DHN loss in source dataset

    fake_sign = tf.sign(fake_feature)
    fake_hashing_loss = h_loss.hash_loss(fake_feature,fake_label_batch,batch_size,omega_size)
    fake_Q_loss = tf.reduce_mean(tf.pow(tf.subtract(fake_feature, fake_sign), 2.0))
    fake_loss = fake_hashing_loss + 0.5 * fake_Q_loss  # DHN loss in target dataset

    target_source_sign = tf.sign(target_source_feature)
    target_fake_sign = tf.sign(target_fake_feature) 
    t_s_Q_loss = tf.reduce_mean(tf.pow(tf.subtract(target_source_sign, target_source_feature), 2.0))
    t_f_Q_loss = tf.reduce_mean(tf.pow(tf.subtract(target_fake_sign, target_fake_feature), 2.0))
    Q_loss = 0.5 * (t_s_Q_loss + t_f_Q_loss)
    #---------------------------------------------------------------------------------------------------------------   
    mmd_t_s = M_loss.MMD_loss(target_source_feature, source_feature) # MK-MMD loss
    mmd_t_f = M_loss.MMD_loss(target_fake_feature, fake_feature)
    distance_loss = tf.reduce_mean(tf.abs(target_source_sign - target_fake_sign))  # consistency loss 
    #---------------------------------------------------------------------------------------------------------------     
    loss = mmd_t_s + mmd_t_f + source_loss + fake_loss + Q_loss  + 1.5*distance_loss # loss
    #--------------------------------------------------------------------------------------------------------------- 
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer_source = opt.minimize(loss,global_step = global_step)
   
    sess.run(tf.global_variables_initializer())
    t_vars = tf.trainable_variables()     
    saver = tf.train.Saver(t_vars,max_to_keep=1)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    print('start train_bottle')
    count=0000  
    try:
        for e in range(50000):
            if coord.should_stop():
                break
            count=count+1                                             
            _,loss_= sess.run([optimizer_source,loss])
            if( (e+1)%10 == 0):
                print("After %d training step(s),the loss is %g." % (e+1,loss_))                      
            if( (e+1)%5000 == 0):
                saver.save(sess,logs_train_dir,global_step=count)
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()


if __name__ == '__main__':
    main()
