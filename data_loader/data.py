import glob
import os.path
import random
import numpy as np
import tensorflow as tf


def read_image(path,num_per):
    data = []
    able = []
    datas = []
    lables=[]

    roi = open(path)
    roi_path = roi.readlines()
    classnum = len(roi_path)//num_per
    trainnum = 0
    for i,image_list in enumerate(roi_path):
        datas.append(image_list[:-1])
        lables.append(int(i/num_per))
        trainnum = trainnum +1

    lable = np.zeros([trainnum, classnum], np.int64)

    i = 0
    for label in lables:
        lable[i][label] = 1
        i = i + 1

    lables = lable.reshape([trainnum*classnum])

    return datas,lables
 

def get_batch(image, label,label_size,batch_size, Capacity,Shuffle):

    classnum=len(label)//len(image)
    image = tf.cast(image, tf.string)
    label = tf.convert_to_tensor(label,tf.int64)
    label = tf.reshape(label,[label_size,classnum])
    
    input_queue = tf.train.slice_input_producer([image, label],shuffle = Shuffle,capacity = Capacity)
    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents, channels=3)
    image = tf.image.resize_images(image, [224, 224])

    image_batch,label_batch= tf.train.batch([image,label],batch_size= batch_size,num_threads= 1, capacity = Capacity)
    
    label_batch = tf.cast(label_batch, tf.float32)
    image_batch = tf.cast(image_batch, tf.float32)
    
    return image_batch, label_batch