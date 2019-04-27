# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 09:48:18 2019

@author: Administrator
"""
import SimpleITK as sitk
import numpy as np
import os
from skimage import io 

#load the image and output the nums of frames and the size
def loadimage(filename):
    ds=sitk.ReadImage(filename)
    img_array=sitk.GetArrayFromImage(ds)
    frame_num,width,height=img_array.shape
    return img_array,frame_num,width,height


def loadfile_and_reshape(PathDicom):#reshape into 100 images
    img_array=[]
    for dirName, subdirList, fileList in os.walk(PathDicom):
        for filename in fileList:
            img,frames,width,height=loadimage(PathDicom+'/'+filename)
            img_array.append(img[0])
    
    nums=np.shape(img_array)[0]
    img_array_1=np.zeros((512,512,100))
    for i in range(100):
        no=round((i+1)*nums/100)-1
        img_array_1[:,:,i]=img_array[no]
    return img_array_1



import tensorflow as tf
img_width=512
img_geight=512

#def get_file(path_to_data1):#return the address of the files
#    images = []
#    nor_img = os.listdir(path_to_data1)
#    for nor in (nor_img):
#        images.append(os.path.join(path_to_data1,nor))
#    temp =np.array([images])
#    temp = temp.transpose()
#    np.random.shuffle(temp)
#    image_list = list(temp[:, 0])
#    return image_list

def get_file(path_to_data1):
    images = []
    labels = []
    nor_img = os.listdir(path_to_data1)
    for nor in (nor_img):
        images.append(os.path.join(path_to_data1,nor))
        letter=nor.split('.')[2]
        if letter=='0':
            labels = np.append(labels,0)
        else:
            labels = np.append(labels,1)
    temp =np.array([images,labels])
    temp = temp.transpose()
    np.random.shuffle(temp)
    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(float(i)) for i in label_list]
    return image_list, label_list

import csv
def read_label(labelfile):
    label_list=[]
    csv_file=csv.reader(open(labelfile,'r'))
    for data in csv_file:
        label_list.append(data[1])
    del label_list[0]
    label_list = [int(float(i)) for i in label_list]
    return label_list

def get_batch(image_list, label_list, img_width, img_height, batch_size, capacity):
    image = tf.cast(image_list, tf.string)
    label = tf.cast(label_list, tf.int32)

    input_queue = tf.train.slice_input_producer([image, label])

    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_png(image_contents, channels=3)

    image = tf.image.resize_image_with_crop_or_pad(image, 512, 512)
    image = tf.image.per_image_standardization(image)  # Standardize the image
    image_batch, label_batch = tf.train.batch([image, label], batch_size=batch_size, num_threads=64, capacity=capacity)
    label_batch = tf.reshape(label_batch, [batch_size])

    return image_batch, label_batch


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def convert_to_tfrecord(image_list,label_list,save_dir,name):
    filename=os.path.join(save_dir,name+'.tfrecords')
    n_samples=len(label_list)
    writer=tf.python_io.TFRecordWriter(filename)
    print('\nTransform start....')
    for i in np.arange(0,n_samples):
        image=loadfile_and_reshape(image_list[i])
        image_raw=image.tostring()
        label=int(label_list[i])
        example=tf.train.Example(features=tf.train.Features(feature={'label':int64_feature(label),'image_raw':bytes_feature(image_raw)}))
        writer.write(example.SerializeToString())
        print('This is over',i)
    writer.close()
    print('Transform done!')

def read_and_decode(tfrecords_file,batch_size):
    filename_queue=tf.train.string_input_producer([tfrecords_file])
    
    reader=tf.TFRecordReader()
    _,serialized_example=reader.read(filename_queue)
    img_features=tf.parse_single_example(serialized_example,features={'label':tf.FixedLenFeature([],tf.int64),'image_raw':tf.FixedLenFeature([],tf.string)})
    image=tf.decode_raw(img_features['image_raw'],tf.uint8)
    image=tf.reshape(image,[512,512,100])
    label=tf.cast(img_features['label'],tf.int32)
    image_batch,label_batch=tf.train.shuffle_batch([image,label],batch_size=batch_size,min_after_dequeue=100,num_threads=64,capacity=2048)
    return image_batch,tf.reshape(label_batch,[batch_size])

def onehot(labels):
    '''one-hot 编码'''
    n_sample = len(labels)
    n_class = 2
    onehot_labels = np.zeros((n_sample, n_class))
    onehot_labels[np.arange(n_sample), labels] = 1
    return onehot_labels


