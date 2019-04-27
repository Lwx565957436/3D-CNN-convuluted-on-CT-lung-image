# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 11:37:58 2019

@author: Administrator
"""

import pydicom
import matplotlib.pyplot as plt
import scipy.misc
import os
import numpy as np
import load_DICOM as ld
import cv2
import tensorflow as tf
PathDicom='G:\AHH\data\ct_lung_cancer\stage1'

def setDicomWinWidthWinCenter(img_data, winwidth, wincenter, rows, cols):
    img_temp = img_data
    img_temp.flags.writeable = True
    min = (2 * wincenter - winwidth) / 2.0 + 0.5
    max = (2 * wincenter + winwidth) / 2.0 + 0.5
    dFactor = 255.0 / (max - min)

            
    a=np.ones([rows,cols])
    img_temp=(img_temp-a*min)*dFactor

    min_index = img_temp < 0
    img_temp[min_index] = 0
    max_index = img_temp > 255
    img_temp[max_index] = 255

    return img_temp




label_list=ld.read_label('stage1_labels.csv')
        

def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


i=0
m=0
name=[]
writer=tf.python_io.TFRecordWriter('G:\\tfrecord\\.exchange_3_1.tfrecords')
for dirName, subdirList, fileList in os.walk(PathDicom):
    num=len(fileList)
    if num==0:
        change=fileList
    else:
        change=[]
        for k in range(100):
            n=round((k+1)/100*num)-1
            change.append(fileList[n])
    image_array=np.zeros([100,256,256],dtype='uint8')
    k=0
    for filename in change:
        in_path=os.path.join(dirName,filename)
        ds = pydicom.read_file(in_path,force=True)  #读取.dcm文件
        img = ds.pixel_array  # 提取图像信息
        img=setDicomWinWidthWinCenter(img, 1500, -400, 512,512)#窗位
        print('Now is',i)
        plt.show()
        img=cv2.resize(img,(256,256))
        image_array[k,:,:]=img
        i=i+1
        k +=1
    label=int(label_list[m])
    image_array=image_array.tostring()
    example=tf.train.Example(features=tf.train.Features(feature={'label':int64_feature(label),'image_raw':bytes_feature(image_array)}))
    if label==1:
        for l in range(15):
            writer.write(example.SerializeToString())
    else:
        writer.write(example.SerializeToString())
    m +=1
writer.close()
print('Transform done')





num=[]
i=0
for dirName, subdirList, fileList in os.walk(PathDicom):
    i=i+1
    num=np.append(num,[len(fileList)])
num=num[1:1596]
label=[]
for k in range(i):
    if label_list[k]==0:
        label=np.append(label,np.zeros((int(num[k]),1)))
    else:
        label=np.append(label,np.ones((int(num[k]),1)))
    if len(label)>96028:
        label=label[0:96028]
        break
label = [int(float(i)) for i in label]
label_list=label




def change(fileList):
    num=len(fileList)
    change=[]
    for i in range(100):
        n=round((i+1)/100*num)-1
        change.append(fileList[n])
    return change
        
 
    

