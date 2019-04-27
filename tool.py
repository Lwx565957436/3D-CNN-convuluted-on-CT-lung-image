# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 12:20:44 2019

@author: Administrator
"""


import SimpleITK as sitk
from PIL import Image
import pydicom
import numpy as np
import cv2

#load the image and output the nums of frames and the size
def loadfile(filename):
    ds=sitk.ReadImage(filename)
    img_array=sitk.GetArrayFromImage(ds)
    frame_num,width,height=img_array.shape
    return img_array,frame_num,width,height

#load the patients' information
def loadFileInformation(filename):
    information={}
    ds=pydicom.read_file(filename)
    information['PatientID']=ds.PatientID
    information['PatientName']=ds.PatientName
    information['PatientBirthDate']=ds.PatientBirthDate
    information['PatientSex']=ds.PatientSex
    information['StudyID']=ds.StudyID
    information['StudyDate']=ds.StudyDate
    information['StudyTime']=ds.StudyTime
    information['InstitutionName']=ds.InstitutionName
    information['Manufacturer']=ds.Manufacturer
    information['NumberOfFrames']=ds.NumberOfFrames
    return information

#show the image
def showImage(img_array,frame_num=0):
    img_bitmap=Image.fromarray(img_array[frame_num])
    return img_bitmap

#CLAHE optimize the graph
def limitedEqualize(img_array,limit=4.0):
    img_array_list=[]
    for img in img_array:
        clahe=cv2.createCLAHE(clipLimit=limit,tileGridSize=(8,8))
        img_array_list.append(clahe.apply(img))
    img_array_limited_equalized=np.array(img_array_list)
    return img_array_limited_equalized




import vtk
from vtk.util import numpy_support
import numpy
 
PathDicom = "./dir_with_dicom_files/"
reader = vtk.vtkDICOMImageReader()
reader.SetDirectoryName(PathDicom)
reader.Update()
 
# Load dimensions using `GetDataExtent`
_extent = reader.GetDataExtent()
ConstPixelDims = [_extent[1]-_extent[0]+1, _extent[3]-_extent[2]+1, _extent[5]-_extent[4]+1]
 
# Load spacing values
ConstPixelSpacing = reader.GetPixelSpacing()
 
# Get the 'vtkImageData' object from the reader
imageData = reader.GetOutput()
# Get the 'vtkPointData' object from the 'vtkImageData' object
pointData = imageData.GetPointData()
# Ensure that only one array exists within the 'vtkPointData' object
assert (pointData.GetNumberOfArrays()==1)
# Get the `vtkArray` (or whatever derived type) which is needed for the `numpy_support.vtk_to_numpy` function
arrayData = pointData.GetArray(0)
 
# Convert the `vtkArray` to a NumPy array
ArrayDicom = numpy_support.vtk_to_numpy(arrayData)
# Reshape the NumPy array to 3D using 'ConstPixelDims' as a 'shape'
ArrayDicom = ArrayDicom.reshape(ConstPixelDims, order='F')




 
PathDicom = "./dir_with_dicom_series/"
lstFilesDCM = []  # create an empty list
for dirName, subdirList, fileList in os.walk(PathDicom):
    for filename in fileList:
        if ".dcm" in filename.lower():  # check whether the file's DICOM
            lstFilesDCM.append(os.path.join(dirName,filename))
            
# Get ref file
RefDs = dicom.read_file(lstFilesDCM[0])
 
# Load dimensions based on the number of rows, columns, and slices (along the Z axis)
ConstPixelDims = (int(RefDs.Rows), int(RefDs.Columns), len(lstFilesDCM))
 
# Load spacing values (in mm)
ConstPixelSpacing = (float(RefDs.PixelSpacing[0]), float(RefDs.PixelSpacing[1]), float(RefDs.SliceThickness))
 
# The array is sized based on 'ConstPixelDims'
ArrayDicom = numpy.zeros(ConstPixelDims, dtype=RefDs.pixel_array.dtype)
 
# loop through all the DICOM files
for filenameDCM in lstFilesDCM:
    # read the file
    ds = dicom.read_file(filenameDCM)
    # store the raw image data
    ArrayDicom[:, :, lstFilesDCM.index(filenameDCM)] = ds.pixel_array
