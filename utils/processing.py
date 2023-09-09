# Copyright 2019 NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

import numpy as np
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

from scipy.interpolate import interp2d
import scipy.io


logging.basicConfig(level=logging.DEBUG,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

NPY_DIR2 = 'npy-PS44'

INPUT_1 = 'R'
INPUT_2 = 'G'
INPUT_3 = 'B'

OUTPUT = 'ts'

def process_llamas(image_dir):

    ## PRe PRE procesamiento-------------------------------------------------------#
    x1 = np.load(os.path.abspath(os.path.join(NPY_DIR2, INPUT_1 + '.npy')))[20:,:,:]
    x2 = np.load(os.path.abspath(os.path.join(NPY_DIR2, INPUT_2 + '.npy')))[20:,:,:]
    x3 = np.load(os.path.abspath(os.path.join(NPY_DIR2, INPUT_3 + '.npy')))[20:,:,:]

    x1 = x1[:,:,:32]
    x2 = x2[:,:,:32]
    x3 = x3[:,:,:32]

    for i in range(len(x2)):
        x_max = np.max([x1[i].max(),x2[i].max(),x3[i].max()])
        x2[i] = x2[i][::-1]/x_max
        x3[i] = x3[i][::-1]/x_max
        x1[i] = x1[i][::-1]/x_max
        
    x1_mean = np.mean(x1[:].mean())
    x1_std = np.mean(x1[:].std())

    x2_mean = np.mean(x2[:].mean())
    x2_std = np.mean(x2[:].std())

    x3_mean = np.mean(x3[:].mean())
    x3_std = np.mean(x3[:].std())

    ## autentica carga de datos-----------------------------------------------------#

    ss = cv2.imread(image_dir,cv2.IMREAD_UNCHANGED)

    ss_new = np.zeros((600,2048,3))
    ss_new[:,:,0] = ss[800:1400,:, 0]
    ss_new[:,:,1] = ss[800:1400,:, 1]
    ss_new[:,:,2] = ss[800:1400,:, 2]

    R_exp=np.zeros_like(ss_new)
    BG = 25.318288541666668
    CONST1 = 0.24409398022534526
    R_exp[:,:,0]=((ss_new[:,:,2]- BG))*CONST1

    CONST2 = 0.6647056891885273
    R_exp[:,:,1]=((ss_new[:,:,1]- BG))*CONST2

    CONST3 = 1.7375992007958856
    R_exp[:,:,2]=((ss_new[:,:,0]- BG))*CONST3

    Py_rot = np.zeros((2048,1536,3))
    Py_rot[:,:,0] = np.rot90(ss[:,:, 0], k=1, axes=(0, 1))
    Py_rot[:,:,1] = np.rot90(ss[:,:, 1], k=1, axes=(0, 1))
    Py_rot[:,:,2] = np.rot90(ss[:,:, 2], k=1, axes=(0, 1))
    r_x = int(150)
    r1 = int(1000)
    r2 = int(1140)
    h_px = int(1300)
    m =  np.where(Py_rot[h_px,r1:r2,1] == Py_rot[h_px,r1:r2,1].min())[0][0]
    center_x = r1 + m
    border_x  = center_x + r_x
    
    Py_rgb  = np.zeros((3,2048,border_x - center_x)) 
    Py_rgb[0,:,:] = Py_rot[:,center_x:border_x, 0]
    Py_rgb[1,:,:] = Py_rot[:,center_x:border_x, 1]
    Py_rgb[2,:,:] = Py_rot[:,center_x:border_x, 2]

    scale = 37294.15914879467
    offset_z = -0.3/100
    nx = r_x     
    Zmin = int(2048)  # initial height to consider
    Zmax = 0  # max height to consider
    offset_z = -0.3/100
    nz = Zmin - Zmax

    r_exp = np.linspace(0, nx - 1, nx) / scale
    z_exp = np.linspace(Zmin - Zmax, 0, nz) / scale + offset_z

    Py_exp_interp = np.empty((3,128,32))

    r, z, Py_exp_interp[0,:,:] = resize_temp(r_exp, z_exp, Py_rgb[0,:,:])
    r, z, Py_exp_interp[1,:,:] = resize_temp(r_exp, z_exp, Py_rgb[1,:,:])
    r, z, Py_exp_interp[2,:,:] = resize_temp(r_exp, z_exp, Py_rgb[2,:,:])

    x_max = np.max([Py_exp_interp])
    Py_exp_interp[0,:,:] = Py_exp_interp[0,:,:][::-1]/x_max
    Py_exp_interp[1,:,:] = Py_exp_interp[1,:,:][::-1]/x_max
    Py_exp_interp[2,:,:] = Py_exp_interp[2,:,:][::-1]/x_max

    Py_exp_interp[0,:,:] = standarize(Py_exp_interp[0,:,:] , x1_mean, x1_std)
    Py_exp_interp[1,:,:] = standarize(Py_exp_interp[1,:,:] , x2_mean, x2_std)
    Py_exp_interp[2,:,:] = standarize(Py_exp_interp[2,:,:] , x3_mean, x3_std)

    #print("Py_exp_interp shape: ", Py_exp_interp.shape)
    return Py_exp_interp

def standarize(data, mean, std):
    return (data - mean) / std

def resize_temp(r, z, Tp):
    # Definir la nueva cuadrícula con dimensiones de 128x40
    new_r = np.linspace(0, 0.32, 32)
    new_z = np.linspace(1, 7.6, 128)
    f = interp2d(r, z, Tp, kind='linear', copy=True, bounds_error=False, fill_value=None)
    new_temp = f(new_r, new_z)
    return new_r, new_z, new_temp

def preprocess_mnist(image, channels=1, height=28, width=28):
    resized_image = image.resize((width, height), Image.ANTIALIAS)
    img_data = np.asarray(resized_image).astype(np.float32)
    
    # Normalización
    mean_vec = np.array([0.1307])
    stddev_vec = np.array([0.3081])

    img_data = (img_data / 255 - mean_vec) / stddev_vec
    
    return img_data

def max_mean(B1):
    B_max = list()
    for i in range(len(B1)):
        B_max.append(B1[i,:,:].max())
    B_max_mean = np.array(B_max).mean()
    B_max_std = np.array(B_max).std()

    print('max_mean: ',B_max_mean, 'Std: ',B_max_std, 'Min: ', np.array(B_max).min(), 'Max: ', np.array(B_max).max())
    return B_max_mean, B_max_std 

def preprocess_imagenet(image, channels=3, height=224, width=224):
    """Pre-processing for Imagenet-based Image Classification Models:
        resnet50, vgg16, mobilenet, etc. (Doesn't seem to work for Inception)

    Parameters
    ----------
    image: PIL.Image
        The image resulting from PIL.Image.open(filename) to preprocess
    channels: int
        The number of channels the image has (Usually 1 or 3)
    height: int
        The desired height of the image (usually 224 for Imagenet data)
    width: int
        The desired width of the image  (usually 224 for Imagenet data)

    Returns
    -------
    img_data: numpy array
        The preprocessed image data in the form of a numpy array

    """
    # Get the image in CHW format
    resized_image = image.resize((width, height), Image.ANTIALIAS)
    img_data = np.asarray(resized_image).astype(np.float32)

    if len(img_data.shape) == 2:
        # For images without a channel dimension, we stack
        img_data = np.stack([img_data] * 3)
        logger.debug("Received grayscale image. Reshaped to {:}".format(img_data.shape))
    else:
        img_data = img_data.transpose([2, 0, 1])

    mean_vec = np.array([0.485, 0.456, 0.406])
    stddev_vec = np.array([0.229, 0.224, 0.225])
    assert img_data.shape[0] == channels

    for i in range(img_data.shape[0]):
        # Scale each pixel to [0, 1] and normalize per channel.
        img_data[i, :, :] = (img_data[i, :, :] / 255 - mean_vec[i]) / stddev_vec[i]

    return img_data


def preprocess_inception(image, channels=3, height=224, width=224):
    """Pre-processing for InceptionV1. Inception expects different pre-processing
    than {resnet50, vgg16, mobilenet}. This may not be totally correct,
    but it worked for some simple test images.

    Parameters
    ----------
    image: PIL.Image
        The image resulting from PIL.Image.open(filename) to preprocess
    channels: int
        The number of channels the image has (Usually 1 or 3)
    height: int
        The desired height of the image (usually 224 for Imagenet data)
    width: int
        The desired width of the image  (usually 224 for Imagenet data)

    Returns
    -------
    img_data: numpy array
        The preprocessed image data in the form of a numpy array

    """
    # Get the image in CHW format
    resized_image = image.resize((width, height), Image.BILINEAR)
    img_data = np.asarray(resized_image).astype(np.float32)

    if len(img_data.shape) == 2:
        # For images without a channel dimension, we stack
        img_data = np.stack([img_data] * 3)
        logger.debug("Received grayscale image. Reshaped to {:}".format(img_data.shape))
    else:
        img_data = img_data.transpose([2, 0, 1])

    return img_data