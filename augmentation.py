import logging
import os
import os.path
import time
import shutil
import tensorflow as tf
import numpy as np
import cv2
import random
from glob import glob
from datetime import datetime
from shutil import copyfile
import imgaug as ia
from imgaug import augmenters as iaa
from scipy.misc import imread
from skimage import exposure
from PIL import Image, ImageOps, ImageEnhance
import math
from math import floor, ceil
import scipy.ndimage

import utils
import image_utils
import configuration as config

def zoom(img, zoom_factor):
    
    height, width = img.shape[:2] # It's also the final desired shape
    new_height, new_width = int(height * zoom_factor), int(width * zoom_factor)

    ### Crop only the part that will remain in the result (more efficient)
    # Centered bbox of the final desired size in resized (larger/smaller) image coordinates
    y1, x1 = max(0, new_height - height) // 2, max(0, new_width - width) // 2
    y2, x2 = y1 + height, x1 + width
    bbox = np.array([y1,x1,y2,x2])
    # Map back to original image coordinates
    bbox = (bbox / zoom_factor).astype(np.int)
    y1, x1, y2, x2 = bbox
    cropped_img = img[y1:y2, x1:x2]

    # Handle padding when downscaling
    resize_height, resize_width = min(new_height, height), min(new_width, width)
    pad_height1, pad_width1 = (height - resize_height) // 2, (width - resize_width) //2
    pad_height2, pad_width2 = (height - resize_height) - pad_height1, (width - resize_width) - pad_width1
    pad_spec = [(pad_height1, pad_height2), (pad_width1, pad_width2)] + [(0,0)] * (img.ndim - 2)

    result = cv2.resize(cropped_img, (resize_width, resize_height))
    result = np.pad(result, pad_spec, mode='constant')
    assert result.shape[0] == height and result.shape[1] == width
    return result

def augmentation_function(images, labels):
    '''
    :param images: A numpy array of shape [batch, X, Y], normalized between 0-1
    :param labels: A numpy array containing a corresponding label mask     
    ''' 
    
    # Define in configuration.py which operations to perform
    do_rotation_range = config.do_rotation_range
    do_fliplr = config.do_fliplr
    do_flipud = config.do_flipud
    crop = config.crop
    do_gamma = config.gamma
    do_blurr = config.blurr
    
    # Probability to perform a generic operation
    prob = config.prob
    if 0.0 <= prob <= 1.0:
        
        new_images = []
        new_labels = []
        num_images = images.shape[0]
        
        for i in range(num_images):
            
            #extract the single image
            img = np.squeeze(images[i,...])
            lbl = np.squeeze(labels[i,...])
            
            # RANDOM ROTATION
            if do_rotation_range:
                coin_flip = np.random.uniform(low=0.0, high=1.0)
                if coin_flip < prob :
                    angle = config.rg
                    random_angle = np.random.uniform(angle[0], angle[1])
                    img = image_utils.rotate_image(img, random_angle)
                    lbl = image_utils.rotate_image(lbl, random_angle, interp=cv2.INTER_NEAREST)
                    
            
            # FLIP Lelf/Right
            if do_fliplr:
                coin_flip = np.random.randint(2)
                if coin_flip == 0:
                    img = np.fliplr(img)
                    lbl = np.fliplr(lbl)
     
                
            # FLIP  up/down
            if do_flipud:
                coin_flip = np.random.randint(2)
                if coin_flip == 0:
                    img = np.flipud(img)
                    lbl = np.flipud(lbl)
                    
                    
            # RANDOM TRANSLATION 5%
         #   if (random.randint(0,1)):
         #               x = random.randint(-11,11)
         #               y = random.randint(-11,11)
         #               M = np.float32([[1,0,x],[0,1,y]])
         #               img = cv2.warpAffine(img,M,(config.image_size[0],config.image_size[1]))
         #               lbl = cv2.warpAffine(lbl,M,(config.image_size[0],config.image_size[1]))
                        
                       
            # RANDOM CROP 5%
            if crop:
                coin_flip = np.random.randint(2)
                if coin_flip == 0:
                    zfactor = round(random.uniform(1,1.05), 2)
                    img = zoom(img, zfactor)
                    lbl = zoom(lbl, zfactor)
            
            # RANDOM GAMMA CORRECTION
            if do_gamma:
                coin_flip = np.random.randint(2)
                if coin_flip == 0:
                    gamma = random.randrange(8,13,1)
                    img = exposure.adjust_gamma(img, gamma/10)
            
            # RANDOM BLURR
            if do_blurr:
                coin_flip = np.random.randint(2)
                if coin_flip == 0:
                    sigma = random.randrange(6,16,2)
                    img = scipy.ndimage.gaussian_filter(img, sigma/10)
            
            new_images.append(img[..., np.newaxis])
            new_labels.append(lbl[...])
        
        sampled_image_batch = np.asarray(new_images)
        sampled_label_batch = np.asarray(new_labels)

        return sampled_image_batch, sampled_label_batch
    
    else:
        logging.warning('Probability must be in range [0.0,1.0]!!')
