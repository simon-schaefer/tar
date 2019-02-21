#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Simon Schaefer
# Description : Collection of tool function for image super resolution.
# =============================================================================
import numpy as np
from PIL import Image

def downsample(image, factor=2, data_type=np.float32): 
    ''' Bicubic image downsampling including anti-aliasing using the PIL
    library imresize. Factor states the downsampling factor (has to be 
    a factor of 2 !). If necessary the input image is cropped so that
    the downsampling operation is "even" (from upper-left-corner).  
    @param[in]      image       image to downsample [np.array]. 
    @param[in]      factor      downsampling factor [integer]. '''
    assert factor % 2 == 0
    assert factor >= 2
    assert len(image.shape) == 2 \
        or (len(image.shape) == 3 and image.shape[2] == 3)
    # Find current image size. 
    w_current = image.shape[0]
    h_current = image.shape[1]
    # Recrop image to "even" size if necessary. 
    w_cropped, h_cropped = w_current, h_current
    if not w_current % factor == 0: 
        w_cropped = int(w_current/factor)*factor
    if not h_current % factor == 0: 
        h_cropped = int(h_current/factor)*factor
    image = image[:w_cropped, :h_cropped]
    # Downsample image. 
    image_down = Image.fromarray(np.uint8(image))
    w_target = int(w_current/factor)
    h_target = int(h_current/factor)
    image_down = image_down.resize((h_target, w_target), 
                                   resample=Image.BICUBIC)
    image_down = np.asarray(image_down, dtype=data_type)
    return image, image_down

def load_image(filename, data_type=np.float32):
    ''' Read image from filename using PIL module, convert and 
    return the image as numpy array. '''
    img = Image.open(filename)
    img.load()
    data = np.asarray(img, dtype=data_type)
    return data

def normalize(image): 
    ''' Normalize image in between interval [-0.5,0.5]. Input image
    should be numpy array ranging from 0 to 255. '''
    data = np.asarray(image, dtype=np.float32)
    assert not np.any(data > 255.0) or not np.any(data < 0.0)
    return data/255.0 - 0.5

def save_image(data, filename) :
    ''' Save image with filename using PIL module, convert from numpy array. '''
    img = Image.fromarray(np.uint8(data))
    img.save(filename)


