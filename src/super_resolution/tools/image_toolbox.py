#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Simon Schaefer
# Description : Collection of tool function for image super resolution.
# =============================================================================
import numpy as np
from PIL import Image
import typing

def downsample(image: np.ndarray, factor: int=2) -> typing.Tuple[np.ndarray, np.ndarray]: 
    ''' Bicubic image downsampling including anti-aliasing using the PIL
    library imresize. Factor states the downsampling factor (has to be 
    a factor of 2 !). If necessary the input image is cropped so that
    the downsampling operation is "even" (from upper-left-corner). '''
    assert factor % 2 == 0
    assert factor >= 2
    assert (len(image.shape) == 3 and (image.shape[0] == 3 or image.shape[0] == 1))
    # Find current image size. 
    w_current, h_current = image.shape[1], image.shape[2]
    # Recrop image to "even" size if necessary. 
    w_cropped, h_cropped = w_current, h_current
    if not w_current % factor == 0: 
        w_cropped = int(w_current/factor)*factor
    if not h_current % factor == 0: 
        h_cropped = int(h_current/factor)*factor
    image = image[:w_cropped, :h_cropped]
    # Downsample image. 
    image_down = numpy_to_pil(image)
    w_target, h_target = int(w_current/factor), int(h_current/factor)
    image_down = image_down.resize((h_target, w_target), 
                                   resample=Image.BICUBIC)
    image_down = pil_to_numpy(image_down)
    return image, image_down

def load_image(filename: str) -> np.ndarray:
    ''' Read image from filename using PIL module, convert and 
    return the image as numpy array. Reformat to 3-dimensional array. '''
    img = Image.open(filename)
    img.load()
    return pil_to_numpy(img)

def normalize(image: np.ndarray) -> np.ndarray: 
    ''' Normalize image in between interval [-0.5,0.5]. Input image
    should be numpy array ranging from 0 to 255. '''
    data = np.asarray(image, dtype=np.float32)
    assert not np.any(data > 255.0) or not np.any(data < 0.0)
    return data/255.0 - 0.5

def numpy_to_pil(data: np.ndarray) -> Image: 
    ''' Transform numpy array to PIL image datatype. Due to training 
    considerations numpy array has an expected shape of ([1,3],w,h). 
    Therefore, before transformation the numpy array is reshaped 
    to the shape (w, h, [1,3]) which is valid for PIL. '''
    w, h = data.shape[1], data.shape[2]
    if data.size == w*h: 
        data = np.reshape(data, (w, h))
    else: 
        data = np.reshape(data, (w, h, 3))
    return Image.fromarray(np.uint8(data))

def pil_to_numpy(image: Image) -> np.ndarray: 
    ''' Transfrom PIL image to numpy array. Due to training 
    considerations numpy array has an expected shape of (3,w,h). 
    Therefore, after transformation the numpy array is reshaped 
    to shape (3, w, h) as PIL image typically have (w, h, 3) shape. '''
    data = np.asarray(image, dtype=np.float32)
    # Reshape if necessary. 
    w, h = data.shape[0], data.shape[1]
    if data.size == w*h: 
        data = np.reshape(data, (1, w, h))
    else: 
        data = np.reshape(data, (3, w, h))
    return data

def random_sub_sample(image: np.ndarray, width: int, height: int) -> np.ndarray:  
    ''' Get random subsample of image with given pixel width and height. 
    Image should be a 3D numpy array of larger size than subsample. '''
    w_image, h_image = image.shape[1], image.shape[2]
    assert w_image >= width and h_image >= height
    dw, dh = w_image - width, h_image - height
    w_0 = np.random.randint(0, dw) if dw > 0 else 0
    h_0 = np.random.randint(0, dh) if dh > 0 else 0
    return image[:, w_0:w_0+width, h_0:h_0+height]

def save_image(data: np.ndarray, filename: str) -> None:
    ''' Save image with filename using PIL module, convert from numpy array. '''
    img = numpy_to_pil(data)
    img.save(filename)

def upsample(image: np.ndarray, factor: int=2) -> typing.Tuple[np.ndarray, np.ndarray]: 
    ''' Bicubic image upsampling using the PIL library imresize. Factor states 
    the upsampling factor (has to be factor of 2 !). '''
    assert factor % 2 == 0
    assert factor >= 2
    assert (len(image.shape) == 3 and (image.shape[0] == 3 or image.shape[0] == 1))
    # Find current image size. 
    w_current, h_current = image.shape[1], image.shape[2]
    # Upsample image. 
    image_up = numpy_to_pil(image)
    w_target, h_target = int(w_current*factor), int(h_current*factor)
    image_up = image_up.resize((h_target, w_target), 
                                resample=Image.BICUBIC)
    image_up = pil_to_numpy(image_up)
    return image, image_up

