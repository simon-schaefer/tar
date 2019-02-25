#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Simon Schaefer
# Description : Data loading, preprocessing and batching in image domain. 
# =============================================================================
import numpy as np
import random
import typing 

from super_resolution.tools import files_toolbox as file_tools
from super_resolution.tools import image_toolbox as img_tools

class DataLoader(object): 

    def __init__(self, train_datasets: typing.List[str], 
                       valid_datasets: typing.List[str], **kwargs): 
        ''' Initialize trainer by loading training and validation 
        datasets. 
        - Train and valid datasets should be lists of absolute 
        paths to the directories containing the training and validation 
        datasets, respectively. In order to be space efficient the overall 
        data are stored as overall list of absolute file paths. 
        - scale_guidance: Scale guidance determines the scale of the 
        resulting LR image w.r.t. the HR image (i.e. 2 = two times smaller). 
        - batch_size: Batch size determines the size of mini-batches 
        used for training. 
        - subsize: To maintain a certain input image size for the model 
        not the whole image is inputted but a random subsample of quadratic
        size, with length subsize. Has to be an even numer (i.e. "evenly" 
        dividable by scale factor). '''
        assert isinstance(train_datasets, (list,)) and len(train_datasets) > 0
        assert isinstance(valid_datasets, (list,)) and len(valid_datasets) > 0
        # Read in training datasets as overall array. 
        self._train_datasets = train_datasets
        self.files_hr_train = self.load_dataset(train_datasets)
        random.shuffle(self.files_hr_train)
        self.num_train_samples = len(self.files_hr_train)
        # Read in validation data. 
        self._valid_datasets = valid_datasets
        self.files_hr_valid = self.load_dataset(valid_datasets)
        self.num_valid_samples = len(self.files_hr_valid)
        # Check and store parameters. 
        kwargs = self.set_default_param(kwargs, "scale_guidance", 2)
        kwargs = self.set_default_param(kwargs, "batch_size", 6)
        kwargs = self.set_default_param(kwargs, "subsize", 96)
        assert kwargs["subsize"] % kwargs["scale_guidance"] == 0
        self._params = kwargs

    def next_batch(self) -> typing.Tuple[np.ndarray, np.ndarray, int]: 
        ''' Get next batch of training data. Return two arrays, the HR images
        as batch (numpy array) and the LR images (numpy array) scaled down 
        by scale_guidance factor and with size = batch_size. 
        To maintain a certain size (constant input size to model) not the 
        whole image but a random subsample is clipped out of the HR image 
        and lower resoluted. 
        Additionally, return flag whether epoch is over (after this batch). '''
        num_samples_remaining = len(self.files_hr_train)
        batch_size = self._params["batch_size"]
        assert num_samples_remaining >= batch_size 
        # Get HR images (randomly as already shuffled list). 
        subsize = self._params["subsize"]
        scale = self._params["scale_guidance"]
        channels = img_tools.load_image(self.files_hr_train[0]).shape[0]
        subsize_lr = int(subsize/scale)
        batch_hr = np.empty((batch_size, channels, subsize, subsize))
        batch_lr = np.empty((batch_size, channels, subsize_lr, subsize_lr))
        files_hr_batch = self.files_hr_train[:batch_size]
        counter = 0
        for x in files_hr_batch: 
            image = img_tools.load_image(x)
            image = img_tools.random_sub_sample(image, subsize, subsize)
            image, image_down = img_tools.downsample(image, factor=scale)
            batch_hr[counter,:,:,:] = image
            batch_lr[counter,:,:,:] = image_down
            counter = counter + 1
        # Update list of training files. 
        self.files_hr_train = self.files_hr_train[batch_size:]
        # Check whether epoch is over. 
        epoch_is_over = num_samples_remaining - batch_size < batch_size
        if epoch_is_over: 
            self.reload_trainig_dataset()
        return batch_hr, batch_lr, epoch_is_over

    def reload_trainig_dataset(self): 
        self.files_hr_train = self.load_dataset(self._train_datasets)
        random.shuffle(self.files_hr_train)

    @staticmethod
    def load_dataset(datasets: typing.List[str]) -> typing.List[str]: 
        ''' Load in dataset from list of sources (absolute paths). 
        Return list of all contained image files (unshuffled !). '''
        assert isinstance(datasets, (list,)) and len(datasets) > 0
        files = []
        for dataset in datasets: 
            x = file_tools.load_files(dataset, "png")
            assert len(x) > 0
            files.extend(x)
        return files

    @staticmethod
    def set_default_param(params: typing.Dict, key: str, value) -> typing.Dict: 
        params[key] = params[key] if key in params.keys() else value
        return params