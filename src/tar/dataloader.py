#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Simon Schaefer
# Description : Loading data front end class.
# =============================================================================
import argparse
import imageio
import importlib
import numpy as np
import os
import glob
import random
import skimage.color as sc
from typing import List, Tuple

from torch import from_numpy, Tensor
from torch.utils.data import ConcatDataset, Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import RandomSampler

import tar.miscellaneous as misc

class _Dataset_(Dataset):

    def __init__(self, args, train: bool, scale: int, name: str=""):
        # Initialize super dataset class.
        super(_Dataset_, self).__init__()
        # Set input parameters.
        self.args = args
        self.name = name
        self.train = train
        self.scale = scale
        self._set_filesystem(args.dir_data)
        list_hr, list_lr = self._scan()
        self.images_hr, self.images_lr = list_hr, list_lr
        max_samples = self.args.max_test_samples
        self.sample_size = min(self.__len__(), max_samples)
        if not self.is_sampled(): self.sample_size = self.__len__()

    # =========================================================================
    # Handling the filesystem
    # =========================================================================
    def _set_filesystem(self, directory: str):
        self.directory = os.path.join(directory, self.name)
        self.dir_hr = os.path.join(self.directory, 'HR')
        self.dir_lr = os.path.join(self.directory, 'LR_bicubic')
        self.dir_lr = os.path.join(self.dir_lr, "X{}".format(self.scale))

    def _scan(self):
        """ Scan given lists of directories for HR and LR images and return
        list of HR and LR absolute file paths. """
        names_hr = sorted(glob.glob(self.dir_hr + "/*" + ".png"))
        # Check if scale == 1, then just return HR images.
        if self.scale == 1: return names_hr, names_hr
        # Otherwise build LR image names for every HR image.
        names_lr = []
        for f in names_hr:
            filename, _ = os.path.splitext(os.path.basename(f))
            names_lr.append(self.dir_lr + "/{}x{}.png".format(filename,self.scale))
        return names_hr, names_lr

    def _load_file(self, idx: int):
        f_hr, f_lr = self.images_hr[idx], self.images_lr[idx]
        filename, _ = os.path.splitext(os.path.basename(f_hr))
        hr, lr = imageio.imread(f_hr), imageio.imread(f_lr)
        lr, hr = self._expand_dimension(lr), self._expand_dimension(hr)
        wl, hl = lr.shape[0]//2*2, lr.shape[1]//2*2
        lr, hr = lr[:wl,:hl,:], hr[:wl*self.scale,:hl*self.scale,:]
        assert hr.shape[2] == lr.shape[2]
        assert hr.shape[0] == self.scale*lr.shape[0]
        assert hr.shape[1] == self.scale*lr.shape[1]
        return lr, hr, filename

    # =========================================================================
    # File loading functions.
    # =========================================================================
    @staticmethod
    def _get_patch(lr: np.ndarray, hr: np.ndarray,
                   scale: int, patch_size: int, do_train: bool):
        if do_train:
            lh, lw = lr.shape[:2]
            hp = patch_size
            lp = hp // scale
            lx = random.randrange(0, lw - lp + 1)
            ly = random.randrange(0, lh - lp + 1)
            hx, hy = scale * lx, scale * ly
            lr = lr[ly:ly + lp, lx:lx + lp, :]
            hr = hr[hy:hy + hp, hx:hx + hp, :]
        else:
            ih, iw = lr.shape[:2]
            hr = hr[0:ih * scale, 0:iw * scale]
        return lr, hr

    @staticmethod
    def _augment(imgs: List[np.ndarray]) -> List[np.ndarray]:
        hflip = random.random() < 0.5
        vflip = random.random() < 0.5
        rot90 = random.random() < 0.5
        def _augment_x(img):
            if hflip: img = img[:, ::-1, :]
            if vflip: img = img[::-1, :, :]
            if rot90: img = img.transpose(1, 0, 2)
            return img
        return [_augment_x(x) for x in imgs]

    @staticmethod
    def _expand_dimension(img: np.ndarray) -> np.ndarray:
        if img.ndim == 2: img = np.stack((img,img,img), axis=2)
        elif img.ndim == 3 and img.shape[2] == 4: img = img[:,:,:3]
        return img

    @staticmethod
    def _set_channel(imgs) -> List[np.ndarray]:
        def _set_channel_x(img):
            if img.ndim == 2: img = np.expand_dims(img, axis=2)
            if img.shape[2] == 1: img = np.concatenate([img] * 3, 2)
            return img
        return [_set_channel_x(x) for x in imgs]

    def _normalize(self, imgs) -> List[np.ndarray]:
        return [misc.normalize(x,
                self.args.norm_min, self.args.norm_max) for x in imgs]

    @staticmethod
    def _np2Tensor(imgs) -> List[Tensor]:
        def _np2Tensor_x(img):
            np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
            tensor = from_numpy(np_transpose).float()
            return tensor
        return [_np2Tensor_x(x) for x in imgs]

    @staticmethod
    def _entcolorize(img, colorspace) -> np.ndarray:
        if colorspace == "ycbcr":  x = sc.rgb2ycbcr(img)[:,:,0]
        elif colorspace == "hsv":  x = sc.rgb2hsv(img)[:,:,0]
        elif colorspace == "gray": x = sc.rgb2gray(img)[:,:]
        else: raise ValueError("Undefined colorspace {} !".format(colorspace))
        return np.expand_dims(x, axis=2)/255.0

    # =========================================================================
    # Miscellaneous
    # =========================================================================
    def __len__(self) -> int:
        return len(self.images_hr)

    def is_sampled(self) -> bool:
        max_samples = self.args.max_test_samples
        return not self.args.valid_only and self.sample_size == max_samples

# =============================================================================
# DATASET EXTENSION FOR IMAGES.
# =============================================================================
class _IDataset_(_Dataset_):
    """ Extension class for torch dataset module, in order to find, search,
    load, preprocess and batch images from datasets. """

    def __init__(self, args, train: bool, scale: int, name: str=""):
        super(_IDataset_,self).__init__(args,train,scale,name)
        self.format = "IMAGE"

    def __getitem__(self, idx: int):
        # Load image file.
        lr, hr, filename = self._load_file(idx)
        # Cut patches from file.
        patch_size = self.args.patch_size
        assert patch_size <= hr.shape[0] and patch_size <= hr.shape[1]
        pair = self._get_patch(lr, hr, self.scale, patch_size, self.train)
        # Normalize patches from rgb_range to [norm_min, norm_max].
        pair = self._normalize(pair)
        # Augment patches (if flag is set).
        if not self.args.no_augment and not self.args.valid_only and self.train:
            pair = self._augment(pair)
        # Set right number of channels.
        pair = self._set_channel(pair)
        # In colorization mode convert "LR" image to YCbCr and take Y-channel.
        if self.args.type=="COLORING":
            pair[0]=self._entcolorize(pair[1].copy(),self.args.color_space)
        # Convert to torch tensor and return.
        pair_t = self._np2Tensor(pair)
        return pair_t[0], pair_t[1], filename

# =============================================================================
# DATASET EXTENSION FOR VIDEO DATA.
# =============================================================================
class _VDataset_(_Dataset_):
    """ Extension class for torch dataset module, in order to find, search,
    load, preprocess and batch video data from datasets. """

    def __init__(self, args, train: bool, scale: int, name: str=""):
        super(_VDataset_, self).__init__(args,train,scale,name)
        self.format = "VIDEO"

    def __getitem__(self, idx: int):
        # Load image file.
        idx = max(min(idx, self.__len__() - 3), 0)
        lr0, hr0, fname0 = self._load_file(idx)
        lr1, hr1, fname1 = self._load_file(idx + 1)
        lr2, hr2, fname2 = self._load_file(idx + 2)
        # Iterate over and process all files in returned array.
        lrs, hrs, fnames = [], [], []
        for lr,hr,filename in zip([lr0,lr1,lr2],[hr0,hr1,hr2],[fname0,fname1,fname2]):
            # Cut patches from file.
            patch_size = self.args.patch_size
            assert patch_size <= hr.shape[0] and patch_size <= hr.shape[1]
            pair = self._get_patch(lr, hr, self.scale, patch_size, self.train)
            # Normalize patches from rgb_range to [norm_min, norm_max].
            pair = self._normalize(pair)
            # Augment patches (if flag is set).
            if not self.args.no_augment and not self.args.valid_only:
                pair = self._augment(pair)
            # Set right number of channels.
            pair = self._set_channel(pair)
            # In colorization mode convert "LR" image to YCbCr and take Y-channel.
            if self.args.type == "COLORING":
                pair[0] = self._entcolorize(pair[1].copy(),self.args.color_space)
            # Convert to torch tensor and return.
            pair_t = self._np2Tensor(pair)
            lrs.append(pair_t[0])
            hrs.append(pair_t[1])
            fnames.append(filename)
        return tuple(lrs), tuple(hrs), tuple(fnames)

# =============================================================================
# DATA LOADING CLASS.
# =============================================================================
class _DataLoader_(DataLoader):
    """ Pytorch data loader to load dataset with the following input arguments:
    - batch_size: number of samples in a batch
    - shuffle: should the dataset be shuffled before loading ?
    - num_workers:  how many subprocesses to use for data loading.
                    0 means that the data will be loaded in the main process.
    - collate_fn: merges a list of samples to form a mini-batch. """

    def __init__(self, dataset,
                 batch_size,
                 shuffle=False,
                 num_workers=1,
                 sampler=None,
                 collate_fn=default_collate):

        super(_DataLoader_, self).__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            sampler=sampler,
            collate_fn=collate_fn,
        )

# =============================================================================
# DATA HANDLING CLASS.
# =============================================================================
class _Data_(object):
    """ Data loading class which allocates all given training and testing
    dataset stated in the input arguments to a loader (and concatenates
    them for training). The resulting loader_test and loader_train can be
    used to load batches from the datasets. """

    def __init__(self, args: argparse.Namespace):
        self.loader_valid = []
        self.loader_train = {}
        if args.type == "SCALING": self.init_scaling(args)
        elif args.type == "COLORING": self.init_coloring(args)
        else: raise ValueError("Invalid program type {}!".format(args.type))

    def init_scaling(self, args: argparse.Namespace):
        # Check input scales, lists have to be a power of 2.
        assert misc.all_power2(args.scales_train)
        # Load validation dataset. In order to get seperated testing results,
        # from each dataset (due to comparability reasons) the testing
        # datasets are each loaded individually.
        for dataset in args.data_valid:
            for s in args.scales_valid:
                vset = self.load_dataset(args, dataset, train=False, scale=s)
                sampler = RandomSampler(vset, replacement=True,
                                        num_samples=args.max_test_samples)
                if not vset.is_sampled(): sampler = None
                self.loader_valid.append(_DataLoader_(
                    vset, 1, num_workers=args.n_threads, sampler=sampler
                ))
        if args.valid_only: return
        # Load training dataset, if not valid only. For training several
        # datasets are trained in one process and therefore, each given
        # training dataset is concatinated to one large dataset (for each scale).
        for dataset in args.data_train:
            self.loader_train[s] = []
            for s in args.scales_train:
                tset = self.load_dataset(args, dataset, train=True, scale=s)
                shuf = args.format != "VIDEO"
                self.loader_train[s].append(_DataLoader_(
                tset, args.batch_size, shuffle=shuf, num_workers=args.n_threads
                ))

    def init_coloring(self, args: argparse.Namespace):
        # Load validation dataset. It is not about scaling here so merely the
        # scale equal one is required.
        for dataset in args.data_valid:
            vset = self.load_dataset(args, dataset, train=False, scale=1)
            sampler = RandomSampler(vset, replacement=True,
                                    num_samples=args.max_test_samples)
            if not vset.is_sampled(): sampler = None
            self.loader_valid.append(_DataLoader_(
                vset, 1, num_workers=args.n_threads, sampler=sampler
            ))
        if args.valid_only: return
        # Load training dataset, if not valid only. For training several
        # datasets are trained in one process and therefore, each given
        # training dataset is concatinated to one large dataset.
        self.loader_train[1] = []
        for dataset in args.data_train:
            tset = self.load_dataset(args, dataset, train=True, scale=1)
            shuffle = args.format != "VIDEO"
            self.loader_train[1].append(_DataLoader_(
            tset, args.batch_size, shuffle=shuffle, num_workers=args.n_threads
        ))

    @staticmethod
    def load_dataset(args, name: str, train: bool, scale: int):
        """ Load dataset from module (in datasets directory). Every module
        loaded should inherit from the _Dataset_ class defined below. """
        try:
            m = importlib.import_module("tar.datasets." + name.lower())
            return getattr(m, name)(args, train=train, scale=scale)
        except ImportError:
            if args.format == "IMAGE":
                return _IDataset_(args,train=train,scale=scale,name=name.upper())
            elif args.format == "VIDEO":
                return _VDataset_(args,train=train,scale=scale,name=name.upper())
            else:
                raise ValueError("Invalid format {} !".format(args.format))
