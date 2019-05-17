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
from torch.utils.data import Dataset, DataLoader
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
        self.split = 'train' if train else 'test'
        self.scale = scale
        self._set_filesystem(args.dir_data)

    # =========================================================================
    # Handling the filesystem
    # =========================================================================
    def _set_filesystem(self, directory: str):
        self.directory = os.path.join(directory, self.name)
        self.dir_hr = os.path.join(self.directory, 'HR')
        self.dir_lr = os.path.join(self.directory, 'LR_bicubic')

    def _scan(self):
        """ Scan given lists of directories for HR and LR images and return
        list of HR and LR absolute file paths. """
        raise NotImplementedError

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
    def _entcolorize(img) -> np.ndarray:
        return np.expand_dims(sc.rgb2ycbcr(img)[:,:,0], axis=2)/255.0

    # =========================================================================
    # Miscellaneous
    # =========================================================================
    def __len__(self) -> int:
        return len(self.images_hr)

# =============================================================================
# DATASET EXTENSION FOR IMAGES.
# =============================================================================
class _IDataset_(_Dataset_):
    """ Extension class for torch dataset module, in order to find, search,
    load, preprocess and batch images from datasets. """

    def __init__(self, args, train: bool, scale: int, name: str=""):
        self.super(_IDataset_,self).__init__(args,train,scale,name)
        list_hr, list_lr = self._scan()
        self.images_hr, self.images_lr = list_hr, list_lr

    def _scan(self):
        names_hr = sorted(glob.glob(self.dir_hr + "/*" + ".png"))
        # Check if scale == 1, then just return HR images.
        if self.scale == 1:
            return names_hr, names_hr
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
        assert hr.shape[2] == lr.shape[2]
        assert hr.shape[0] == self.scale*lr.shape[0]
        assert hr.shape[1] == self.scale*lr.shape[1]
        return lr, hr, filename

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
        if not self.args.augment: pair = self._augment(pair)
        # Set right number of channels.
        pair = self._set_channel(pair)
        # In colorization mode convert "LR" image to YCbCr and take Y-channel.
        if self.args.type == "COLORING": pair[0] = self._entcolorize(pair[1].copy())
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
        list_hr, list_lr = self._scan()
        self.images_hr, self.images_lr = list_hr, list_lr

    def _scan(self):
        hrs = sorted(glob.glob(self.dir_hr + "/*" + ".png"))
        hr_1 = [x for x in hrs if x.find("_1") > 0]
        hr_2 = [x for x in hrs if x.find("_2") > 0]
        assert len(hr_1) == len(hr_2) == len(hr_f)
        names_hr = [(hr_1[i], hr_2[i], hr_f[i]) for i in range(len(hr_1))]
        # Check if scale == 1, then just return HR images.
        if self.scale == 1:
            return names_hr, names_hr
        # Otherwise build LR image names for every HR image.
        names_lr = []
        for f in names_hr:
            lrs = []
            for ff in f:
                filename, _ = os.path.splitext(os.path.basename(ff))
                lrs.append(self.dir_lr + "/{}x{}.png".format(filename,self.scale))
            names_lr.append(tuple(lrs))
        return names_hr, names_lr

    def _load_file(self, idx: int):
        f_hr, f_lr = self.images_hr[idx], self.images_lr[idx]
        lrs, hrs, filenames = [], [], []
        for ff_hr, ff_lr in zip(f_hr, f_lr):
            filename, _ = os.path.splitext(os.path.basename(ff_hr))
            hr, lr = imageio.imread(ff_hr), imageio.imread(ff_lr)
            lr = self._expand_dimension(lr)
            hr = self._expand_dimension(hr)
            assert hr.shape[2] == lr.shape[2]
            assert hr.shape[0] == self.scale*lr.shape[0]
            assert hr.shape[1] == self.scale*lr.shape[1]
            lrs.append(lr); hrs.append(hr); filenames.append(filename)
        return lrs, hrs, filenames

    def __getitem__(self, idx: int):
        # Load image file.
        lrs, hrs, filenames = self._load_file(idx)
        # Iterate over and process all files in returned array.
        returns = []
        for lr,hr,filename in zip(lrs, hrs, filenames):
            # Cut patches from file.
            patch_size = self.args.patch_size
            assert patch_size <= hr.shape[0] and patch_size <= hr.shape[1]
            pair = self._get_patch(lr, hr, self.scale, patch_size, self.train)
            # Normalize patches from rgb_range to [norm_min, norm_max].
            pair = self._normalize(pair)
            # Augment patches (if flag is set).
            if not self.args.augment: pair = self._augment(pair)
            # Set right number of channels.
            pair = self._set_channel(pair)
            # In colorization mode convert "LR" image to YCbCr and take Y-channel.
            if self.args.type == "COLORING": pair[0] = self._entcolorize(pair[1].copy())
            # Convert to torch tensor and return.
            pair_t = self._np2Tensor(pair)
            returns.append((pair_t[0], pair_t[1], filename))
        return tuple(returns)

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
        self.loader_test = []
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
                if args.max_test_samples > len(vset): sampler = None
                self.loader_valid.append(_DataLoader_(
                    vset, 1, num_workers=args.n_threads, sampler=sampler
                ))
        if args.valid_only: return
        # Load testing dataset(s), if not valid only
        for s in args.scales_train:
            for dataset in args.data_test:
                tset = self.load_dataset(args, dataset, train=False, scale=s)
                sampler = RandomSampler(tset, replacement=True,
                                        num_samples=args.max_test_samples)
                if args.max_test_samples > len(tset): sampler = None
                self.loader_test.append(_DataLoader_(
                    tset, 1, num_workers=args.n_threads, sampler=sampler
                ))
        # Load training dataset, if not valid only. For training several
        # datasets are trained in one process and therefore, each given
        # training dataset is concatinated to one large dataset (for each scale).
        for s in args.scales_train:
            tset = self.load_dataset(args, dataset, train=True, scale=s)
            self.loader_train[s] = _DataLoader_(
                tset, args.batch_size, shuffle=True, num_workers=args.n_threads
            )

    def init_coloring(self, args: argparse.Namespace):
        # Load validation dataset. It is not about scaling here so merely the
        # scale equal one is required.
        for dataset in args.data_valid:
            vset = self.load_dataset(args, dataset, train=False, scale=1)
            sampler = RandomSampler(vset, replacement=True,
                                    num_samples=args.max_test_samples)
            if args.max_test_samples > len(vset): sampler = None
            self.loader_valid.append(_DataLoader_(
                vset, 1, num_workers=args.n_threads, sampler=sampler
            ))
        if args.valid_only: return
        # Load testing dataset(s), if not valid only
        for dataset in args.data_test:
            tset = self.load_dataset(args, dataset, train=False, scale=1)
            sampler = RandomSampler(tset, replacement=True,
                                    num_samples=args.max_test_samples)
            if args.max_test_samples > len(tset): sampler = None
            self.loader_test.append(_DataLoader_(
                tset, 1, num_workers=args.n_threads, sampler=sampler
            ))
        # Load training dataset, if not valid only. For training several
        # datasets are trained in one process and therefore, each given
        # training dataset is concatinated to one large dataset.
        tset = self.load_dataset(args, dataset, train=True, scale=1)
        self.loader_train[1] = _DataLoader_(
            tset, args.batch_size, shuffle=True, num_workers=args.n_threads
        )

    @staticmethod
    def load_dataset(args, name: str, train: bool, scale: int) -> _IDataset_:
        """ Load dataset from module (in datasets directory). Every module loaded
        should inherit from the _IDataset_ class defined below. """
        m = importlib.import_module("tar.datasets." + name.lower())
        return getattr(m, name)(args, train=train, scale=scale)
