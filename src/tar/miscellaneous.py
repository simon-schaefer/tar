#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Simon Schaefer
# Description : Collection of miscellaneous helper functions.
# =============================================================================
import argparse
import csv
import glob
import imageio
import math
from multiprocessing import Process, Queue
import numpy as np
import os
import random
import sys
import time
from typing import Dict, List

import cv2

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch

class _Checkpoint_(object):
    """ Logging class for model training, including saving the model,
    optimizer state and loss curve (in a parallel threat). """

    def __init__(self, args: argparse.Namespace):
        super(_Checkpoint_, self).__init__()
        # Initializing checkpoint module.
        self.ready = False
        self.args  = args
        self.log   = torch.Tensor()
        # Building model directory based on name and time.
        now = time.strftime("%H_%M_%S_%d_%b", time.gmtime())
        tag = args.template + "_" + now
        self.dir = os.path.join(os.environ['SR_PROJECT_OUTS_PATH'], tag)
        if args.valid_only: self.dir = self.dir + "_valid"
        self.dir_load = None
        # If previous training/model should be loaded, build loading path
        # from input arguments and read loaded config file.
        if not args.load == "":
            assert len(args.load.split("x")) == 2
            path, tag = args.load.split("x")
            assert path in ["outs", "models"]
            if path == "outs":
                path = os.environ['SR_PROJECT_OUTS_PATH']
            else:
                path = os.environ['SR_PROJECT_MODELS_PATH']
            self.dir_load = os.path.join(path, tag)
            if not os.path.exists(self.dir_load):
                raise ValueError("Loading path %s does not exists !" % self.dir_load)
            self.args_load = {}
            with open(self.get_load_path("config.txt"), "r") as f:
                for line in f:
                    x = line.rstrip('\n')
                    if not len(x.split(":")) == 2: continue
                    key, val = x.split(":")
                    self.args_load[key] = val.replace(" ", "")
        # Creating output directories for model.
        os.makedirs(self.dir, exist_ok=True)
        os.makedirs(self.get_path('model'), exist_ok=True)
        # Create output directory for test datasets.
        os.makedirs(self.get_path('results_{}'.format(args.data_test)),exist_ok=True)
        # Create output directory for validation datasets.
        for d in args.data_valid:
            os.makedirs(self.get_path('results_{}'.format(d)), exist_ok=True)
        # Create output directory for logging data and write config.
        open_type = 'a' if os.path.exists(self.get_path('log.txt')) else 'w'
        self.log_file = open(self.get_path('log.txt'), open_type)
        with open(self.get_path('config.txt'), open_type) as f:
            for arg in vars(args):
                f.write('{}: {}\n'.format(arg, getattr(args, arg)))
        # Build list containing names of test and validation datasets
        # for logging and plotting purposes.
        self.log_datasets = [self.args.data_test]
        for dv in self.args.data_valid:
            for sv in self.args.scales_valid:
                self.log_datasets.append(dv + "x" + str(sv))
        # Set number of logging threats.
        self.n_processes  = 8
        self.iter_is_best = False
        self.ready        = True
        self.write_log("Building model module ...")
        if not args.load == "":
            self.write_log("... continue from model in {}...".format(self.dir_load))
        self.write_log("... successfully built checkpoint module !")

    def step(self, **kwargs):
        self.add_log(torch.zeros(1, len(self.log_datasets), kwargs["nlogs"]))
        return self.ready

    # =========================================================================
    # Saving
    # =========================================================================
    def save(self, trainer, epoch: int):
        trainer.model.save(self.get_path('model'),epoch,is_best=self.iter_is_best)
        trainer.loss.save(self.dir)
        trainer.loss.plot_loss(self.dir, epoch, scaling="linear")
        trainer.loss.plot_loss(self.dir, epoch, scaling="logarithmic")
        trainer.optimizer.save(self.dir)
        torch.save(self.log, self.get_path('psnr_log.pt'))
        # Plot peak signal-to-noise ratio (PSNR) plot.
        if self.log.shape[1] != len(self.log_datasets): return
        axis = np.linspace(1, epoch, epoch)
        labels = trainer.log_description()
        for id, d in enumerate(self.log_datasets):
            fig = plt.figure()
            label = "PSNR on {}".format(d)
            plt.title(label)
            axes = plt.gca()
            ymin, ymax = 15, 60
            axes.set_ylim([0,ymax])
            for i in range(len(labels)):
                log = torch.clamp(self.log[:, id, i], ymin+1, ymax-1)
                plt.plot(axis,log.numpy(),label="{}".format(labels[i]))
            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel('PSNR')
            plt.grid(True)
            plt.savefig(self.get_path("psnr_{}.pdf".format(d)))
            plt.close(fig)

    def save_results(self, save_list: List[torch.Tensor], desc_list: List[str],
                     filename: str, dataset, scale: int):
        filename = self.get_path(
            'results_{}'.format(dataset.dataset.name),
            '{}_x{}_'.format(filename, scale)
        )
        assert len(save_list) == len(desc_list)
        for v, p in zip(save_list, desc_list):
            normalized = v[0]
            r = 255/(self.args.norm_max - self.args.norm_min)
            normalized = normalized.add(-self.args.norm_min).mul(r)
            normalized = normalized.clamp(0, 255).round()
            tensor_cpu = normalized.byte().permute(1, 2, 0).cpu()
            self.queue.put(('{}{}.png'.format(filename, p), tensor_cpu))

    def clear_results(self, dataset):
        directory = self.get_path("results_{}".format(dataset.dataset.name))
        files = glob.glob(directory + "/*.png")
        for f in files: os.remove(f)

    def save_validations(self, valids: List[Dict[str,str]]):
        file_path = self.get_path('validations.csv')
        csv_data  = []
        csv_data.append(sorted(valids[0].keys(),reverse=True))
        for v in valids:
            csv_data.append(x for (_, x) in sorted(v.items(),reverse=True))
        with open(file_path, 'w+') as f:
            writer = csv.writer(f)
            writer.writerows(csv_data)
        f.close()

    # =========================================================================
    # Logging.
    # =========================================================================
    def add_log(self, log):
        self.log = torch.cat([self.log, log])

    def write_log(self, log, refresh=False):
        self.log_file.write(log + '\n')
        if refresh:
            self.log_file.close()
            self.log_file = open(self.get_path('log.txt'), 'a')
        if self.args.verbose: print(log)

    def done(self):
        self.ready = False
        self.log_file.close()

    # =========================================================================
    # Multithreading.
    # =========================================================================
    def begin_background(self):
        self.queue = Queue()

        def bg_target(queue):
            while True:
                if not queue.empty():
                    filename, tensor = queue.get()
                    if filename is None: break
                    imageio.imwrite(filename, tensor.numpy())

        self.process = [
            Process(target=bg_target, args=(self.queue,)) \
            for _ in range(self.n_processes)
        ]
        for p in self.process: p.start()

    def end_background(self):
        for _ in range(self.n_processes): self.queue.put((None, None))
        while not self.queue.empty(): time.sleep(1)
        for p in self.process: p.join()

    def get_path(self, *subdir):
        return os.path.join(self.dir, *subdir)

    def get_load_path(self, *subdir):
        return os.path.join(self.dir_load, *subdir)

# =============================================================================
# Timer class.
# =============================================================================
class _Timer_(object):
    """ Time logging class based on time library. """

    def __init__(self):
        super(_Timer_, self).__init__()
        self.acc = 0
        self.tic()

    def tic(self):
        self.t0 = time.time()

    def toc(self, restart=False):
        diff = time.time() - self.t0
        if restart: self.t0 = time.time()
        return diff

    def hold(self):
        self.acc += self.toc()

    def release(self):
        ret = self.acc
        self.reset()
        return ret

    def reset(self):
        self.acc = 0

# =============================================================================
# Miscellaneous utility functions.
# =============================================================================
def print_header() -> None:
    header_file = os.environ['SR_PROJECT_SCRIPTS_PATH'] + "/header.bash"
    os.system("bash " + header_file)

def progress_bar(iteration: int, num_steps: int, bar_length: int=50) -> int:
    """ Draws progress bar showing the number of executed
    iterations over the overall number of iterations.
    Increments the iteration and returns it. """
    status = ""
    progress = float(iteration) / float(num_steps)
    if progress >= 1.0:
        progress, status = 1.0, "\r\n"
    block = int(round(bar_length * progress))
    text = "\r[{}] {:.0f}% {}".format(
        "#" * block + "-" * (bar_length - block), round(progress*100, 0),
        status)
    sys.stdout.write(text)
    sys.stdout.flush()
    return iteration + 1

def calc_psnr(x: torch.Tensor, y: torch.Tensor,
              patch_size: int=None, rgb_range: float=255.0) -> float:
    """ Determine peak signal to noise ratio between to tensors
    (mostly images, given as torch tensors), according to the formula in
    https://www.mathworks.com/help/vision/ref/psnr.html. If patch size
    is None the PSNR will be determined over the full tensors, otherwise
    a random patch of give patch size is determined and the PSNR is calculated
    with respect to this patch. The tensors have an expected shape of (b,c,h,w). """
    mse = torch.pow(x - y, 2).mean().item()
    # if patch_size is not None:
    #     h, w = x.shape[2:4]
    #     lp = int(patch_size)
    #     lx = random.randrange(0, w - lp + 1)
    #     ly = random.randrange(0, h - lp + 1)
    #     px = x[:, :, ly:ly + lp, lx:lx + lp]
    #     py = y[:, :, ly:ly + lp, lx:lx + lp]
    if mse == 0: return 100.0
    return 20 * math.log10(rgb_range/np.sqrt(mse))

def discretize(img: torch.Tensor, norm_range: List[float]) -> torch.Tensor:
    """ Discretize image (given as torch tensor) in defined range of
    pixel values (e.g. 255 or 1.0), i.e. smart rounding. """
    pixel_range = 255 * (norm_range[1] - norm_range[0])
    img_dis = img.add(-norm_range[0]) # denormalization
    img_dis = img_dis.mul(pixel_range).clamp(0, 255).round().div(pixel_range)
    img_dis = img_dis.add(norm_range[0]) # normalization
    return img_dis

def normalize(img, norm_min, norm_max):
    """ Normalize numpy array or torch tensor from RGB range
    to given norm range [norm_min, norm_max]. """
    assert norm_max > norm_min
    norm_range = norm_max - norm_min
    return img/255.0*norm_range + norm_min

def unnormalize(img, norm_min, norm_max):
    """ Unnormalize numpy array or torch tensor from given norm
    range [norm_min, norm_max] to RGB range. """
    assert norm_max > norm_min
    norm_range = norm_max - norm_min
    return (img - norm_min)/norm_range*255.0

def is_power2(num):
    return num != 0 and ((num & (num - 1)) == 0)

def all_power2(numbers):
    return all([is_power2(x) for x in numbers])


# =============================================================================
# Flow to color functions (https://github.com/georgegach/flow2image).
# =============================================================================
def convert_flow_to_color(flow):
    u,v = _normalizeFlow(flow)
    img = _computeColor(u, v)
    return img

def _colorWheel():
    # Original inspiration: http://members.shaw.ca/quadibloc/other/colint.htm
    RY = 15; YG = 6; GC = 4; CB = 11; BM = 13; MR = 6
    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros([ncols, 3]) # RGB
    col = 0
    #RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255*np.arange(0, RY, 1)/RY)
    col += RY
    #YG
    colorwheel[col : YG + col, 0] = 255 - np.floor(255*np.arange(0, YG, 1)/YG)
    colorwheel[col : YG + col, 1] = 255
    col += YG
    #GC
    colorwheel[col : GC + col, 1] = 255
    colorwheel[col : GC + col, 2] = np.floor(255*np.arange(0, GC, 1)/GC)
    col += GC
    #CB
    colorwheel[col : CB + col, 1] = 255 - np.floor(255*np.arange(0, CB, 1)/CB)
    colorwheel[col : CB + col, 2] = 255
    col += CB
    #BM
    colorwheel[col : BM + col, 2] = 255
    colorwheel[col : BM + col, 0] = np.floor(255*np.arange(0, BM, 1)/BM)
    col += BM
    #MR
    colorwheel[col : MR + col, 2] = 255 - np.floor(255*np.arange(0, MR, 1)/MR)
    colorwheel[col : MR + col, 0] = 255
    return colorwheel

def _computeColor(u, v):
    colorwheel = _colorWheel()
    idxNans = np.where(np.logical_or(
        np.isnan(u),
        np.isnan(v)
    ))
    u[idxNans], v[idxNans] = 0, 0
    ncols = colorwheel.shape[0]
    radius = np.sqrt(np.multiply(u, u) + np.multiply(v, v))
    a = np.arctan2(-v, -u) / np.pi
    fk = (a+1) / 2 * (ncols - 1)
    k0 = fk.astype(np.uint8)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    img = np.empty([k1.shape[0], k1.shape[1], 3])
    ncolors = colorwheel.shape[1]
    for i in range(ncolors):
        tmp = colorwheel[:, i]
        col0 = tmp[k0] / 255
        col1 = tmp[k1] / 255
        col = (1-f) * col0 + f * col1
        idx = radius <= 1
        col[idx] = 1 - radius[idx] * (1 - col[idx])
        col[~idx] *= 0.75
        img[:, :, i] = np.floor(255 * col).astype(np.uint8) # RGB
    return img.astype(np.uint8)

def _normalizeFlow(flow):
    UNKNOWN_FLOW_THRESH = 1e9
    UNKNOWN_FLOW = 1e10
    height, width, nBands = flow.shape
    u = flow[:, :, 0]
    v = flow[:, :, 1]
    # Fix unknown flow
    idxUnknown = np.where(np.logical_or(
        abs(u) > UNKNOWN_FLOW_THRESH,
        abs(v) > UNKNOWN_FLOW_THRESH
    ))
    u[idxUnknown], v[idxUnknown] = 0, 0
    maxu = max([-999, np.max(u)])
    maxv = max([-999, np.max(v)])
    minu = max([999, np.min(u)])
    minv = max([999, np.min(v)])
    rad = np.sqrt(np.multiply(u, u) + np.multiply(v, v))
    maxrad = max([-1, np.max(rad)])
    eps = np.finfo(np.float32).eps
    u = u/(maxrad + eps)
    v = v/(maxrad + eps)
    return u,v
