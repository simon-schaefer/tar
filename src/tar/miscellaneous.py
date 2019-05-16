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
        if not args.load:
            tag = args.model + "_" + now
            self.dir = os.path.join(os.environ['SR_PROJECT_OUTS_PATH'], tag)
        else:
            assert len(args.load.split("x")) == 2
            path, tag = args.load.split("x")
            assert path in ["outs", "models"]
            if path == "outs":
                path = os.environ['SR_PROJECT_OUTS_PATH']
            else:
                path = os.environ['SR_PROJECT_MODELS_PATH']
            self.dir = os.path.join(path, tag)
            if not os.path.exists(self.dir):
                raise ValueError("Loading path %s does not exists !" % self.dir)
            self.log = torch.load(self.get_path('psnr_log.pt'))
        # Creating output directories for model.
        os.makedirs(self.dir, exist_ok=True)
        os.makedirs(self.get_path('model'), exist_ok=True)
        # Create output directory for test datasets.
        for d in args.data_test:
            os.makedirs(self.get_path('results_{}'.format(d)), exist_ok=True)
        # Create output directory for validation datasets.
        for d in args.data_valid:
            os.makedirs(self.get_path('results_{}'.format(d)), exist_ok=True)
        # Create output directory for logging data and write config.
        open_type = 'a' if os.path.exists(self.get_path('log.txt')) else 'w'
        self.log_file = open(self.get_path('log.txt'), open_type)
        if not args.valid_only:
            with open(self.get_path('config.txt'), open_type) as f:
                f.write(now + '\n\n')
                for arg in vars(args):
                    f.write('{}: {}\n'.format(arg, getattr(args, arg)))
                f.write('\n')
        # Build list containing names of test and validation datasets
        # for logging and plotting purposes.
        self.log_datasets = self.args.data_test.copy()
        for dv in self.args.data_valid:
            for sv in self.args.scales_valid:
                self.log_datasets.append(dv + "x" + str(sv))
        # Set number of logging threats.
        self.n_processes  = 8
        self.iter_is_best = False
        self.ready        = True
        self.write_log("Building model module ...")
        if args.load:
            self.write_log('... continue from epoch {}...'.format(len(self.log)))
        self.write_log("... successfully built checkpoint module !")

    def step(self, **kwargs):
        self.add_log(torch.zeros(1, len(self.log_datasets), 2))
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
        labels = ("SHRT", "SLR")
        for id, d in enumerate(self.log_datasets):
            fig = plt.figure()
            label = "PSNR on {}".format(d)
            plt.title(label)
            for i in range(self.log.shape[2]):
                plt.plot(axis,self.log[:, id, i].numpy(),label="{}".format(labels[i]))
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

def convert_flow_to_color(flow):
    hsv = np.zeros((flow.shape[0],flow.shape[1],3), dtype=np.uint8)
    hsv[:,:,1] = 255
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[:,:,0] = ang*180/np.pi/2
    hsv[:,:,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    return cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

def is_power2(num):
    return num != 0 and ((num & (num - 1)) == 0)

def all_power2(numbers):
    return all([is_power2(x) for x in numbers])
