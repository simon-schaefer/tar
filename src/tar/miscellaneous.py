#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Simon Schaefer
# Description : Collection of miscellaneous helper functions. 
# =============================================================================
import argparse
import csv
import imageio
import math
from multiprocessing import Process, Queue
import numpy as np
import os
import random
import sys
import time
from typing import Dict, List

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
        self.args = args
        self.log = torch.Tensor()
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
        if type(args.data_test) == str: 
            args.data_test = [args.data_test]
        for d in args.data_test:
            os.makedirs(self.get_path('results-{}'.format(d)), exist_ok=True)
        # Create output directory for validation datasets. 
        if type(args.data_valid) == str: 
            args.data_valid = [args.data_valid]
        for d in args.data_valid: 
            os.makedirs(self.get_path('results-{}'.format(d)), exist_ok=True)
        # Create output directory for logging data and write config. 
        open_type = 'a' if os.path.exists(self.get_path('log.txt')) else 'w'
        self.log_file = open(self.get_path('log.txt'), open_type)
        if not args.valid_only: 
            with open(self.get_path('config.txt'), open_type) as f:
                f.write(now + '\n\n')
                for arg in vars(args):
                    f.write('{}: {}\n'.format(arg, getattr(args, arg)))
                f.write('\n')
        # Set number of logging threats. 
        self.n_processes = 8
        self.ready = True
        self.write_log("Building model module ...")
        if args.load: 
            self.write_log('... continue from epoch {}...'.format(len(self.log)))
        self.write_log("... successfully built checkpoint module !")

    # =========================================================================
    # Saving 
    # =========================================================================
    def save(self, trainer, epoch: int, scaling: int, is_best: bool=False):
        trainer.model.save(self.get_path('model'), epoch, is_best=is_best)
        trainer.loss.save(self.dir)
        trainer.loss.plot_loss(self.dir, epoch, scaling="linear")
        trainer.loss.plot_loss(self.dir, epoch, scaling="logarithmic")
        trainer.optimizer.save(self.dir)
        torch.save(self.log, self.get_path('psnr_log.pt'))
        # Plot peak signal-to-noise ratio (PSNR) plot. 
        axis = np.linspace(1, epoch, epoch)
        labels = ("HR", "LR")
        for id, d in enumerate(self.args.data_test):
            fig = plt.figure()
            label = "PSNR on {}".format(d)
            plt.title(label)
            for i in range(self.log.shape[2]):             
                plt.plot(axis,self.log[:, id, i].numpy(),
                        label="{}: scaling {}".format(labels[i], scaling))
            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel('PSNR')
            plt.grid(True)
            plt.savefig(self.get_path("test_{}.pdf".format(d)))
            plt.close(fig)
        
    def save_results(self, save_list: List[torch.Tensor], desc_list: List[str], 
                     filename: str, dataset, scale: int):
        filename = self.get_path(
            'results-{}'.format(dataset.dataset.name),
            '{}_x{}_'.format(filename, scale)
        )
        assert len(save_list) == len(desc_list)
        for v, p in zip(save_list, desc_list):
            normalized = v[0]
            if not self.args.no_normalize: 
                r = 255/(self.args.norm_max - self.args.norm_min)
                normalized = normalized.add(-self.args.norm_min).mul(r)
            else: 
                normalized = normalized.mul(255/self.args.rgb_range)
            normalized = normalized.clamp(0, 255).round()
            tensor_cpu = normalized.byte().permute(1, 2, 0).cpu()
            self.queue.put(('{}{}.png'.format(filename, p), tensor_cpu))

    def save_validations(self, valids: List[Dict[str,str]]): 
        file_path = self.get_path('validations.csv')
        csv_data  = [] 
        csv_data.append(sorted(valids[0].keys()))
        for v in valids: 
            csv_data.append(x for (_, x) in sorted(v.items()))
        with open(file_path, 'w') as f:
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
    if x.nelement() == 1: return 0
    px, py = None, None
    if patch_size is None: px, py = x, y
    else: 
        h, w = x.shape[2:4]
        lp = int(patch_size)
        lx = random.randrange(0, w - lp + 1)
        ly = random.randrange(0, h - lp + 1)    
        px = x[:, :, ly:ly + lp, lx:lx + lp]
        py = y[:, :, ly:ly + lp, lx:lx + lp]
    mse = torch.dist(px, py, 2).pow(2).mean()
    return 10 * math.log10(rgb_range**2/mse)

def discretize(img: torch.Tensor, rgb_range: float, 
               normalized: bool, norm_range: List[float]) -> torch.Tensor:
    """ Discretize image (given as torch tensor) in defined range of
    pixel values (e.g. 255 or 1.0), i.e. smart rounding. """
    pixel_range = 255 * (norm_range[1] - norm_range[0])
    img_dis = img if not normalized else img.add(-norm_range[0])
    img_dis = img_dis.mul(pixel_range).clamp(0, 255).round().div(pixel_range)
    img_dis = img_dis if not normalized else img_dis.add(norm_range[0])
    return img_dis

def normalize(img, rgb_range, norm_min, norm_max): 
    """ Normalize numpy array or torch tensor from RGB range
    to given norm range [norm_min, norm_max]. """
    assert norm_max > norm_min 
    norm_range = norm_max - norm_min
    return img/rgb_range*norm_range + norm_min

def unnormalize(img, rgb_range, norm_min, norm_max): 
    """ Unnormalize numpy array or torch tensor from given norm
    range [norm_min, norm_max] to RGB range. """
    assert norm_max > norm_min 
    norm_range = norm_max - norm_min
    return (img - norm_min)/norm_range*rgb_range