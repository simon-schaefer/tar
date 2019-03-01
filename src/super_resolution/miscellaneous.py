#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Simon Schaefer
# Description : Collection of miscellaneous helper functions. 
# =============================================================================
import argparse
import imageio
import math
from multiprocessing import Process, Queue
import os
import sys
import time
import typing

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch

class _Checkpoint_(object):
    ''' Logging class for model training, including saving the model, 
    optimizer state and loss curve (in a parallel threat). '''

    def __init__(self, args: argparse.Namespace):
        super(_Checkpoint_, self).__init__()
        # Initializing checkpoint module. 
        print("Building checkpoint module ...")
        self.ready = False
        self.args = args
        self.log = torch.Tensor()
        # Building model directory based on name and time. 
        now = time.strftime("%H_%M_%S_%d_%b", time.gmtime())
        tag = args.model + "_" + now
        if not args.load:
            self.dir = os.path.join(os.environ['SR_PROJECT_OUTS_PATH'], tag)
        else:
            self.dir = os.path.join(os.environ['SR_PROJECT_OUTS_PATH'], args.load)
            if not os.path.exists(self.dir):
                raise ValueError("Loading path %s does not exists !" % self.dir)
            self.log = torch.load(self.get_path('psnr_log.pt'))
            print('Continue from epoch {}...'.format(len(self.log)))
        # Creating output directories for model, results, logging, config, etc. 
        os.makedirs(self.dir, exist_ok=True)
        os.makedirs(self.get_path('model'), exist_ok=True)
        for d in args.data_test:
            os.makedirs(self.get_path('results-{}'.format(d)), exist_ok=True)
        open_type = 'a' if os.path.exists(self.get_path('log.txt')) else 'w'
        self.log_file = open(self.get_path('log.txt'), open_type)
        with open(self.get_path('config.txt'), open_type) as f:
            f.write(now + '\n\n')
            for arg in vars(args):
                f.write('{}: {}\n'.format(arg, getattr(args, arg)))
            f.write('\n')
        # Set number of logging threats. 
        self.n_processes = 8
        self.ready = True
        print("... successfully built checkpoint module !")

    # =========================================================================
    # Saving 
    # =========================================================================
    def save(self, trainer, epoch: int, is_best: bool=False):
        trainer.model.save(self.get_path('model'), epoch, is_best=is_best)
        trainer.loss.save(self.dir)
        trainer.loss.plot_loss(self.dir, epoch)
        trainer.optimizer.save(self.dir)
        torch.save(self.log, self.get_path('psnr_log.pt'))
        # Plot peak signal-to-noise ratio (PSNR) plot. 
        axis = np.linspace(1, epoch, epoch)
        for idx_data, d in enumerate(self.args.data_test):
            label = 'SR on {}'.format(d)
            fig = plt.figure()
            plt.title(label)
            plt.plot(axis,self.log[:, 0, 0].numpy(),
                     label='Scale {}'.format(scale))
            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel('PSNR')
            plt.grid(True)
            plt.savefig(self.get_path('test_{}.pdf'.format(d)))
            plt.close(fig)
        
    def save_results(self, dataset, filename: str, save_list: typing.List[str], scale: int):
        if self.args.save_results:
            filename = self.get_path(
                'results-{}'.format(dataset.dataset.name),
                '{}_x{}_'.format(filename, scale)
            )
            postfix = ('SR', 'LR', 'HR')
            for v, p in zip(save_list, postfix):
                normalized = v[0].mul(255 / self.args.rgb_range)
                tensor_cpu = normalized.byte().permute(1, 2, 0).cpu()
                self.queue.put(('{}{}.png'.format(filename, p), tensor_cpu))

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
    ''' Time logging class based on time library. '''

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
    ''' Draws progress bar showing the number of executed 
    iterations over the overall number of iterations. 
    Increments the iteration and returns it. '''
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

def calc_psnr(x: torch.Tensor, y: torch.Tensor, scale: int, rgb_range: float) -> float:
    ''' Determine peak signal to noise ratio between to tensors 
    (mostly images, given as torch tensors), based on scale. '''
    if x.nelement() == 1: return 0
    diff = (x - y) / rgb_range
    shave = scale + 6
    valid = diff[..., shave:-shave, shave:-shave]
    mse = valid.pow(2).mean()
    return -10 * math.log10(mse)

def discretize(img: torch.Tensor, rgb_range: float) -> torch.Tensor:
    ''' Discretize image (given as torch tensor) in defined range of
    pixel values (e.g. 255 or 1.0). '''
    pixel_range = 255 / rgb_range
    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)