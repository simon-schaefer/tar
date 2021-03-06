import numpy as np
from PIL import Image
import glob
import os
import torch
from torch.utils.data.dataset import Dataset
import math
import random

class TrainsetLoader(Dataset):
    def __init__(self, trainsets, upscale_factor, patch_size, n_iters):
        super(TrainsetLoader).__init__()
        self.trainsets = trainsets
        self.upscale_factor = upscale_factor
        self.patch_size = patch_size
        self.n_iters = n_iters

    def __getitem__(self, idx):
        path = os.environ["SR_PROJECT_DATA_PATH"]
        idx_video = random.randint(0, len(self.trainsets)-1)
        idx_frame = random.randint(0, 28)
        lr_dir = os.path.join(path, self.trainsets[idx_video], "LR_bicubic")
        lr_dir = lr_dir + '/X' + str(self.upscale_factor)
        hr_dir = os.path.join(path, self.trainsets[idx_video], "HR")
        # read HR & LR frames
        sc = "x" + str(self.upscale_factor)
        LR0 = Image.open(lr_dir + '/hr' + str(idx_frame) + sc + '.png')
        LR1 = Image.open(lr_dir + '/hr' + str(idx_frame + 1) + sc + '.png')
        LR2 = Image.open(lr_dir + '/hr' + str(idx_frame + 2) + sc + '.png')
        HR0 = Image.open(hr_dir + '/hr' + str(idx_frame) + '.png')
        HR1 = Image.open(hr_dir + '/hr' + str(idx_frame + 1) + '.png')
        HR2 = Image.open(hr_dir + '/hr' + str(idx_frame + 2) + '.png')

        LR0 = np.array(LR0, dtype=np.float32) / 255.0
        LR1 = np.array(LR1, dtype=np.float32) / 255.0
        LR2 = np.array(LR2, dtype=np.float32) / 255.0
        HR0 = np.array(HR0, dtype=np.float32) / 255.0
        HR1 = np.array(HR1, dtype=np.float32) / 255.0
        HR2 = np.array(HR2, dtype=np.float32) / 255.0
        # extract Y channel for LR inputs
        HR0 = rgb2y(HR0)
        HR1 = rgb2y(HR1)
        HR2 = rgb2y(HR2)
        LR0 = rgb2y(LR0)
        LR1 = rgb2y(LR1)
        LR2 = rgb2y(LR2)
        # crop patchs randomly
        HR0, HR1, HR2, LR0, LR1, LR2 = random_crop(HR0, HR1, HR2, LR0, LR1, LR2, self.patch_size, self.upscale_factor)

        HR0 = HR0[:, :, np.newaxis]
        HR1 = HR1[:, :, np.newaxis]
        HR2 = HR2[:, :, np.newaxis]
        LR0 = LR0[:, :, np.newaxis]
        LR1 = LR1[:, :, np.newaxis]
        LR2 = LR2[:, :, np.newaxis]

        HR = np.concatenate((HR0, HR1, HR2), axis=2)
        LR = np.concatenate((LR0, LR1, LR2), axis=2)
        # data augmentation
        LR, HR = augumentation()(LR, HR)
        return toTensor(LR), toTensor(HR)

    def __len__(self):
        return self.n_iters


class TestsetLoader(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super(TestsetLoader).__init__()
        self.dataset_dir = dataset_dir
        self.upscale_factor = upscale_factor
        path = os.environ["SR_PROJECT_DATA_PATH"]
        lr_dir = os.path.join(path, self.dataset_dir, "LR_bicubic")
        lr_dir = lr_dir + '/X' + str(self.upscale_factor)
        self.frame_list = sorted(glob.glob(lr_dir + "/*" + ".png"))

    def __getitem__(self, idx):
        LR0 = Image.open(self.frame_list[idx])
        LR1 = Image.open(self.frame_list[idx+1])
        LR2 = Image.open(self.frame_list[idx+2])
        W, H = LR1.size
        # H and W should be divisible by 2
        W = int(W // 2) * 2
        H = int(H // 2) * 2
        LR0 = LR0.crop([0, 0, W, H])
        LR1 = LR1.crop([0, 0, W, H])
        LR2 = LR2.crop([0, 0, W, H])

        LR1_bicubic = LR1.resize((W*self.upscale_factor, H*self.upscale_factor), Image.BICUBIC)
        LR1_bicubic = np.array(LR1_bicubic, dtype=np.float32) / 255.0

        LR0 = np.array(LR0, dtype=np.float32) / 255.0
        LR1 = np.array(LR1, dtype=np.float32) / 255.0
        LR2 = np.array(LR2, dtype=np.float32) / 255.0
        # extract Y channel for LR inputs
        LR0_y, _, _ = rgb2ycbcr(LR0)
        LR1_y, _, _ = rgb2ycbcr(LR1)
        LR2_y, _, _ = rgb2ycbcr(LR2)

        LR0_y = LR0_y[:, :, np.newaxis]
        LR1_y = LR1_y[:, :, np.newaxis]
        LR2_y = LR2_y[:, :, np.newaxis]
        LR = np.concatenate((LR0_y, LR1_y, LR2_y), axis=2)

        LR = toTensor(LR)
        # generate Cr, Cb channels using bicubic interpolation
        _, SR_cb, SR_cr = rgb2ycbcr(LR1_bicubic)
        return LR, SR_cb, SR_cr

    def __len__(self):
        return self.frame_list.__len__() - 2

class augumentation(object):
    def __call__(self, input, target):
        if random.random()<0.5:
            input = input[:, ::-1, :]
            target = target[:, ::-1, :]
        if random.random()<0.5:
            input = input[::-1, :, :]
            target = target[::-1, :, :]
        if random.random()<0.5:
            input = input.transpose(1, 0, 2)
            target = target.transpose(1, 0, 2)
        return np.ascontiguousarray(input), np.ascontiguousarray(target)

def random_crop(HR0, HR1, HR2, LR0, LR1, LR2, patch_size_lr, upscale_factor):
    h_hr, w_hr = HR0.shape
    h_lr = h_hr // upscale_factor
    w_lr = w_hr // upscale_factor
    idx_h = random.randint(10, h_lr - patch_size_lr - 10)
    idx_w = random.randint(10, w_lr - patch_size_lr - 10)

    h_start_hr = (idx_h - 1) * upscale_factor
    h_end_hr = (idx_h - 1 + patch_size_lr) * upscale_factor
    w_start_hr = (idx_w - 1) * upscale_factor
    w_end_hr = (idx_w - 1 + patch_size_lr) * upscale_factor

    h_start_lr = idx_h - 1
    h_end_lr = idx_h - 1 + patch_size_lr
    w_start_lr = idx_w - 1
    w_end_lr = idx_w - 1 + patch_size_lr

    HR0 = HR0[h_start_hr:h_end_hr, w_start_hr:w_end_hr]
    HR1 = HR1[h_start_hr:h_end_hr, w_start_hr:w_end_hr]
    HR2 = HR2[h_start_hr:h_end_hr, w_start_hr:w_end_hr]
    LR0 = LR0[h_start_lr:h_end_lr, w_start_lr:w_end_lr]
    LR1 = LR1[h_start_lr:h_end_lr, w_start_lr:w_end_lr]
    LR2 = LR2[h_start_lr:h_end_lr, w_start_lr:w_end_lr]
    return HR0, HR1, HR2, LR0, LR1, LR2

def toTensor(img):
    img = torch.from_numpy(img.transpose((2, 0, 1)))
    img.float().div(255)
    return img

def rgb2ycbcr(img_rgb):
    ## the range of img_rgb should be (0, 1)
    img_y = 0.257 * img_rgb[:, :, 0] + 0.504 * img_rgb[:, :, 1] + 0.098 * img_rgb[:, :, 2] + 16 / 255.0
    img_cb = -0.148 * img_rgb[:, :, 0] - 0.291 * img_rgb[:, :, 1] + 0.439 * img_rgb[:, :, 2] + 128 / 255.0
    img_cr = 0.439 * img_rgb[:, :, 0] - 0.368 * img_rgb[:, :, 1] - 0.071 * img_rgb[:, :, 2] + 128 / 255.0
    return img_y, img_cb, img_cr

def ycbcr2rgb(img_ycbcr):
    ## the range of img_ycbcr should be (0, 1)
    img_r = 1.164 * (img_ycbcr[:, :, 0] - 16 / 255.0) + 1.596 * (img_ycbcr[:, :, 2] - 128 / 255.0)
    img_g = 1.164 * (img_ycbcr[:, :, 0] - 16 / 255.0) - 0.392 * (img_ycbcr[:, :, 1] - 128 / 255.0) - 0.813 * (img_ycbcr[:, :, 2] - 128 / 255.0)
    img_b = 1.164 * (img_ycbcr[:, :, 0] - 16 / 255.0) + 2.017 * (img_ycbcr[:, :, 1] - 128 / 255.0)
    img_r = img_r[:, :, np.newaxis]
    img_g = img_g[:, :, np.newaxis]
    img_b = img_b[:, :, np.newaxis]
    img_rgb = np.concatenate((img_r, img_g, img_b), 2)
    return img_rgb

def rgb2y(img_rgb):
    ## the range of img_rgb should be (0, 1)
    image_y = 0.257 * img_rgb[:, :, 0] + 0.504 * img_rgb[:, :, 1] + 0.098 * img_rgb[:, :, 2] +16 / 255.0
    return image_y
