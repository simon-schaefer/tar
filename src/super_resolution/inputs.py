#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Simon Schaefer
# Description : Collection of input arguments. 
# =============================================================================
import argparse
import os

parser = argparse.ArgumentParser(description="super_resolution")
parser.add_argument("--template", default=".",
                    help="set various templates in option.py")
parser.add_argument("--verbose", action="store_true",
                    help="print in log file and terminal")

# =============================================================================
# Hardware specifications.  
# =============================================================================
parser.add_argument("--n_threads", type=int, default=6,
                    help="number of threads for data loading")
parser.add_argument("--cpu", action="store_true",
                    help="use cpu only (default=False)")
parser.add_argument("--cuda_device", type=str, default="cuda:0", 
                    help="index name of used GPU")
parser.add_argument("--n_gpus", type=int, default=1, 
                    help="number of GPUs (unblocked device_count gives error)")
parser.add_argument("--seed", type=int, default=1,
                    help="random seed")

# =============================================================================
# Data specifications.
# =============================================================================
parser.add_argument("--dir_data", type=str, default=os.environ["SR_PROJECT_DATA_PATH"],
                    help="dataset directory")
parser.add_argument("--data_train", type=str, default="DIV2K",
                    choices=("DIV2K", "MNIST"), 
                    help="training datasets name i.e. either str or list")
parser.add_argument("--data_test", type=str, default="DIV2K", 
                    choices=("DIV2K", "MNIST"), 
                    help="testing datasets name i.e. either str or list")
parser.add_argument("--data_range", type=str, default="1-700/701-800",
                    help="train/test data range")
parser.add_argument("--ext", type=str, default="img",
                    choices=("img"),
                    help="dataset file extension")
parser.add_argument("--scale", type=int, default=2,
                    help="super resolution scale")
parser.add_argument("--patch_size", type=int, default=96,
                    help="input patch size")
parser.add_argument("--rgb_range", type=int, default=255,
                    help="maximum pixel intensity value")
parser.add_argument("--n_colors", type=int, default=3,
                    help="number of color channels to use")
parser.add_argument("--no_normalize", action="store_true",
                    help="not normalize inputs to [-0.5, 0.5] (default=False)")
parser.add_argument("--no_augment", action="store_true",
                    help="not use data augmentation (default=False)")

# =============================================================================
# Model specifications.
# =============================================================================
parser.add_argument("--model", default="AETAD_1D",
                    help="model name")
parser.add_argument("--model_type", type=str, default="", 
                    choices=("", "TAD"),
                    help="kind of model specifying opt. core")

# =============================================================================
# Training specifications. 
# =============================================================================
parser.add_argument("--reset", action="store_true",
                    help="reset the training (default=False)")
parser.add_argument("--test_every", type=int, default=1000,
                    help="do test per every N batches")
parser.add_argument("--epochs", type=int, default=3000,
                    help="number of epochs to train")
parser.add_argument("--batch_size", type=int, default=6,
                    help="input batch size for training")
parser.add_argument("--test_only", action="store_true",
                    help="set this option to test the model (default=False)")
parser.add_argument("--precision", type=str, default="single",
                    choices=("single", "half"),
                    help="floating point precision(single | half)")

# =============================================================================
# Optimization specifications.
# =============================================================================
parser.add_argument("--lr", type=float, default=1e-3,
                    help="learning rate")
parser.add_argument("--decay", type=str, default="100",
                    help="learning rate decay type")
parser.add_argument("--gamma", type=float, default=0.6,
                    help="learning rate decay factor for step decay")
parser.add_argument("--optimizer", default="ADAM",
                    choices=("ADAM"),
                    help="optimization class")
parser.add_argument("--momentum", type=float, default=0.9,
                    help="SGD momentum")
parser.add_argument("--betas", type=tuple, default=(0.9, 0.999),
                    help="ADAM beta")
parser.add_argument("--epsilon", type=float, default=1e-8,
                    help="ADAM epsilon for numerical stability")
parser.add_argument("--weight_decay", type=float, default=0,
                    help="weight decay")
parser.add_argument("--gclip", type=float, default=0,
                    help="gradient clipping threshold (0 = no clipping)")

# =============================================================================
# Loss specifications.
# =============================================================================
parser.add_argument("--loss", type=str, default="HR*1*L1",
                    help="loss function configuration")

# =============================================================================
# Log specifications.
# =============================================================================
parser.add_argument("--save", type=str, default="test",
                    help="file name to save")
parser.add_argument("--load", type=str, default="",
                    help="file name to load")
parser.add_argument("--resume", type=int, default=0,
                    help="resume from specific checkpoint")
parser.add_argument("--save_models", action="store_true",
                    help="save all intermediate models (default=False)")
parser.add_argument("--print_every", type=int, default=100,
                    help="how many batches to wait before logging training status")
parser.add_argument("--save_results", action="store_false",
                    help="save output results (default=True)")
parser.add_argument("--save_gt", action="store_false",
                    help="save low-resolution and high-resolution images together (default=True)")

# =============================================================================
# TEMPLATES. 
# =============================================================================
def set_template(args):
    if args.template.find("IM_AE_TAD_MNIST") >= 0:
        args.model      = "AETAD_1D"
        args.model_type = "TAD"
        args.scale      = 2
        args.optimizer  = "ADAM"
        args.batch_size = 6
        args.data_train = "MNIST"
        args.data_test  = "MNIST"
        args.n_colors   = 1
        args.patch_size = 28
        args.loss       = "HR*10*L1+LR*1*L1"

    if args.template.find("IM_AE_TAD_DIV2K") >= 0:
        args.model      = "AETAD_3D"
        args.model_type = "TAD"
        args.scale      = 2
        args.optimizer  = "ADAM"
        args.batch_size = 6
        args.data_train = "DIV2K"
        args.data_test  = "DIV2K"
        args.n_colors   = 3
        args.patch_size = 96
        args.loss       = "HR*1*L1+LR*1*L1"

    if args.template.find("IM_AE_TAD_DIV2K_SMALL") >= 0:
        args.model      = "AETAD_3D_SMALL"
        args.model_type = "TAD"
        args.scale      = 2
        args.optimizer  = "ADAM"
        args.batch_size = 6
        args.data_train = "DIV2K"
        args.data_test  = "DIV2K"
        args.n_colors   = 3
        args.patch_size = 24
        args.loss       = "HR*1*L1+LR*1*L1"

    if args.template.find("IM_AE_TEST") >= 0:
        args.model      = "AETRIAL_NOSR"
        args.scale      = 1
        args.optimizer  = "ADAM"
        args.batch_size = 6
        args.data_train = "MNIST"
        args.data_test  = "MNIST"
        args.n_colors   = 1
        args.patch_size = 28
        args.loss       = "HR*1*MSE"

# =============================================================================
# MAIN. 
# =============================================================================
args = parser.parse_args()
set_template(args)

# Reformat arguments. 
args.data_train = args.data_train.split("+")
args.data_test = args.data_test.split("+")
args.epochs = 1e8 if (args.epochs == 0) else args.epochs   
for arg in vars(args):
    if vars(args)[arg] == "True":
        vars(args)[arg] = True
    elif vars(args)[arg] == "False":
        vars(args)[arg] = False

