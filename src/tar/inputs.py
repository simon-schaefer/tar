#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Simon Schaefer
# Description : Collection of input arguments.
# =============================================================================
import argparse
import os

parser = argparse.ArgumentParser(description="tar")
parser.add_argument("--template", default=".",
                    help="set various templates in option.py")
parser.add_argument("--verbose", action="store_false",
                    help="print in log file and terminal (default=True)")

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
                    choices=("DIV2K"),
                    help="training dataset name (= 1 dataset!)")
parser.add_argument("--data_test", type=str, default="DIV2K",
                    choices=("DIV2K"),
                    help="testing datasets name (>= 1 dataset!)")
parser.add_argument("--data_valid", default=["SET5","SET14"],
                    choices=("URBAN100","SET5","SET14","BSDS100", "VDIV2K"),
                    help="validation datasets names (>= 1 dataset!)")
parser.add_argument("--data_range", type=str, default="",
                    help="train/test data range")
parser.add_argument("--scales_train", type=list, default=[2],
                    help="super resolution scales for training/testing")
parser.add_argument("--scales_guidance", type=list, default=[2],
                    help="subset of training in which guidance image should be added")
parser.add_argument("--scales_valid", type=list, default=[2,4],
                    help="list of validation scales")
parser.add_argument("--patch_size", type=int, default=96,
                    help="input patch size")
parser.add_argument("--rgb_range", type=int, default=255,
                    help="maximum pixel intensity value")
parser.add_argument("--n_colors", type=int, default=3,
                    help="number of color channels to use")
parser.add_argument("--no_normalize", action="store_true",
                    help="not normalize inputs to [norm_min, norm_max] (default=False)")
parser.add_argument("--norm_min", type=float, default=0.0,
                    help="normalization lower border")
parser.add_argument("--norm_max", type=float, default=1.0,
                    help="normalization upper border")
parser.add_argument("--no_augment", action="store_false",
                    help="not use data augmentation (default=True)")

# =============================================================================
# Model specifications.
# =============================================================================
parser.add_argument("--model", default="AETAD_3D",
                    help="model name")
parser.add_argument("--model_type", type=str, default="",
                    choices=("", "TAD"),
                    help="kind of model specifying opt. core")

# =============================================================================
# Training specifications.
# =============================================================================
parser.add_argument("--reset", action="store_true",
                    help="reset the training (default=False)")
parser.add_argument("--epochs_base", type=int, default=500,
                    help="number of epochs to train on base scale")
parser.add_argument("--epochs_zoom", type=int, default=0,
                    help="number of epochs to train on further scale")
parser.add_argument("--fine_tuning", type=int, default=400,
                    help="start fine tuning model at epoch")
parser.add_argument("--batch_size", type=int, default=16,
                    help="input batch size for training")
parser.add_argument("--valid_only", action="store_true",
                    help="validate only, no training (default=False)")
parser.add_argument("--precision", type=str, default="single",
                    choices=("single", "half"),
                    help="floating point precision(single | half)")

# =============================================================================
# Optimization specifications.
# =============================================================================
parser.add_argument("--lr", type=float, default=1e-3,
                    help="learning rate")
parser.add_argument("--decay", type=str, default="100",
                    help="learning rate decay interval (epochs)")
parser.add_argument("--gamma", type=float, default=0.9,
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
                    help="weight decay factor for model simplification")
parser.add_argument("--gclip", type=float, default=0,
                    help="gradient clipping threshold (0 = no clipping)")

# =============================================================================
# Loss specifications.
# =============================================================================
parser.add_argument("--loss", type=str, default="HR*10*L1+LR*1*L1",
                    help="loss function configuration")

# =============================================================================
# Log specifications.
# =============================================================================
parser.add_argument("--save", type=str, default="test",
                    help="directory name to save")
parser.add_argument("--load", type=str, default="",
                    help="directory to load, format [outs,models]xdir_name")
parser.add_argument("--resume", type=int, default=-2,
                    help="resume from specific checkpoint (-1=latest, -2=best)")
parser.add_argument("--save_models", action="store_true",
                    help="save all intermediate models (default=False)")
parser.add_argument("--print_every", type=int, default=20,
                    help="how many batches to wait before logging training status")
parser.add_argument("--save_results", action="store_false",
                    help="save output results (default=True)")
parser.add_argument("--save_every", type=int, default=10,
                    help="save output/models every x steps if save_result flag is set")

# =============================================================================
# TEMPLATES.
# =============================================================================
def set_template(args):
    if args.template.find("IM_AE_TAD_DIV2K") >= 0:
        args.model      = "AETAD_3D"
        args.model_type = "TAD"
        args.optimizer  = "ADAM"
        args.data_train = "DIV2K"
        args.data_test  = "DIV2K"
        args.data_range  =  "1-700/1-10" #"1-700/701-800"
        args.n_colors   = 3
        args.patch_size = 96

    if args.template.find("IM_AE_SIMPLE") >= 0:
        args.model      = "AETAD_3D"
        args.model_type = "TAD"
        args.optimizer  = "ADAM"
        args.data_train = "SIMPLE"
        args.data_test  = "SIMPLE"
        args.n_colors   = 3
        args.patch_size = 96

# =============================================================================
# MAIN.
# =============================================================================
args = parser.parse_args()
set_template(args)

def reformat_to_list(inputs):
    if type(inputs) == list: return inputs
    elif type(inputs) == int: return [inputs]
    elif type(inputs) == str: return [int(x) for x in inputs.split(",")]

# Reformat arguments.
args.data_train = args.data_train.split("+")
args.data_test = args.data_test.split("+")
args.scales_train = reformat_to_list(args.scales_train)
args.scales_guidance = reformat_to_list(args.scales_guidance)
args.scales_valid = reformat_to_list(args.scales_valid)
if type(args.data_valid) == str: args.data_valid = [args.data_valid]
if type(args.data_test) == str: args.data_test = [args.data_test]
for arg in vars(args):
    if vars(args)[arg] == "True":
        vars(args)[arg] = True
    elif vars(args)[arg] == "False":
        vars(args)[arg] = False
