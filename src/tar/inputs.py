#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Simon Schaefer
# Description : Collection of input arguments.
# =============================================================================
import argparse
import os

parser = argparse.ArgumentParser(description="tar")
parser.add_argument("--type", type=str, default="SCALING",
                    choices=("SCALING" "COLORING"))
parser.add_argument("--format", type=str, default="IMAGE",
                    choices=("IMAGE", "VIDEO"))
parser.add_argument("--external", type=str, default="",
                    choices=("SOFVSR"))
parser.add_argument("--template", default="valid",
                    help="set various templates in option.py")
parser.add_argument("--verbose", action="store_false",
                    help="print in log file and terminal (default=True)")

# =============================================================================
# Hardware specifications.
# =============================================================================
parser.add_argument("--n_threads", type=int, default=5,
                    help="number of threads for data loading")
parser.add_argument("--cpu", action="store_true",
                    help="use cpu only (default=False)")
parser.add_argument("--cuda_device", type=str, default="cuda:0",
                    help="index name of used GPU")
parser.add_argument("--n_gpus", type=int, default=1,
                    help="number of GPUs (unblocked device_count gives error)")

# =============================================================================
# Data specifications.
# =============================================================================
parser.add_argument("--dir_data", type=str, default=os.environ["SR_PROJECT_DATA_PATH"],
                    help="dataset directory")
parser.add_argument("--data_train", type=str, default="DIV2K",
                    choices=("DIV2K", "NTIAASPEN"),
                    help="training dataset name (= 1 dataset!)")
parser.add_argument("--data_valid", default="SET5:SET14",
                    help="validation datasets names (>= 1 dataset!), \
                    choices=URBAN100,SET5,SET14,BSDS100,VDIV2K,CUSTOM, \
                    CALENDAR,NTIAASPEN,WALK,CITY,FOLIAGE")
parser.add_argument("--scales_train", type=str, default=[2],
                    help="super resolution scales for training/testing")
parser.add_argument("--scales_guidance", type=str, default=[1,2,4,8,16],
                    help="subset of training in which guidance image should be added")
parser.add_argument("--scales_valid", type=str, default=[2,4],
                    help="list of validation scales")
parser.add_argument("--patch_size", type=int, default=96,
                    help="input patch size")
parser.add_argument("--max_test_samples", type=int, default=20,
                    help="maximal number of testing samples (non saving epoch)")
parser.add_argument("--norm_min", type=float, default=0.0,
                    help="normalization lower border")
parser.add_argument("--norm_max", type=float, default=1.0,
                    help="normalization upper border")
parser.add_argument("--no_augment", action="store_true",
                    help="use data augmentation (default=False)")

# =============================================================================
# Model specifications.
# =============================================================================
parser.add_argument("--model", default="AETAD",
                    help="model name")

# =============================================================================
# Training specifications.
# =============================================================================
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

# =============================================================================
# Optimization specifications.
# =============================================================================
parser.add_argument("--lr", type=float, default=4e-4,
                    help="learning rate")
parser.add_argument("--decay", type=str, default="20-100-200",
                    help="learning rate decay interval (epochs)")
parser.add_argument("--gamma", type=float, default=0.25,
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
parser.add_argument("--load", type=str, default="",
                    help="directory to load model from, training is not \
                    continued to not overwrite, format [outs,models]xdir_name")
parser.add_argument("--resume", type=int, default=-2,
                    help="resume from specific checkpoint (-1=latest, -2=best)")
parser.add_argument("--save_models", action="store_true",
                    help="save all intermediate models (default=False)")
parser.add_argument("--save_results", action="store_false",
                    help="save output results (default=True)")
parser.add_argument("--save_every", type=int, default=20,
                    help="save output/models every x steps if save_result flag is set")
parser.add_argument("--print_every", type=int, default=20,
                    help="how many batches to wait before logging training status")

# =============================================================================
# TEMPLATES.
# =============================================================================
def set_template(args):

    # Training dataset.
    if args.template.count("DIV2K") > 0:
        args.data_train = "DIV2K"
    if args.template.count("NTIAASPEN") > 0:
        args.data_train = "NTIAASPEN"
        args.data_valid = "CALENDAR"
    if args.template.count("INTIAASPEN") > 0:
        args.data_train = "INTIAASPEN"
        args.data_valid = "CALENDAR"
    # Model.
    if args.template.count("AETAD") > 0:
        args.model      = "AETAD"
    if args.template.count("AETAD_SMALL") > 0:
        args.model      = "AETAD_SMALL"
    if args.template.count("AETAD_LARGE") > 0:
        args.model      = "AETAD_LARGE"
    # Type.
    if args.template.count("ISCALE") > 0:
        args.type       = "SCALING"
        args.format     = "IMAGE"
    elif args.template.count("VSCALE") > 0:
        args.type       = "SCALING"
        args.format     = "VIDEO"
    elif args.template.count("ICOLOR") > 0:
        args.type       = "COLORING"
        args.format     = "IMAGE"
        args.model      = "AETAD_COLOR"
    # Scales.
    if args.template.count("_4") > 0:
        args.scales_train = [4]

    # Specific types.
    if args.template.find("ICOLOR_AETAD_DIV2K") >= 0:
        args.loss         = "COL*100*L1+GRY*1*L1"
        args.scales_train = [1]
        args.scales_valid = [1]

    if args.template.find("ISCALE_AETAD_NTIAASPEN_4") >= 0:
        args.scales_valid = [4]

    if args.template.find("VSCALE_AETAD_SOFVSR") >= 0:
        args.load       = "modelsxiscale4"
        args.format     = "VIDEO"
        args.type       = "SCALING"
        args.loss       = "HR*1*L1+LR*1*L1+EXT*100*L1"
        args.external   = "SOFVSR"
        args.scales_train = [4]
        args.scales_valid = [4]
        args.batch_size = 6

# =============================================================================
# MAIN.
# =============================================================================
args = parser.parse_args()
set_template(args)

def reformat_to_list(inputs):
    if type(inputs) == list: return inputs
    elif type(inputs) == int: return [inputs]
    elif type(inputs) == str: return [int(x) for x in inputs[1:-1].split(",")]

# Reformat arguments.
args.scales_train = reformat_to_list(args.scales_train)
args.scales_guidance = reformat_to_list(args.scales_guidance)
args.scales_valid = reformat_to_list(args.scales_valid)
args.data_valid = args.data_valid.split(":")
#args.data_valid.append(args.data_train)
for arg in vars(args):
    if vars(args)[arg] == "True":
        vars(args)[arg] = True
    elif vars(args)[arg] == "False":
        vars(args)[arg] = False
