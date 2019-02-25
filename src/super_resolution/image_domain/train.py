#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Simon Schaefer
# Description : Main training environment for image domain. 
# Input arg   : parameters file name.   
# =============================================================================
import os
import sys
import time 
import yaml

import torch
import torchvision

import super_resolution.image_domain.dataloader as dataloader
import super_resolution.image_domain.models as models
import super_resolution.tools.image_toolbox as img_tools
import super_resolution.tools.miscellaneous as misc

# Load parameters. 
assert len(sys.argv) >= 2
filename = os.environ['SR_PROJECT_PARAMS_PATH'] + "/" + sys.argv[1] + ".yaml"
stream = open(filename, 'r')
params = yaml.load(stream)

# Print introduction and determine model tag.  
misc.print_header()
tag = params["model"] + "_" + time.strftime("%H_%M_%S_%d_%b", time.gmtime())

# Load datasets in data loader.  
print("Loading training and validation datasets ...")
datasets_path = os.environ['SR_PROJECT_DATA_PATH']
datasets_train = [datasets_path + "/" + x for x in params["datasets_train"]]
datasets_valid = [datasets_path + "/" + x for x in params["datasets_valid"]]
data = dataloader.DataLoader(
        datasets_train, datasets_valid,
        scale_guidance=params["scale_guidance"], 
        batch_size=params["batch_size"], subsize=params["subsize"])

# Loading model.
print("Building model ...") 
model = models.models_by_name(params["model"]).cuda()

# Defining training optimization. 
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), 
                lr=float(params["learning_rate"]),
                weight_decay=float(params["weight_decay"]))
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                step_size=params["lr_decay_steps"], gamma=params["lr_decay_gamma"])

# Training loop. 
print("Starting training loop ...")
for epoch in range(params["num_epochs"]):
    epoch_is_over = False
    iteration = 0
    while not epoch_is_over: 
        # Load and normalize batch. 
        batch_hr, batch_lr, epoch_is_over = data.next_batch()
        batch_hr = img_tools.normalize(batch_hr)
        batch_lr = img_tools.normalize(batch_lr)
        # Convert to torch tensors. 
        batch_hr = torch.from_numpy(batch_hr).float().cuda()
        batch_lr = torch.from_numpy(batch_lr).float().cuda()
        batch_hr = torch.autograd.Variable(batch_hr).cuda()
        # Forward pass. 
        output_lr = model.encode(batch_hr)
        output_hr = model.forward(batch_hr)
        loss = criterion(output_hr, batch_hr) \
             + params["loss_lambda"]*criterion(output_lr, batch_lr)
        # Backward pass. 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Print progress. 
        iteration = misc.progress_bar(iteration, data.num_train_samples)
    # Scheduling. 
    scheduler.step()
    # Logging loss results and epoch. 
    print('epoch [{}/{}], loss:{:.4f}'
        .format(epoch+1, params["num_epochs"], loss.data))
    # Every few epochs save picture of decoder outputs. 
    if epoch % 10 == 0:
        pic_path = os.environ['SR_PROJECT_OUTS_PATH'] + "/" + tag
        torchvision.utils.save_image(output_hr.cpu().data, 
                pic_path + "_epoch_{}.png".format(epoch), normalize=True)

# Save model. 
model_path = os.environ['SR_PROJECT_OUTS_PATH'] + "/" + tag + ".pth"
torch.save(model.state_dict(), model_path)
print("Stored model parameters to output directory !")