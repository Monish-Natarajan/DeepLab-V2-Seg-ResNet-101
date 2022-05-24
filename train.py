#!/usr/bin/env python
# coding: utf-8
#
# Author: Kazuto Nakashima
# URL:    https://kazuto1011.github.io
# Date:   07 January 2019

from __future__ import absolute_import, division, print_function

import json
import multiprocessing
import os

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm

#To change
from model.deeplabv2 import DeepLabV2
from utils import DenseCRF, PolynomialLR, scores
from model.multiscale import MSC
from model.model_utils import resize_labels, get_device, get_params 

from data.pascal_voc import VOC_seg
import data.transforms as Trs

tr_transforms = Trs.Compose([
    Trs.RandomScale(0.5, 1.5),
    Trs.ResizeRandomCrop((321, 321)), 
    Trs.RandomHFlip(0.5), 
    Trs.ColorJitter(0.5,0.5,0.5,0),
    Trs.Normalize_Caffe(),
    ])

def makedirs(dirs):
    if not os.path.exists(dirs):
        os.makedirs(dirs)

def DeepLabV2_ResNet101_MSC(CONFIG):
    return MSC(
        base=DeepLabV2(n_classes=CONFIG.DATASET.N_CLASSES,
                       n_blocks=CONFIG.MODEL.N_BLOCKS,
                       atrous_rates=CONFIG.MODEL.ATROUS_RATES
                       ),
       scales=[0.5, 0.75],
    )

def main(config_path):
    """
    Training and evaluation
    """
    print("Initializing Training")
    train(config_path, cuda=True)


def train(config_path, cuda):
    """
    Training DeepLab by v2 protocol
    """

    # Configuration
    CONFIG = OmegaConf.load(config_path)
    device = get_device(cuda)
    torch.backends.cudnn.benchmark = True  #WHAT FOR???

    #Custom Data Handler
    mydataset = VOC_seg(data_root = CONFIG.DATASET.ROOT, data_mode = 'train_weak',transforms=tr_transforms)

    # DataLoader
    loader = torch.utils.data.DataLoader(
        dataset=mydataset,
        batch_size=CONFIG.SOLVER.BATCH_SIZE.TRAIN,
        num_workers=CONFIG.DATALOADER.NUM_WORKERS,
        shuffle=True,
    )
    loader_iter = iter(loader)

    # Model setup
    model = DeepLabV2_ResNet101_MSC(CONFIG)
    state_dict = torch.load(CONFIG.MODEL.INIT_MODEL)
    model.base.load_state_dict(state_dict, strict=False)
    model = nn.DataParallel(model)                        
    model.to(device)

    # Loss definition
    criterion = nn.CrossEntropyLoss(ignore_index=CONFIG.DATASET.IGNORE_LABEL)
    criterion.to(device)

    # Optimizer
    optimizer = torch.optim.SGD(
        # cf lr_mult and decay_mult in train.prototxt
        params=[
            {
                "params": get_params(model.module, key="1x"),
                "lr": CONFIG.SOLVER.LR,
                "weight_decay": CONFIG.SOLVER.WEIGHT_DECAY,
            },
            {
                "params": get_params(model.module, key="10x"),
                "lr": 10 * CONFIG.SOLVER.LR,
                "weight_decay": CONFIG.SOLVER.WEIGHT_DECAY,
            },
            {
                "params": get_params(model.module, key="20x"),
                "lr": 20 * CONFIG.SOLVER.LR,
                "weight_decay": 0.0,
            },
        ],
        momentum=CONFIG.SOLVER.MOMENTUM,
    )

    # Learning rate scheduler
    scheduler = PolynomialLR(
        optimizer=optimizer,
        step_size=CONFIG.SOLVER.LR_DECAY,
        iter_max=CONFIG.SOLVER.ITER_MAX,
        power=CONFIG.SOLVER.POLY_POWER,
    )

    # Path to save models
    checkpoint_dir = os.path.join(CONFIG.EXP.OUTPUT_DIR, "weights")
    makedirs(checkpoint_dir)
    print("Checkpoint dst:", checkpoint_dir)

    # Freeze the batch norm pre-trained on COCO
    model.train()
    model.module.base.freeze_bn()

    for iteration in tqdm(
        range(1, CONFIG.SOLVER.ITER_MAX + 1),
        total=CONFIG.SOLVER.ITER_MAX,
        dynamic_ncols=True,
    ):

        # Clear gradients
        optimizer.zero_grad()

        loss = 0
        for _ in range(CONFIG.SOLVER.ITER_SIZE):
            try:
                images, labels = next(loader_iter)
            except:
                loader_iter = iter(loader)
                images, labels = next(loader_iter)

            # Propagate forward
            logits = model(images.to(device))
            iter_loss = 0

            for logit in logits:
                # Resize labels for {100%, 75%, 50%, Max} logits
                _, _, H, W = logit.shape
                labels_ = resize_labels(labels[0], size=(H, W))
                iter_loss = criterion(logit, labels_.to(device))
            
            # Propagate backward (just compute gradients)
            iter_loss /= CONFIG.SOLVER.ITER_SIZE
            iter_loss.backward()

            loss += float(iter_loss)

        # average_loss.add(loss)
        print('loss=',loss)

        optimizer.step()
        scheduler.step(epoch=iteration)

        # Saving model at regular intervals
        if iteration % CONFIG.SOLVER.ITER_SAVE == 0:
            torch.save(
                model.module.state_dict(),
                os.path.join(checkpoint_dir, "checkpoint_{}.pth".format(iteration)),
            )

    torch.save( model.module.state_dict(), os.path.join(checkpoint_dir, "checkpoint_final.pth"))

if __name__ == '__main__':
  config_path = '/content/DeepLab-V2-Seg-ResNet-101-/config/deeplabv2.yaml'
  main(config_path)