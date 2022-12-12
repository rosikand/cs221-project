"""
File: configs.py 
----------------------
Specifies config parameters. 
"""

import data 
import models
import experiments
import torchplate
import rsbox 
import wandb
from rsbox import ml, misc
import torch.optim as optim
import torch
from torch import nn
import pickle 
import timm


class BaseConfig:
    experiment = experiments.BaseExp
    trainloader, test_set = data.get_data()
    model = models.MLP()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    # logger = None
    logger = wandb.init(project = "221-proj-supervised", entity = "rosikand", name = misc.timestamp())


class Resnet:
    experiment = experiments.BaseExp
    trainloader, test_set = data.get_data()
    model = timm.create_model('resnet18', pretrained=True, num_classes=2, in_chans=3)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    logger = None
    # logger = wandb.init(project = "221-proj", entity = "rosikand", name = misc.timestamp())

    