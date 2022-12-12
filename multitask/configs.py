"""
File: configs.py 
----------------------
Specifies config parameters. 
"""


import datasets 
import models
import experiments
import torch
from torch import nn 
from torch import optim
from rsbox import misc, ml
import wandb


class BaseConfig:
    exp_name = "DRO-" + misc.timestamp()
    experiment = experiments.DROExp
    trainloader, testloader = datasets.get_dataloaders()
    model = models.MultiMLP()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    # logger = None
    logger = wandb.init(project = "221-proj-multi-task", entity = "rosikand", name = misc.timestamp())


class Nondro:
    # weights 
    exp_name = "Non-DRO-" + misc.timestamp()
    experiment = experiments.NonDROExp
    trainloader, testloader = datasets.get_dataloaders() 
    model = models.MultiMLP()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    # logger = None
    logger = wandb.init(project = "221-proj-multi-task", entity = "rosikand", name = exp_name)



class Nondroweighted:
    # weights 
    exp_name = "Non-DRO-weighted-" + misc.timestamp()
    experiment = experiments.NonDROExp
    trainloader, testloader = datasets.get_dataloaders() 
    model = models.MultiMLP()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    # logger = None
    logger = wandb.init(project = "221-proj-multi-task", entity = "rosikand", name = exp_name)
    skin_type_one_weight = 0.5/6
    skin_type_two_weight = 0.5/6
    skin_type_three_weight = 0.5/6
    skin_type_four_weight = 1.5/6
    skin_type_five_weight = 1.5/6
    skin_type_six_weight = 1.5/6
