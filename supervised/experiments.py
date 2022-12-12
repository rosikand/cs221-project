"""
File: experiments.py
------------------
This file holds the experiments which are
subclasses of torchplate.experiment.Experiment. 
"""

import numpy as np
import torchplate
from torchplate import (
        experiment,
        utils
    )
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import models
import pprint
import pdb




class BaseExp(experiment.Experiment):
    # prints relevant group accuracies out 
    def __init__(self, config): 
        self.cfg = config
        self.model = self.cfg.model
        self.optimizer = self.cfg.optimizer
        self.criterion = self.cfg.loss_fn
        self.trainloader = self.cfg.trainloader
        self.test_set = self.cfg.test_set
        self.logger = self.cfg.logger
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(self.device)
        self.model = self.model.to(self.device)


        super().__init__(
            model = self.model,
            optimizer = self.optimizer,
            trainloader = self.trainloader,
            wandb_logger = self.logger,
            verbose = True
        )

    
    # provide this abstract method to calculate loss 
    def evaluate(self, batch):
        x, y = batch
        x = x.float()
        x = x.to(self.device)
        y = y.to(self.device)
        logits = self.model(x)
        loss_val = self.criterion(logits, y)
        return loss_val


    def test(self):
        accuracy_count = 0
        one_accuracy_count = 0
        two_accuracy_count = 0
        three_accuracy_count = 0
        four_accuracy_count = 0
        five_accuracy_count = 0
        six_accuracy_count = 0

        one_total_count = 0
        two_total_count = 0
        three_total_count = 0
        four_total_count = 0
        five_total_count = 0
        six_total_count = 0


        for key in self.test_set.keys():
            dist = self.test_set[key]
            for batch in dist:
                x, y = batch
                x = torch.from_numpy(x).unsqueeze(dim=0).float()
                assert len(x.shape) == 4
                x = x.to(self.device)
                logits = self.model(x)
                pred = torch.argmax(F.softmax(logits, dim=1)).item()
                
                correct_pred = False
                # print(f"Prediction: {pred}, True: {y}")
                if pred == y:
                    accuracy_count += 1
                    correct_pred = True
                
                if key == '1':
                    if correct_pred:
                        one_accuracy_count += 1
                    one_total_count += 1
                elif key == '2':
                    if correct_pred:
                        two_accuracy_count += 1
                    two_total_count += 1
                elif key == '3':
                    if correct_pred:
                        three_accuracy_count += 1
                    three_total_count += 1
                elif key == '4':
                    if correct_pred:
                        four_accuracy_count += 1
                    four_total_count += 1
                elif key == '5':
                    if correct_pred:
                        five_accuracy_count += 1
                    five_total_count += 1
                elif key == '6':
                    if correct_pred:
                        six_accuracy_count += 1
                    six_total_count += 1


        accuracies = {}
        overall_accuracy = (one_accuracy_count + two_accuracy_count + three_accuracy_count + four_accuracy_count + five_accuracy_count + six_accuracy_count)/(one_total_count + two_total_count + three_total_count + four_total_count + five_total_count + six_total_count)
        accuracies['skin-type-1-accuracy'] = one_accuracy_count/one_total_count
        accuracies['skin-type-2-accuracy'] = two_accuracy_count/two_total_count
        accuracies['skin-type-3-accuracy'] = three_accuracy_count/three_total_count
        accuracies['skin-type-4-accuracy'] = four_accuracy_count/four_total_count
        accuracies['skin-type-5-accuracy'] = five_accuracy_count/five_total_count
        accuracies['skin-type-6-accuracy'] = six_accuracy_count/six_total_count
        accuracies['123-group-accuracy'] = (accuracies['skin-type-1-accuracy'] + accuracies['skin-type-2-accuracy'] + accuracies['skin-type-3-accuracy'])/3
        accuracies['456-group-accuracy'] = (accuracies['skin-type-4-accuracy'] + accuracies['skin-type-5-accuracy'] + accuracies['skin-type-6-accuracy'])/3
        accuracies['overall-accuracy'] = overall_accuracy

        pprint.pprint(accuracies)

        if self.logger is not None:
            self.logger.log(accuracies)
    

    def on_epoch_end(self):
        self.test()
