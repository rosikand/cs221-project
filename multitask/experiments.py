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
import datasets
import pprint
import rsbox 
from rsbox import ml, misc 



class DROExp(experiment.Experiment):
    # Implements multi-task DRO 
    # prints relevant group accuracies out 
    def __init__(self, config): 
        self.cfg = config
        self.model = self.cfg.model
        self.optimizer = self.cfg.optimizer
        self.criterion = self.cfg.loss_fn
        self.trainloader = self.cfg.trainloader
        self.testloader = self.cfg.testloader
        self.logger = self.cfg.logger
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(self.device)
        self.model = self.model.to(self.device)


        # metrics 
        self.one_epoch_loss = ml.MeanMetric()
        self.two_epoch_loss = ml.MeanMetric()
        self.three_epoch_loss = ml.MeanMetric()
        self.four_epoch_loss = ml.MeanMetric()
        self.five_epoch_loss = ml.MeanMetric()
        self.six_epoch_loss = ml.MeanMetric()



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
        one_logits, two_logits, three_logits, four_logits, five_logits, six_logits = self.model(x['1'].to(self.device), x['2'].to(self.device), x['3'].to(self.device), x['4'].to(self.device), x['5'].to(self.device), x['6'].to(self.device))
        
        one_loss = self.criterion(one_logits, y['1'].to(self.device))
        two_loss = self.criterion(two_logits, y['2'].to(self.device))
        three_loss = self.criterion(three_logits, y['3'].to(self.device))
        four_loss = self.criterion(four_logits, y['4'].to(self.device))
        five_loss = self.criterion(five_logits, y['5'].to(self.device))
        six_loss = self.criterion(six_logits, y['6'].to(self.device))

        group_dro_multitask_loss = max(one_loss, two_loss, three_loss, four_loss, five_loss, six_loss)


        # metrics 
        self.one_epoch_loss.update(one_loss)
        self.two_epoch_loss.update(two_loss)
        self.three_epoch_loss.update(three_loss)
        self.four_epoch_loss.update(four_loss)
        self.five_epoch_loss.update(five_loss)
        self.six_epoch_loss.update(six_loss)



        return group_dro_multitask_loss


    def test(self):
        one_accuracy_count = 0
        two_accuracy_count = 0
        three_accuracy_count = 0
        four_accuracy_count = 0
        five_accuracy_count = 0
        six_accuracy_count = 0

        # one_total_count = 0
        # two_total_count = 0
        # three_total_count = 0
        # four_total_count = 0
        # five_total_count = 0
        # six_total_count = 0


        for batch in self.testloader:
            x, y = batch
            one_logits, two_logits, three_logits, four_logits, five_logits, six_logits = self.model(x['1'].to(self.device), x['2'].to(self.device), x['3'].to(self.device), x['4'].to(self.device), x['5'].to(self.device), x['6'].to(self.device))

            one_pred = torch.argmax(F.softmax(one_logits, dim=1)).item()
            two_pred = torch.argmax(F.softmax(two_logits, dim=1)).item()
            three_pred = torch.argmax(F.softmax(three_logits, dim=1)).item()
            four_pred = torch.argmax(F.softmax(four_logits, dim=1)).item()
            five_pred = torch.argmax(F.softmax(five_logits, dim=1)).item()
            six_pred = torch.argmax(F.softmax(six_logits, dim=1)).item()


            if one_pred == y['1']:
                one_accuracy_count += 1
            if two_pred == y['2']:
                two_accuracy_count += 1
            if three_pred == y['3']:
                three_accuracy_count += 1
            if four_pred == y['4']:
                four_accuracy_count += 1
            if five_pred == y['5']:
                five_accuracy_count += 1
            if six_pred == y['6']:
                six_accuracy_count += 1



        accuracies = {}
        overall_acc = (one_accuracy_count + two_accuracy_count + three_accuracy_count + four_accuracy_count + five_accuracy_count + six_accuracy_count)/(6*len(self.testloader))
        accuracies['skin-type-1-accuracy'] = one_accuracy_count/len(self.testloader)
        accuracies['skin-type-2-accuracy'] = two_accuracy_count/len(self.testloader)
        accuracies['skin-type-3-accuracy'] = three_accuracy_count/len(self.testloader)
        accuracies['skin-type-4-accuracy'] = four_accuracy_count/len(self.testloader)
        accuracies['skin-type-5-accuracy'] = five_accuracy_count/len(self.testloader)
        accuracies['skin-type-6-accuracy'] = six_accuracy_count/len(self.testloader)
        accuracies['123-group-accuracy'] = (accuracies['skin-type-1-accuracy'] + accuracies['skin-type-2-accuracy'] + accuracies['skin-type-3-accuracy'])/3
        accuracies['456-group-accuracy'] = (accuracies['skin-type-4-accuracy'] + accuracies['skin-type-5-accuracy'] + accuracies['skin-type-6-accuracy'])/3
        accuracies['overall-accuracy'] = overall_acc

        pprint.pprint(accuracies)

        if self.logger is not None:
            self.logger.log(accuracies)
    
    
    def on_epoch_end(self):

        # metrics 
        log_metrics_dict = {
                "skin_type_one_loss": self.one_epoch_loss.get(),
                "skin_type_two_loss": self.two_epoch_loss.get(),
                "skin_type_three_loss": self.three_epoch_loss.get(),
                "skin_type_four_loss": self.four_epoch_loss.get(),
                "skin_type_five_loss": self.five_epoch_loss.get(),
                "skin_type_six_loss": self.six_epoch_loss.get()
        }

        if self.logger is not None:
            self.logger.log(log_metrics_dict)
        

        self.one_epoch_loss.reset()
        self.two_epoch_loss.reset()
        self.three_epoch_loss.reset()
        self.four_epoch_loss.reset()
        self.five_epoch_loss.reset()
        self.six_epoch_loss.reset()


        # test 
        self.test()


# ---------------------------------------------------------



class NonDROExp(experiment.Experiment):
    # Implements regular multi-task learning 
    # prints relevant group accuracies out 
    def __init__(self, config): 
        print("Running Non-DRO experiment.")
        self.cfg = config
        self.model = self.cfg.model
        self.optimizer = self.cfg.optimizer
        self.criterion = self.cfg.loss_fn
        self.trainloader = self.cfg.trainloader
        self.testloader = self.cfg.testloader
        self.logger = self.cfg.logger
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(self.device)
        self.model = self.model.to(self.device)


        # metrics 
        self.one_epoch_loss = ml.MeanMetric()
        self.two_epoch_loss = ml.MeanMetric()
        self.three_epoch_loss = ml.MeanMetric()
        self.four_epoch_loss = ml.MeanMetric()
        self.five_epoch_loss = ml.MeanMetric()
        self.six_epoch_loss = ml.MeanMetric()



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
        one_logits, two_logits, three_logits, four_logits, five_logits, six_logits = self.model(x['1'].to(self.device), x['2'].to(self.device), x['3'].to(self.device), x['4'].to(self.device), x['5'].to(self.device), x['6'].to(self.device))
        
        one_loss = self.criterion(one_logits, y['1'].to(self.device))
        two_loss = self.criterion(two_logits, y['2'].to(self.device))
        three_loss = self.criterion(three_logits, y['3'].to(self.device))
        four_loss = self.criterion(four_logits, y['4'].to(self.device))
        five_loss = self.criterion(five_logits, y['5'].to(self.device))
        six_loss = self.criterion(six_logits, y['6'].to(self.device))

        multitask_loss = sum([one_loss, two_loss, three_loss, four_loss, five_loss, six_loss])/6

        # metrics 
        self.one_epoch_loss.update(one_loss)
        self.two_epoch_loss.update(two_loss)
        self.three_epoch_loss.update(three_loss)
        self.four_epoch_loss.update(four_loss)
        self.five_epoch_loss.update(five_loss)
        self.six_epoch_loss.update(six_loss)



        return multitask_loss


    def test(self):
        one_accuracy_count = 0
        two_accuracy_count = 0
        three_accuracy_count = 0
        four_accuracy_count = 0
        five_accuracy_count = 0
        six_accuracy_count = 0

        # one_total_count = 0
        # two_total_count = 0
        # three_total_count = 0
        # four_total_count = 0
        # five_total_count = 0
        # six_total_count = 0


        for batch in self.testloader:
            x, y = batch
            one_logits, two_logits, three_logits, four_logits, five_logits, six_logits = self.model(x['1'].to(self.device), x['2'].to(self.device), x['3'].to(self.device), x['4'].to(self.device), x['5'].to(self.device), x['6'].to(self.device))

            one_pred = torch.argmax(F.softmax(one_logits, dim=1)).item()
            two_pred = torch.argmax(F.softmax(two_logits, dim=1)).item()
            three_pred = torch.argmax(F.softmax(three_logits, dim=1)).item()
            four_pred = torch.argmax(F.softmax(four_logits, dim=1)).item()
            five_pred = torch.argmax(F.softmax(five_logits, dim=1)).item()
            six_pred = torch.argmax(F.softmax(six_logits, dim=1)).item()


            if one_pred == y['1']:
                one_accuracy_count += 1
            if two_pred == y['2']:
                two_accuracy_count += 1
            if three_pred == y['3']:
                three_accuracy_count += 1
            if four_pred == y['4']:
                four_accuracy_count += 1
            if five_pred == y['5']:
                five_accuracy_count += 1
            if six_pred == y['6']:
                six_accuracy_count += 1



        accuracies = {}
        overall_acc = (one_accuracy_count + two_accuracy_count + three_accuracy_count + four_accuracy_count + five_accuracy_count + six_accuracy_count)/(6*len(self.testloader))
        accuracies['skin-type-1-accuracy'] = one_accuracy_count/len(self.testloader)
        accuracies['skin-type-2-accuracy'] = two_accuracy_count/len(self.testloader)
        accuracies['skin-type-3-accuracy'] = three_accuracy_count/len(self.testloader)
        accuracies['skin-type-4-accuracy'] = four_accuracy_count/len(self.testloader)
        accuracies['skin-type-5-accuracy'] = five_accuracy_count/len(self.testloader)
        accuracies['skin-type-6-accuracy'] = six_accuracy_count/len(self.testloader)
        accuracies['123-group-accuracy'] = (accuracies['skin-type-1-accuracy'] + accuracies['skin-type-2-accuracy'] + accuracies['skin-type-3-accuracy'])/3
        accuracies['456-group-accuracy'] = (accuracies['skin-type-4-accuracy'] + accuracies['skin-type-5-accuracy'] + accuracies['skin-type-6-accuracy'])/3
        accuracies['overall-accuracy'] = overall_acc

        pprint.pprint(accuracies)

        if self.logger is not None:
            self.logger.log(accuracies)
    
    
    def on_epoch_end(self):

        # metrics 
        log_metrics_dict = {
                "skin_type_one_loss": self.one_epoch_loss.get(),
                "skin_type_two_loss": self.two_epoch_loss.get(),
                "skin_type_three_loss": self.three_epoch_loss.get(),
                "skin_type_four_loss": self.four_epoch_loss.get(),
                "skin_type_five_loss": self.five_epoch_loss.get(),
                "skin_type_six_loss": self.six_epoch_loss.get()
        }

        if self.logger is not None:
            self.logger.log(log_metrics_dict)
        

        self.one_epoch_loss.reset()
        self.two_epoch_loss.reset()
        self.three_epoch_loss.reset()
        self.four_epoch_loss.reset()
        self.five_epoch_loss.reset()
        self.six_epoch_loss.reset()


        # test 
        self.test()



# ---------------------------------------------------------


class NonDROWeightedExp(experiment.Experiment):
    # Implements regular multi-task learning with weighing tasks 
    def __init__(self, config): 
        print("Running Non-DRO experiment.")
        self.cfg = config
        self.model = self.cfg.model
        self.optimizer = self.cfg.optimizer
        self.criterion = self.cfg.loss_fn
        self.trainloader = self.cfg.trainloader
        self.testloader = self.cfg.testloader
        self.logger = self.cfg.logger
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(self.device)
        self.model = self.model.to(self.device)



        # weights 
        self.one_weight = self.cfg.skin_type_one_weight
        self.two_weight = self.cfg.skin_type_two_weight
        self.three_weight = self.cfg.skin_type_three_weight
        self.four_weight = self.cfg.skin_type_four_weight
        self.five_weight = self.cfg.skin_type_five_weight
        self.six_weight = self.cfg.skin_type_six_weight


        # metrics 
        self.one_epoch_loss = ml.MeanMetric()
        self.two_epoch_loss = ml.MeanMetric()
        self.three_epoch_loss = ml.MeanMetric()
        self.four_epoch_loss = ml.MeanMetric()
        self.five_epoch_loss = ml.MeanMetric()
        self.six_epoch_loss = ml.MeanMetric()



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
        one_logits, two_logits, three_logits, four_logits, five_logits, six_logits = self.model(x['1'].to(self.device), x['2'].to(self.device), x['3'].to(self.device), x['4'].to(self.device), x['5'].to(self.device), x['6'].to(self.device))
        
        one_loss = self.criterion(one_logits, y['1'].to(self.device))
        two_loss = self.criterion(two_logits, y['2'].to(self.device))
        three_loss = self.criterion(three_logits, y['3'].to(self.device))
        four_loss = self.criterion(four_logits, y['4'].to(self.device))
        five_loss = self.criterion(five_logits, y['5'].to(self.device))
        six_loss = self.criterion(six_logits, y['6'].to(self.device))
        multitask_loss = (self.one_weight * one_loss) + (self.two_weight * two_loss) + (self.three_weight * three_loss) + (self.four_weight * four_loss) + (self.five_weight * five_loss) + (self.six_weight * six_loss)


        # metrics 
        self.one_epoch_loss.update(one_loss)
        self.two_epoch_loss.update(two_loss)
        self.three_epoch_loss.update(three_loss)
        self.four_epoch_loss.update(four_loss)
        self.five_epoch_loss.update(five_loss)
        self.six_epoch_loss.update(six_loss)



        return multitask_loss


    def test(self):
        one_accuracy_count = 0
        two_accuracy_count = 0
        three_accuracy_count = 0
        four_accuracy_count = 0
        five_accuracy_count = 0
        six_accuracy_count = 0

        # one_total_count = 0
        # two_total_count = 0
        # three_total_count = 0
        # four_total_count = 0
        # five_total_count = 0
        # six_total_count = 0


        for batch in self.testloader:
            x, y = batch
            one_logits, two_logits, three_logits, four_logits, five_logits, six_logits = self.model(x['1'].to(self.device), x['2'].to(self.device), x['3'].to(self.device), x['4'].to(self.device), x['5'].to(self.device), x['6'].to(self.device))

            one_pred = torch.argmax(F.softmax(one_logits, dim=1)).item()
            two_pred = torch.argmax(F.softmax(two_logits, dim=1)).item()
            three_pred = torch.argmax(F.softmax(three_logits, dim=1)).item()
            four_pred = torch.argmax(F.softmax(four_logits, dim=1)).item()
            five_pred = torch.argmax(F.softmax(five_logits, dim=1)).item()
            six_pred = torch.argmax(F.softmax(six_logits, dim=1)).item()


            if one_pred == y['1']:
                one_accuracy_count += 1
            if two_pred == y['2']:
                two_accuracy_count += 1
            if three_pred == y['3']:
                three_accuracy_count += 1
            if four_pred == y['4']:
                four_accuracy_count += 1
            if five_pred == y['5']:
                five_accuracy_count += 1
            if six_pred == y['6']:
                six_accuracy_count += 1



        accuracies = {}
        overall_acc = (one_accuracy_count + two_accuracy_count + three_accuracy_count + four_accuracy_count + five_accuracy_count + six_accuracy_count)/(6*len(self.testloader))
        accuracies['skin-type-1-accuracy'] = one_accuracy_count/len(self.testloader)
        accuracies['skin-type-2-accuracy'] = two_accuracy_count/len(self.testloader)
        accuracies['skin-type-3-accuracy'] = three_accuracy_count/len(self.testloader)
        accuracies['skin-type-4-accuracy'] = four_accuracy_count/len(self.testloader)
        accuracies['skin-type-5-accuracy'] = five_accuracy_count/len(self.testloader)
        accuracies['skin-type-6-accuracy'] = six_accuracy_count/len(self.testloader)
        accuracies['123-group-accuracy'] = (accuracies['skin-type-1-accuracy'] + accuracies['skin-type-2-accuracy'] + accuracies['skin-type-3-accuracy'])/3
        accuracies['456-group-accuracy'] = (accuracies['skin-type-4-accuracy'] + accuracies['skin-type-5-accuracy'] + accuracies['skin-type-6-accuracy'])/3
        accuracies['overall-accuracy'] = overall_acc

        pprint.pprint(accuracies)

        if self.logger is not None:
            self.logger.log(accuracies)
    
    
    def on_epoch_end(self):

        # metrics 
        log_metrics_dict = {
                "skin_type_one_loss": self.one_epoch_loss.get(),
                "skin_type_two_loss": self.two_epoch_loss.get(),
                "skin_type_three_loss": self.three_epoch_loss.get(),
                "skin_type_four_loss": self.four_epoch_loss.get(),
                "skin_type_five_loss": self.five_epoch_loss.get(),
                "skin_type_six_loss": self.six_epoch_loss.get()
        }

        if self.logger is not None:
            self.logger.log(log_metrics_dict)
        

        self.one_epoch_loss.reset()
        self.two_epoch_loss.reset()
        self.three_epoch_loss.reset()
        self.four_epoch_loss.reset()
        self.five_epoch_loss.reset()
        self.six_epoch_loss.reset()


        # test 
        self.test()


