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
from tqdm.auto import tqdm
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import models
import pprint
import pdb
from rsbox import ml, misc
import os




class BaseExp:
    # prints relevant group accuracies out 
    def __init__(self, config): 
        print("Group DRO running.")
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
        self.verbose = self.cfg.verbose

    
    # group dro 
    def train(self, num_epochs):
        """
        Training loop. 
        """
        self.model.train()
        epoch_num = 0

        for epoch in range(num_epochs):
            epoch_num += 1
            one_loss = 0.0
            two_loss = 0.0
            three_loss = 0.0
            four_loss = 0.0
            five_loss = 0.0
            six_loss = 0.0

            one_count = 0
            two_count = 0
            three_count = 0
            four_count = 0
            five_count = 0
            six_count = 0

            
            tqdm_loader = tqdm(self.trainloader)
            for batch in tqdm_loader:
                tqdm_loader.set_description(f"Epoch {epoch_num}")

                x, y, g = batch
                x = x.to(self.device)
                y = y.to(self.device)
                g = g[0]  # stringerize 
                x = x.float()
                logits = self.model(x)
                loss_val = self.criterion(logits, y)
                
                if g == '1':
                    one_loss += loss_val
                    one_count += 1
                elif g == '2':
                    two_loss += loss_val
                    two_count += 1
                elif g == '3':
                    three_loss += loss_val
                    three_count += 1
                elif g == '4':
                    four_loss += loss_val
                    four_count += 1
                elif g == '5':
                    five_loss += loss_val
                    five_count += 1
                elif g == '6':
                    six_loss += loss_val
                    six_count += 1

                
            # params update
            one_loss_avg = one_loss/one_count
            two_loss_avg = two_loss/two_count
            three_loss_avg = three_loss/three_count
            four_loss_avg = four_loss/four_count
            five_loss_avg = five_loss/five_count
            six_loss_avg = six_loss/six_count
            max_loss_avg = max(one_loss_avg, two_loss_avg, three_loss_avg, four_loss_avg, five_loss_avg, six_loss_avg)  # worst group loss

            self.optimizer.zero_grad()
            max_loss_avg.backward()
            self.optimizer.step()

            # metrics dict 
            metrics_dict = {
                "skin_type_one_loss": one_loss_avg,
                "skin_type_two_loss": two_loss_avg,
                "skin_type_three_loss": three_loss_avg,
                "skin_type_four_loss": four_loss_avg,
                "skin_type_five_loss": five_loss_avg,
                "skin_type_six_loss": six_loss_avg,
                "Max loss (group DRO)": max_loss_avg
            }



            if self.logger is not None:
                self.logger.log(metrics_dict)
                # self.logger.log({"Training loss": max_loss_avg})
            if self.verbose:
                print("Training Loss (epoch " + str(epoch_num) + "):", max_loss_avg)

            
            # test 
            self.test()
            self.model.train()
        
        self.model.eval()
        print('Finished Training!')



    def save_weights(self, save_path=None):
        """
        Function to save model weights at 'save_path'. 
        Arguments:
        - save_path: path to save the weights. If not given, defaults to current timestamp. 
        """ 
        if save_path is None:
            if not os.path.exists("saved"):
                os.makedirs("saved")
            save_path = "saved/" + misc.timestamp() + ".pth"
        torch.save(self.model.state_dict(), save_path)
        print("Model weights saved at: " + str(save_path))
        


    def test(self):
        self.model.eval()
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
    

        
