"""
File: datasets.py
------------------
This file holds various dataset and dataloading
functions. 
"""

import cloudpickle as cp
import torch
from torch.utils.data import Dataset
import torchplate
from torchplate import utils as tp_utils
import requests 
from urllib.request import urlopen
import rsbox
from rsbox import misc
import pdb
import random
import pickle




def augment_data(data_dists):
    # to make the data work with multi-task learning! 
    # takes in {
    # task1: [(x1, y1), (x2, y2), ...],
    # task2: [(x1, y1), (x2, y2), ...],
    # ...
    # }
    # and balances it out such that each task has the same number of samples
    # by repeating the samples in the task with the least number of samples
    # returns the balanced data distribution
    # input: dict of data distributions
    # output: dict of balanced data distributions

    rkey = random.choice(list(data_dists.keys()))
    max_len = len(data_dists[rkey])
    # get maximum length of elems in data_dists
    for key in data_dists:
        if len(data_dists[key]) > max_len:
            max_len = len(data_dists[key])

    # repeat the samples in the task with the least number of samples
    # until all tasks have the same number of samples
    for key in data_dists:
        while len(data_dists[key]) < max_len:
            data_dists[key] += data_dists[key]
        
    # remove the extra samples
    for key in data_dists:
        # shuffle 
        random.shuffle(data_dists[key])
        data_dists[key] = data_dists[key][:max_len]
   

   # note: works only for multiples of ten
    return data_dists



class MultiTaskDataset(Dataset):
    def __init__(self, data_set_dict):
        # input: list of dataset distributions 
        # change to dict for key mapping purpose to avoid mis-odering 
        assert len(data_set_dict) > 0
        self.data_dists = data_set_dict

        # get min len (and randomly shuffle)
        # randomly select a key from self.data_dists
        key = random.choice(list(self.data_dists.keys()))
        min_len = len(self.data_dists[key])
        for key in self.data_dists:
            random.shuffle(self.data_dists[key]) # list 
            if len(self.data_dists[key]) < min_len:
                min_len = len(self.data_dists[key])
            
        self.min_length = min_len



        
    def __getitem__(self, index):
        x = {}
        y = {}
        
        for key in self.data_dists:
            sample = self.data_dists[key][index % self.min_length][0]
            label = self.data_dists[key][index % self.min_length][1]
            sample = torch.tensor(sample, dtype=torch.float)
            label = torch.tensor(label)
            x[key] = sample
            y[key] = label
        
        return x, y

        
    def __len__(self):
        return self.min_length * 10



def get_dataloaders():
    train_dict = pickle.load(open("../data/train.pkl", 'rb'))
    train_dict = augment_data(train_dict)
    test_dict = pickle.load(open("../data/test.pkl", 'rb'))
    train_set = MultiTaskDataset(train_dict)
    test_set = MultiTaskDataset(test_dict)
    trainloader = torch.utils.data.DataLoader(train_set)
    testloader = torch.utils.data.DataLoader(test_set)
    return trainloader, testloader




# test 

# def main():
#     loader = get_dataloaders()
#     for x, y in loader:
#         pdb.set_trace()
#         print(x)
#         print(y)
#         break

#     pdb.set_trace()



# if __name__ == "__main__":
#     main()