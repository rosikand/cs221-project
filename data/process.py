"""
File: process.py 
-------------- 
Processes raw data into numpy arrays and pickles them up. 
""" 


import numpy as np 
import rsbox
from rsbox import ml 
import pdb
from glob import glob
import pickle


class_labels = ["1", "2", "3", "4", "5", "6"]


# labelize 
f = open("labels.txt")

label_dict = {}
for line in f:
    filename, label = line.strip().split()
    label_dict[filename] = label



# load all subdirectories 

def load_with_labels(dirpath):
    dirpath = dirpath + "/*"
    sub_set = glob(dirpath)
    
    dset = []
    for elem in sub_set:
        hash_string = elem[elem.rfind(r'/') + 1:]
        label = label_dict[hash_string]
        assert label == '1' or label == '-1'
        if label == '-1':
            label = 0
        if label == '1':
            label = 1
        img = ml.load_image(elem, resize=None, normalize=True)
        pair = (img, label)
        dset.append(pair)
    
    return dset



dataset = {}
for label in class_labels:
    train_path = "train/" + label
    dset = load_with_labels(train_path)
    dataset[label] = dset


# pickle dataset 
with open("train.pkl", "wb") as f:
    pickle.dump(dataset, f)


# now for test 
tdataset = {}
for label in class_labels:
    test_path = "test/" + label
    dset = load_with_labels(test_path)
    tdataset[label] = dset


# pickle dataset 
with open("test.pkl", "wb") as f:
    pickle.dump(tdataset, f)
