# process data 
import torchplate 
import rsbox 
from rsbox import ml, misc 
import pickle
import torch
from torch.utils.data import Dataset




class BaseDataset(Dataset):
    def __init__(self, data_set):
        self.data_distribution = data_set
        
    def __getitem__(self, index):
        sample = self.data_distribution[index % len(self.data_distribution)][0]
        label = self.data_distribution[index % len(self.data_distribution)][1]
        group = self.data_distribution[index % len(self.data_distribution)][2]
        sample = torch.tensor(sample, dtype=torch.float)
        label = torch.tensor(label)
        return (sample, label, group) 
        
    def __len__(self):
        return len(self.data_distribution) * 1

 

def get_data():
    new_list = []
    dicts = pickle.load(open("../data/train.pkl", 'rb'))
    test_iter = pickle.load(open("../data/test.pkl", 'rb'))

    for k, v in dicts.items():
        new_v = []
        for elem in v:
            elem = (elem[0], elem[1], k)  # add group to tuple
            new_v.append(elem)
        new_list += new_v


    trainloader = torch.utils.data.DataLoader(BaseDataset(new_list))

    return trainloader, test_iter
