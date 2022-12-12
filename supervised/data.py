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
        sample = torch.tensor(sample, dtype=torch.float)
        label = torch.tensor(label)
        return (sample, label) 
        
    def __len__(self):
        return len(self.data_distribution) * 1

 

def get_data():
    new_list = []
    dicts = pickle.load(open("../data/train.pkl", 'rb'))
    test_iter = pickle.load(open("../data/test.pkl", 'rb'))

    for k, v in dicts.items():
        new_list += v


    trainloader = torch.utils.data.DataLoader(BaseDataset(new_list))

    return trainloader, test_iter
