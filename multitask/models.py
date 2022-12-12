"""
File: models.py
------------------
This file holds the torch.nn modules. 
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pdb


def build_mlp(layer_sizes, output_dim):
    """layer_sizes[0] should be input dim.""" 

    layers = []
    for i in range(1, len(layer_sizes) + 1):
        if i == len(layer_sizes):
            # for last layer, no relu and ouput 1 for shape size 
            layers.append(nn.Linear(layer_sizes[i-1], output_dim))
        else:
            layers.append(nn.Linear(layer_sizes[i-1], layer_sizes[i]))
            layers.append(nn.ReLU())

    return nn.Sequential(*layers)



class MultiMLP(nn.Module):
    def __init__(self):
        super().__init__()

        layer_sizes = [64*64*3, 120, 84, 32]
        output_dim = 2

        # nets 
        self.skin_type_one = build_mlp(layer_sizes, output_dim)
        self.skin_type_two = build_mlp(layer_sizes, output_dim)
        self.skin_type_three = build_mlp(layer_sizes, output_dim)
        self.skin_type_four = build_mlp(layer_sizes, output_dim)
        self.skin_type_five = build_mlp(layer_sizes, output_dim)
        self.skin_type_six = build_mlp(layer_sizes, output_dim)

       
    def forward(self, one_x, two_x, three_x, four_x, five_x, six_x):
        # the model takes in 6 images and predicts 6 logits, one for each class (skin type) 

        # 1. flatten
        one_x = torch.flatten(one_x, 1)
        two_x = torch.flatten(two_x, 1)
        three_x = torch.flatten(three_x, 1)
        four_x = torch.flatten(four_x, 1)
        five_x = torch.flatten(five_x, 1)
        six_x = torch.flatten(six_x, 1)

        # maybe shared embeddings here? 

        # 2. forward pass 
        one_logits = self.skin_type_one(one_x)
        two_logits = self.skin_type_two(two_x)
        three_logits = self.skin_type_three(three_x)
        four_logits = self.skin_type_four(four_x)
        five_logits = self.skin_type_five(five_x)
        six_logits = self.skin_type_six(six_x)

        return one_logits, two_logits, three_logits, four_logits, five_logits, six_logits
    

