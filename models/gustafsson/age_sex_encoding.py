# -*- coding: utf-8 -*-
"""
Encodes age+sex to be added as predictors in addition to the ECG data.
Adapted from script by daniel.gedon@it.uu.se
"""


import torch
import torch.nn as nn


class AgeSexEncoding(nn.Module):
    """Encode age+sex input into a higher dimension space"""
    def __init__(self, output_dim = 128):
        super(AgeSexEncoding, self).__init__()
        self.output_dim = output_dim
        
        # Fully connected linear layer to map from 2 (age+sex) to output_dim
        # features
        # Apply ReLU to capture non-linearities
        self.linear = nn.Linear(2, self.output_dim)
        self.relu = nn.ReLU()
    
    def forward(self, age_sex):
        out = self.linear(age_sex)
        out = self.relu(out)
        return out
