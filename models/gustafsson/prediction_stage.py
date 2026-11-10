# -*- coding: utf-8 -*-
"""
Final fully connected layer of the model producing the logits
Adapted from script by daniel.gedon@it.uu.se
"""

import torch
import torch.nn as nn


class LinearPredictionStage(nn.Module):
    def __init__(self, prev_layer_dim, n_outcomes):
        super(LinearPredictionStage, self).__init__()
        self.lin_classifier = nn.Linear(prev_layer_dim, n_outcomes)
    
    def forward(self, x):
        # Fully connected layer going from prev_layer_dim to n_outcomes
        x = self.lin_classifier(x)
        return x
