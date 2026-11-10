# -*- coding: utf-8 -*-
"""
Code to setup an instance of ResNet1d. Ensemble version also defined.
"""
__author__ = "Stefan Gustafsson"
__email__  = "stefan.gustafsson@medsci.uu.se"

import os
import copy
import torch
import torch.nn as nn

from resnet import ResNet1d
from prediction_stage import LinearPredictionStage
from age_sex_encoding import AgeSexEncoding


class ECGModel(nn.Module):
    def __init__(self, config):
        super(ECGModel, self).__init__()
        # Setup for ResNet
        self.resnet = ResNet1d(
            input_dim = (config.n_leads, config.seq_length), # (8, 4096) as default
            blocks_dim = list(zip(config.net_filter_size, config.net_seq_length)),
            n_outcomes = config.n_outcomes,
            kernel_size = config.kernel_size,
            dropout_rate = config.dropout_rate,
            activation = config.activation_function
        )
        self.device = config.device
        
        # Setup for final prediction stage (ECG ResNet + age + sex)
        # Fully connected layer going from combined_output_dim to n_outcomes
        final_resnet_filter_size = config.net_filter_size[-1]
        final_resnet_length = config.net_seq_length[-1]
        self.age_sex_emb = AgeSexEncoding(config.agesex_dim)
        
        combined_output_dim = final_resnet_filter_size * final_resnet_length + config.agesex_dim
        self.lin = LinearPredictionStage(
            prev_layer_dim = combined_output_dim,
            n_outcomes = config.n_outcomes
        )
    
    
    def forward(self, indata):
        age_sex, ecg = indata
        
        # Apply the ResNet forward function to perform all transformations
        # of the ECG indata but before applying the last layer
        x_ecg = self.resnet(ecg)
        x_ecg = x_ecg.view(x_ecg.size(0), -1) # Flatten array
        
        # Apply the age+sex forward function to produce the age+sex embeddings
        x_age_sex = self.age_sex_emb(age_sex)
        
        # Combine the two and apply the final prediction layer
        x = torch.cat([x_age_sex, x_ecg], dim = 1)
        
        # Fully connected linear layer
        logits = self.lin(x)
        
        # Return logits, i.e. the linear predictor X*w^T + b (w=coeff, b=bias)
        # Sigmoid/softmax applied downstream to get predicted probabilities
        return logits


class EnsembleECGModel(ECGModel):
    def __init__(self, config, model_dir):
        super(EnsembleECGModel, self).__init__(config)
        
        self.model_dir = model_dir
        self.model_list = self.load_ensembles(config)
    
    def load_ensembles(self, config):
        """Load best model for each ensemble"""
        loaded = []
        for ensemble_nr in range(1, config.n_ensembles + 1):
            # Generic model in eval mode.
            ens_model = ECGModel(config)
            ens_model.eval()
            
            # Load trained best model parameters of given ensemble member.
            try:
                best = torch.load(
                    os.path.join(self.model_dir, 'model_' + str(ensemble_nr) + '.pth'), 
                    map_location = self.device,
                    weights_only = False
                )
            except TypeError:
                best = torch.load(
                    os.path.join(self.model_dir, 'model_' + str(ensemble_nr) + '.pth'), 
                    map_location = self.device
                )
            ens_model.load_state_dict(best['model'])
            
            loaded.append(copy.deepcopy(ens_model))
        return loaded
    
    def forward(self, x):
        """Forward implemented for combining ensembles"""
        logits_comb = []
        
        for ens_model in self.model_list:
            ens_model.to(self.device)
            logits = ens_model.forward(x)
            logits_comb.append(logits)
        
        # Average logits across ensembles
        return torch.stack(logits_comb).mean(dim = 0)
