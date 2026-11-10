# -*- coding: utf-8 -*-
"""
Misc help functions
Adapted from script by antonior92@gmail.com
"""
__author__ = "Stefan Gustafsson"
__email__  = "stefan.gustafsson@medsci.uu.se"

import random
import numpy as np
import pandas as pd
import torch
import os
import datetime
import json
from torch.utils.tensorboard import SummaryWriter


def seed_everything(seed):
    """Set seed for all relevant functions for reproducable results"""
    seed = int(seed)
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    return seed


def set_output_folder(main_folder):
    """Create (if not exists) a timestamped output folder in the specified main folder"""
    timestamp = datetime.datetime.now(datetime.timezone.utc)
    run_folder_name = 'run_' + timestamp.strftime('%Y%m%d_%H%M%S')
        
    output_folder = os.path.join(main_folder, run_folder_name)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    return output_folder


def save_config(args, full_filename):
    """Save arguments used to control the script to file"""
    with open(full_filename, 'w') as f:
        json.dump(vars(args), f, indent = 4)


def save_dset(data, full_filename):
    """Save a pandas dataframe with train/valid/test data to CSV"""
    data.reset_index().to_csv(full_filename, index = False)


class Logger(object):
    """Tensorboard logger as well as log selected metrics over epochs to CSV"""
    
    def __init__(self, log_folder):
        """Initiate logger with output folder"""
        self.folder = log_folder
    
    def init_tensorboardlog(self, suffix = ''):
        """Init tensorboard writer"""
        self.logger = SummaryWriter(log_dir = self.folder, filename_suffix = suffix)
    
    def tensorboardlog_tofile_single(self, metric_name, metric_val, epoch_nr):
        """Log single metric to tensorboard"""
        self.logger.add_scalar(metric_name, metric_val, epoch_nr)
    
    def tensorboardlog_tofile(self, metric_data, epoch_nr):
        """Log multiple metrics to tensorboard"""
        for metric_name in metric_data:
            metric_val = metric_data[metric_name]
            # Add some structure for different classes of metrics.
            # Just a prefix for structuring results.
            if "loss" in metric_name:
                metric_name = 'loss/' + metric_name
            else:
                metric_name = 'other/' + metric_name
            
            # log individually
            self.tensorboardlog_tofile_single(metric_name, metric_val, epoch_nr)
    
    def init_csvlog(self):
        """Init metrics data frame to be saved as CSV"""
        self.history = pd.DataFrame()
    
    def csvlog_tofile(self, new_log_data, outfile):
        """Append added log data and save to CSV"""
        self.history = pd.concat([self.history, pd.DataFrame([new_log_data])], ignore_index = True)
        self.history.to_csv(os.path.join(self.folder, outfile), index = False)
        
    def predictions_tofile(self, pred, file_prefix, ensemble_nr = None, epoch_nr = None):
        """Prediction result data frame to CSV"""
        outfile = file_prefix + '_' + str(ensemble_nr) if ensemble_nr is not None else file_prefix
        outfile = outfile + '_' + str(epoch_nr) if epoch_nr is not None else outfile
        outfile = outfile + '.csv'
        save_dset(pred, outfile)


def net_param_map(n_blocks):
    """Different ResNet-CNN layer setups to select from, based on the number of residual blocks selected"""
    # Each network layer setup (4) is defined by:
    # - Network size (depth) corresponding to number of layers
    # - Feature map size/spatial resolution which is gradually reduced
    # - Filter/kernel size which is gradually increased to cover more and more 
    #   of the feature map, first capturing local patterns, then more global, complex patterns
    # Linked together by index.
    network_sizes = [2, 4, 8, 16]
    net_seq_length_list = [
        [4096, 256, 16],
        [4096, 1024, 256, 64, 16],
        [4096, 2048, 1024, 512, 256, 128, 64, 32, 16],
        [4096, 2048, 2048, 1024, 1024, 512, 512, 256, 256, 128, 128, 64, 64, 32, 32, 16, 16]
    ]
    net_filter_size_list = [
        [64, 196, 320],
        [64, 128, 196, 256, 320],
        [64, 128, 128, 196, 256, 256, 320, 512, 512],
        [64, 64, 64, 64, 64, 128, 128, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512]
    ]
    if n_blocks in network_sizes:
        indx = network_sizes.index(n_blocks)
        return net_filter_size_list[indx], net_seq_length_list[indx]
    else:
        raise ValueError(f"Network size {n_blocks} not in pre-defined set of sizes {network_sizes}")
