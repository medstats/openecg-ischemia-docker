# -*- coding: utf-8 -*-
"""
Setup ECG data from HDF5 together with outcomes and split sets from a text file.
Dataloader fetching data in batches.
Adapted from script by antonior92@gmail.com
"""
__author__ = "Stefan Gustafsson"
__email__  = "stefan.gustafsson@medsci.uu.se"

import sys
import h5py
from collections import OrderedDict
import torch
import pandas as pd
import numpy as np


class OMIDataset:
    """Setup traces, outcomes and split sets"""
    def __init__(self, path_to_h5, path_to_txt, col_outcome, 
        traces_dset = 'ecg_normalized', record_id_dset = 'id_record', 
        test = False, test_name = 'test', split_col = 'split',
        age_col = 'age', male_col = 'male', age_mean = None, age_sd = None):
        
        # Traces with dimensions (nsamples, 4096, 8) from HDF5
        try:
            f = h5py.File(path_to_h5, 'r')
        except (FileNotFoundError, OSError) as e:
            raise ValueError(f"Failed to open HDF5 file '{path_to_h5}': {e}")
        self.f = f
        self.traces = f[traces_dset]
        self.record_ids = np.array(f[record_id_dset])
        
        # Outcomes and split sets (training/validation/test) from plain text file
        try:
            df = pd.read_csv(path_to_txt, sep = '\t')
        except (FileNotFoundError, OSError) as e:
            raise ValueError(f"Failed to open outcome file '{path_to_txt}': {e}")
        
        # These columns may not contain missing values
        nona_cols = [split_col, age_col, male_col] + col_outcome
        cols_with_na = [col for col in nona_cols if df[col].isna().any()]
        if cols_with_na:
            raise ValueError(f"Missing values found in columns: {', '.join(cols_with_na)}")
        
        # Extract split sets into separate variables
        # Build a outcome dataset with 0/1 indicators for each outcome
        # Set the numeric id_record as index and remove all non-outcome columns
        csv_index = df['id_record']
        self.csv_record_ids = csv_index
        df = df.set_index('id_record')
        if set(self.record_ids) != set(df.index):
            raise ValueError("Different record IDs found in the HDF5 vs outcome file!")
        df = df.reindex(self.record_ids, fill_value = 0, copy = True) # Same record sorting as traces
        
        # Get age and sex
        # Mean-center + scale age to unit variance
        age_raw = df[age_col]
        if age_mean is None:
            age_mean = age_raw.astype(float).mean()
            age_sd = age_raw.astype(float).std()
        if age_sd is None or age_sd == 0:
            raise ValueError("SD of age must not be empty or 0!")
        self.age = (age_raw - age_mean) / age_sd
        self.male = df[male_col]
        
        # T/F flags if records belong in given set
        if test:
            self.test = np.array(df[split_col] == test_name)
            self.valid = np.empty(0)
            self.train = np.empty(0)
        else:
            self.test = np.empty(0)
            self.valid = np.array(df[split_col] == 'valid')
            self.train = np.array(df[split_col] == 'train')
        
        # Now restrict to 0/1 outcome columns.
        if not set(col_outcome).issubset(df.columns):
            missing_cols = set(col_outcome) - set(df.columns)
            raise ValueError(f"These outcome columns were not found in the outcome file: {missing_cols}!")
        col_excl = df.columns.difference(col_outcome)
        df.drop(col_excl, axis = 1, inplace = True)
        df = df[col_outcome] # Enforce same column order as in vector
        
        self.outcomes = df
    
    def __del__(self):
        """Close HDF5 file"""
        if hasattr(self, "f"):
            self.f.close()
    
    def __len__(self):
        """Get number of records"""
        return len(self.outcomes.index)
    
    def getbatch(self, extract_indx = None):
        """Fetch a batch of ECG and corresponding outcome data, age, and sex"""
        if extract_indx is None:
            extract_indx = range(len(self))
        
        return self.traces[extract_indx], self.outcomes.iloc[extract_indx].values, self.age.iloc[extract_indx].values, self.male.iloc[extract_indx].values


class BatchDataloader:
    """Iterator loading batches of data"""
    def __init__(self, dset, batch_size, mask = None):
        """Get indices of target records and number of batches needed for these"""
        self.dset = dset
        self.batch_size = batch_size
        if mask is None:
            mask = np.ones(len(dset), dtype = bool)
        self.all_indx = np.nonzero(mask)[0].tolist() # indices of targets
        self.current_batch = 0
        self.n_batches = -(len(self.all_indx) // -self.batch_size)
    
    def __next__(self):
        """Get indices of current batch"""
        self.current_batch += 1
        if self.current_batch > self.n_batches:
            self.current_batch = 0
            raise StopIteration
        start = (self.current_batch-1)*self.batch_size
        end = min(self.current_batch*self.batch_size, len(self.all_indx))
        current_indx = self.all_indx[start:end]
        batch = self.dset.getbatch(current_indx)
        return [torch.tensor(b, dtype = torch.float32) for b in batch]
    
    def __iter__(self):
        return self
    
    def __len__(self):
        """Get total number of batches for tqdm progress"""
        return self.n_batches


def test():
    """Test the batch dataloader"""
    dset = OMIDataset(
        '/proj/sens2020005/omi/data/omi_analysis_data_20250217.hdf5',
        '/proj/sens2020005/omi/data/omi_analysis_data_20250217.txt',
        ['control_nomyoperi', 'control_myoperi'] # two columns for test
    )
    
    incl = np.zeros(len(dset), dtype = bool)
    # 53 records, 10 in each batch, 6 batches in total
    incl[:50] = True
    incl[[88, 90, 92]] = True
    loader = BatchDataloader(dset, 10, incl)
    for traces, outcomes, age, sex in loader:
        print("Loading...")
        print(traces.shape)
        print(outcomes.shape)
        print(age.shape)
        print(sex.shape)


if __name__ == "__main__":
    test()
