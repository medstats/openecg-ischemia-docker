# -*- coding: utf-8 -*-
"""
From a list of XML files with ECG tracings, extract and normalize the data.
Store in HDF5 format for analysis.
Adapted from script by antonior92@gmail.com
"""

import argparse
import os
import pandas as pd
import numpy as np
import h5py
from tqdm import tqdm

from ecg_processing import read_gemuse, read_mortara, read_basel_json, bandpass_filter, powerline_filter, normalize



def main():
    """Create a HDF5 file from GE MUSE or Mortara XML files."""
    parser = argparse.ArgumentParser(description = 'Create a HDF5 file from GE MUSE or Mortara XML files.')
    parser.add_argument('--record_list', type = str, 
        help = 'Tab-separated file with GE MUSE XML filenames (full_filename, filename, id_record).')
    parser.add_argument('--hdf5', type = str, 
        help = 'HDF5 output file with extracted data.')
    parser.add_argument('--transpose', action = 'store_true', 
        help = 'Store traces with dimensions (patients, signals, leads) instead of default (patients, leads, signals).')
    parser.add_argument('--lead_order', type = str, nargs = '+', default = ['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'],
        help = 'Store leads in this order instead of default [I,II,V1,V2,V3,V4,V5,V6].')
    parser.add_argument('--new_rate', type = int, default = 400,
        help = 'Final sampling rate of signal.')
    parser.add_argument('--padded_length', type = int, default = 4096,
        help = 'Final padded length of the normalized signal.')
    parser.add_argument('--subset_length', type = int, default = -1,
        help = 'Truncate all leads to this length (default: no truncation).')
    parser.add_argument('--overwrite', action = 'store_true', 
        help = 'Overwrite the HDF5 if it exists?')
    parser.add_argument('--format', type = str, default = 'gemuse',
        help = 'ECG input format (gemuse, mortara, or basel).')
    parser.add_argument('--muse_filters', action = 'store_true', 
        help = 'Enforce the GE MUSE filters used in the Swedish data')
    parser.add_argument('--powerline', type = float, default = 50.0,
        help = 'Powerline frequency depending on country (50/60 Hz).')
    args = parser.parse_args()
    
    # Check of arguments
    if not os.path.isfile(args.record_list):
        raise ValueError('Could not find input file.')
    if os.path.isfile(args.hdf5) and not args.overwrite:
        raise ValueError('Output HDF5 file exists. Will not overwrite')
    n_leads = len(args.lead_order)
    if n_leads == 0 or n_leads > 12:
        raise ValueError('Only a 1-12 lead setup is supported. ' + str(n_leads) + ' given')
    if len(args.lead_order) != len(set(args.lead_order)):
        raise ValueError('Duplicates in lead list not allowed')
    valid_leads = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    if not set(args.lead_order).issubset(valid_leads):
        raise ValueError('Only these lead names are supported: I,II,III,AVR,AVL,AVF,V1,V2,V3,V4,V5,V6')
    if args.new_rate <= 0:
        raise ValueError('New sampling rate expected to be > 0')
    if args.padded_length <= 0:
        raise ValueError('Padded length expected to be > 0')
    if args.subset_length != -1 and args.subset_length <= 0:
        raise ValueError('Subset length expected to be > 0')
    if args.subset_length > args.padded_length:
        raise ValueError('Subset length expected to be less than padded length')
    valid_formats = ['gemuse', 'mortara', 'basel']
    if args.format not in valid_formats:
        raise ValueError('Only these formats are supported: gemuse,mortara,basel')
    # 8 leads given in GE MUSE, these 4 have to be calculated
    add4 = False
    if any(x in ['III', 'AVR', 'AVL', 'AVF'] for x in args.lead_order):
        add4 = True
    
    # Records to use.
    all_records = pd.read_csv(args.record_list, sep = '\t', 
        usecols = ['full_filename', 'filename', 'id_record'])
    
    # Initiate the HDF5 file.
    # Dimension and datatype for data to be stored.
    f = h5py.File(args.hdf5, 'w')
    n = len(all_records)
    if args.transpose:
        d_ecg = f.create_dataset('ecg_normalized', (n, args.padded_length, n_leads), dtype = 'f8') # float64
    else:
        d_ecg = f.create_dataset('ecg_normalized', (n, n_leads, args.padded_length), dtype = 'f8') # float64
    d_filename = f.create_dataset('xml_filename', (n,), dtype = 'S100') # fixed length 100 allocation will fit filename, not path
    d_id_record = f.create_dataset('id_record', (n,), dtype = 'i8') # int64
    
    # Now add the actual data.
    for i in tqdm(range(len(all_records))):
        current_record = all_records.iloc[i]
        
        # Read and normalize the ECG data of the record.
        if args.format == 'mortara':
            ecg, old_rate = read_mortara(current_record['full_filename'], 
                lead_order = args.lead_order)
        elif args.format == 'basel':
            ecg, old_rate = read_basel_json(current_record['full_filename'], 
                lead_order = args.lead_order)
        else:
            ecg, old_rate = read_gemuse(current_record['full_filename'], 
                lead_order = args.lead_order, add4 = add4)
        
        # Enforce GE MUSE filters used in Swedish data
        if args.muse_filters:
            ecg = bandpass_filter(ecg, old_rate, low_cutoff = 0.16, high_cutoff = 150)
            ecg = powerline_filter(ecg, old_rate, args.powerline)
        
        # Default normalization applied.
        ecg_preprocessed = normalize(ecg, old_rate, args.new_rate, args.padded_length, args.subset_length)
        
        # Store
        if args.transpose:
            d_ecg[i, :, :] = np.transpose(ecg_preprocessed, (1, 0))
        else:
            d_ecg[i, :, :] = ecg_preprocessed
        d_filename[i] = np.string_(current_record['filename'])
        d_id_record[i] = current_record['id_record']
    
    f.close()

if __name__ == "__main__":
    main()
