# -*- coding: utf-8 -*- 
"""
Help functions to extract ECG data from the GE MUSE XML format and normalize.
Adapted from script by antonior92@gmail.com
"""

import xml.etree.ElementTree as ET
import numpy as np
import base64
import scipy.signal as sgn
import re
import json



def read_gemuse(xml_filename, lead_order = ['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'], add4 = False):
    """Read a GE MUSE XML file and extract the 8 lead data on mV scale to a stacked array together with the sampling rate."""
    # Parse the XML tree.
    tree = ET.parse(xml_filename)
    root = tree.getroot()
    
    # Get the Waveform node containing all lead data.
    # Two Waveform types: WaveformType = {Median,Rhytm}
    # Rhytm is continuous 10s samplings and Median is the median ECG tracing for these.
    # Get Rhythm.
    rhythm_waveform = root.findall("Waveform[WaveformType='Rhythm']")[0]
    sample_rate = float(rhythm_waveform.find('SampleBase').text)
    
    ecg_data = {}
    for lead in rhythm_waveform.findall('LeadData'):
        lead_id = lead.find('LeadID').text
        
        scaling = float(lead.find('LeadAmplitudeUnitsPerBit').text.replace(',', '.'))
        unit = lead.find('LeadAmplitudeUnits').text.lower()
        if unit == 'microvolts':
            scaling = scaling / 1000
        elif unit == 'millivolts':
            pass
        else:
            raise ValueError('LeadAmplitudeUnit = {} not handled!'.format(unit))
        
        # Get the waveform data: base64 decode and collect the 2-byte signed integer array (little endian).
        waveform_data = lead.find('WaveFormData')
        if waveform_data is not None:
            if waveform_data.text:
                lead_bytes = base64.b64decode(waveform_data.text)
                lead_array = np.frombuffer(lead_bytes, dtype = 'i2') * scaling
                ecg_data[lead_id] = lead_array
    
    lead_lengths = [len(ecg_data[lead]) for lead in ecg_data.keys()]
    if len(set(lead_lengths)) > 1:
        raise ValueError("All leads are not of the same length")
    
    # Leads available in the GE MUSE format: I,II,V1,V2,V3,V4,V5,V6
    # The remaining 4 in a 12-lead setup are a function of the 8, hence redundant.
    # Add them if requested:
    if add4:
        required_leads = ['I', 'II']
        missing = [lead for lead in required_leads if lead not in ecg_data]
        if missing:
            raise ValueError('I and II has to be present in order to calculate additional four leads.')
        ecg_data['III'] = np.subtract(ecg_data['II'], ecg_data['I']) # III = II - I
        ecg_data['AVF'] = np.subtract(ecg_data['II'], 0.5*ecg_data['I']) # aVF = II – 0.5*I
        ecg_data['AVR'] = np.add(ecg_data['I'], ecg_data['II'])*(-0.5) # aVR = –0.5*I – 0.5*II
        ecg_data['AVL'] = np.subtract(ecg_data['I'], 0.5*ecg_data['II']) # aVL = I – 0.5*II
    
    # All leads have to be present
    if not set(lead_order).issubset(set(ecg_data)):
        raise ValueError("Not all required leads are present")
    
    # Stack the lead data in the specified order
    ecg = np.stack([ecg_data[x] for x in lead_order], axis = 0)
    return ecg, sample_rate



def read_mortara(xml_filename, lead_order = ['I', 'II', 'III', 'AVF', 'AVR', 'AVL', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']):
    """Read a Mortara ECG XML file and extract the lead data in mV scale to a stacked array with the sampling rate."""
    # Parse the XML tree
    tree = ET.parse(xml_filename)
    root = tree.getroot()
    
    # Get the continuous sampling data
    ecg_data = {}
    sample_rates = set() # keep track of sampling frequency per lead
    lead_order = [x.upper() for x in lead_order]
    for channel in root.findall('CHANNEL'):
        # Only process leads in extract list
        lead_id = channel.get('NAME').upper()
        if lead_id not in lead_order:
            continue
        
        # Get sampling frequency per lead
        sample_rate = float(channel.get('SAMPLE_FREQ'))
        sample_rates.add(sample_rate)
        
        # Get scaling factor (signal in mV which is the target unit)
        scaling = 1.0 / float(channel.get('UNITS_PER_MV'))
        
        # base64 decode the signed 16b integer array
        waveform_data = channel.get('DATA')
        if waveform_data:
            waveform_data = re.sub(r'\s', '', waveform_data) # Make sure no whitespace character causes issues
            lead_bytes = base64.b64decode(waveform_data)
            lead_array = np.frombuffer(lead_bytes, dtype = 'i2') * scaling
            ecg_data[lead_id] = lead_array
    
    # All leads have to be present
    if set(lead_order) != set(ecg_data):
        raise ValueError("Not all leads are present")
    
    # Verify that all leads are recorded using the same sampling rate
    if len(sample_rates) > 1:
        raise ValueError("Multiple sampling rates are not supported")
    
    # Verify all leads have the same length
    lead_lengths = [len(ecg_data[lead]) for lead in ecg_data.keys()]
    if len(set(lead_lengths)) > 1:
        raise ValueError("All leads are not of the same length")
    
    # Stack the lead data in the specified order
    ecg = np.stack([ecg_data[x] for x in lead_order if x in ecg_data], axis = 0)
    return ecg, sample_rate



def read_basel_json(json_filename, lead_order = ['I', 'II', 'III', 'AVF', 'AVR', 'AVL', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']):
    """Read a Basel ECG JSON file with leads in mV scale and return stacked array with sampling rate."""
    # Lead order to upper case
    lead_order = [x.upper() for x in lead_order]
    
    # Parse the JSON data
    with open(json_filename, 'r') as f:
        data = json.load(f)
    leads_list = data.get('leads', [])
    if not isinstance(leads_list, list):
        raise ValueError("The data did not contain a leads list")
    
    # Get the continuous sampling data
    ecg_data = {}
    sample_rates = set() # keep track of sampling frequency per lead
    for lead_item in leads_list:
        lead_id = lead_item.get('lead_name', '').upper()
        if lead_id not in lead_order:
            continue
            
        # Get sampling frequency
        sample_rate = float(lead_item.get('sampling_frequency', 0))
        if sample_rate <= 0:
            raise ValueError("Invalid sampling frequency for lead")
        sample_rates.add(sample_rate)
        
        # Get signal (already in mV)
        waveform_data = lead_item.get('signal')
        if waveform_data is None or not isinstance(waveform_data, list):
            raise ValueError("Missing or invalid signal data for lead")
        
        # Convert to numpy array (float since already in mV)
        lead_array = np.array(waveform_data, dtype = np.float64)
        
        # Optional offset
        offset = float(lead_item.get('offset', 0))
        if offset != 0:
            lead_array = lead_array + offset
        
        ecg_data[lead_id] = lead_array
    
    # All leads have to be present
    if set(lead_order) != set(ecg_data):
        raise ValueError("Not all leads are present")
    
    # Verify that all leads are recorded using the same sampling rate
    if len(sample_rates) > 1:
        raise ValueError("Multiple sampling rates are not supported")
    if len(sample_rates) == 0:
        raise ValueError("No sampling rate found")
    
    # Verify all leads have the same length
    lead_lengths = [len(ecg_data[lead]) for lead in ecg_data]
    if len(set(lead_lengths)) > 1:
        raise ValueError("All leads are not of the same length")
    
    # Stack the lead data in the specified order
    ecg = np.stack([ecg_data[x] for x in lead_order], axis = 0)
    
    sample_rate = float(next(iter(sample_rates)))
    
    return ecg, sample_rate



def remove_baseline_filter(sample_rate):
    """For a given sampling rate """
    fc = 0.8  # [Hz], cutoff frequency
    fst = 0.2  # [Hz], rejection band
    rp = 0.5  # [dB], ripple in passband
    rs = 40  # [dB], attenuation in rejection band
    wn = fc / (sample_rate / 2)
    wst = fst / (sample_rate / 2)
    
    filterorder, aux = sgn.ellipord(wn, wst, rp, rs)
    sos = sgn.iirfilter(filterorder, wn, rp, rs, btype = 'high', ftype = 'ellip', output = 'sos')
    
    return sos



def bandpass_filter(ecg, sample_rate, low_cutoff = 0.16, high_cutoff = 150, order = 4):
    """Zero-phase bandpass filter (Butterworth) with cutoffs according to the Swedish GE MUSE data."""
    nyquist = 0.5 * sample_rate
    low = low_cutoff / nyquist
    high = high_cutoff / nyquist
    b, a = sgn.butter(order, [low, high], btype = "band")
    
    return sgn.filtfilt(b, a, ecg, axis = -1)



def powerline_filter(ecg, sample_rate, powerline_hz, quality = 30.0):
    """Remove powerline interference at typically 50 or 60 Hz."""
    b, a = sgn.iirnotch(powerline_hz, quality, fs = sample_rate)
    return sgn.filtfilt(b, a, ecg, axis = -1)



def normalize(ecg, old_rate, new_rate, padded_length, subset_length = -1):
    """Take a stacked array with all lead data, remove the baseline, resample to new_rate, and zero pad to padded_length."""
    # Remove baseline.
    sos = remove_baseline_filter(old_rate)
    ecg_nobaseline = sgn.sosfiltfilt(sos, ecg, padtype = 'constant', axis = -1)
    
    # Resample to new_rate Hz.
    ecg_resampled = sgn.resample_poly(ecg_nobaseline, up = new_rate, down = old_rate, axis = -1)
    
    # Optional: truncate the signal to the first subset_length samples
    if subset_length != -1:
        n_leads, lead_len = ecg_resampled.shape
        if subset_length > lead_len:
            raise ValueError("The requested subset length is larger than the current lead length")
        if subset_length > padded_length:
            raise ValueError("The requested subset length is larger than the padded length")
        ecg_resampled = ecg_resampled[:, :subset_length]
    
    # Zero pad from original length to padded_length to match the CNN design used.
    # The signal is centered with equal padding on both sides.
    n_leads, lead_len = ecg_resampled.shape
    if padded_length < lead_len:
        raise ValueError("The requested padded length is less than the current lead length")
    ecg_zeropadded = np.zeros([n_leads, padded_length])
    pad = (padded_length - lead_len) // 2
    ecg_zeropadded[..., pad:lead_len+pad] = ecg_resampled
    
    return ecg_zeropadded
