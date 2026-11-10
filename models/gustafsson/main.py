# -*- coding: utf-8 -*-
"""
Train a convolutional neural network with residual blocks to use 8 leads from
a 12 lead ECG setup to predict a myocardial infarction subtypes.
Can predict non-overlapping or overlapping outcome classes or a combination thereof.
Two modes: train+validate or test
Adapted from script by antonior92@gmail.com
"""
__author__ = "Stefan Gustafsson"
__email__  = "stefan.gustafsson@medsci.uu.se"

import torch
import os
from tqdm import tqdm
import torch.optim as optim
import pandas as pd
import numpy as np
import json
import copy
import argparse
from warnings import warn

# User defined
from model import ECGModel, EnsembleECGModel
from dataloader import OMIDataset, BatchDataloader
import utils
from metrics import EcgMetrics


def train_epoch(epoch_nr, dataloader, optimiser, model, w_bin, w_cat, indx_bin, indx_cat, loss_fn_cat, loss_fn_bin, device):
    """Training of model in batches, summing up accumulated loss."""
    model.train()
    
    # Separate loss functions for the categorical (CE) and binary outcomes (BCE) which
    # are summed up as a weighted sum.
    # By setting w_bin or w_cat to 0 only one of the loss functions is applied.
    
    # For a given epoch, sum up total loss and number of ECGs processed over batches
    if w_bin > 0 and w_cat > 0:
        total_loss_bin = 0
        total_loss_cat = 0
    total_loss = 0
    n_ecgs = 0
    
    # Training progress bar
    train_pbar = tqdm(
        dataloader, desc = "Training, epoch {epoch:2d}".format(epoch = epoch_nr), 
        leave = False, initial = 0, position = 0
    )
    
    # Process batches of the given epoch
    for ecgs_cpu, outcomes_cpu, age_cpu, male_cpu in train_pbar:
        # Data to device (GPU if available)
        # Combine age and sex into a (n_batch, 2) tensor
        ecgs, outcomes = ecgs_cpu.to(device), outcomes_cpu.to(device)
        age_sex = torch.cat(
            [
                torch.tensor(male_cpu, dtype = torch.float32).unsqueeze(1), 
                torch.tensor(age_cpu, dtype = torch.float32).unsqueeze(1)
            ], 
            dim = 1
        )
        age_sex = age_sex.to(device = device)
        
        # Reset gradients for each batch
        model.zero_grad()
        
        # Forward pass, making predictions from model
        pred_logits = model((age_sex, ecgs))
        
        # Calculate loss from observed vs predicted
        
        # Categorical
        loss_cat = torch.tensor(0.0, dtype = torch.float32)
        if w_cat > 0:
            loss_cat = loss_fn_ce(pred_logits[:, indx_cat], outcomes[:, indx_cat]) 
        # Binary
        loss_bin = torch.tensor(0.0, dtype = torch.float32)
        if w_bin > 0:
            loss_bin = loss_fn_bce(pred_logits[:, indx_bin], outcomes[:, indx_bin])
        loss = w_cat*loss_cat + w_bin*loss_bin
        
        # Backward pass to calculate gradients of the loss with respect to model parameters
        loss.backward()
        
        # Update model parameters using the gradients
        optimiser.step()
        
        # Sum up loss and ECGs processed so far
        if w_bin > 0 and w_cat > 0:
            total_loss_cat += loss_cat.detach().cpu().numpy()
            total_loss_bin += loss_bin.detach().cpu().numpy()
        total_loss += loss.detach().cpu().numpy()
        n_ecgs += len(ecgs)
        
    train_pbar.close()
    
    loss_metrics = {'tot': total_loss/n_ecgs}
    if w_bin > 0 and w_cat > 0:
        loss_metrics.update({'bin': total_loss_bin/n_ecgs, 'cat': total_loss_cat/n_ecgs})
    return loss_metrics


def evaluate_epoch(epoch_nr, dataloader, n_records, n_outcomes, model, w_bin, w_cat, indx_bin, indx_cat, loss_fn_cat, loss_fn_bin, device):
    """Validation or test of model calculating accumulated loss and predicted probabilities."""
    model.eval()
    
    # Separate loss functions for the categorical (CE) and binary outcomes (BCE) which
    # are summed up as a weighted sum.
    # By setting w_bin or w_cat to 0 only one of the loss functions is applied.
    
    # For a given epoch, sum up total loss and number of ECGs processed over batches
    if w_bin > 0 and w_cat > 0:
        total_loss_bin = 0
        total_loss_cat = 0
    total_loss = 0
    n_ecgs = 0
    
    # Allocate storage of predicted probabilities
    pred_prob = np.zeros((n_records, n_outcomes))
    batch_end = 0
    
    # Validation/test progress bar
    eval_pbar = tqdm(
        dataloader, 
        desc = "Evaluation, epoch {epoch:2d}".format(epoch = epoch_nr), 
        leave = False, initial = 0, position = 0
    )
    
    # Process batches of the given epoch
    for ecgs_cpu, outcomes_cpu, age_cpu, male_cpu in eval_pbar:
        # Data to device (GPU if available)
        # Combine age and sex into a [n_batch, 2] tensor
        ecgs, outcomes = ecgs_cpu.to(device), outcomes_cpu.to(device)
        age_sex = torch.cat(
            [
                torch.tensor(male_cpu).long().unsqueeze(1), 
                torch.tensor(age_cpu).unsqueeze(1)
            ], 
            dim = 1
        )
        age_sex = age_sex.to(device = device)
        batch_start = batch_end
        
        # Only predict, don't bother with gradients
        with torch.no_grad():
            pred_logits = model((age_sex, ecgs))
            
            # Calculate loss from observed vs predicted
            
            # Categorical
            loss_cat = torch.tensor(0.0, dtype = torch.float32)
            if w_cat > 0:
                loss_cat = loss_fn_ce(pred_logits[:, indx_cat], outcomes[:, indx_cat]) 
            # Binary
            loss_bin = torch.tensor(0.0, dtype = torch.float32)
            if w_bin > 0:
                loss_bin = loss_fn_bce(pred_logits[:, indx_bin], outcomes[:, indx_bin])
            loss = w_cat*loss_cat + w_bin*loss_bin
            
            # Calculate and store predicted probabilities
            batch_end = min(batch_start + len(pred_logits), pred_prob.shape[0])
            pred_prob[batch_start:batch_end, indx_cat] = torch.nn.Softmax(dim = 1)(pred_logits[:, indx_cat]).detach().cpu().numpy()
            pred_prob[batch_start:batch_end, indx_bin] = torch.nn.Sigmoid()(pred_logits[:, indx_bin]).detach().cpu().numpy()
            
            # Sum up loss and ECGs processed so far
            if w_bin > 0 and w_cat > 0:
                total_loss_cat += loss_cat.detach().cpu().numpy()
                total_loss_bin += loss_bin.detach().cpu().numpy()
            total_loss += loss.detach().cpu().numpy()
            n_ecgs += len(ecgs)
            
    eval_pbar.close()
    
    loss_metrics = {'tot': total_loss/n_ecgs}
    if w_bin > 0 and w_cat > 0:
        loss_metrics.update({'bin': total_loss_bin/n_ecgs, 'cat': total_loss_cat/n_ecgs})
    return loss_metrics, pred_prob


def run_train_validate(arg, loss_fn_cat, loss_fn_bin, logger, metrics):
    """Run training + validation"""
    
    ############################################################################
    # Preparing input
    ############################################################################
    
    tqdm.write('Setting up data...', end = '')
    
    dset = OMIDataset(
        path_to_h5 = arg.hdf5,
        path_to_txt = arg.txt,
        col_outcome = arg.col_outcome,
        split_col = arg.split_col,
        age_col = arg.age_col, male_col = arg.male_col, 
        age_mean = arg.age_mean, age_sd = arg.age_sd
    )
    
    # Get index of outcome columns
    indx_bin = [dset.outcomes.columns.get_loc(col_x) for col_x in arg.outcomes_bin]
    indx_cat = [dset.outcomes.columns.get_loc(col_x) for col_x in arg.outcomes_cat]
    
    # Number of records per set
    n_train = sum(dset.train)
    n_valid = sum(dset.valid)
    
    # Save copy of train and validation set
    utils.save_dset(dset.outcomes[dset.train], os.path.join(arg.folder, 'observed_data_train.csv'))
    utils.save_dset(dset.outcomes[dset.valid], os.path.join(arg.folder, 'observed_data_valid.csv'))
    
    tqdm.write('done')
    tqdm.write('Found {} training and {} validation records\n'.format(n_train, n_valid))
    
    
    # Multiple runs of the model with different initial guesses, leading
    # to multiple slightly different versions of the trained model (ensembles).
    # Combined into the final model
    for ensemble_nr in range(1, arg.n_ensembles + 1):
        best_loss = np.Inf
        
        train_loader = BatchDataloader(dset, arg.batch_size, mask = dset.train)
        valid_loader = BatchDataloader(dset, arg.batch_size, mask = dset.valid)
        
        true_outcomes_valid = dset.outcomes[dset.valid]
        
        model = ECGModel(arg)
        model.to(device = arg.device)
        
        if arg.optim_algo == 'SGD':
            optimiser = optim.SGD(model.parameters(), lr = arg.lr, weight_decay = arg.weight_decay)
        else:
            optimiser = optim.Adam(model.parameters(), lr = arg.lr, weight_decay = arg.weight_decay)
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimiser,
            patience = arg.patience,
            min_lr = arg.lr_factor * arg.min_lr,
            factor = arg.lr_factor
        )
        
        if arg.n_ensembles > 1:
            tqdm.write('\nRunning ensemble {} out of {}\n'.format(ensemble_nr, arg.n_ensembles))
            model_out = os.path.join(arg.folder, 'model_' + str(ensemble_nr) + '.pth')
            tensorboard_out = str(ensemble_nr)
            csv_out = 'performance_metrics_' + str(ensemble_nr) + '.csv'
        else:
            model_out = os.path.join(arg.folder, 'model.pth')
            tensorboard_out = ''
            csv_out = 'performance_metrics.csv'
        
        logger.init_tensorboardlog(suffix = tensorboard_out)
        logger.init_csvlog()
        
        for epoch_nr in range(1, arg.epochs + 1):
            train_loss = train_epoch(epoch_nr, train_loader, optimiser, model, arg.w_bin, arg.w_cat, indx_bin, indx_cat, loss_fn_cat, loss_fn_bin, arg.device)
            valid_loss, valid_pred = evaluate_epoch(epoch_nr, valid_loader, n_valid, arg.n_outcomes, model, arg.w_bin, arg.w_cat, indx_bin, indx_cat, loss_fn_cat, loss_fn_bin, arg.device)
            
            # Save validation predictions at each ensemble+epoch iteration.
            if arg.save_progress:
                logger.predictions_tofile(valid_pred, 'valid', epoch_nr, ensemble_nr)
            
            # Compute validation metrics
            if not arg.no_metrics:
                valid_metrics = metrics.compute_metrics(valid_pred, true_outcomes_valid)
            
            # Save the model with the current best validation loss
            if valid_loss['tot'] < best_loss:
                torch.save({
                    'epoch': epoch_nr,
                    'model': model.state_dict(),
                    'valid_loss': valid_loss['tot'],
                    'optimiser': optimiser.state_dict()
                }, model_out)
                best_loss = valid_loss['tot']
                is_current_best = True
            else:
                is_current_best = False
            
            # Break if minimum learning rate has been passed
            for param_group in optimiser.param_groups:
                current_learning_rate = param_group["lr"]
            if current_learning_rate < arg.min_lr:
                tqdm.write("Stopped since learning rate minimum is reached!")
                break
            
            # Loss metrics to terminal.
            tqdm.write(
                'Epoch {epoch:2d}: '
                '\033[91mTrain Loss\033[0m [tot={train_loss:.5f}] | '
                '\033[91mValid Loss\033[0m [tot={valid_loss:.5f}] | '
                '\033[91mLearn Rate\033[0m [{lr:.7f}] | '
                '\033[91mBest\033[0m [{best}]'
                .format(
                    epoch = epoch_nr,
                    train_loss = train_loss['tot'],
                    valid_loss = valid_loss['tot'],
                    lr = current_learning_rate,
                    best = is_current_best
                )
            )
            
            # All metrics to log.
            log_data = {
                "ensemble": ensemble_nr,
                "epoch": epoch_nr,
                "lr": current_learning_rate
            }
            log_data.update(train_loss)
            log_data.update(valid_loss)
            if not arg.no_metrics:
                log_data.update(valid_metrics)
            
            logger.tensorboardlog_tofile(log_data, epoch_nr)
            logger.csvlog_tofile(log_data, csv_out)
            
            # Update learning rate with lr-scheduler
            scheduler.step(valid_loss['tot'])


def run_test(arg, loss_fn_cat, loss_fn_bin, logger, metrics):
    """Evaluate model in test set"""
    
    ############################################################################
    # Preparing input
    ############################################################################
    
    tqdm.write('Setting up data...', end = '')
    
    dset = OMIDataset(
        path_to_h5 = arg.hdf5,
        path_to_txt = arg.txt,
        split_col = arg.split_col,
        col_outcome = arg.col_outcome,
        test = arg.test, test_name = arg.test_name,
        age_col = arg.age_col, male_col = arg.male_col, 
        age_mean = arg.age_mean, age_sd = arg.age_sd
    )
    
    # Get index of outcome columns
    indx_bin = [dset.outcomes.columns.get_loc(col_x) for col_x in arg.outcomes_bin]
    indx_cat = [dset.outcomes.columns.get_loc(col_x) for col_x in arg.outcomes_cat]
    
    # Number of records in set
    n_test = sum(dset.test)
    
    # Save copy of testset
    utils.save_dset(dset.outcomes[dset.test], os.path.join(arg.folder, 'observed_data_' + arg.test_name + '.csv'))
    
    # Observed/true outcomes
    true_outcomes_test = dset.outcomes[dset.test]
    
    tqdm.write('done')
    tqdm.write('Found {} test records for "{}"\n'.format(n_test, arg.test_name))
    
    
    ############################################################################
    # Data loader
    ############################################################################
    
    test_loader = BatchDataloader(dset, arg.batch_size, mask = dset.test)
    
    
    ############################################################################
    # Setup model
    ############################################################################
    
    tqdm.write('Loading model...', end = '')
    
    if (arg.n_ensembles == 1):
        model = ECGModel(arg)
    else:
        model = EnsembleECGModel(arg, arg.model_path)
    
    tqdm.write('done\n')
    
    
    ############################################################################
    # Evaluate model
    ############################################################################
    
    # Predict in testset and calculate evaluation metrics
    test_loss, test_pred = evaluate_epoch(0, test_loader, n_test, arg.n_outcomes, model, arg.w_bin, arg.w_cat, indx_bin, indx_cat, loss_fn_cat, loss_fn_bin, arg.device)
    
    if not arg.no_metrics:
        test_metrics = metrics.compute_metrics(test_pred, true_outcomes_test)
    
    # Predictions to log
    test_pred = pd.DataFrame(
        data = test_pred,
        index = true_outcomes_test.index.values.tolist(),
        columns = ['pr_' + x for x in true_outcomes_test.columns.values.tolist() if isinstance(x, str)]
    )
    test_pred.index.name = true_outcomes_test.index.name
    logger.predictions_tofile(test_pred, os.path.join(arg.folder, 'predicted_data_' + arg.test_name))
    
    # Evaluation metrics to log
    logger.init_csvlog()
    log_data = test_loss
    if not arg.no_metrics:
        log_data.update(test_metrics)
    logger.csvlog_tofile(log_data, arg.test_name + '_metrics.csv')


def parse_arguments():
    """Parse command-line arguments from sys.argv, used to control the script"""
    # Paths and system settings.
    # Saved to sys.json
    sys_parser = argparse.ArgumentParser(add_help = False)
    sys_parser.add_argument('--model_path', type = str, 
        help = 'Pre-existing path with model.pth and model_config.json to be loaded.')
    sys_parser.add_argument('--hdf5', type = str, default = '',
        help = 'HDF5 file with ECGs used in training/validation/test.')
    sys_parser.add_argument('--hdf5_dset_ecg', 
        help = 'Dataset with ECGs in the HDF5 file.')
    sys_parser.add_argument('--hdf5_dset_recordid', 
        help = 'Dataset with record IDs in the HDF5 file.')
    sys_parser.add_argument('--txt', type = str, default = '',
        help = 'Tab-separated file with record ID, outcome classes, and test sets.')
    sys_parser.add_argument('--split_col', type = str, default = 'split',
        help = 'Name of test set split column in tab-separated file.')
    sys_parser.add_argument('--age_col', type = str, default = 'age',
        help = 'Name of age column in tab-separated file.')
    sys_parser.add_argument('--male_col', type = str, default = 'male',
        help = 'Name of male column in tab-separated file.')
    sys_parser.add_argument('--test', action = 'store_true',
        help = 'Test of model instead of default train/validate')
    sys_parser.add_argument('--test_name', type = str, default = 'test',
        help = 'Name of test set split in tab-separated file.')
    sys_parser.add_argument('--out_dir', type = str, 
        help = 'Output folder for logs and saved model.')
    sys_parser.add_argument('--cpu', action = 'store_true',
        help = 'CPU based calculations instead of default GPU.')
    sys_parser.add_argument('--gpu_index', type = int, default = 0,
        help = 'CUDA GPU index')
    sys_parser.add_argument('--save_progress', action = 'store_true',
        help = 'Save current training status continuously')
    sys_parser.add_argument('--outcomes_json', type = str, default = None,
        help = 'JSON file with outcome columns.')
    sys_parser.add_argument('--no_metrics', action = 'store_true',
        help = 'Do not calculate evaluation metrics such as C-statistic.')
    
    # CNN setup arguments
    # Saved in model_config.json
    model_parser = argparse.ArgumentParser(add_help = False)
    model_parser.add_argument('--seed', type = int, default = 1234567,
        help = 'Seed for the PRNG for reproducable results.')
    model_parser.add_argument('--outcomes_cat', type = str, nargs = '+', default = [],
        help = 'List of categorical outcome columns in tab-separated.')
    model_parser.add_argument('--outcomes_bin', type = str, nargs = '+', default = [],
        help = 'List of binary outcome columns in tab-separated.')
    # Training arguments
    model_parser.add_argument('--epochs', type = int, default = 70,
        help = 'Maximum number of epochs')
    model_parser.add_argument('--batch_size', type = int, default = 32,
        help = 'Batch size.')
    model_parser.add_argument('--lr', type = float, default = 0.001,
        help = 'Initial learning rate')
    model_parser.add_argument("--patience", type = int, default = 7,
        help = 'Maximum number of epochs at plateau without reducing the learning rate')
    model_parser.add_argument("--min_lr", type = float, default = 1e-7,
        help = 'Minimum learning rate')
    model_parser.add_argument("--lr_factor", type = float, default = 0.1,
        help = 'Reducing factor for learning rate at plateau')
    model_parser.add_argument('--weight_decay', type = float, default = 0.01,
        help = 'Weight decay in the optimisation')
    # Model arguments
    model_parser.add_argument('--seq_length', type = int, default = 4096,
        help = 'Length of all stacked ECG leads')
    model_parser.add_argument('--n_residual_block', type = int, 
        help = 'Number of residual blocks')
    model_parser.add_argument('--net_filter_size', type = int, nargs = '+', default = [64, 128, 196, 256, 320],
        help = 'Filter size in resnet layers')
    model_parser.add_argument('--net_seq_length', type = int, nargs ='+', default = [4096, 1024, 256, 64, 16],
        help = 'Number of samples per resnet layer')
    model_parser.add_argument('--dropout_rate', type = float, default = 0.8,
        help = 'Dropout proportion')
    model_parser.add_argument('--kernel_size', type = int, default = 17,
        help = 'Kernel size in convolutional layers')
    model_parser.add_argument('--activation_function', type = str, default = 'ReLU',
        help = 'Activation function used on model')
    model_parser.add_argument('--optim_algo', type = str, default = 'ADAM',
        help = 'Optimization algorithm')
    model_parser.add_argument('--w_bin_cat_ratio', type = float, default = 1.0,
        help = 'Ratio of weights in the weighted loss (bin/cat).')
    model_parser.add_argument('--n_ensembles', type = int, default = 1,
        help = 'Number of ensembles.')
    model_parser.add_argument('--n_leads', type = int, default = 8,
        help = 'Number of leads to use.')
    model_parser.add_argument('--agesex_dim', type = int, default = 64,
        help = 'Dimensions to represent age and sex with.')
    model_parser.add_argument('--age_mean', type = float, default = None,
        help = 'Mean of age for standardization.')
    model_parser.add_argument('--age_sd', type = float, default = None,
        help = 'SD of age for standardization.')
    
    
    arg_sys, arg_nonsys = sys_parser.parse_known_args()
    
    # Set defaults according to stored settings
    if arg_sys.model_path:
        reuse_model = True
        with open(os.path.join(arg_sys.model_path, 'model_config.json'), 'r') as f:
            model_config = json.load(f)
        if model_config is not None:
            model_parser.set_defaults(**model_config)
    else:
        reuse_model = False
    
    arg_model, arg_unknown = model_parser.parse_known_args(arg_nonsys)
    model_parser = argparse.ArgumentParser(
        parents = [model_parser, sys_parser],
        description = 'Train+validate or test model.'
    )
    _, arg_unknown = model_parser.parse_known_args(arg_unknown)
    if arg_unknown:
        warn("Unknown arguments: " + str(arg_unknown) + ".")
    
    # Generate output folder if needed
    if arg_sys.test:
        arg_sys.folder = arg_sys.out_dir
    else:
        arg_sys.folder = utils.set_output_folder(arg_sys.out_dir)
    
    # Set device
    device = 'cuda:' + str(arg_sys.gpu_index) if not arg_sys.cpu and torch.cuda.is_available() else 'cpu'
    arg_model.device = device
    
    # Outcomes
    # The order of outcomes is locked in training. The same order is enforced for test.
    if not arg_sys.test:
        if not arg_sys.outcomes_json is None:
            with open(arg_sys.outcomes_json, "r") as file:
                data = json.load(file)
                arg_model.outcomes_cat = data['outcomes_cat'] if 'outcomes_cat' in data else arg_model.outcomes_cat
                arg_model.outcomes_bin = data['outcomes_bin'] if 'outcomes_bin' in data else arg_model.outcomes_bin
    
    if arg_sys.test:
        for key in ['outcomes_cat', 'outcomes_bin']:
            setattr(arg_model, key, model_config[key])
    
    arg_model.col_outcome = arg_model.outcomes_cat + arg_model.outcomes_bin
    arg_model.n_outcomes = len(arg_model.col_outcome)
    
    return arg_sys, arg_model, arg_unknown


if __name__ == "__main__":
    """Start training+validation or test"""
    
    ############################################################################
    # Process user-supplied arguments and setup
    ############################################################################
    
    tqdm.write('Initiating...', end = '')
    
    arg_sys, arg_model, _ = parse_arguments()
    
    # Save config file
    # Keep all arguments in one list from now on.
    if not arg_sys.test:
        utils.save_config(arg_model, os.path.join(arg_sys.folder, 'model_config.json'))
        utils.save_config(arg_sys, os.path.join(arg_sys.folder, 'sys.json'))
    else:
        utils.save_config(arg_sys, os.path.join(arg_sys.folder, 'sys_' + arg_sys.test_name + '.json'))
    arg_all = argparse.Namespace(**vars(arg_sys), **vars(arg_model))
    
    if(arg_all.n_residual_block is not None and arg_all.n_residual_block in [2, 4, 8, 16]):
        arg_all.net_filter_size, arg_all.net_seq_length = utils.net_param_map(arg_all.n_residual_block)
    
    # Set seed
    utils.seed_everything(arg_all.seed)
    
    tqdm.write('done')
    tqdm.write('Using seed {}'.format(str(arg_all.seed)))
    tqdm.write('Using device {}'.format(arg_all.device))
    if not arg_sys.test:
        tqdm.write('Model settings saved in {}'.format(os.path.join(arg_all.folder, 'model_config.json')))
        tqdm.write('System settings saved in {}\n'.format(os.path.join(arg_all.folder, 'sys.json')))
    
    
    ############################################################################
    # Loss functions
    ############################################################################
    
    # Cross entropy loss (softmax)
    # Multi-class, single-label (non-overlapping classes, i.e. class prob sum up to 1)
    # Logits as input
    loss_fn_ce = torch.nn.CrossEntropyLoss()
    
    # Binary cross entropy loss (sigmoid)
    # Multi-label (overlapping classes, i.e. class prob does not have to sum to 1, instead sum up to 1 to n_classes)
    # Logits as input
    loss_fn_bce = torch.nn.BCEWithLogitsLoss() # Numerically more stable than Sigmoid+BCELoss
    
    # Weighted loss
    # w_bin*loss_bin + w_cat*loss_cat
    # w_bin + w_cat = 1
    # w_bin / w_cat = ratio
    if len(arg_all.outcomes_cat) == 0:
        arg_all.w_bin = 1
        arg_all.w_cat = 0
    elif len(arg_all.outcomes_bin) == 0:
        arg_all.w_bin = 0
        arg_all.w_cat = 1
    else:
        arg_all.w_bin = arg_all.w_bin_cat_ratio / (arg_all.w_bin_cat_ratio+1)
        arg_all.w_cat = 1 / (arg_all.w_bin_cat_ratio+1)
    
    
    ############################################################################
    # Logger and metrics
    ############################################################################
    
    tqdm.write('Set up logger and metrics...', end = '')
    metrics = EcgMetrics(arg_all.col_outcome)
    logger = utils.Logger(arg_all.folder)
    tqdm.write('done')
    tqdm.write('Logs and output written to "{}"\n'.format(arg_all.folder))
    
    
    ############################################################################
    # Run train+validate or test
    ############################################################################
    
    # Two possible paths:
    
    # Test:
    if arg_all.test:
        tqdm.write('Testing model...\n')
        run_test(arg_all, loss_fn_ce, loss_fn_bce, logger, metrics)
    
    # ... or ...
    
    # Training+validation:
    else:
        tqdm.write('Training and validating model...\n')
        run_train_validate(arg_all, loss_fn_ce, loss_fn_bce, logger, metrics)
    
    tqdm.write('\nall done')
