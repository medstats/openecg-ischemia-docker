# -*- coding: utf-8 -*-
"""
Discrimination and calibration metrics
"""
__author__ = "Stefan Gustafsson"
__email__  = "stefan.gustafsson@medsci.uu.se"

from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


def auroc_one_vs_rest(y_obs, y_pred_prob, target_names):
    """AUROC/C-statistic for target classes vs all other classes (discrimination)"""
    if not isinstance(y_obs, pd.DataFrame):
        raise TypeError("observed outcomes for AUROC should be stored in a pandas dataframe")
    if not isinstance(y_pred_prob, np.ndarray):
        raise TypeError("predicted probabilites for AUROC should be stored in a numpy array")
    if not all(name in y_obs.columns for name in target_names):
        raise ValueError("Some expected columns not found in AUROC input data")
    
    # Get indices of targets by name from a dataframe (y_obs)
    # This dataframe is expected to have the same column order as y_pred_prob
    target_colindx = np.where(y_obs.columns.isin(target_names))[0].tolist()
    truth_comb = y_obs.to_numpy()[:,target_colindx].max(axis = 1)
    pred_comb = y_pred_prob[:,target_colindx].sum(axis = 1)
    
    # AUROC only calculated for non-constant columns.
    if not np.all(np.isin(truth_comb, [0, 1])):
        raise ValueError("Combined output label should only contain [0,1]")
    if len(np.unique(truth_comb)) <= 1:
        return(np.array([None], dtype = np.float64))
    else:
        auroc = roc_auc_score(truth_comb, pred_comb)
        return(auroc)


def compute_brier_score(y_obs, y_pred_prob):
    """Calculate the Brier score (discrimination+calibration)"""
    # from https://stats.stackexchange.com/questions/403544/how-to-compute-the-brier-score-for-more-than-two-classes
    # the scikit-learn version actually averages over the classes and does not sum over them.
    
    bs = np.mean(np.sum((y_pred_prob - y_obs) ** 2, axis = 1), dtype = np.float64)
    return bs


class EcgMetrics(object):
    """Discrimination/calibration metrics"""
    
    def __init__(self, col_outcome):
        if not isinstance(col_outcome, (list, tuple)) or not all(isinstance(x, str) for x in col_outcome):
            raise TypeError("outcome columns must be a list of strings")
        
        # One-vs-rest AUROC for each outcome column
        self.col_outcome = col_outcome
        self.metric_names = ['auroc_' + outcome for outcome in col_outcome]
    
    def compute_metrics(self, y_pred_prob, y_obs):
        """Compute all discrimination/calibration metrics from predicted probabilities vs observed outcomes"""
        metrics = dict()
        
        # AUROC/C-statistic for target vs rest
        for outcome in self.col_outcome:
            metrics['auroc_' + outcome] = auroc_one_vs_rest(y_obs, y_pred_prob, [outcome])
        
        return metrics
