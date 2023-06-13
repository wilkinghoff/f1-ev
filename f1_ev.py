import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import argparse
import pandas as pd
from tqdm import tqdm
import os
from scipy.stats import hmean


def f1_ev(y_true, pred, bounded=False):
    # define thresholds
    thresholds = np.unique(pred)
    if bounded:
        normal_scores = pred[y_true==0]
        # use Bathia-Davis inequality with sample mean as estimate
        est_var_bound = (np.max(normal_scores)-np.mean(normal_scores))*(np.mean(normal_scores)-np.min(normal_scores))
        # define bound for threshold estimated with standard deviation
        alpha = 1.28
        est_upper_bound = np.mean(normal_scores)+alpha*np.sqrt(est_var_bound)
        est_lower_bound = np.mean(normal_scores)
        thresholds = np.concatenate([np.expand_dims(est_lower_bound, axis=0), thresholds[(thresholds<est_upper_bound)*(thresholds>est_lower_bound)], np.expand_dims(est_upper_bound, axis=0)], axis=0)
    # compute f1-scores
    f1_scores = np.zeros(thresholds.shape)
    for k, threshold in enumerate(thresholds):
        f1_scores[k] = metrics.f1_score(y_true, pred > threshold)
    # integrate using Riemann sum to get expected value of F1 score
    ds = np.diff(thresholds/ (thresholds[-1] - thresholds[0]))
    df1 = f1_scores[:-1]
    return np.sum(ds*df1), np.max(f1_scores)


def compute_performance(pred_files_path, ref_files_path):
    # load predictions
    pred_files = os.listdir(pred_files_path)
    pred_files_cleaned = []
    dec_files = []
    # remove decision results
    for k in np.arange(len(pred_files)):
        if pred_files[k].startswith('anomaly_score'):
            pred_files_cleaned.append(pred_files[k])
        else:
            dec_files.append(pred_files[k])
    pred_files = pred_files_cleaned
    # load ground truth
    ref_files = os.listdir(ref_files_path)
    auc = np.zeros(len(pred_files))
    pauc = np.zeros(len(pred_files))
    f1ev = np.zeros(len(pred_files))
    maxf1 = np.zeros(len(pred_files))
    f1ev_bounded = np.zeros(len(pred_files))
    estf1 = np.zeros(len(pred_files))
    subf1 = np.zeros(len(pred_files))
    for k in tqdm(np.arange(len(pred_files))):
        pred = np.array(pd.read_csv(pred_files_path+pred_files[k]))[:,1].astype(np.float32)
        dec = np.array(pd.read_csv(pred_files_path+dec_files[k]))[:,1].astype(np.float32)
        y_true = np.array(pd.read_csv(ref_files_path+ref_files[k]))[:,1].astype(np.float32)
        # compute AUC
        auc[k] = metrics.roc_auc_score(y_true, pred)
        # compute pAUC
        pauc[k] = metrics.roc_auc_score(y_true, pred, max_fpr=0.1)
        # computet F1-EV
        f1ev[k], maxf1[k] = f1_ev(y_true, pred)
        # compute bounded F1-EV
        f1ev_bounded[k], _ = f1_ev(y_true, pred, bounded=True)
        # estimate threshold and compute F1-score
        normal_scores = pred[y_true==0]
        alpha = 1.28
        threshold = np.mean(normal_scores) + alpha*np.std(normal_scores)
        estf1[k] = metrics.f1_score(y_true, pred>threshold)
        # submitted F1-score
        subf1[k] = metrics.f1_score(y_true, dec)
    print('threshold-independent metrics:')
    print('harmonic mean of AUCs: ' + str(hmean(auc)))
    print('harmonic mean of pAUCs: ' + str(hmean(pauc)))
    print('harmonic mean of F1-EVs: ' + str(hmean(f1ev)))
    print('harmonic mean of bounded F1-EVs: ' + str(hmean(f1ev_bounded)))
    print('threshold-dependent metrics:')
    print('submitted F1-score: ' + str(hmean(subf1)))
    print('estimated F1-score: ' + str(hmean(estf1)))
    print('optimal F1-score: ' + str(hmean(maxf1)))
    return


if __name__ == "__main__":
    # example: python f1_ev.py -pred_files_path ./dcase-2022/team_fkie/ -ref_files_path ./dcase-2022/ground_truth_data/
    parser = argparse.ArgumentParser()
    parser.add_argument('-pred_files_path', type=str, help='path to the folder containing the prediction files')
    parser.add_argument('-ref_files_path', type=str, help='path to the folder containing the ground truth files')
    args = parser.parse_args()
    compute_performance(args.pred_files_path, args.ref_files_path)