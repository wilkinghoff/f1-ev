import numpy as np
from sklearn import metrics
import argparse
import pandas as pd
from tqdm import tqdm
import os
from scipy.stats import hmean, gamma, pearsonr
import matplotlib.pyplot as plt


def store_results(x1, x2, filepath):
    res = pd.DataFrame()
    res['x1'], res['x2'] = [x1, x2]
    res.to_csv(filepath, encoding='utf-8', sep=' ', index=False, header=True, line_terminator='\n')
    return


def estimate_decision_threshold(pred, dec):
    thresholds = pred[:-1] + np.diff(pred)/2
    similarities = np.zeros(thresholds.shape[0])
    for k, threshold in enumerate(thresholds):
        similarities[k] = np.sum((pred>threshold)==dec)
    return thresholds[np.argmax(similarities)]


def f1_ev(y_true, pred, bounded=False, orig_threshold=None, alpha=0.2515):
    # define thresholds
    thresholds = np.unique(pred)
    if bounded:
        normal_scores = pred[y_true==0]
        std_normal = np.std(normal_scores)
        # define bounds for threshold estimated with standard deviation
        est_upper_bound = orig_threshold+alpha*std_normal
        est_lower_bound = np.mean(normal_scores)-alpha*std_normal
        thresholds = np.concatenate([np.expand_dims(est_lower_bound, axis=0), thresholds[(thresholds<est_upper_bound)*(thresholds>est_lower_bound)], np.expand_dims(est_upper_bound, axis=0)], axis=0)
    # compute f1-scores
    f1_scores = np.zeros(thresholds.shape)
    for k, threshold in enumerate(thresholds):
        f1_scores[k] = metrics.f1_score(y_true, pred > threshold)
    # integrate using Riemann sum to get expected value of F1 score
    ds = np.diff(thresholds/ (thresholds[-1] - thresholds[0]))
    df1 = f1_scores[:-1]
    return np.sum(ds*df1), np.max(f1_scores), thresholds[np.argmax(f1_scores)]


def test_alpha_values(pred_files_path, ref_files_path):
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

    alphas = np.arange(0.025, 1.025, 0.025)
    f1ev_bounded = np.zeros((len(pred_files), alphas.shape[0]))
    estf1 = np.zeros(len(pred_files))
    subf1 = np.zeros(len(pred_files))
    for k in np.arange(len(pred_files)):
        pred = np.array(pd.read_csv(pred_files_path+pred_files[k]))[:,1].astype(np.float32)
        dec = np.array(pd.read_csv(pred_files_path+dec_files[k]))[:,1].astype(np.float32)
        y_true = np.array(pd.read_csv(ref_files_path+ref_files[k]))[:,1].astype(np.float32)
        # compute optimal threshold
        _, _, threshold = f1_ev(y_true, pred)
        if np.array_equal(dec, dec.astype(bool)):
            for j, alpha in enumerate(alphas):
                f1ev_bounded[k,j], _, _ = f1_ev(y_true, pred, bounded=True, orig_threshold=threshold, alpha=alpha)
    return alphas, f1ev_bounded


def compute_performance(pred_files_path, ref_files_path, verbose=False):
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
    for k in np.arange(len(pred_files)):
        pred = np.array(pd.read_csv(pred_files_path+pred_files[k]))[:,1].astype(np.float32)
        dec = np.array(pd.read_csv(pred_files_path+dec_files[k]))[:,1].astype(np.float32)
        y_true = np.array(pd.read_csv(ref_files_path+ref_files[k]))[:,1].astype(np.float32)
        # compute AUC
        auc[k] = metrics.roc_auc_score(y_true, pred)
        # compute pAUC
        pauc[k] = metrics.roc_auc_score(y_true, pred, max_fpr=0.1)
        # computet F1-EV
        f1ev[k], maxf1[k], optimal_threshold = f1_ev(y_true, pred)
        if np.array_equal(dec, dec.astype(bool)):
            # estimate threshold and compute F1-score
            #threshold = estimate_decision_threshold(pred, dec)  # this would be treshold-dependent
            threshold = optimal_threshold  # to have a threshold-independent metric!
            # compute bounded F1-EV
            f1ev_bounded[k], _, _ = f1_ev(y_true, pred, bounded=True, orig_threshold=threshold)
            # submitted F1-score
            subf1[k] = metrics.f1_score(y_true, dec)
    if verbose:
        print('##########################################')
        print(pred_files_path)
        print('##########################################')
        print('threshold-independent metrics:')
        print('harmonic mean of AUCs: ' + str(hmean(auc)))
        print('harmonic mean of pAUCs: ' + str(hmean(pauc)))
        print('harmonic mean of F1-EVs: ' + str(hmean(f1ev)))
        print('harmonic mean of bounded F1-EVs: ' + str(hmean(f1ev_bounded)))
        print('threshold-dependent metrics:')
        print('harmonic mean of submitted F1-scores: ' + str(hmean(subf1)))
        print('harmonic mean of optimal F1-score: ' + str(hmean(maxf1)))
    return np.array([auc, pauc, f1ev, f1ev_bounded, subf1, maxf1])


if __name__ == "__main__":
    # example: python f1_ev.py -pred_files_path ./dcase-2023/teams/ -ref_files_path ./dcase-2023/ground_truth_data/
    parser = argparse.ArgumentParser()
    parser.add_argument('-pred_files_path', type=str, help='path to the folder containing the prediction files')
    parser.add_argument('-ref_files_path', type=str, help='path to the folder containing the ground truth files')
    args = parser.parse_args()
    """
    # compute bounded F1-EV for different values of alpha
    alpha_results = []
    for team_dir in tqdm(os.listdir(args.pred_files_path)):
        for submission_dir in os.listdir(args.pred_files_path + '/' + team_dir + '/'):
            alphas, alpha_result = test_alpha_values(args.pred_files_path + '/' + team_dir + '/' + submission_dir + '/', args.ref_files_path)
            alpha_results.append(alpha_result)
    alpha_results = np.mean(np.concatenate(alpha_results), axis=0)
    store_results(alphas, alpha_results, 'alpha_vs_f1-ev-bounded.txt')
    """
    # compute all metrics for all files
    results = []
    for team_dir in tqdm(os.listdir(args.pred_files_path)):
        for submission_dir in os.listdir(args.pred_files_path + '/' + team_dir + '/'):
            results.append(compute_performance(args.pred_files_path + '/' + team_dir + '/' + submission_dir + '/', args.ref_files_path))
    results = np.array(results)

    # remove submissions without threshold/F1-score
    auc = np.ravel(results[:,0])
    pauc = np.ravel(results[:,1])
    f1ev = np.ravel(results[:,2])
    f1ev_bounded = np.ravel(results[:,3])
    f1_sub = np.ravel(results[:,4])
    f1_opt = np.ravel(results[:,5])

    valid = f1_sub>0
    auc = auc[valid]
    pauc = pauc[valid]
    f1ev = f1ev[valid]
    f1ev_bounded = f1ev_bounded[valid]
    f1_sub = f1_sub[valid]
    f1_opt = f1_opt[valid]

    # output results
    print('Pearson correlation coefficients:')
    print('AUC-ROC vs. F1-EV: ' + str(np.round(pearsonr(auc, f1ev)[0], 3)))
    print('AUC-ROC vs. F1-EV_bounded: ' + str(np.round(pearsonr(auc, f1ev_bounded)[0], 3)))
    print('F1-EV vs. F1-EV_bounded: ' + str(np.round(pearsonr(f1ev, f1ev_bounded)[0], 3)))
    print('AUC-ROC vs. F1 score (as submitted): ' + str(np.round(pearsonr(auc, f1_sub)[0], 3)))
    print('F1-EV vs. F1 score (as submitted): ' + str(np.round(pearsonr(f1ev, f1_sub)[0], 3)))
    print('F1-EV_bounded vs. F1 score (as submitted): ' + str(np.round(pearsonr(f1ev_bounded, f1_sub)[0], 3)))
    print('AUC-ROC vs. optimal F1 score: ' + str(np.round(pearsonr(auc, f1_opt)[0], 3)))
    print('F1-EV vs. optimal F1 score: ' + str(np.round(pearsonr(f1ev, f1_opt)[0], 3)))
    print('F1-EV_bounded vs. optimal F1 score: ' + str(np.round(pearsonr(f1ev_bounded, f1_opt)[0], 3)))

    """
    no correlation: 0 to 0.3
    weak correlation: 0.3 to 0.5
    moderate correlation: 0.5 to 0.7
    high correlation: 0.7 to 0.9
    very high correlation: 0.9 to 1
    """

    # store results
    store_results(auc, f1ev, 'auc-roc_vs_f1-ev.txt')
    store_results(auc, f1ev_bounded, 'auc-roc_vs_f1-ev-bounded.txt')
    store_results(f1ev, f1ev_bounded, 'f1-ev_vs_f1-ev-bounded.txt')
    store_results(auc, f1_sub, 'auc-roc_vs_f1-sub.txt')
    store_results(f1ev, f1_sub, 'f1-ev_vs_f1-sub.txt')
    store_results(f1ev_bounded, f1_sub, 'f1-ev-bounded_vs_f1-sub.txt')
    store_results(auc, f1_opt, 'auc-roc_vs_f1-opt.txt')
    store_results(f1ev, f1_opt, 'f1-ev_vs_f1-opt.txt')
    store_results(f1ev_bounded, f1_opt, 'f1-ev-bounded_vs_f1-opt.txt')