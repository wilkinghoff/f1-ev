# F1-EV Score: Measuring the Likelihood of Estimating a Good Decision Threshold for Semi-Supervised Anomaly Detection

Implementation of the threshold-independent performance measure F1-EV and its bounded version for semi-supervised anomaly detection. The script is designed to be evaluated with output files of the anomaly detection tasks of the [DCASE Challenge](https://dcase.community/challenge2023/task-first-shot-unsupervised-anomalous-sound-detection-for-machine-condition-monitoring).

## Instructions

Just run the script and pass the folders containing the predictions and the folder containing the labels as arguments:
python f1_ev.py -pred_files_path ./path-to-pred-files/ -ref_files_path ./path-to-ref-files/

## Reference

When reusing (parts of) the code, a reference to the following paper would be appreciated:

@unpublished{wilkinghoff2023design,
  author = {Wilkinghoff, Kevin},
  title  = {F1-EV Score: Measuring the Likelihood of Estimating a Good Decision Threshold for Semi-Supervised Anomaly Detection},
  note = {Preprint},
  year   = {2023}
}
