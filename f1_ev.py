import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt


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
    print(thresholds)
    print(f1_scores)
    # integrate using Riemann sum to get expected value of F1 score
    ds = np.diff(thresholds/ (thresholds[-1] - thresholds[0]))
    #df1 = (f1_scores[1:]+f1_scores[:-1])/2.
    df1 = f1_scores[:-1]
    return np.sum(ds*df1), np.max(f1_scores)

pred = np.array([1,3,5,7,11,13,17,19])
y = np.array([0, 0, 1, 0, 0, 1, 1, 1])
#pred = np.array([0., 0.5, 1., 1.1, 1.2])
#y = np.array([0, 0, 1, 1, 1])
#pred = np.random.rand(100)
#y = np.round(pred+np.random.rand(100)*0.1-np.random.rand(100)*0.1)
#pred[y==1] +=0.5
#pred[y==0] += -0.5

fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=1, drop_intermediate=False)  # pos_label needs to be modified!
thresholds[0] = thresholds[1]+1e-16  # correct manually included extra threshold to be only slightly higher, impact should be very minor (or even non-existant?) for final score
#fpr, tpr, thresholds = metrics.precision_recall_curve(y, pred, pos_label=1)  # pos_label needs to be modified!
#end_idx = np.argmax(tpr>=1)
#tpr = tpr[:end_idx+1]
#fpr = fpr[:end_idx+1]
#thresholds = thresholds[:end_idx+1]

print(thresholds)
#print(tpr)
#print(fpr)

f1_scores = np.zeros(thresholds.shape)
for k, threshold in enumerate(thresholds):
    f1_scores[k] = metrics.f1_score(y, pred>=threshold)
print(f1_scores)

thresholds = thresholds/(thresholds[0]-thresholds[-1])
dx=np.diff(fpr)
ds=np.diff(-thresholds)
dy=(tpr[1:]+tpr[:-1])/2
df1 = (f1_scores[1:]+f1_scores[:-1])/2

#print(dx)
#print(dy)
print(ds)
print(df1)
#print(dy*dx*ds)


f1_ev_score, f1_max = f1_ev(y_true=y, pred=pred)
print('F1-EV is ' + str(f1_ev_score))
print('Optimal F1-score is ' + str(f1_max))
print('AUC is ' + str(np.sum(dy*dx)))
print('TA-AUC is ' + str(np.sum(dy*dx*ds)))  # replace with square-root of dx*dy?
#print('TA-AUC is ' + str(np.sum(dy*ds)))  # replace with square-root of dx*dy?
#print('TA-AUC is ' + str(np.sum(dy*np.sqrt(dx*ds))))  # replace with square-root of dx*dy?
#display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=np.sum(dy*dx), estimator_name='dx sy')
#display = metrics.RocCurveDisplay(fpr=thresholds, tpr=tpr, roc_auc=np.sum(dy*ds), estimator_name='dy ds')
#display = metrics.RocCurveDisplay(fpr=fpr, tpr=thresholds, roc_auc=np.sum(dx*ds), estimator_name='dx ds')
#display = metrics.RocCurveDisplay(fpr=thresholds, tpr=f1_scores, roc_auc=np.sum(df1*ds), estimator_name='dx sy')
#display.plot()
#plt.show()