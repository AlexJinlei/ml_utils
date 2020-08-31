import datetime
import pytz

# Print time stamp with user defined message.
def time_msg(msg=None):
    t = datetime.datetime.now(pytz.timezone('America/New_York'))
    if msg is not None:
        print(t.strftime('%Y-%m-%d %H:%M:%S') + ' - ' + str(msg))
    else:
        print(t.strftime('%Y-%m-%d %H:%M:%S'))
    return t

import numpy as np
from sklearn.utils.extmath import stable_cumsum

#========================================================================================

def confusion_matrix_with_thresholds(y_true, y_score):

    # Sort scores and corresponding truth values in descending order.
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    # Keep the last index of y_true.
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # Accumulate the true positives with decreasing threshold.
    # Decreasing threashold leads to increasing true positives.
    tps = stable_cumsum(y_true)[threshold_idxs]
    #fps = stable_cumsum(1 - y_true)[threshold_idxs] # equivalent.
    fps = 1 + threshold_idxs - tps

    # The total number of negative samples is equal to fps[-1], 
    # thus true negatives are given by fps[-1] - fps.
    tns = fps[-1] - fps
    # The total number of positive samples is equal to tps[-1],
    # thus false negatives are given by tps[-1] - tps.
    fns = tps[-1] - tps
    
    # Trim. After tps reaches the maximum, recall won't change any more even threshold decreases.
    slice_ = slice(None, tps.searchsorted(tps[-1])+1, 1)
    TP = tps[slice_]
    FP = fps[slice_]
    TN = tns[slice_]
    FN = fns[slice_]
    
    precision = TP / (TP + FP)
    precision[np.isnan(precision)] = 0
    recall = TP / TP[-1]
    
    TSS = recall - FP / (TP + TN)
    F1 = 2 * precision * recall / (precision + recall)
    # PFA = FP / (TP + FP) = 1 - precision # Probability of False Alarm.
    
    scores_dict = {}
    scores_dict['TP'] = TP
    scores_dict['FP'] = FP
    scores_dict['TN'] = TN
    scores_dict['FN'] = FN
    scores_dict['recall'] = recall
    scores_dict['precision'] = precision
    scores_dict['PFA'] = 1 - precision
    scores_dict['TSS'] = TSS
    scores_dict['F1'] = F1
    scores_dict['thresholds'] = y_score[threshold_idxs][slice_]
    
    return scores_dict
