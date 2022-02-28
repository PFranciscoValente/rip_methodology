# -*- coding: utf-8 -*-
"""
PACKAGE FOR COMPUTATION OF PERFORMANCE METRICS 

@author: Francisco Valente (paulo.francisco.valente@gmail.com)
2021
"""

## Import packages

from sklearn import metrics
import math
import numpy as np

#------------------------------------------------------------------------------------#

## Function to obtain performance metrics 

def performance_metrics(predicted, true_label, threshold):
    
    fpr, tpr, thresholds = metrics.roc_curve(true_label, predicted)
    
    # gmeans = np.sqrt(tpr * (1-fpr))
    # ix = np.argmax(gmeans)
    # thr = thresholds[ix]
    # binary_predicted = predicted>=thr
    
    binary_predicted = predicted>=threshold
    
    tn, fp, fn, tp = metrics.confusion_matrix(true_label, binary_predicted).ravel()
    
    spec = tn / (tn+fp)
    sens = tp / (tp+fn)
    gm = math.sqrt(sens*spec)
    auc = metrics.auc(fpr, tpr)
    
    
    return auc, gm, spec, sens