# -*- coding: utf-8 -*-
"""
PACKAGE TO CREATE RULES BASED ON CENTROIDS OF FEATURES AND THRESHOLDS, AND APPLY THEM TO NEW SAMPLES 

@author: Francisco Valente (paulo.francisco.valente@gmail.com)
2021
"""

## Import packages

import numpy as np
from scipy.spatial.distance import cdist

#------------------------------------------------------------------------------------#

## Function to create rules based on centroids and generate the training outputs

def get_outputs(X_train, y_train):
    
    rules_outputs = []
    cent_negative = []
    cent_positive = []
    
    # Create a rule per feature, based on positive and negative centroids
    
    for i in range(0,X_train.shape[1]):
        
        X = X_train[:,i]
        idx_negative = np.where(y_train == 0)[0] # negative samples
        idx_positive = np.where(y_train == 1)[0] # positive samples
        feat_negative = X[idx_negative]
        feat_positive = X[idx_positive]
        
        # Obtain the positive and negative centroids (means of each group)
        centroid_negative = np.mean(feat_negative)
        centroid_positive = np.mean(feat_positive)
        
        cent_negative.append(centroid_negative)
        cent_positive.append(centroid_positive)
        
        # Get the normalized distance for each training sample
        
        dist_norm = []
        
        for t in range(0,len(X)):
            
            # distances of point t to each centroid
            d_neg = np.linalg.norm(X[t]-centroid_negative)
            d_pos = np.linalg.norm(X[t]-centroid_positive)
            # noramlized distance of point t
            d_normalized = 1 - (d_pos/(d_pos+d_neg))
            dist_norm.append(d_normalized)

        # Obtain the traning outputs for rule i
        feat_outputs = np.array(dist_norm) >= 0.5
        feat_outputs = feat_outputs*1
        
        rules_outputs.append(feat_outputs)
        
    rules_outputs = np.stack(rules_outputs, axis=1)

    return rules_outputs, cent_negative, cent_positive

#------------------------------------------------------------------------------------#

## Function to apply the rules based on centroids to new samples

def predict_outputs(X_test, cent_negative, cent_positive):
    
    rules_outputs = []
    
    for i in range(0,X_test.shape[1]):
        
        X = X_test[:,i]

        # Get the positive and negative centroids of rule i
        centroid_negative = cent_negative[i]
        centroid_positive = cent_positive[i]
            
        # Get the normalized distance for each training sample
        
        dist_norm = []
        
        for t in range(0,len(X)):
            
            # distances of point t to each centroid 
            d_neg = np.linalg.norm(X[t]-centroid_negative)
            d_pos = np.linalg.norm(X[t]-centroid_positive)
            # noramlized distance of point t
            d_normalized = 1 - (d_pos/(d_pos+d_neg))
            dist_norm.append(d_normalized)

         # Obtain the testing outputs for rule i
        feat_outputs = np.array(dist_norm) >= 0.5
        feat_outputs = feat_outputs*1
        
        rules_outputs.append(feat_outputs)
        
    rules_outputs = np.stack(rules_outputs, axis=1)

    return rules_outputs