# -*- coding: utf-8 -*-
"""
PACKAGE TO AID THE COMPUTATION OF THE APPROACHES FROM THE MAIN FILE

@author: Francisco Valente (paulo.francisco.valente@gmail.com)
2021
"""

## Import packages

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, LassoCV, LogisticRegressionCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier, GradientBoostingClassifier, ExtraTreesClassifier
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics, feature_selection
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.feature_selection import RFE, mutual_info_classif
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
from scipy import stats
from sklearn import tree, calibration
import matplotlib.pyplot as plt
import re
import math
from sklearn.impute import KNNImputer
import random
from scipy.signal import argrelextrema
from sklearn.neural_network import MLPClassifier
import import_dataset
import extract_rules
import use_rules
from iteration_utilities import unique_everseen
from itertools import permutations  
from sklearn.model_selection import RandomizedSearchCV
import developed_predictions
import rules_selection
import generate_rules
import extract_rules
from sklearn.feature_selection import SelectKBest


#------------------------------------------------------------------------------------#

## Function to compute the developed approach (rules + personalization through acceptance prediction)

def compute_ourApproach(X_train, X_test, y_train, y_test, features_names, all_rules, rules_outputs, idx_best20, k):
    
    # selected rules
    idx_selected = idx_best20[:k]
    rules_selected = [all_rules[i] for i in list(idx_selected)]
    # print(rules_selected)
    
    # training and testing rules' outputs
    selected_output = rules_outputs[:,idx_selected] 
    suggested_output = use_rules.apply_rules(rules_selected, X_test)
    
    # get the rules acceptance in the training dataset
    rules_acceptance = selected_output == y_train.reshape(len(y_train), 1) # RULES ACCEPTANCE
    rules_acceptance = rules_acceptance.astype('float')
    
    # to be used if there are "nans" in rules_acceptance (exploratory methods)
    # all_nans = np.argwhere(np.isnan(selected_output))
    # for xx in range(0, len(all_nans)):
    #     my_x = all_nans[xx][0]
    #     my_y = all_nans[xx][1]
    #     rules_acceptance[my_x][my_y] = float("NAN")
    
    # Report analysis of the selected rules
    # rules_corr = np.corrcoef(selected_output) # correlation between the used rules' outputs
    
    # Features used in the selected rules
    used_features = rules_selection.obtain_features(rules_selected) 
    # (use only the features that are also used by the rules)
    new_X_train = pd.DataFrame(X_train , columns=used_features)
    new_X_test = pd.DataFrame(X_test , columns=used_features)
    
    # Get the (predicted) acceptances for training and testing dataset
    predicted_acceptance, predicted_acceptance_train = developed_predictions.get_acceptances(rules_acceptance, new_X_train, new_X_test)
    # predicted_acceptance, predicted_acceptance_train = developed_predictions.get_acceptances_balancing(rules_acceptance, new_X_train, new_X_test)
    
    # Get the training and testing predictions
    # (method used in ICML-IMLH paper)
    predictions = developed_predictions.get_prediction(suggested_output, predicted_acceptance)
    predictions_train = developed_predictions.get_prediction(selected_output, predicted_acceptance_train)
    # (method used in EMBC paper)
    # predictions = developed_predictions.get_prediction_embc(suggested_output, predicted_acceptance)
    # predictions_train = developed_predictions.get_prediction_embc(selected_output, predicted_acceptance_train)
    
    # Calibrate the predictions
    predictions_train, predictions = developed_predictions.calibration_step(y_train, predictions_train, predictions)
    
    return predictions, rules_selected, suggested_output, predicted_acceptance

#------------------------------------------------------------------------------------#

## Function to compute the predictions by simple averaging of suggested outputs of the k best rules

def compute_meanRules(all_rules, X_test, idx_best20, k):
    
    # selected rules
    idx_selected = idx_best20[:k]
    rules_selected = [all_rules[i] for i in list(idx_selected)]
    suggested_output = use_rules.apply_rules(rules_selected, X_test)
    
    # obtained predictions (mean)
    predictions = np.nanmean(suggested_output, axis=1)
        
    return predictions

#------------------------------------------------------------------------------------#

## Function to compute the predictions by weighted averaging of suggested outputs of the k best rules

def compute_weightMeanRules(all_rules, X_test, idx_best20, k, wgt):
    
    # selected rules
    idx_selected = idx_best20[:k]
    
    # obtained predictions (weighted mean)
    rules_selected = [all_rules[i] for i in list(idx_selected)]
    suggested_output = use_rules.apply_rules(rules_selected, X_test)
    predictions = np.average(suggested_output, axis=1, weights=wgt)
    
    return predictions

#------------------------------------------------------------------------------------#

## Function to compute an optimized Random Forest WITH NO constraints
# i.e., get the parameters that lead to the best model, based on a random CV search

def compute_rfOptimized(X_train, X_test, y_train, features_names):
    
    rf = RandomForestClassifier()
    
    # Find best parameters to optimize random forest
    # Number of trees in random forest
    n_estimators = [10,20,50,100,150,200,250,300]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [1,2,3,4,5,6,7,10,12,15,20]
    # max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    # min_samples_leaf = [1, 2, 3, 5, 10]
    # Method of selecting samples for training each tree
    bootstrap = [1, 0]
    
    random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split}
    
    # Get the best parameters
    rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 10, cv = 3, verbose=2, random_state=42, n_jobs = -1)

    model = rf_random.fit(X_train, y_train)
    selected_param = model.best_params_
    
    best_rf = RandomForestClassifier(class_weight = 'balanced_subsample', 
                                     n_estimators = selected_param['n_estimators'],
                                     max_features = selected_param['max_features'],
                                     max_depth = selected_param['max_depth'],
                                     min_samples_split = selected_param['min_samples_split'])
    # .best_estimator_
    best_model = best_rf.fit(X_train, y_train)
    pred = best_model.predict_proba(X_test.values)
    predictions = pred[:,1]
    # predictions2 = best_model.predict(X_test.values)
    
    # Extract total number of rules of the selected model
    rules_rf = generate_rules.rules_extraction([best_model],features_names, 'naomeu')
    # rules_rf = list(unique_everseen(rules_rf)) # remover duplicados
    
    # Save the depth, nr of trees and total number of rules of the selected model
    depth_rf = rf_random.best_params_['max_depth']
    nrTrees_rf = rf_random.best_params_['n_estimators']
    nrRules_rf = len(rules_rf)
        
    # Save the trees of the selected model
    tree_list = best_model.estimators_
    
    return predictions, rules_rf, depth_rf , nrTrees_rf, nrRules_rf, best_rf, tree_list

#------------------------------------------------------------------------------------#

## Function to compute an optimized Random Forest WITH constraints (limited number of trees and depth)
# i.e., get the parameters that lead to the best model, based on a random CV search

def compute_rfsOptimized(X_train, X_test, y_train, features_names):
    
    rf = RandomForestClassifier()
    
    # Find best parameters to optimize random forest
    # Number of trees in random forest
    n_estimators = [2,3,4]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [1,2]
    # max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    # min_samples_leaf = [1, 2, 3, 5, 10]
    # Method of selecting samples for training each tree
    bootstrap = [1, 0]
    
    random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split}
    
    # Get the best parameters
    rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 5, verbose=2, random_state=42, n_jobs = -1)

    model = rf_random.fit(X_train, y_train)
    selected_param = model.best_params_
    
    best_rf = RandomForestClassifier(class_weight = 'balanced_subsample', 
                                     n_estimators = selected_param['n_estimators'],
                                     max_features = selected_param['max_features'],
                                     max_depth = selected_param['max_depth'],
                                     min_samples_split = selected_param['min_samples_split'])
    # .best_estimator_
    best_model = best_rf.fit(X_train, y_train)
    pred = best_model.predict_proba(X_test.values)
    predictions = pred[:,1]
    # predictions2 = best_model.predict(X_test.values)
    
    # Extract total number of rules of the selected model
    rules_rf = generate_rules.rules_extraction([best_model],features_names, 'naomeu')
    # rules_rf = list(unique_everseen(rules_rf)) # remover duplicados
    
    # Save the depth, nr of trees and total number of rules of the selected model
    depth_rf = rf_random.best_params_['max_depth']
    nrTrees_rf = rf_random.best_params_['n_estimators']
    nrRules_rf = len(rules_rf)
    
    # Save the trees of the selected model
    tree_list = best_model.estimators_
    
    return predictions, rules_rf, depth_rf , nrTrees_rf, nrRules_rf, best_rf, tree_list

#------------------------------------------------------------------------------------#

## Function to compute an optimized Decision Tree WITH NO constraints
# i.e., get the parameters that lead to the best model, based on a random CV search

def compute_dtOptimized(X_train, X_test, y_train, features_names):
    
    my_tree = tree.DecisionTreeClassifier()
    
    # Find best parameters to optimize decision tree
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [1,2,3,4,5,6,7,10,12,15,20]
    # max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    # min_samples_leaf = [1, 2, 3, 5, 10]
    # Method of selecting samples for training each tree
    bootstrap = [1, 0]
    
    random_grid = {'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split}
    
    # Get the best parameters
    tree_random = RandomizedSearchCV(estimator = my_tree, param_distributions = random_grid, n_iter = 100, cv = 5, verbose=2, random_state=42, n_jobs = -1)

    model = tree_random.fit(X_train, y_train)
    selected_param = model.best_params_
    
    best_tree = DecisionTreeClassifier(max_features = selected_param['max_features'],
                                     max_depth = selected_param['max_depth'],
                                     min_samples_split = selected_param['min_samples_split'])
    # .best_estimator_
    best_model = best_tree.fit(X_train, y_train)
    pred = best_model.predict_proba(X_test.values)
    predictions = pred[:,1]
    # predictions2 = best_model.predict(X_test.values)
    
    # Extract total number of rules of the selected model
    rules_tree = extract_rules.extract_rules_from_tree(best_model.tree_,features_names, 'naomeu')
    # rules_rf = list(unique_everseen(rules_rf)) # remover duplicados
    
    # Save the depth and number of rules of the selected model
    depth_tree = tree_random.best_params_['max_depth']
    nrRules_tree = len(rules_tree)
    
    return predictions, rules_tree, depth_tree, nrRules_tree, best_tree

#------------------------------------------------------------------------------------#

## Function to compute an optimized Support Vector Machine WITH NO constraints
# i.e., get the parameters that lead to the best model, based on a random CV search

def compute_svmOptimized(X_train, X_test, y_train, features_names):
    
    svm = SVC()
    
    kernels = ['rbf', 'poly', 'sigmoid', 'linear']
    
    Cs = [0.1,1, 10, 100]
    
    gammas = [1,0.1,0.01,0.001]
    
    random_grid = {'kernel': kernels,
               'C': Cs,
               'gamma': gammas}
    
    # Get the best parameters
    svm_random = RandomizedSearchCV(estimator = svm, param_distributions = random_grid, cv = 3, verbose=2, random_state=42, n_jobs = -1)

    model = svm_random.fit(X_train, y_train)
    
    # .best_estimator_
    pred = model.predict_proba(X_test.values)
    predictions = pred[:,1]
    # predictions2 = best_model.predict(X_test.values)

    return predictions

#------------------------------------------------------------------------------------#

## Function to compute an optimized Logistic Regression WITH NO constraints
# i.e., get the parameters that lead to the best model, based on a random CV search

def compute_lrOptimized(X_train, X_test, y_train, features_names):
    
    lr = LogisticRegression()

    random_grid = {'penalty' : ['l1', 'l2'],
                  'C' : np.logspace(-4, 4, 5),
                  'solver' : ['liblinear']},
    
    # Get the best parameters
    lr_random = RandomizedSearchCV(estimator = lr, param_distributions = random_grid, cv = 3, verbose=2, random_state=42, n_jobs = -1)

    # model = lr_random.fit(X_train, y_train)
    model = lr.fit(X_train, y_train)

    # .best_estimator_
    
    pred = model.predict_proba(X_test.values)
    predictions = pred[:,1]
    # predictions2 = best_model.predict(X_test.values)

    return predictions
   
#------------------------------------------------------------------------------------#

## Function to compute the reliability for every patients

def compute_reliability(acceptances, outputs):
    
    all_reliabs = []
    
    # Compute  the reliability of a single patient
    
    for i in range(0, len(acceptances)):
        
        acceptance = acceptances[i] # vector of outputs' acceptance
        output = outputs[i] # vector of outputs
        
        idx_out0 = np.where(output==0) # idx of negative (out=0) rules
        idx_out1 = np.where(output==1) # idx of positive (out=1) rules
        idx_acep0 = acceptance[idx_out0] # acceptance of negative (out=0) rules
        idx_acep1 = acceptance[idx_out1] # acceptance of positive (out=0) rules
        
        mean0 = np.mean(idx_acep0) # mean acceptance of negative rules
        mean1 = np.mean(idx_acep1) # mean acceptance of positive rules
        
        reliab = abs(mean1-mean0) # obtained reliability for patient i
        all_reliabs.append(reliab)
        
    return all_reliabs

#------------------------------------------------------------------------------------#

## Function to get the misclassification ratios per group of relaibility values

def plot_reliability(predictions, threshold, y_true, reliability):
    
    # Get the misclassifications
    pred = predictions>=threshold # binarizing the predictions (if necessary)
    classifications_are_right = pred == y_true # vector with the correctness of classifications
    is_misclassification = np.invert(classifications_are_right) # vector with the INcorrectness of classifications
    
    # Stratifify the reliability values
    bins = np.linspace(0, 1, 11) # create intervals of 10%
    put_in_bin = np.digitize(reliability, bins) # divide the reliability values by those intervals
    
    
    # Get the misclassification ratios by interval
    
    misc_ratios = []
    
    for i in range(0,len(bins)):
        
        idx = np.where(put_in_bin==i) # idx that belong to interval i
        
        if len(idx[0])==0: # if there are no realibility values in interval i
            
            misc_ratios.append(np.nan)
            
        else:
                
            # Compute misclassification ratio for interval i
            misc = np.where(is_misclassification[idx]==True)
            misc_ratio = len(misc[0])/len(idx[0])
            
            misc_ratios.append(misc_ratio)

    return misc_ratios


#------------------------------------------------------------------------------------#

## Function to get the predictions of every tree in a list of trees

def get_trees_predictions(trees, X_train, X_test):
    
    train_outputs =  np.empty([len(X_train.index), len(trees)])
    test_outputs =  np.empty([len(X_test.index), len(trees)])
    
    count = 0
    
    # Compute the predictions of a single tree for the set of train and test samples
    for tree in trees:
        
        output_train = tree.predict(X_train.values) 
        output_test = tree.predict(X_test.values)
        train_outputs[:,count] = output_train
        test_outputs[:,count] = output_test
        count = count+1
   
    return train_outputs, test_outputs

#------------------------------------------------------------------------------------#

## Function to compute the occurence (ratio) of each class in the prediction vectors
# i.e., compute how many times (percentage) each class is predicted by the rules for each patient

def compute_mode(suggested_output):
    
    def get_probs(matrix):
        
        classes = np.unique(matrix) # possible classes
        all_probs = []
        
        for i in range(0, len(matrix)):
            
            row = matrix[i,:] 
            row_probs = []
            for c in classes:
                prob = (row == c).sum()/len(row) # percentage of each class for patient i
                row_probs.append(prob)
            all_probs.append(np.asarray(row_probs))
            
        return np.asarray(all_probs)
    
    predictions = get_probs(suggested_output)[:,1]
    return predictions