# -*- coding: utf-8 -*-
"""
MAIN FILE TO RUN THE METHDOLOGIES
USED FOR ICML-IMLH AND EMBC PAPERS

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
import import_dataset
import extract_rules
import use_rules
from iteration_utilities import unique_everseen
from itertools import permutations  
from sklearn.model_selection import RandomizedSearchCV
import rules_selection
import generate_rules
import extract_rules
import auxiliary_main
from sklearn.feature_selection import SelectKBest

import warnings
warnings.filterwarnings("ignore")


#------------------------------------------------------------------------------------#

# Auxiliary function to calibrate the predictions

def calibration_step(y_train, predictions_train, predictions_test):
        
    clf = LogisticRegression(solver='liblinear')
    calibrator = clf.fit(predictions_train.reshape(-1,1), y_train)
    calibrated_predictions = calibrator.predict_proba(predictions_test.reshape(-1,1))[:,1]
    
    return calibrated_predictions

#------------------------------------------------------------------------------------#

# Auxiliary function to compute the predcted confidence interval

def confidence_interval(data):

    alpha = 0.05                       # significance level = 5%
    df = len(data) - 1                  # degress of freedom = 20
    t = stats.t.ppf(1 - alpha/2, df)   # t-critical value for 95% CI = 2.093
    s = np.nanstd(data, ddof=1)            # sample standard deviation = 2.502
    n = len(data)
    
    lower = np.nanmean(data) - (t * s / np.sqrt(n))
    upper = np.nanmean(data) + (t * s / np.sqrt(n))
    interval = (t * s / np.sqrt(n))
    
    return interval

#------------------------------------------------------------------------------------#

# Dataset loading

# in order to see the available datasets and their names, see the import_dataset file
my_data, my_label = import_dataset.dataset_info('heart_disease') 
binary_target = my_label
features_names = list(my_data.columns)
features = my_data.values

#------------------------------------------------------------------------------------#

# Variables to save results

# area under the roc curve vectors
all_auc_rf = []
all_auc_rfs_probs = []
all_auc_rfs = []
all_auc_dt = []
all_auc_m3 = []
all_auc_m5 = []
all_auc_m10 = []
all_auc_m15 = []
all_auc_m20 = []
all_auc_m3m = []
all_auc_m5m = []
all_auc_m10m = []
all_auc_m15m = []
all_auc_m20m = []
all_auc_mm = []
all_auc_m10wm = []

# balanced accuracy vectors
all_ba_rf = []
all_ba_rfs_probs = []
all_ba_rfs = []
all_ba_dt = []
all_ba_m3 = []
all_ba_m5 = []
all_ba_m10 = []
all_ba_m15 = []
all_ba_m20 = []
all_ba_m3m = []
all_ba_m5m = []
all_ba_m10m = []
all_ba_m15m = []
all_ba_m20m = []
all_ba_mm = []
all_ba_m10wm = []

# other metrics

rf_nrRules = []
rf_treeDepth = []
rf_nrTrees = []
rfs_nrRules = []
dt_nrRules = []
rf_nrFeatures = []
rfs_nrFeatures = []
dt_nrFeatures = []
m3_nrFeatures = []
m5_nrFeatures = []
m10_nrFeatures = []
m15_nrFeatures = []
m20_nrFeatures = []

all_preds = []
all_reliabs = []
all_reliability20 = []
all_reliability = []
all_pred = []
all_ytest= []


#------------------------------------------------------------------------------------#

# Run Monte-Carlos cross validation

kf = StratifiedKFold(n_splits=5, shuffle =True)
count= 1

mccv_runs = 5 # define number of Monte Carlo cross validation runs
for i in range(0,mccv_runs):
    
    print ('RUN NUMBER: ', count)
    count= count+1

    for train_index, test_index in kf.split(features, binary_target):
        
        # Division into training and testing datasets
        
        X_train, X_test = features[train_index], features[test_index]
        y_train, y_test = binary_target[train_index], binary_target[test_index]
        my_X_train = pd.DataFrame(X_train, columns = features_names)
        my_X_test = pd.DataFrame(X_test, columns = features_names)
        
        # binarization thershold is defined as the rate of positive events
        # i.e., if 5% of patients die, than if the predicted probability of death is higher than 0.5, then the patient is predicted to die
        num_events = (y_train == 1).sum()
        threshold = num_events/len(y_train) 
        
        
        #------------------------------------------------------------------------------------#
        
        # RUN MACHINE LEARNING COMPARISON METHODS (RANDOM FOREST, DECISION TREE)
        
        #### RANDOM FOREST OPTIMIZED #####
        
        # print('performing rf optimized...')
        
        # predictions_rf, rules_rf, depth_rf, nrTrees_rf, nrRules_rf, best_rf, trees_rf = auxiliary_main.compute_rfOptimized(my_X_train, my_X_test, y_train, features_names)
        # auc_rf = roc_auc_score(y_test, predictions_rf)
        # all_auc_rf.append(auc_rf)
        # ba_rf = balanced_accuracy_score(y_test, predictions_rf>=threshold)
        # all_ba_rf.append(ba_rf)
        
        # # total number of rules used by the ensemble
        # rf_nrRules.append(nrRules_rf) 
        # # depth of the trees used by the ensemble
        # rf_treeDepth.append(depth_rf) 
        # # total number of trees used by the ensemble
        # rf_nrTrees.append(nrTrees_rf) 
        # # total number of features used by the ensemble
        # rf_nrFeatures.append(len(RulesSelection.obtain_features(rules_rf)))
        
        
        #### RANDOM FOREST SIMPLE #####
        
        # print('performing rf 5 trees with max_depth=3...')
        
        # predictions_rfs, rules_rfs, depth_rfs, nrTrees_rfs, nrRules_rfs, best_rfs, trees_rfs = auxiliary_main.compute_rfsOptimized(my_X_train, my_X_test, y_train, features_names)
        # auc_rfs = roc_auc_score(y_test, predictions_rfs)
        # ba_rfs = balanced_accuracy_score(y_test, predictions_rfs>=threshold)
        # all_auc_rfs.append(auc_rfs)
        # all_ba_rfs.append(all_ba_rfs)
        
        # # total number of rules used by the ensemble
        # rfs_nrRules.append(len(rules_rfs))
        # # total number of features used by the ensemble
        # rfs_nrFeatures.append(len(RulesSelection.obtain_features(rules_rfs)))
        
        
        #### DECISION TREE #####
        
        # print('performing decision tree..')
        
        # predictions_dt, rules_dt, depth_dt, nrRules_dt, best_dt = auxiliary_main.compute_dtOptimized(my_X_train, my_X_test, y_train, features_names)
        # auc_dt = roc_auc_score(y_test, predictions_dt)
        # all_auc_dt.append(auc_dt)
        # ba_dt = balanced_accuracy_score(y_test, predictions_dt>=threshold)
        # all_ba_dt.append(ba_dt)
        
        # # total number of rules used by the tree
        # dt_nrRules.append(len(rules_dt))
        # # total number of features used by the tree
        # dt_nrFeatures.append(len(RulesSelection.obtain_features(rules_dt)))


         #------------------------------------------------------------------------------------#
         
         # RUN THE DEVELOPED APPROACH
        
        #### GENERATE, EXTRACT AND SELECT DECISION RULES ####
        
        # Generate ensemble models
        fitted_models = generate_rules.generate_treeModels(X_train, y_train)
        # Extract rules from the trees of the ensemble models
        print('extracting rules...')
        all_rules = generate_rules.rules_extraction(fitted_models, features_names, 'include_subRules')
        # all_rules, rules_groups = generate_rules.rules_extraction_withGroups(fitted_models, features_names, 'include_subRules')
        # Remove duplicated rules
        all_rules = list(unique_everseen(all_rules))
        nr_total_rules = len(all_rules)
        
        # Generate the output of all the rules
        rules_outputs = use_rules.apply_rules(all_rules, my_X_train)
        rules_apply = use_rules.do_rules_apply(all_rules, my_X_train)
        
        # Select the subset of the (20) best decision rules
        idx_best20 = rules_selection.lasso_kbest(rules_apply, y_train, 20)
        # idx_best20 = rules_selection.sglasso_kbest(rules_outputs, rules_groups, y_train, 20)
        # idx_best20 = rules_selection.mutual_information(rules_outputs, y_train, 20)
        # idx_best20 = rules_selection.best_SupportAccuracy(all_rules, rules_outputs, y_train, 20)
        
        # index of best 3 rules, best 5 rules, etc...
        idx_selected3 = idx_best20[:3] 
        idx_selected5 = idx_best20[:5]
        idx_selected10 = idx_best20[:10]
        idx_selected15 = idx_best20[:15]
        idx_selected20 = idx_best20[:20]
        

        ##### DEVELOPED PREDICTION AND RELIABILITY METHOD (for rules = 3, 5, 10, 15, 20) #####
        # acceptance is the correctness of each rule for each patient
        
        print('performing rules with acceptance...')
        
        # Compute the predictions using the developed approach 
        # predictions3, rules_m3, suggested_output, predicted_acceptance = auxiliary_main.compute_ourApproach(my_X_train, my_X_test, y_train, y_test, features_names, all_rules, rules_outputs, idx_best20, 3)
        # predictions5, rules_m5, suggested_output, predicted_acceptance = auxiliary_main.compute_ourApproach(my_X_train, my_X_test, y_train, y_test, features_names, all_rules, rules_outputs, idx_best20, 5)
        predictions10, rules_m10, suggested_output10, predicted_acceptance10 = auxiliary_main.compute_ourApproach(my_X_train, my_X_test, y_train, y_test, features_names, all_rules, rules_outputs, idx_best20, 10)
        # predictions15, rules_m15, suggested_output15, predicted_acceptance15 = auxiliary_main.compute_ourApproach(my_X_train, my_X_test, y_train, y_test, features_names, all_rules, rules_outputs, idx_best20, 15)
        # predictions20, rules_m20, suggested_output20, predicted_acceptance20 = auxiliary_main.compute_ourApproach(my_X_train, my_X_test, y_train, y_test, features_names, all_rules, rules_outputs, idx_best20, 20)
        
        # Get the reliability values
        reliability10 = auxiliary_main.compute_reliability(predicted_acceptance10, suggested_output10)
        misc_ratios = auxiliary_main.plot_reliability(predictions10, threshold, y_test, reliability10)
        all_reliability20.append(misc_ratios)
        
        # Auxiliary saves
        all_preds.append(predictions10)
        all_reliabs.append(reliability10)
        
        # Binarization of the output
        binary_predicted = predictions10>=threshold
        all_reliability.append(reliability10)
        all_pred.append(binary_predicted)
        all_ytest.append(y_test)
        
        # Performance metrics (auc and balanced accuracy)
        # auc_m3 = roc_auc_score(y_test, predictions3)
        # auc_m5 = roc_auc_score(y_test, predictions5)
        auc_m10 = roc_auc_score(y_test, predictions10)
        # auc_m15 = roc_auc_score(y_test, predictions15)
        # auc_m20 = roc_auc_score(y_test, predictions20)
        # all_auc_m3.append(auc_m3)
        # all_auc_m5.append(auc_m5)
        all_auc_m10.append(auc_m10)
        # all_auc_m15.append(auc_m15)
        # all_auc_m20.append(auc_m20)
        # ba_m3 = balanced_accuracy_score(y_test, predictions3>=threshold)
        # ba_m5 = balanced_accuracy_score(y_test, predictions5>=threshold)
        ba_m10 = balanced_accuracy_score(y_test, predictions10>=threshold)
        # ba_m15 = balanced_accuracy_score(y_test, predictions15>=threshold)
        # ba_m20 = balanced_accuracy_score(y_test, predictions20>=threshold)
        # all_ba_m3.append(ba_m3)
        # all_ba_m5.append(ba_m5)
        all_ba_m10.append(ba_m10)
        # all_ba_m15.append(ba_m15)
        # all_ba_m20.append(ba_m20)
        
        
        #------------------------------------------------------------------------------------#

        # COMPUTE THE MEANS OF THE SELECTED RULES
        
        #### OUTPUTS' MEAN OF THE SELECTED RULES (for rules = 3, 5, 10, 15, 20) ####
        
        print('performing rules mean...')
        
        # Compute the predictions using the non weighted mean
        # predictions3m = auxiliary_main.compute_meanRules(all_rules, my_X_test, idx_best20, 3) 
        # predictions5m = auxiliary_main.compute_meanRules(all_rules, my_X_test, idx_best20, 5)
        predictions10m_train = auxiliary_main.compute_meanRules(all_rules, my_X_train, idx_best20, 10)
        predictions10m = auxiliary_main.compute_meanRules(all_rules, my_X_test, idx_best20, 10)
        predictions10m = calibration_step(y_train, predictions10m_train, predictions10m) # calibration of the predictions
        # predictions15m = auxiliary_main.compute_meanRules(all_rules, my_X_test, idx_best20, 15)
        # predictions20m = auxiliary_main.compute_meanRules(all_rules, my_X_test, idx_best20, 20)
        
        # Performance metrics (auc and balanced accuracy)
        # auc_m3m = roc_auc_score(y_test, predictions3m)
        # auc_m5m = roc_auc_score(y_test, predictions5m)
        auc_m10m = roc_auc_score(y_test, predictions10m)
        # auc_m15m = roc_auc_score(y_test, predictions15m)
        # auc_m20m = roc_auc_score(y_test, predictions20m)
        # all_auc_m3m.append(auc_m3m)
        # all_auc_m5m.append(auc_m5m)
        all_auc_m10m.append(auc_m10m)
        # all_auc_m15m.append(auc_m15m)
        # all_auc_m20m.append(auc_m20m)
        # ba_m3m = balanced_accuracy_score(y_test, predictions3m>=threshold)
        # ba_m5m = balanced_accuracy_score(y_test, predictions5m>=threshold)
        ba_m10m = balanced_accuracy_score(y_test, predictions10m>=threshold)
        # ba_m15m = balanced_accuracy_score(y_test, predictions15m>=threshold)
        # ba_m20m = balanced_accuracy_score(y_test, predictions20m>=threshold)
        # all_ba_m3m.append(ba_m3m)
        # all_ba_m5m.append(ba_m5m)
        all_ba_m10m.append(ba_m10m)
        # all_ba_m15m.append(ba_m15m)
        # all_ba_m20m.append(ba_m20m)

        
        #### OUTPUTS' WEIGHTED MEAN OF THE SELECTED RULES (for rules = 3, 5, 10, 15, 20) ####
        
        # Here is computed the non personalized weighted mean
        # i.e, the mean is weighted by accuracy of the rules in the training dataset
        
        print('performing rules weighted mean...')
        
        selected_output = rules_outputs[:,idx_selected10] # rules' outputs
        
        # Get the weight for each rule
        rules_correct = selected_output == y_train.reshape(len(y_train), 1)
        rules_correct = rules_correct.astype('float')
        column_sums = rules_correct.sum(axis=0)
        weigths = column_sums/len(y_train)     
        
        # Compute the predictions using the non personalized weighted mean
        predictions10wm = auxiliary_main.compute_weightMeanRules(all_rules, my_X_test, idx_best20, 10, weigths)
        predictions10wm_train = auxiliary_main.compute_weightMeanRules(all_rules, my_X_train, idx_best20, 10, weigths)
        predictions10wm = calibration_step(y_train, predictions10wm_train, predictions10wm) # calibration of the predictions
        
        # Performance metrics (auc and balanced accuracy)
        auc_m10wm = roc_auc_score(y_test, predictions10wm)
        all_auc_m10wm.append(auc_m10wm)
        ba_m10wm = balanced_accuracy_score(y_test, predictions10wm>=threshold)
        all_ba_m10wm.append(ba_m10wm)
        
        # note: here it was only done for 10 rules. 
        
        
        #------------------------------------------------------------------------------------#
        
        # #### TESTED APPROACH: PREDICT ACCEPTANCE OF EACH DECISION TREE ####
        
        # Here it was tested an approach were we get trees from ensembles and then we try to predict when they were correct
        # it is basically the same approach as in "RUN THE DEVELOPED APPROACH" section, but a "decision tree" is a "decision rule"
        
        # note: to run this section it is necessary that one of the sections "RANDOM FOREST OPTIMIZED" or "RANDOM FOREST SIMPLE" are uncommented
        
        # print('performing trees with probabilities')

        # train_outputs, test_outputs = auxiliary_main.get_trees_predictions(trees_rfs, my_X_train, my_X_test) # get trees' predicions
        # predictions_rfs_probs, predictions_train, predicted_acceptance, predicted_acceptance_train = ComputeApproaches.compute_PFCV(my_X_train, my_X_test, y_train, y_test, [], [], rules_rfs, rules_rfs, train_outputs, test_outputs)    
        # auc_rfs_probs = roc_auc_score(y_test, predictions_rfs_probs)
        # ba_rfs_probs = balanced_accuracy_score(y_test, predictions_rfs_probs>=threshold)
        # all_auc_rfs_probs.append(auc_rfs_probs)
        # all_ba_rfs_probs.append(all_ba_rfs_probs)
        
#------------------------------------------------------------------------------------#

# PREDICTION PERFORMANCE RESULTS
# comment/uncomment the methods runned above
    
print('\n---------- Area under the ROC curve ---------------')

# print('Decision tree')
# print('AUC: %.2f +- %.2f' % (np.mean(all_auc_dt), confidence_interval(all_auc_dt)))
# print('Random Forest (no constraints)')
# print('AUC: %.2f +- %.2f' % (np.mean(all_auc_rf), confidence_interval(all_auc_rf)))
# print('Random Forest (simple)')
# print('AUC: %.2f +- %.2f' % (np.mean(all_auc_rfs), confidence_interval(all_auc_rfs)))
# print('Developed approach 3 rules')
# print('AUC: %.2f +- %.2f' % (np.mean(all_auc_m3), confidence_interval(all_auc_m3)))
# print('Developed approach 5 rules')
# print('AUC: %.2f +- %.2f' % (np.mean(all_auc_m5), confidence_interval(all_auc_m5)))
print('Developed approach 10 rules')
print('AUC: %.2f +- %.2f' % (np.mean(all_auc_m10), confidence_interval(all_auc_m10)))
# print('Developed approach 15 rules')
# print('AUC: %.2f +- %.2f' % (np.mean(all_auc_m15), confidence_interval(all_auc_m15)))
# print('Developed approach 20 rules')
# print('AUC: %.2f +- %.2f' % (np.mean(all_auc_m20), confidence_interval(all_auc_m20)))
# print('Mean of 3 best rules')
# print('AUC: %.2f +- %.2f' % (np.mean(all_auc_m3m), confidence_interval(all_auc_m3m)))
# print('Mean of 5 best rules')
# print('AUC: %.2f +- %.2f' % (np.mean(all_auc_m5m), confidence_interval(all_auc_m5m)))
print('Mean of 10 best rules')
print('AUC: %.2f +- %.2f' % (np.mean(all_auc_m10m), confidence_interval(all_auc_m10m)))
# print('Mean of 15 best rules')
# print('AUC: %.2f +- %.2f' % (np.mean(all_auc_m15m), confidence_interval(all_auc_m15m)))
# print('Mean of 20 best rules')
# print('AUC: %.2f +- %.2f' % (np.mean(all_auc_m20m), confidence_interval(all_auc_m20m)))
print('Weighted mean of 10 best rules')
print('AUC: %.2f +- %.2f' % (np.mean(all_auc_m10wm), confidence_interval(all_auc_m10wm)))
# print('Trees with probabilities')
# print('AUC: %.2f +- %.2f' % (np.mean(all_auc_rfs_probs), confidence_interval(all_auc_rfs_probs)))

print('\n----------  Balanced accuracy ---------------')

# print('Decision tree')
# print('BA: %.2f +- %.2f' % (np.mean(all_ba_dt), confidence_interval(all_ba_dt)))
# print('Random Forest (no constraints)')
# print('BA: %.2f +- %.2f' % (np.mean(all_ba_rf), confidence_interval(all_ba_rf)))
# print('Random Forest (simple)')
# print('BA: %.2f +- %.2f' % (np.mean(all_ba_rfs), confidence_interval(all_ba_rfs)))
# print('Developed approach 3 rules')
# print('BA: %.2f +- %.2f' % (np.mean(all_ba_m3), confidence_interval(all_ba_m3)))
# print('Developed approach 5 rules')
# print('BA: %.2f +- %.2f' % (np.mean(all_ba_m5), confidence_interval(all_ba_m5)))
print('Developed approach 10 rules')
print('BA: %.2f +- %.2f' % (np.mean(all_ba_m10), confidence_interval(all_ba_m10)))
# print('Developed approach 15 rules')
# print('BA: %.2f +- %.2f: %.2f' % (np.mean(all_ba_m15), confidence_interval(all_ba_m15)))
# print('Developed approach 20 rules')
# print('BA: %.2f +- %.2f: %.2f' % (np.mean(all_ba_m20), confidence_interval(all_ba_m20)))
# print('Mean of 3 best rules')
# print('BA: %.2f +- %.2f' % (np.mean(all_ba_m3m), confidence_interval(all_ba_m3m)))
# print('Mean of 5 best rules')
# print('BA: %.2f +- %.2f' % (np.mean(all_ba_m5m), confidence_interval(all_ba_m5m)))
print('Mean of 10 best rules')
print('BA: %.2f +- %.2f' % (np.mean(all_ba_m10m), confidence_interval(all_ba_m10m)))
# print('MMean of 15 best rules')
# print('BA: %.2f +- %.2f' % (np.mean(all_ba_m15m), confidence_interval(all_ba_m15m)))
# print('Mean of 20 best rules')
# print('BA: %.2f +- %.2f' % (np.mean(all_ba_m20m), confidence_interval(all_ba_m20m)))
print('Weighted mean of 10 best rules')
print('BA: %.2f +- %.2f' % (np.mean(all_ba_m10wm), confidence_interval(all_ba_m10wm)))
# print('Trees with probabilities')
# print('BA: %.2f +- %.2f' % (np.mean(all_ba_rfs_probs), confidence_interval(all_ba_rfs_probs)))



#------------------------------------------------------------------------------------#

# AUXILIARY PLOTS AND PRINTS
# mainly used for further analysis purposes


#### ANALYSE HOW THE PREDICTED ACCEPTANCE RELATES TO THE CORRECTNESS OF RULES ####

# true acceptance: if the rule is indeed correct or not
true_acceptance = suggested_output10 == y_test.reshape(len(y_test), 1) # rules' acceptance
true_acceptance = true_acceptance*1
# true acceptance for correct (acep1) and incorrect (acep0) rules
acep0 = np.where(true_acceptance==0)
acep1 = np.where(true_acceptance==1)
# predicted acceptance for correct (pa1) and incorrect (pa0) rules
pa0 = predicted_acceptance10[acep0[0],acep0[1]]
pa1 = predicted_acceptance10[acep1[0],acep1[1]]
# print(np.mean(pa0)) # mean acceptance of incorrect rules
# print(np.mean(pa1)) # mean acceptance of correct rules

# Plot of the relation between predicted acceptance and correctness of the rules
fig, (ax1, ax2) = plt.subplots(2, 1)
fig.suptitle('Predicted acceptances')
ax1.set_title('rules that are incorrect')
ax2.set_title('rules that are correct')
ax1.hist(pa0)
ax2.hist(pa1)
ax2.set_xlabel('Predicted acceptance')


#### MIN, MAX AND MEANS OF SOME VECTORS ####

# print(np.min(m3_nrFeatures))
# print(np.max(m3_nrFeatures))
# print(np.mean(m3_nrFeatures))
# print('----------------')
# print(np.min(m5_nrFeatures))
# print(np.max(m5_nrFeatures))
# print(np.mean(m5_nrFeatures))
# print('----------------')
# print(np.min(m10_nrFeatures))
# print(np.max(m10_nrFeatures))
# print(np.mean(m10_nrFeatures))
# print('----------------')
# print(np.min(m15_nrFeatures))
# print(np.max(m15_nrFeatures))
# print(np.mean(m15_nrFeatures))
# print('----------------')
# print(np.min(m20_nrFeatures))
# print(np.max(m20_nrFeatures))
# print(np.mean(m20_nrFeatures))
# print('----------------')
# print(np.min(m10_nrFeatures))
# print(np.max(m10_nrFeatures))
# print(np.mean(m10_nrFeatures))
# print('----------------')
# print(np.mean(m15_nrFeatures))
# print('----------------')
# print(np.mean(rf_nrRules))
# print(np.mean(rfs_nrRules))
# print(np.mean(dt_nrRules))


#### AUC OF "ACCEPTANCES" ####

# auc_all = []
# for i in range(0, predicted_acceptance10.shape[1]):
#     auc_aux = roc_auc_score(true_acceptance[:,i], predicted_acceptance10[:,i])
#     auc_all.append(auc_aux)
#     print(auc_aux)
# print('mean: ', np.mean(auc_all))
# print('median: ', np.median(auc_all))


#### SOME PLOTS RELATED TO RELIABILITY ####

# RELIABILITY PLOT 1 : RELIABILITY VS MISCLASSIFICATIONS
# Here, the groups were created per run, and then it's computed their means

# means of each group of reliability
ttt = np.array(all_reliability20)
ttt_means = np.nanmean(ttt, axis=0)
# confidence interval for each group
ttt_cis = []
for xx in range(0, len(ttt_means)):
    ci = confidence_interval(ttt[:,xx])
    ttt_cis.append(ci)
    
bins = np.linspace(0, 1, 11)
plt.figure()
plt.scatter(bins, ttt_means)
plt.errorbar(bins, ttt_means, yerr=ttt_cis, linestyle="None")
plt.show()
plt.xlabel('reliability')
plt.ylabel('rate of misclassifications')

# RELIABILITY PLOT 2 : RELIABILITY VS MISCLASSIFICATIONS
# Here, the values of all runs are joined and only then the groups and their means are computed 

all_pred = np.concatenate(all_pred, axis=0 )
all_ytest = np.concatenate(all_ytest, axis=0 )
all_reliability = np.concatenate(all_reliability, axis=0 )

misc_ratios = auxiliary_main.plot_reliability(all_pred, threshold, all_ytest, all_reliability)

bins = np.linspace(0, 1, 11)
plt.figure()
plt.scatter(bins, misc_ratios)
plt.show()
plt.xlabel('reliability')
plt.ylabel('rate of misclassifications')

# RELIABILITY PLOT 3: RELIABILITY VS PREDICTED OUTPUT

all_preds = np.concatenate(all_preds, axis=0 )
all_reliabs = np.concatenate(all_reliabs, axis=0 )

plt.figure()
plt.scatter(all_preds, all_reliabs)
plt.ylabel('Output prediction')
plt.xlabel('Reliability estimation')
plt.show()





