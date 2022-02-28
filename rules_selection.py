# -*- coding: utf-8 -*-
"""
PACKAGE FOR THE SELECTION OF A SUB-SET OF BEST RULES, ACCORDING TO A GIVEN CRITERION
SEVERAL METHODS AVAILABLE

@author: Francisco Valente (paulo.francisco.valente@gmail.com)
2021
"""

## Import packages

from sklearn.linear_model import LogisticRegression, LassoCV, LogisticRegressionCV
import numpy as np
from sklearn.feature_selection import RFE, mutual_info_classif
from sklearn import metrics, feature_selection
from sklearn.svm import SVC
# import groupyr as gpr
import math


#------------------------------------------------------------------------------------#

# Selection of the k best features using LASSO 

def lasso_kbest(rules_outputs, y_train, k):
    
    # lcv_selec = LogisticRegressionCV(penalty='l1', fit_intercept=False, solver='saga', cv=3, scoring='balanced_accuracy')
    lcv_selec = LogisticRegression(penalty='l1',solver='liblinear', fit_intercept=False)

    lcv_selec.fit(rules_outputs, y_train) # fit lasso to data
    coef_rules = lcv_selec.coef_[0]
    
    # select the features (rules) with the k-best coefficients
    idx_selected = abs(coef_rules).argsort()[-k :][::-1]
    
    return idx_selected

#------------------------------------------------------------------------------------#

# Selection of the features with not null coefficients using LASSO 

def lasso_notnull(rules_outputs, y_train):    
    
    lcv_selec = LogisticRegressionCV(penalty='l1', fit_intercept=False, solver='liblinear', cv=5, scoring='balanced_accuracy')
    lcv_selec.fit(rules_outputs, y_train) # fit lasso to data
    coef_rules = lcv_selec.coef_[0]
    
    # select the features (rules) with non null coefficients
    idx_selected = np.where(coef_rules != 0)[0]
    
    return idx_selected

#------------------------------------------------------------------------------------#

# Selection of the k best features using LASSO (regression)

def lasso_reg(rules_outputs, y_train, k):
    
    lcv_selec = LassoCV(cv=3)
    lcv_selec.fit(rules_outputs, y_train) # fit lasso to data
    coef_rules = lcv_selec.coef_
    
    # select the features (rules) with the k-best coefficients
    idx_selected = abs(coef_rules).argsort()[-k :][::-1]
    
    return idx_selected

#------------------------------------------------------------------------------------#

# Selection of the k best features using sparse group LASSO (SG LASSO)

def sglasso_kbest(rules_outputs, rules_groups, y_train, k):
    
    lcv_selec = gpr.LogisticSGLCV(cv=2, groups=rules_groups, scoring='balanced_accuracy')
    lcv_selec.fit(rules_outputs, y_train) # fit sglasso to data
    coef_rules = lcv_selec.coef_[0]
    
    # select the features (rules) with the k-best coefficients
    idx_selected = abs(coef_rules).argsort()[-k :][::-1]
    
    return idx_selected

#------------------------------------------------------------------------------------#

# Selection of the best features using Recursive Feature Elimination (RFE) w/ cross validation

def rfecv(rules_outputs, y_train):
    
    estimator = LogisticRegression(penalty='l1', fit_intercept=False, solver='liblinear') 
    # estimator = SVC(kernel="linear", probability=True)
    lcv_selec = feature_selection.RFECV(estimator, scoring = 'roc_auc', cv=20) 
    lcv_selec.fit(rules_outputs, y_train)
    
    # print("Optimal number of features : %d" % lcv_selec.n_features_)
    
    # # # Plot number of features VS. cross-validation scores
    # # plt.figure()
    # # plt.xlabel("Number of features selected")
    # # plt.ylabel("Cross validation score (nb of correct classifications)")
    # # plt.plot(range(1,
    # #                 len(lcv_selec.grid_scores_) + 1),
    # #           lcv_selec.grid_scores_)
    # # plt.show()
    
    # select the features (rules) 
    idx_selected = np.where(lcv_selec.ranking_==1)[0]
    
    return idx_selected

#------------------------------------------------------------------------------------#

# Selection of the k-best features using Recursive Feature Elimination (RFE) w/o cross validation

def rfe_notcv(rules_outputs, y_train, k):

    estimator = LogisticRegression(penalty='l1', fit_intercept=False, solver='liblinear')
    # estimator = SVC(kernel="linear", probability=True)
    lcv_selec = RFE(estimator, n_features_to_select=k, step=1)
    lcv_selec.fit(rules_outputs, y_train)
    
    # select of the k-best features (rules) 
    idx_selected = np.where(lcv_selec.ranking_==1)[0]
    
    return idx_selected
    
#------------------------------------------------------------------------------------#

# Selection of the k-best features using Mutual Information

def mutual_information(rules_outputs, y_train, k):
    
    mi = mutual_info_classif(rules_outputs, y_train, discrete_features=True)
    
    # select of the k-best features (rules) 
    idx_selected = mi.argsort()[-k:][::-1]
    
    return idx_selected

#------------------------------------------------------------------------------------#

# Selection of the k features (rules) with the best geometric mean

def best_gm(all_gms, k):
    
    idx_selected = np.argpartition(all_gms, -k)[-k:]
    
    return idx_selected

#------------------------------------------------------------------------------------#

# Selection of the best trade-off between support and accuracy

# support: it is the ratio of instances to which the condition of a rule applies
# accuracy: it is ratio of correct class prediction for the  instances to which the condition of a rule applies

def best_SupportAccuracy(all_rules, rules_outputs, y_train, k):

    # auxiliary function to obtain the supports of the rules
    
    def get_supports(all_rules, rules_outputs):
    
        rules_if = [rule['Rule predictions']['IF'] for rule in all_rules]
        supports = [] 
        
        for i in range(0, len(rules_if)):
            frequency = (rules_outputs[:,i] == rules_if[i]).sum()/len(rules_outputs)
            supports.append(frequency)
    
        return np.array(supports)

    # obtain the support of the rules
    
    supports = get_supports(all_rules, rules_outputs)
    
    # obtain the accuracy of the rules
    
    accuracy = []
    for ac in range(0, rules_outputs.shape[1]):
        acc = np.sum(rules_outputs[:,ac] == y_train)/len(y_train)
        accuracy.append(acc)
    accuracy =  np.asarray(accuracy)
    
    # get a balanced metric between accuracy and supports (its geometric mean)
    aux = []
    for t in range(0, len(all_rules)):
        aux.append( math.sqrt(accuracy[t]* supports[t]))
    aux =  np.asarray(aux)
    
    # select of the features (rules) with the k best geometric mean
    idx_selected = aux.argsort()[-k:][::-1]
    
    return idx_selected

#------------------------------------------------------------------------------------#

# Function to obtain the sub-set of features used by the selected rules
# i.e, not all of the initial features are used by the set of selected rules

def obtain_features(all_rules):
        
    used_features = []
    
    for rule in all_rules:
        rule_conditions = rule['Rule conditions']
        for condition in rule_conditions:
            feat_name = condition[0]
            used_features.append(feat_name)
            
    # eliminate duplicates
    used_features = list(set(used_features))
    
    return used_features


