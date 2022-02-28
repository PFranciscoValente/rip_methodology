# -*- coding: utf-8 -*-
"""
PACKAGE TO COMPUTE THE RULES' ACCEPTANCE AND PREDICTION (DEVELOPED METHODOLOGY)

@author: Francisco Valente (paulo.francisco.valente@gmail.com)
2021
"""

## Import packages

import numpy as np
from sklearn.linear_model import LogisticRegression, LassoCV, LogisticRegressionCV, Perceptron, LinearRegression, Lasso
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, VotingClassifier, StackingClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from scipy import stats
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
# import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn import tree

#------------------------------------------------------------------------------------#


# Auxiliary function to calibrate the predictions

def calibration_step(y_train, predictions_train, predictions_test):
        
    clf = LogisticRegression(solver='liblinear')
    calibrator = clf.fit(predictions_train.reshape(-1,1), y_train)
    calibrated_predictions_train = calibrator.predict_proba(predictions_train.reshape(-1,1))[:,1]
    calibrated_predictions_test = calibrator.predict_proba(predictions_test.reshape(-1,1))[:,1]
    
    return calibrated_predictions_train, calibrated_predictions_test
        
#------------------------------------------------------------------------------------#

## Functions to get the prediction of the developed approach
# several approaches are presented, the ones used in the papers and some exploratory ones

# Auxiliary function to normalize (range [0,1]) the ouput
def get_value_inRange(OldValue):
    OldMax = 1; OldMin = -1; NewMin = 0; NewMax = 1
    OldRange = (OldMax - OldMin)  
    NewRange = (NewMax - NewMin)  
    NewValue = (((OldValue - OldMin) * NewRange) / OldRange) + NewMin
    return NewValue
    

# Function to get the predictions of the developed approach
# (this one is the one used in the papers of Artificial Intelligence in Medicine journal and ICML-IMLH conference)

def get_prediction(outputs,acceptances):
              
    outputs = np.where(outputs==0, -1, outputs) # replace outputs=0 by outputs=-1
    pointwise_mult = np.multiply(outputs, acceptances) # pointwise multiplication of each rule's output by its predicted acceptance
    all_pred = []

    # Compute the prediction for each patient
    
    for p in range (0,len(outputs)):
        
        probs =  pointwise_mult[p,:] # pointwise multiplication for sample p
        # used_probs = np.delete(probs, np.where(probs==0)) # usar só as regras que foram utilizadas
        prediction = np.mean(probs) # get the prediction as the mean of 
        prediction = get_value_inRange(prediction) # normalize the output
    
        all_pred.append(prediction)
        
    all_pred = np.array(all_pred)
    
    return all_pred


# Function to get the predictions of the developed approach
# (this one is the one used in the paper of EMBC conference)

def get_prediction_embc(outputs,acceptances):

    all_pred = []
    
     # Compute the prediction for each patient
     
    for p in range (0,len(outputs)):
        
        # Define the weight of each rule
        peso = acceptances[p,:]+1
        # here if acceptance=0, weight=1; if if acceptance=0, weight=2

        # Compute the predicted as weighted mean
        prediction = np.average(outputs[p,:], weights=peso)

        all_pred.append(prediction)

    all_pred = np.array(all_pred)
    
    return all_pred


# Function to get the predictions of the developed approach
# (this one was used as exploration of other methods)

def get_prediction_3(outputs,acceptances):

    all_pred = []
    
    for p in range (0,len(outputs)):
        
        idx0 = np.where(outputs[p,:]==0) # idx of negative (output=0) rules
        idx1 = np.where(outputs[p,:]==1) # idx of positive (output=1) rules
        acep0 = acceptances[p,idx0] # acceptance of negative rules
        acep1 = acceptances[p,idx1] # acceptance of positive rules
        
        total_acep = np.nanmean(acep0)+np.nanmean(acep1)
        prediction = np.nanmean(acep1)/total_acep
        all_pred.append(prediction)

    all_pred = np.array(all_pred)
    return all_pred


# Function to get the predictions of the developed approach
# (this one was used as exploration of other methods)

def get_prediction_4(outputs,acceptances):

    all_pred = []
    
     # Compute the prediction for each patient
     
    for p in range (0,len(outputs)):
        
        # rules_accept = np.where(acceptances[p,:] == 1)
        # used_output = outputs[p,rules_accept] 
        
        # arr = []
        
        # for t in range(0, len(acceptances[p,:])):
        #     if acceptances[p,t]==1:
        #         arr.append(outputs[p,t])
        #     else:
        #         if outputs[p,t]==1:
        #             arr.append(0)
        #         else:
                    # arr.append(1)
        
        # if len(used_output[0])==0:
        #     prediction = np.mean(outputs[p,:])
        # else:
        #     prediction = sum(used_output[0])/len(used_output[0])
    
        # print(acceptances)
        
        # Define the weight of each rule
        peso = acceptances[p,:]+1
        # here if acceptance=0, weight=1; if if acceptance=0, weight=2
        
        # ii = ~np.isnan(outputs[p,:])

        # Compute the predicted as weighted mean
        prediction = np.average(outputs[p,:], weights=peso)
          
        # prediction = np.average(outputs[p,ii], weights = peso[ii])
        # prediction = 1
        
        # prediction = np.mean(arr)

        # if len(used_output[0])==0:
        #     prediction = stats.mode(outputs[p,:])[0]
        # else:
        #     prediction = stats.mode(used_output[0])[0]
            
        all_pred.append(prediction)

    all_pred = np.array(all_pred)
    
    return all_pred

#------------------------------------------------------------------------------------#

## Functions to get the predicted acceptances for each rule
# several approaches are presented, the used one (function "get_acceptances") and some exploratory ones


# Function to get the predicted acceptances for each rule
# (this one is the main one - the one used in the papers)

def get_acceptances (rules_acceptance, X_train, X_test):

    
    # Choose the classification model
    clf = LogisticRegression(penalty='l1', solver='liblinear')
    # clf = tree.DecisionTreeClassifier()
    # clf = SVC(probability=True)
    # clf = RandomForestClassifier(n_estimators=100, random_state=1)
    # clf = RandomForestClassifier(n_estimators=10, random_state=0, max_leaf_nodes=2, class_weight='balanced')
    # clf = xgb.XGBClassifier()
    # clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(8, 16), random_state=1)

    # Auxiliary function to optimize a logistic regression for each rule
    def apply_lr(X_train, y_train):
        lr = LogisticRegression()
        
        random_grid = {'penalty' : ['l1', 'l2'],
                      'C' : np.logspace(-4, 4, 5),
                      'solver' : ['liblinear']},
        
        lr_random = RandomizedSearchCV(estimator = lr, param_distributions = random_grid, cv = 3, verbose=2, random_state=42, n_jobs = -1)
        model = lr_random.fit(X_train, y_train)
        
        return model
     
    predicted_acceptance =  np.empty([len(X_test.index), rules_acceptance.shape[1]])  
    predicted_acceptance_train =  np.empty([len(X_train.index),rules_acceptance.shape[1]])  
    
    # Train a model for each rule and obtain the predicted acceptances
    
    for r in range(0, rules_acceptance.shape[1]):
        
        # new label (rules_acceptance) of rule r
        label_rule = rules_acceptance[:,r]
        
        # Train the classification model for rule r
        model = clf.fit(X_train,label_rule) # train a simple model
        # model = apply_lr(X_train, label_rule) # train an optimized logistic regression
        
        # Obtain the test acceptance predictions
        predictions = model.predict_proba(X_test.values)
        predicted_acceptance[:,r] = predictions[:,1]
                
        # Obtain the train acceptance predictions
        predictions_train = model.predict_proba(X_train.values)
        predicted_acceptance_train[:,r] = predictions_train[:,1]
        
    return predicted_acceptance, predicted_acceptance_train


# Function to get the predicted acceptances for each rule
# (this one was used as exploration of other methods)
# this one is adapted to be used if there are nans in the training rules' acceptances

def get_acceptances_nans (rules_acceptance, X_train, X_test):
    
    # Choose the classification model
    clf = LogisticRegressionCV(penalty='l1', solver='liblinear')
    
    predicted_acceptance =  np.empty([len(X_test.index), rules_acceptance.shape[1]])  
    predicted_acceptance_train =  np.empty([len(X_train.index),rules_acceptance.shape[1]]) 
    
    # Train a model for each rule and obtain the predicted acceptances
    
    for r in range(0, rules_acceptance.shape[1]):
        
        # new label (rules_acceptance) of rule r
        label_rule = rules_acceptance[:,r]
        
        # get nans in the rules
        rule_nans = np.stack(np.argwhere(np.isnan(label_rule)),axis=1)[0]
        # remove nans
        label_rule = label_rule[~np.isnan(label_rule)]
        X_train2 = X_train.drop(rule_nans)
        
        # Train the classification model for rule r
        model = clf.fit(X_train2,label_rule)
        
        # Obtain the test acceptance predictions
        predictions = model.predict_proba(X_test.values)
        predicted_acceptance[:,r] = predictions[:,1]
        
        # Obtain the train acceptance predictions
        predictions_train = model.predict_proba(X_train2.values)
        predicted_acceptance_train = 0
        
    return predicted_acceptance, predicted_acceptance_train


# Function to get the predicted acceptances for each rule
# (this one was used as exploration of other methods)
# this one does a balancing of the training classes

def get_acceptances_balancing (rules_acceptance, X_train, X_test):

    # Choose the classification model
    clf = LogisticRegressionCV(penalty='l1', solver='liblinear')
    # clf = SVC(probability=True)
    # clf = RandomForestClassifier(n_estimators=10, random_state=0, max_depth=2, class_weight='balanced')
    # clf = xgb.XGBClassifier()
    # clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(8, 16), random_state=1)
    
    # Choose the balancing model
    # sampler = RandomUnderSampler(sampling_strategy='majority')
    # sampler = RandomOverSampler(sampling_strategy='minority')
    sampler = SMOTE()
    
    
    predicted_acceptance =  np.empty([len(X_test.index), rules_acceptance.shape[1]])  
    predicted_acceptance_train =  np.empty([len(X_train.index),rules_acceptance.shape[1]])  
    predicted_acceptance_train[:] = np.nan
    
    # Train a balanced model for each rule and obtain the predicted acceptances
    
    for r in range(0, rules_acceptance.shape[1]):

        # new label (rules_acceptance) of rule r
        label_rule = rules_acceptance[:,r]
        
        # perform sampling (balancing)
        X_train_aux, label_rule_aux = sampler.fit_resample(X_train, label_rule)
        
        # Train the classification model for rule r
        model = clf.fit(X_train_aux,label_rule_aux)
        
        # Obtain the test acceptance predictions
        predictions = model.predict_proba(X_test.values)
        predicted_acceptance[:,r] = predictions[:,1]
        
        # Obtain the train acceptance predictions
        predictions_train = model.predict_proba(X_train.values)
        predicted_acceptance_train[:,r] = predictions_train[:,1]
        
    return predicted_acceptance, predicted_acceptance_train


#------------------------------------------------------------------------------------#

## Functions to get the predicted acceptances for each rule (muticlassification scenario)
# used for exploratory purposes

def get_acceptances_multi (rules_acceptance, X_train, X_test):

    # Choose the classification model
    clf = LogisticRegressionCV(penalty='l1', solver='liblinear')
    # clf = SVC(probability=True)
    # clf = RandomForestClassifier(n_estimators=10, random_state=0, max_depth=2, class_weight='balanced')
    # clf = xgb.XGBClassifier()
    # clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(8, 16), random_state=1)

    # Choose the balancing model
    # sampler = RandomUnderSampler(sampling_strategy='majority')
    # sampler = RandomOverSampler(sampling_strategy='minority')
    # sampler = SMOTE()
    
    predicted_acceptance =  np.empty([len(X_test.index), rules_acceptance.shape[1]])  
    predicted_acceptance_train =  np.empty([len(X_train.index),rules_acceptance.shape[1]])  
    
    # Train a model for each rule and obtain the predicted acceptances
    
    for r in range(0, rules_acceptance.shape[1]):

        # new label (rules_acceptance) of rule r
        label_rule = rules_acceptance[:,r]
        
        # Remove nans
        idx_notnan = np.argwhere(~np.isnan(label_rule))
        notnan = []
        for xx in idx_notnan:
            notnan.append(xx[0])
        X_train_aux = X_train.iloc[notnan, :]
        label_rule_aux = label_rule[~np.isnan(label_rule)]
        
        # perform sampling (balancing)
        # X_train_aux, label_rule_aux = sampler.fit_resample(X_train, label_rule)
        
        # Train the classification model for rule r
        model = clf.fit(X_train_aux,label_rule_aux)
        
        # Obtain the test acceptance predictions
        predictions = model.predict_proba(X_test.values)
        predicted_acceptance[:,r] = predictions[:,1] 
        
        # Obtain the train acceptance predictions
        predictions_train = model.predict_proba(X_train.values)
        predicted_acceptance_train[:,r] = predictions_train[:,1]
        
    return predicted_acceptance, predicted_acceptance_train

#------------------------------------------------------------------------------------#

## Functions to get the predicted acceptances for each rule (regression scenario)
# used for exploratory purposes

def get_acceptances_reg (rules_acceptance, X_train, X_test):

    # Choose the classification model
    # clf = LinearRegression()
    clf = Lasso()
    # clf = SVC(probability=True)    # clf = xgb.XGBClassifier()
    # clf = RandomForestRegressor(n_estimators=100, random_state=0, max_depth=3,  oob_score = True,max_samples = 0.90)

    predicted_acceptance =  np.empty([len(X_test.index), rules_acceptance.shape[1]])  
    predicted_acceptance_train =  np.empty([len(X_train.index),rules_acceptance.shape[1]])
    
    # Train a model for each rule and obtain the predicted acceptances
    
    for r in range(0, rules_acceptance.shape[1]):

        # new label (rules_acceptance) of rule r
        label_rule = rules_acceptance[:,r]
        
        # Remove nans
        idx_notnan = np.argwhere(~np.isnan(label_rule))
        notnan = []
        for xx in idx_notnan:
            notnan.append(xx[0])
        X_train_aux = X_train.iloc[notnan, :]
        label_rule_aux = label_rule[~np.isnan(label_rule)]
        
        # perform sampling (balancing)
        # X_train_aux, label_rule_aux = sampler.fit_resample(X_train, label_rule)
        
        # Train the classification model for rule r
        model = clf.fit(X_train_aux,label_rule_aux)
        
        # Obtain the test acceptance predictions
        predictions = model.predict(X_test.values)
        predicted_acceptance[:,r] = predictions
        
        # Obtain the train acceptance predictions
        predictions_train = model.predict(X_train.values)
        predicted_acceptance_train[:,r] = predictions_train
        
    return predicted_acceptance, predicted_acceptance_train