# -*- coding: utf-8 -*-
"""
PACKAGE TO CREATE A LIST OF TREES OF ENSEMBLES OF TREES, AND FOR CREATE A LIST
OF THE DECISION RULES EXTRACTED FROM THOSE TREES

@author: Francisco Valente (paulo.francisco.valente@gmail.com)
2021
"""

## Import packages

import extract_rules
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier, GradientBoostingClassifier, ExtraTreesClassifier, RandomForestRegressor
from sklearn import tree, calibration
import performance_metrics
import numpy as np

#------------------------------------------------------------------------------------#


## Auxiliary function to run a list of trees and extract all their decision rules

def rules_extraction(fitted_models, features_names, tipo):
        
    all_rules = []
    
    for model in fitted_models:
        
        tree_list = model.estimators_
        for my_tree in tree_list:
            rules = extract_rules.extract_rules_from_tree(my_tree.tree_,features_names, tipo)
            all_rules = all_rules + rules
            
            # Plot tree
            # tree.plot_tree(my_tree, feature_names = features_names)
        
    return all_rules
   
#------------------------------------------------------------------------------------#

## Auxiliary function to run a list of trees and extract all their decision rules, by groups
# i.e., instead of join the decision rules of all trees in a single list (like the function rules_extraction)
# it creates a group of decision rules for each tree

def rules_extraction_withGroups(fitted_models, features_names, tipo):
        
    all_rules = []
    groups = []
    count = 0
    
    for model in fitted_models:
        
        tree_list = model.estimators_
        for my_tree in tree_list:
            rules = extract_rules.extract_rules_from_tree(my_tree.tree_,features_names, tipo)
            all_rules = all_rules + rules
            
            new_group = np.arange(count,count+len(rules))
            groups.append(new_group)
            
            count= count+len(rules)
            
            # PLOT TREE
            # tree.plot_tree(my_tree, feature_names = features_names)
        
    return all_rules, groups

#------------------------------------------------------------------------------------#

## Function to generate Ensembles of Classification Trees and create a single list of all trees of those models 

def generate_treeModels(X_train, y_train):

    ## Create ensembles

    # rf = GradientBoostingClassifier(n_estimators=10)
    rf1 = RandomForestClassifier(n_estimators=100, random_state=0, max_depth=3,  min_samples_leaf=5, oob_score = True,max_samples = 0.75)
    # rf2 = RandomForestClassifier(n_estimators=40, random_state=0, max_depth=2,  oob_score = True,max_samples = 0.40)
    # rf3 = RandomForestClassifier(n_estimators=20, random_state=0, max_depth=1, class_weight='balanced_subsample', oob_score = True,max_samples = 0.90)
    # rf = ExtraTreesClassifier(n_estimators=1000, random_state=0, max_leaf_nodes=5, class_weight='balanced')
    # rf = xgb.XGBClassifier()
    # tree = DecisionTreeClassifier()
    
    # Join all the trees
    
    list_of_trees = []
    
    # for a single ensemble of trees
    for tree in [rf1]:
        model = tree.fit(X_train, y_train) 
        list_of_trees.append(model)
        
    # # for several ensembles of trees (example)
    # # we may used this one to use models with depth=1, depth=2 and depth=3 e.g.
    # for tree in [rf1, rf2, rf3]:
    #     model = tree.fit(X_train, y_train) 
    #     list_of_trees.append(model)    
        
    return list_of_trees

#------------------------------------------------------------------------------------#

## Function to generate Ensembles of Regression Trees and create a single list of all trees of those models 

def generate_treeModels_regression(X_train, y_train):

    ## Create ensembles
    
    rf1 = RandomForestRegressor(n_estimators=10, random_state=0, max_depth=3,  oob_score = True,max_samples = 0.70)
    # rf2 = RandomForestRegressor(n_estimators=40, random_state=0, max_depth=2,  oob_score = True,max_samples = 0.40)
    
    # Join all the trees
    
    list_of_trees = []
    
    for tree in [rf1]:
        model = tree.fit(X_train, y_train) 
        list_of_trees.append(model)
        
    return list_of_trees

#------------------------------------------------------------------------------------#

# Get the performance metrics (discrimination) of each rule 

def evaluate_rules(rules_outputs, y_train):
    
    # auc: area under the roc curve
    # sens: sensitivity
    # spec: specificity
    # gm: geometric mean of sens and spec
    
    all_gms = []
    
    for t in range(0,rules_outputs.shape[1]):
        this_rule_outputs = rules_outputs[:,t]
        auc, gm, spec, sens = performance_metrics.performance_metrics(this_rule_outputs, y_train, 0.5)
        all_gms.append(gm)
        
    return all_gms
    


