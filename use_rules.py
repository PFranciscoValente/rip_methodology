# -*- coding: utf-8 -*-
"""
PACKAGE TO APPLY THE LIST OF DECISION RULES TO NEW SAMPLES , OBTAINING THE SUGGESTED OUTPUTS

@author: Francisco Valente (paulo.francisco.valente@gmail.com)
2021
"""

## Import packages

import pandas as pd
import numpy as np


#------------------------------------------------------------------------------------#

# Auxiliary function to verify the idx where the rule conditions agree

def verify_rule(rule_condition, X_train):

    feat_name = rule_condition[0]
    operator = rule_condition[1]
    threshold = rule_condition[2]
    
    values = X_train[feat_name]
    values.to_numpy()
    
    if operator == '<=':
        idx = np.where(values <= threshold)
    else: 
        idx = np.where(values > threshold)
    return idx


#------------------------------------------------------------------------------------#

# Auxiliary function to get the idx of the samples that satisfy all the condition of a given rule and the ones doesn't

def satisfy_condition(rule, X_train):

    # get rule condition (IF part) and rule predictions (THEN and ELSE parts)
    rule_conditions = rule['Rule conditions']
    # rule_predictions = rule['Rule predictions']
    
    # Get elements that satisfy the rule's partial conditions
    all_idx = []
    for condition in rule_conditions:
        
        new_idx = verify_rule(condition, X_train)
        all_idx.append(new_idx[0])

    list_idx = [arr.tolist() for arr in all_idx] # convert list of arrays to list of lists
    
    # list of elements that DO satisfy all the conditions
    satisfies = list(set.intersection(*map(set, list_idx)))
    # list of elements that DO NOT satisfy  the conditions
    all_elements = np.arange(0, len(X_train), 1).tolist()
    dontSatisfies = list(set(all_elements)-set(satisfies))
    
    return satisfies, dontSatisfies
    
    
#------------------------------------------------------------------------------------#

## Function to apply the rules to new samples

def apply_rules(all_rules, X_train):
    
    all_rules_out =  np.empty([len(X_train.index), len(all_rules)])   
    
    contag = 0 
    
    for rule in all_rules:
                
        # get rule condition (IF part) and rule predictions (THEN and ELSE parts)
        rule_conditions = rule['Rule conditions']
        rule_predictions = rule['Rule predictions']
        
        # Get elements that satisfy the rule's conditions and ones that doesn't
        satisfies, dontSatisfies = satisfy_condition(rule, X_train)
        
        # RULE OUTPUT
        rule_out = np.zeros(len(X_train.index))
        rule_out[satisfies] = rule_predictions['IF']
        ## classification
        rule_out[dontSatisfies] = rule_predictions['ELSE']
        ## regression
        #rule_out[dontSatisfies] = float("NAN")
        
        all_rules_out[:,contag] = rule_out

        contag = contag +1

    return all_rules_out        
        
#------------------------------------------------------------------------------------#

## Function to retrieve which rules apply (IF-part) to each sample
   
def do_rules_apply(all_rules, X_train):
    
    all_rules_out =  np.empty([len(X_train.index), len(all_rules)])   
     
    contag = 0 
    
    for rule in all_rules:
                
        # get rule condition (IF part) and rule predictions (THEN and ELSE parts)
        rule_conditions = rule['Rule conditions']
        rule_predictions = rule['Rule predictions']
        
       # Get elements that satisfy the rule's conditions and ones that doesn't
        satisfies, dontSatisfies = satisfy_condition(rule, X_train)
        
        # RULE APPLY OR NOT
        rule_out = np.zeros(len(X_train.index))
        rule_out[satisfies] = 1
        ## classification
        rule_out[dontSatisfies] = 0
        ## regression
        # rule_out[dontSatisfies] = float("NAN")
        
        all_rules_out[:,contag] = rule_out

        contag = contag +1

    return all_rules_out        
        
