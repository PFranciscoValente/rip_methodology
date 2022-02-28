# -*- coding: utf-8 -*-
"""
PACKAGE TO EXTRACT EACH INDIVIDUAL RULE FROM A GIVEN DECISION TREE

@author: Francisco Valente (paulo.francisco.valente@gmail.com)
2021

Note: several parts of this script were retrieved or adapted from:
https://github.com/christophM/rulefit/blob/master/rulefit/rulefit.py
"""

from sklearn import tree

"""Recursively extract rules from each tree in the ensemble
        """

class RuleCondition():
    """Class for binary rule condition

    Warning: this class should not be used directly.
    """

    def __init__(self,
                 feature_index,
                 threshold,
                 operator,
                 support,
                 feature_name = None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.operator = operator
        self.support = support
        self.feature_name = feature_name


    def __repr__(self):
        return self.__str__()

    def __str__(self):
        if self.feature_name:
            feature = self.feature_name
        else:
            feature = self.feature_index
        return "%s %s %s" % (feature, self.operator, self.threshold)
    
    def get_condition(self):
        if self.feature_name:
            feature = self.feature_name
        else:
            feature = self.feature_index
        return [feature, self.operator, self.threshold]
        # return self.feature
        
        
    
class Rule():
    """Class for binary Rules from list of conditions

    Warning: this class should not be used directly.
    """
    def __init__(self,
                 rule_conditions,prediction_value):
        # self.conditions = set(rule_conditions)
        self.conditions = rule_conditions
        # self.support = min([x.support for x in rule_conditions])
        self.prediction_value=prediction_value
        self.rule_direction=None
    def transform(self, X):
        """Transform dataset.

        Parameters
        ----------
        X: array-like matrix

        Returns
        -------
        X_transformed: array-like matrix, shape=(n_samples, 1)
        """
        rule_applies = [condition.transform(X) for condition in self.conditions]
        return reduce(lambda x,y: x * y, rule_applies)

    def __str__(self):
        return  " & ".join([x.__str__() for x in self.conditions]) + ' || predicted value: ' + str(self.prediction_value)
    
    # def __str__(self):
    #     print(x.__str__() for x in self.conditions)
    #     return  " & ".join([x.__str__() for x in self.conditions]) + ' > ' + str(self.prediction_value)
    
    def __repr__(self):
        return self.__str__()
    
    # def get_rulesList(self):
    #     return  [re.split('( > )|( <= )', x.__str__()) for x in self.conditions]

  
#------------------------------------------------------------------------------------#
  
## main (recursive) funtion used to extract the rules from a decision tree

# input "tipo" - se for "include_subRules" quer dizer que tambem inclui as sub-regras
# i.e. vamos supor que o full path de uma tree é IF A>x AND B<=y AND C>z THEN k1
# isso é uma regra que pode ser decomposta em outras 2 sub-regras (partial path)
# 1. IF A>x THEN k2
# 2. IF A>x AND B<=y THEN k3

def extract_rules_from_tree(tree, feature_names=None, tipo='standard'):
    """Helper to turn a tree into as set of rules
    """
    rules = list() # my final list of extracted rules

    def traverse_nodes(node_id=0,
                       operator=None,
                       threshold=None,
                       feature=None,
                       conditions=[]):
        if node_id != 0:

            if feature_names is not None:
                feature_name = feature_names[feature]
            else:
                feature_name = feature
            rule_condition = RuleCondition(feature_index=feature,
                                           threshold=threshold,
                                           operator=operator,
                                           support = tree.n_node_samples[node_id] / float(tree.n_node_samples[0]),
                                           feature_name=feature_name)

            new_conditions  = conditions + [rule_condition]

        else:
            new_conditions = []

        ## if not terminal node
        if tree.children_left[node_id] != tree.children_right[node_id]:
            feature = tree.feature[node_id]
            threshold = tree.threshold[node_id]

            if len(new_conditions)>0:

                my_conditions  = []
                for cond in new_conditions:
                    indiv_condition = cond.get_condition()
                    my_conditions.append(indiv_condition)
                    
                    
                predictions = {}    
                
                ## CLASSIFICATION
                
                # this vector orders the number of elements in each group
                # so the first one will be the larger class (IF) and the second 
                # one will be the lower class in case of binary classification or
                # the second class in case of multiclass classification (ELSE)
                pred = (-tree.value[node_id]).argsort()
                
                # Get IF and ELSE
                predictions['IF'] = pred[0][0]
                predictions['ELSE'] = pred[0][1]
                # predictions['ELSE'] = -1
                
                ## REGRESSION
                # predictions['IF'] = tree.value[node_id][0][0]
                
                new_rule = {}
                new_rule['Rule conditions'] = my_conditions
                new_rule['Rule predictions'] = predictions
                # my_conditions = Rule(new_conditions,np.argmax(tree.value[node_id][0])) # vai buscar o output dessa regra para dps usar no IF ELSE
                
                if tipo=='include_subRules': # in this case, it also includes sub-rules
                    rules.append(new_rule)
                    
            else:
                pass #tree only has a root node!
            
            left_node_id = tree.children_left[node_id]
            traverse_nodes(left_node_id, "<=", threshold, feature, new_conditions)

            right_node_id = tree.children_right[node_id]
            traverse_nodes(right_node_id, ">", threshold, feature, new_conditions)
            
            
        else: # a leaf node

            if len(new_conditions)>0:

                my_conditions  = []
                for cond in new_conditions:
                    indiv_condition = cond.get_condition()
                    my_conditions.append(indiv_condition)  
                
                predictions = {}

                
                ## CLASSIFICATION
                    
                # this vector orders the number of elements in each group
                # so the first one will be the larger class (IF) and the second 
                # one will be the lower class in case of binary classification or
                # the second class in case of multiclass classification (ELSE)
                pred = (-tree.value[node_id]).argsort()
                
                # Get IF and ELSE
                predictions['IF'] = pred[0][0]
                predictions['ELSE'] = pred[0][1]
                # predictions['ELSE'] = -1
                
                
                ## REGRESSION
                #predictions['IF'] = tree.value[node_id][0][0]
                
                new_rule = {}
                new_rule['Rule conditions'] = my_conditions
                new_rule['Rule predictions'] = predictions
                # my_conditions = Rule(new_conditions,np.argmax(tree.value[node_id][0])) # vai buscar o output dessa regra para dps usar no IF ELSE
                
                rules.append(new_rule)
                
            else:
                pass #tree only has a root node!
            return None

    traverse_nodes()

    return rules

#------------------------------------------------------------------------------------#

def get_rules(self, exclude_zero_coef=False, subregion=None):
        """Return the estimated rules

        Parameters
        ----------
        exclude_zero_coef: If True (default), returns only the rules with an estimated
                           coefficient not equalt to  zero.

        subregion: If None (default) returns global importances (FP 2004 eq. 28/29), else returns importance over
                           subregion of inputs (FP 2004 eq. 30/31/32).

        Returns
        -------
        rules: pandas.DataFrame with the rules. Column 'rule' describes the rule, 'coef' holds
               the coefficients and 'support' the support of the rule in the training
               data set (X)
        """

        n_features= len(self.coef_) - len(self.rule_ensemble.rules)
        rule_ensemble = list(self.rule_ensemble.rules)
        output_rules = []
        ## Add coefficients for linear effects
        for i in range(0, n_features):
            if self.lin_standardise:
                coef=self.coef_[i]*self.friedscale.scale_multipliers[i]
            else:
                coef=self.coef_[i]
            if subregion is None:
                importance = abs(coef)*self.stddev[i]
            else:
                subregion = np.array(subregion)
                importance = sum(abs(coef)* abs([ x[i] for x in self.winsorizer.trim(subregion) ] - self.mean[i]))/len(subregion)
            output_rules += [(self.feature_names[i], 'linear',coef, 1, importance)]

        ## Add rules
        for i in range(0, len(self.rule_ensemble.rules)):
            rule = rule_ensemble[i]
            coef=self.coef_[i + n_features]

            if subregion is None:
                importance = abs(coef)*(rule.support * (1-rule.support))**(1/2)
            else:
                rkx = rule.transform(subregion)
                importance = sum(abs(coef) * abs(rkx - rule.support))/len(subregion)

            output_rules += [(rule.__str__(), 'rule', coef,  rule.support, importance)]
        rules = pd.DataFrame(output_rules, columns=["rule", "type","coef", "support", "importance"])
        if exclude_zero_coef:
            rules = rules.ix[rules.coef != 0]
        return rules

def split(delimiters, string, maxsplit=0):
    import re
    regexPattern = '(|)'.join(map(re.escape, delimiters))
    return re.split(regexPattern, string, maxsplit)