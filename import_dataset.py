# -*- coding: utf-8 -*-
"""
PACKAGE FOR THE IMPORT OF THE DATASETS
If necessary, it is also performed pre-processing here (e.g., missing imputation)

Note1: to see which names should be passed as input for each dataset, please see the
'dataset_name' variable on the function 'dataset_info' (bottom of the script)

@author: Francisco Valente (paulo.francisco.valente@gmail.com)
2021

Note2:
All the files were obtained from public repositories (UCI, kaggle), except the lookAfterRisk and Friend datasets                                                
UCI repository: https://archive.ics.uci.edu/ml/index.php
Kaggle: https://www.kaggle.com/
"""

## Import packages

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer
from sklearn.datasets import load_wine
import os


#------------------------------------------------------------------------------------#

## Function for imputation of missing data

def impute_missing(old_dataset):
    
    imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    new_dataset_aux = imputer.fit_transform(old_dataset)
    new_dataset = pd.DataFrame(data=new_dataset_aux, columns=old_dataset.columns)
    
    return  new_dataset

#------------------------------------------------------------------------------------#

# Auxiliary function for data loading

def get_filePath(file_name):
    
    file_loc = '\\datasets\\'
    file = file_loc + file_name
    path = os.getcwd()+file
    
    return path

#------------------------------------------------------------------------------------#

## DATASETS IMPORTING (and pre-processing if necessary)

def lookAfterRisk():
    
    file = get_filePath("ACS_DATASET.csv")
    data1 = pd.read_csv(file, sep=';', decimal=',')
    
    # names_feat = ['idade', 'pressao sistolica', 'frequencia cardiaca','classe killip','avc/ait', 'diagnóstico', 'sexo', 'fumador', 'diabetes', 'função ve', 'creatinina', 'glicose']
    names_feat = ['idade', 'pressao sistolica', 'frequencia cardiaca','classe killip','avc/ait', 'diagnóstico', 'função ve']
    names_feat = list(map(str.upper, names_feat))
    
    # PATIENTS TO USE
    idx_col = data1.columns.get_loc('FOLLOW-UP 30 D')
    idx = data1.index[data1['FOLLOW-UP 30 D'] == 1].tolist()
    idx.remove(93)
    idx.remove(760)
    
    data2 = data1.iloc[idx, :]
    data2 = data2.reset_index()
    
    # LABEL
    idx_death = data2.index[(data2['MORTE'] == 1) & (data2['DIAS MORTE'] <= 30)].tolist()
    label = np.zeros((len(data2),), dtype=int)
    label[idx_death] = 1
    
    # SELECT VARIABLES
    all_variables = list(data2.columns)
    select = [all_variables.index(i) for i in all_variables if i in names_feat]
    data3 = data2.iloc[:, select]
    
    # impute missing data
    imputer = KNNImputer(n_neighbors=5)
    data_imputed = imputer.fit_transform(data3)
    data4 = pd.DataFrame(data=data_imputed , columns = names_feat)
    
    return data4, label


def adult_income():

    file = get_filePath("adult_income.data")
    col_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income-label']
    data1 =  pd.read_table(file, sep = ",", names = col_names)
    
    # find and remove nans
    data1= data1.replace(' ?', np.NaN)
    data1 = data1.dropna()
     
    data2 = data1.drop(labels='income-label', axis=1)
    label = data1.iloc[:,-1:]
    
    # transform categorical to numerical using one hot encoding
    categorical_feat = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
    data3 = pd.get_dummies(data2, columns = categorical_feat)
    
    # label
    label = label.replace(' <=50K', 0)
    label = label.replace(' >50K', 1)
    
    label = label['income-label'].values

    return data3, label


def breast_cancer():

    file = get_filePath('breast_cancer_wisconsin.csv')
    data1 = pd.read_csv(file)
    
    # find and remove nans
    # data1= data1.replace(' ?', np.NaN)
    # data1 = data1.dropna()
    
    # remove not interesting columns
    data2 = data1.iloc[:, :-1]
    data2 = data2.drop(['id', 'diagnosis'], axis=1)

    # label
    label = data1['diagnosis']
    label = label.replace('B', 0)
    label = label.replace('M', 1)
    
    label = label.values

    return data2, label


def diabetes():

    file = get_filePath("diabetes.csv")
    data1 = pd.read_csv(file)
    # find and remove nans
    # data1= data1.replace(' ?', np.NaN)
    # data1 = data1.dropna()
    
    # remove not interesting columns
    data2 = data1.drop(labels='Outcome', axis=1)

    # label
    label = data1['Outcome']
    label = label.values

    return data2, label


def australian_credit():
    
    file = get_filePath("australian.dat")
    col_names = ['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9', 'a10', 'a11', 'a12', 'a13', 'a14', 'a15']
    data1 =  pd.read_table(file, sep = " ", names = col_names)
    
    data2 = data1.drop(labels='a15', axis=1)
    label = data1.iloc[:,-1:]
    
    # transform categorical to binary using one hot encoding
    categorical_feat = ['a4', 'a5', 'a6', 'a8', 'a9', 'a11', 'a12']
    data3 = pd.get_dummies(data2, columns = categorical_feat)
    
    label = label['a15'].values
    
    return data3, label


def mammo():

    file = get_filePath("mammo.data")
    col_names = ['a1', 'a2', 'a3', 'a4', 'a5', 'a6']
    data1 =  pd.read_table(file, sep = ",", names = col_names, na_values='?')
    
    # find and remove nans
    data1= data1.replace('?', np.NaN)
    data1 = data1.dropna()
     
    data2 = data1.drop(labels='a6', axis=1)
    label = data1.iloc[:,-1:]
    label = label['a6'].values
    
    # # transform categorical to numerical using one hot encoding
    # categorical_feat = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
    # data3 = pd.get_dummies(data2, columns = categorical_feat)
    
    # # label
    # label = label.replace(' <=50K', 0)
    # label = label.replace(' >50K', 1)
    
    # label = label['income-label'].values
    
    return data2, label


def heart_disease():
    
    file = get_filePath("cleveland_heart_disease.data")
    col_names = ['age', 'sex', 'cp', 'trestbps', 'chols', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'prediction']
    data1 =  pd.read_table(file, sep = "," , decimal='.', names = col_names, na_values='?')
    
    # find and remove nans
    # data1 = data1.dropna()
    
    # replace nans
    data1 = impute_missing(data1)

    data2 = data1.drop(labels='prediction', axis=1)
    
    # transform categorical to numerical using one hot encoding
    data3 = pd.get_dummies(data2, columns=['cp', 'restecg', 'slope'])
    # data3 =data2
    
    #label
    label = data1.iloc[:,-1:]
    label2 = label.replace([1,2,3,4], 1)
    label2 = label2['prediction'].values
    
    return data3, label2


def cervical_cancer():
    
    file = get_filePath("cervicalCancer.xlsx")
    data1 =  pd.read_excel(file)
    
    # find and remove nans
    # data1 = data1.dropna()
    
    # remove samples with more than 40% of missing values
    nulls = data1.isnull().sum(axis=1)/data1.shape[1]
    toremove = np.where(nulls>0.4)[0]
    data1 = data1.drop(toremove,axis=0)
    
    # replace nans
    data1 = impute_missing(data1)
    
    data2 = data1.drop(labels=['Hinselmann', 'Schiller', 'Citology','Biopsy'], axis=1)

    # transform categorical to numerical using one hot encoding
    # data3 = pd.get_dummies(data2, columns=['cp', 'restecg', 'slope'])
    data3 =data2
    #label
    label = data1.iloc[:,-1:]
    label = label['Biopsy'].values
    
    return data3, label


def wine():
    
    # 3 classes
    dataset = load_wine()
    data1 = pd.DataFrame(data=dataset.data , columns = dataset.feature_names)
    
    label = dataset.target
    
    return data1, label


def wine_quality():
    
    file = get_filePath('white_wine_quality.csv')
    data1 = pd.read_csv(file, sep=';', decimal='.')
    data2 = data1.drop(labels='quality', axis=1)
    
    #label
    label = data1.iloc[:,-1:]
    label = label['quality'].values
    
    return data2, label


def thoracic():
    
    file = get_filePath('thoracic_surgery.csv')
    data1 = pd.read_csv(file, sep=',', decimal='.')
    
    # replace trues ands falses
    data1 = data1.replace('F', 0)
    data1 = data1.replace('T', 1)
    # PRE14
    data1 = data1.replace('OC11', 1)
    data1 = data1.replace('OC12', 2)
    data1 = data1.replace('OC13', 3)
    data1 = data1.replace('OC14', 4)
    # PRE6
    data1 = data1.replace('PRZ0', 0)
    data1 = data1.replace('PRZ1', 1)
    data1 = data1.replace('PRZ2', 2)
    # DGN
    data1 = data1.replace('DGN1', 1)
    data1 = data1.replace('DGN2', 2)
    data1 = data1.replace('DGN3', 3)
    data1 = data1.replace('DGN4', 4)
    data1 = data1.replace('DGN5', 5)
    data1 = data1.replace('DGN6', 6)
    data1 = data1.replace('DGN8', 8)
    
    data2 = data1.drop(labels='Risk1Yr', axis=1)
    # PRE6 and DGN
    # transform categorical to numerical using one hot encoding
    categorical_feat = ['DGN', 'PRE6']
    data3 = pd.get_dummies(data2, columns = categorical_feat)
    
    #label
    label = data1.iloc[:,-1:]
    label = label['Risk1Yr'].values
    
    return data3, label


def heart_disease_multi(): # heart disease multiclass
    
    file = get_filePath("cleveland_heart_disease.data")
    col_names = ['age', 'sex', 'cp', 'trestbps', 'chols', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'prediction']
    data1 =  pd.read_table(file, sep = "," , decimal='.', names = col_names, na_values='?')
    
    # find and remove nans
    data1 = data1.dropna()

    data2 = data1.drop(labels='prediction', axis=1)
    
    # transform categorical to numerical using one hot encoding
    data3 = pd.get_dummies(data2, columns=['cp', 'restecg', 'slope'])
    
    #label
    label = data1.iloc[:,-1:]
    label = label['prediction'].values
    label = label.replace('F', 0)
    label = label.replace('T', 1)
    
    return data3, label


def car_evaluation():
    
    file = get_filePath("car_evaluation.data")
    col_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
    data1 =  pd.read_table(file, sep = "," , decimal='.', names = col_names, na_values='?')
    
    # find and remove nans
    data1 = data1.dropna()

    data2 = data1.drop(labels='class', axis=1)
    
    # transform categorical to numerical using one hot encoding
    data3 = pd.get_dummies(data2, columns=['buying', 'maint', 'lug_boot', 'safety'])
    data3 = data3.replace('5more', 5)
    data3 = data3.replace('more', 5)
    data3 = data3.astype('int32')
    
    #label
    label = data1.iloc[:,-1:]
    label = label.replace('unacc', 0)
    label = label.replace('acc', 1)
    label = label.replace('good', 2)
    label = label.replace('vgood', 3)
    label = label['class'].values
    
    return data3, label


def cardiotocography(label_type):
    
    file = get_filePath('cardiotocography.csv')
    data1 = pd.read_csv(file)
    
    data2 = data1.drop(labels=['CLASS', 'NSP'], axis=1)
    
    labels = data1.iloc[:,-2:]
    
    #  mp : morphologic pattern
    if label_type == 'mp':
        label2 = labels['CLASS'].values
        label2 = label2-1
    # NSP : fetal state class code 
    elif label_type == 'nsp':
        label2 = labels['NSP'].values
        label2 = label2-1
    return data2, label2


def forest_fires():

    file = get_filePath('forestfires.csv')
    data1 = pd.read_csv(file)
    
    # find and remove nans
    # data1= data1.replace(' ?', np.NaN)
    # data1 = data1.dropna()
    
    data2 = data1.drop(labels='area', axis=1)
    
    # transform categorical to numerical using one hot encoding
    data3 = pd.get_dummies(data2, columns=['month', 'day'])
    
    # label
    label = data1['area']
    
    label = label.values

    return data3, label

    
def students(label_type):
    
    def data_process(data1):
    
        dummy_columns = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 
                                           'reason', 'guardian', 'schoolsup', 'famsup', 'paid', 'activities', 
                                           'nursery', 'higher', 'internet', 'romantic']
        
        data2 = data1.drop(labels=['G1', 'G2', 'G3'], axis=1)
        
        # transform categorical to numerical using one hot encoding
        data3 = pd.get_dummies(data2, columns=dummy_columns)
        
        # label
        label = data1['G3']
        label = label.values
        
        return data3, label
    
    
    #  pt : portugues
    if label_type == 'pt':
        file = get_filePath('student-por.csv')
        data = pd.read_csv(file, sep=';')
        dataf, labelf = data_process(data)
        
    # mat : matematica
    elif label_type == 'mat':   
        file = get_filePath('student-mat.csv')
        data = pd.read_csv(file, sep=';')
        dataf, labelf = data_process(data)

    return dataf, labelf


def friend():
    
    file = get_filePath('Friend.csv')
    data1 = pd.read_csv(file, sep = ';', decimal=',')
    
    # find and remove nans
    data1 = data1.dropna()

    data2 = data1.drop(labels='vo2_ml_kg_min', axis=1)
    
    # label
    label = data1['vo2_ml_kg_min']
    label = label.values
    
    return data2, label
    

#------------------------------------------------------------------------------------#

## Main function to import the desired dataset

def dataset_info(dataset_name):
    
    if dataset_name == 'lookafterrisk':
        data, label = lookAfterRisk()
        
    elif dataset_name == 'adult_income':
        data, label = adult_income()
        
    elif dataset_name == 'australian_credit':
        data, label = australian_credit()
        
    elif dataset_name == 'breast_cancer':
        data, label = breast_cancer()
        
    elif dataset_name == 'diabetes':
        data, label = diabetes()
        
    elif dataset_name == 'thoracic':
        data, label = thoracic()
        
    elif dataset_name == 'cervical_cancer':
        data, label = cervical_cancer()
        
    elif dataset_name == 'heart_disease':
        data, label = heart_disease()
        
    elif dataset_name == 'heart_disease_multi':
        data, label = heart_disease_multi()
        
    elif dataset_name == 'wine':
        data, label = wine()
        
    elif dataset_name == 'cardiotcg_mp':
        data, label = cardiotocography('mp')
        
    elif dataset_name == 'cardiotcg_nsp':
        data, label = cardiotocography('nsp')
        
    elif dataset_name == 'car_evaluation':
        data, label = car_evaluation()
    
    elif dataset_name == 'forest_fires':
        data, label = forest_fires()
    
    elif dataset_name == 'students_mat':
        data, label = students('mat')
        
    elif dataset_name == 'students_pt':
        data, label = students('pt')
    
    elif dataset_name == 'friend':
        data, label = friend()
    
    elif dataset_name == 'mammo':
        data, label = mammo()
        
    return data, label
        


    
    


