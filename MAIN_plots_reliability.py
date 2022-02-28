# -*- coding: utf-8 -*-
"""
MAIN FILE FOR THE PLOTTING OF RELIABILITY CURVES

@author: Francisco Valente (paulo.francisco.valente@gmail.com)
2021
"""

## Import packages

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os

#------------------------------------------------------------------------------------#

## Change the font type (required for paper submission)

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

#------------------------------------------------------------------------------------#

# LOAD DATA

def load_file(file_name):
    
    file_loc = '\\results_files\\'
    file = file_loc + file_name
    path = os.getcwd()+file
    
    my_file = np.load(path)*100

    return my_file
    
# Mammo dataset
mammo_m = load_file('mammo_means.npy')
mammo_ci = load_file('mammo_cis.npy')
# Heart dataset
heart_m = load_file('heart_means.npy')
heart_ci = load_file('heart_cis.npy')
# Breast dataset
breast_m = load_file('breast_means2.npy')
breast_ci = load_file('breast_cis2.npy')

# mammo = np.load('ratios_mammo.npy')*100
# heart = np.load('ratios_heart.npy')*100
# breast = np.load('ratios_breast.npy')*100


# PLOT RELIABILITY CURVES

bins = np.linspace(0, 1, 11)

# intervals
bins = ['no', '0-10', '10-20', '20-30', '30-40', '40-50','50-60', '60-70', '70-80', '80-90', '90-100']

plt.figure()
plt.scatter(bins, heart_m, label=r'Heart, AUC: 0.89 , BA: 0.81')
plt.errorbar(bins, heart_m, yerr=heart_ci, linestyle='--', alpha=0.5)
plt.scatter(bins, breast_m, label='Breast, AUC: 0.99 , BA: 0.95')
plt.errorbar(bins, breast_m, yerr=breast_ci, linestyle='--', alpha=0.5)
plt.scatter(bins, mammo_m, label='Mammo, AUC: 0.90 , BA: 0.83')
plt.errorbar(bins, mammo_m, yerr=mammo_ci, linestyle='--', alpha=0.5)
plt.legend(loc="best")
plt.ylabel('Ratio of Misclassifications (%)',fontweight='bold')
plt.xlabel('Reliability Estimation (%)', fontweight='bold')
plt.show()