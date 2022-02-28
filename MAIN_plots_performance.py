# -*- coding: utf-8 -*-
"""
MAIN FILE FOR PLOTTING OF MODELS' PERFORMANCE

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

## RESULTS FOR THE ICML-IMLH PAPER

# Plot performance (mean vs weighted mean vs personalized weighted mean) 

# auc heart disease
hd =[0.83, 0.85, 0.90]
# auc breast cancer
bc =[0.98, 0.99, 0.99]
# auc mammo
mm =[0.84, 0.86, 0.92]

all_aucs = [hd, mm, bc]

x_label = ['Non-weighted mean', 'Non personalized weighted mean', 'Personalized weight mean']
colours = ['b', 'r', 'g']
y_label = ['heart', 'mammo', 'breast']

x = np.arange(start=1, stop=len(x_label), step=1)

matplotlib.pyplot.figure()

for i in range(0, len(x_label)):
    
    aucs = all_aucs[i]
    matplotlib.pyplot.plot(x_label, aucs, '-o', label=y_label[i], color=colours[i])
    
matplotlib.pyplot.legend(loc="best")
matplotlib.pyplot.show()
matplotlib.pyplot.xlabel('Averaging method')
matplotlib.pyplot.ylabel('AUC')
matplotlib.pyplot.ylim(0.80, 1)

#------------------------------------------------------------------------------------#

## RESULTS FOR THE EMBC PAPER

# Plot performance (mean vs personalized weighted mean) 

# auc heart disease
hd_wmean = [0.82, 0.85, 0.89, 0.90, 0.90]
hd_mean = [0.76, 0.80, 0.85, 0.87, 0.88]
# auc breast cancer
bc_wmean = [0.97, 0.98, 0.99, 0.99, 0.99]
bc_mean = [0.94, 0.97, 0.98, 0.99, 0.99]
# auc diabetes
db_wmean = [0.70, 0.74, 0.79, 0.80, 0.80]
db_mean = [0.64, 0.67, 0.72, 0.74, 0.76]

all_aucs = [hd_wmean, hd_mean, bc_wmean, bc_mean, db_wmean, db_mean]

names = ['Heart weighted mean', 'Heart mean', 'Breast weighted mean', 'Breast mean', 'Diabetes weighted mean', 'Diabetes mean']

# colours = ['lightsalmon', 'salmon', 'lightblue', 'steelblue', 'lightgreen', 'mediumseagreen']
colours = ['b', 'b', 'r', 'r', 'g', 'g']

rules_nr = ['3', '5', '10', '15', '20']
x = np.arange(start=1, stop=len(rules_nr), step=1)

matplotlib.pyplot.figure()

for i in range(0, len(names)):
    
    aucs = all_aucs[i]
    if (i % 2) != 0:
        matplotlib.pyplot.plot(rules_nr, aucs, '--o', label=names[i], color=colours[i])
    else:
        matplotlib.pyplot.plot(rules_nr, aucs, '-o', label=names[i], color=colours[i])
    
matplotlib.pyplot.legend(loc="best")
matplotlib.pyplot.show()
matplotlib.pyplot.xlabel('Number of rules')
matplotlib.pyplot.ylabel('AUC')
matplotlib.pyplot.ylim(0.5, 1)

