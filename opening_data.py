# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 12:41:50 2021

@author: Ciaran
"""

# import modules
import os # to access directories
import pandas as pd # for data analysis
import numpy as np
import matplotlib.pyplot as plt # plotting

# Laura's code from:
# (https://github.com/hayesla/flare_forecast_proj/blob/main/forecast_tests/metric_utils.py)
from metric_utils import plot_reliability_curve, plot_roc_curve


# the directory of the data
data_dir = './Leka2019_files/Data/CSV/' 

# downloaded from:
#
# source: Leka, K. D.; Park, Sung-Hong, 2019, 
#         "A Comparison of Flare Forecasting Methods II: Data and Supporting Code", 
#          https://doi.org/10.7910/DVN/HYP74O, Harvard Dataverse, V1, 
#          UNF:6:yz1noMojlzL7SZM+9flXhQ== [fileUNF]
# link: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/HYP74O

forecast_dir = os.listdir(data_dir) # every file in the directory

# observed C and M events (from 1996 - 2017!!)
#
# These are given as .txt files, and are located at index 7 and 8
all_C_events = pd.read_csv(data_dir + forecast_dir[7], names=['Date', 'Event']) # C events
all_M_events = pd.read_csv(data_dir + forecast_dir[8], names=['Date', 'Event']) # M evenets

# lets get the events from 2016 onwards -- the ones that were examined in Leka 2019
#
# first lets find the index of data frame where the year becomes 2016 for 
# C and M class events, respectively.
C_events_from_2016_index = np.where(all_C_events['Date'].str.contains('2016'))[0][0]
M_events_from_2016_index = np.where(all_M_events['Date'].str.contains('2016'))[0][0]

# now lets locate the events from 2016 onwards in the original dfs
C_events = all_C_events.iloc[C_events_from_2016_index:].loc[:, "Event"].values
M_events = all_M_events.iloc[M_events_from_2016_index:].loc[:, "Event"].values

#
# ==============================================================
#
# Now lets get the models
#
# They are the CSV files in directory, so to differentiate between the .txt files,
# load the files enidng with ".csv"
models = [x for x in forecast_dir if x.endswith(".csv")]

all_models = [] # list to store models

# Dataframes for each individual model, stored in all_models
for file in models:
    df = pd.read_csv(data_dir+file) # load the model
    all_models.append(df) # store the model

# ==============================================================
#
# Now try plotting reliability and ROC curves
#

# ROC function I made
#
# def get_ROC(probabilites, observed, threshold):
#     '''
#     Funtion that will compute values for ROC curve
    
#     Inputs:
#     probabilites: list of model predictions
#     observed: list of observed events
#     threshold: probability threshold (Pth) value, above which a probabilistic
#                prediction will be converted to a 'yes', and below, a 'no'
#
#     Outputs:
#     x: proability of false detection (x value of ROC curve)
#     y: probability of detection      (y value of ROC curve)
#     '''
#  
#     #               predicted
#     # observed    event     no event
#     #  event       TP         FN
#     # no event     FP         TN
#    
#     TP = [] # true positive (hit)
#     FN = [] # false negative (miss)
#    
#     FP = [] # false positive (false alarm)
#     TN = [] # true negative (correct negative)
#    
#    
#     # see if values are greater than Pth
#     # if yes: convert to 1
#     # if not: convert to 0
#     yesno_vals = (probabilites > threshold).astype(int)
#    
#     for i in range(len(yesno_vals)):
#         if yesno_vals[i] == observed[i] and yesno_vals[i] == 1:
#             TP.append(1)
#         elif yesno_vals[i] == observed[i] and yesno_vals[i] == 0:
#             TN.append(1)
#         elif yesno_vals[i] != observed[i] and yesno_vals[i] == 1:
#             FP.append(1)
#         elif yesno_vals[i] != observed[i] and yesno_vals[i] == 0:
#             FN.append(1)
#    
#       
#     x = len(FP) / float(len(FP) + len(TN))
#     y = len(TP) / float(len(TP) + len(FN))
#    
#     # print(x,y,'\n')
#    
#     return x, y
        
        
bins = 20 # number of bins for reliability curve, same as Leka 2019

# now loop through all models and plot their ROC and reliability curves

for j in range(len(models)):
    
    test = all_models[j] # the test model
 
    test_M = test.loc[:,"M1+(0-24hr)"].values # probabilities given by model
    test_M[test_M < 0] = 0  # map any negative values to 0 
                            # sklearn.calibration.calibration_curve() (used in
                            # metric_utils.plot_reliability_curve()) doesnt accept -
                            # ve vals and dont want to normalise to interval [0,1]
    
    # try:
    # plot_reliability_curve(C_events, test_C, n_bins=bins, 
    #                        title="Reliability plot for {} model for C1+(0-24hr)".format(models[j][:-12]))
        
    # except:
    #     print(models[j]," Didnt work\n")
    

    plot_roc_curve(M_events, test_M, 
                    title="ROC plot for {} model for M1+(0-24hr)".format(models[j][:-12]))
    
