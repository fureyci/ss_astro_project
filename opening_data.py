# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 12:41:50 2021

@author: Ciaran

First file, messing around with data, preparing methods and plots to 
potentially use in future scripts.
"""

# import modules
import os # to access directories
import pandas as pd # for data analysis
import numpy as np
import matplotlib.pyplot as plt # plotting

# Laura's code from:
# (https://github.com/hayesla/flare_forecast_proj/blob/main/forecast_tests/metric_utils.py)
from metric_utils import (plot_reliability_curve, plot_roc_curve,
                          calculate_bss, calculate_tss_threshold)

# the directory of the data
DATA_DIR = './CSV/' 

# probability threshold, for calculating TSS, vary for different tss results
P_TH = 0.5

# number of bins for reliability curve, same as Leka 2019
BINS = 20

# ==============================================================
#
# DATA
#
# downloaded from:
# source: Leka, K. D.; Park, Sung-Hong, 2019,
# "A Comparison of Flare Forecasting Methods II: Data and Supporting Code",
# https://doi.org/10.7910/DVN/HYP74O, Harvard Dataverse, V1,
# UNF:6:yz1noMojlzL7SZM+9flXhQ== [fileUNF]
# link: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DV
# N/HYP74O

forecast_dir = os.listdir(DATA_DIR) # every file in the directory

# Observed C and M events (from 1996 - 2017!!)
# These are given as .txt files, and are located at index 7 and 8
all_C_events = pd.read_csv(DATA_DIR + forecast_dir[7],
                           names=['Date', 'Event']) # C events

all_M_events = pd.read_csv(DATA_DIR + forecast_dir[8],
                           names=['Date', 'Event']) # M evenets

# Let's get the events from 2016 onwards -- the ones that were examined in
# Leka 2019. First lets find the index of data frame where the year becomes
# 2016 for C and M class events, respectively.
C_events_from_2016_index = np.where(all_C_events['Date'].str.contains('2016'))[0][0]
M_events_from_2016_index = np.where(all_M_events['Date'].str.contains('2016'))[0][0]

# now lets locate the events from 2016 onwards in the original dfs
C_events = all_C_events.iloc[C_events_from_2016_index:].loc[:, "Event"].values
M_events = all_M_events.iloc[M_events_from_2016_index:].loc[:, "Event"].values


# ==============================================================
# Now let's get the models

# They are the CSV files in directory, so to differentiate between the .txt
# files, load the files ending with ".csv"
model_name_list = [x for x in forecast_dir if x.endswith(".csv")]

# Since each file ends with "_release.csv," can get rid of it to have model
# name only.
model_names_only = [x[:-12] for x in model_name_list]

model_csv_list = [] # list to store models

# Dataframes for each individual model, stored in all_models
for model in model_name_list:
    df = pd.read_csv(DATA_DIR+model) # load the model
    model_csv_list.append(df) # store the model

# ==============================================================
#
# Now try plotting reliability and ROC curves for M1+(0-24hr)
#
# loop through all models and plot their ROC and reliability curves

bss_list = [] # list to store Brier skill score (bss)
tss_list = [] # list to store True skill score (tss)

for i, model in enumerate(model_name_list):

    test = model_csv_list[i] # the test model

    test_M = test.loc[:,"M1+(0-24hr)"].values # model forecast

    # map any negative values to 0. sklearn.calibration.calibration_curve(),
    # which is used in metric_utils.plot_reliability_curve(), doesn't accept
    # negative values and, don't want to normalise to interval [0,1]
    test_M[test_M < 0] = 0

    # plot the reliability curve for current model
    plot_reliability_curve(
        M_events,
        test_M,
        n_bins=BINS,
        title="Reliability plot for {} model for "
              "M1+(0-24hr)".format(model_names_only[i])
        )

    # plot roc curve for current model
    plot_roc_curve(M_events,
                    test_M,
                    title="ROC plot for {} model for "
                    "M1+(0-24hr)".format(model_names_only[i])
                    )

    # store bss of current model
    bss_list.append(calculate_bss(M_events, test_M))

    # store tss of current model
    tss_list.append(calculate_tss_threshold(M_events, test_M, P_TH))

# ==============================================================
#
# Let's make a simple ensemble that will calculate the average of each forecast
# that is, an equal weighted linear combination.
#
# Firstly, let's make a function that will combine the forecasts of each model
# into a single dataframe. This will allow us to calculate linear combination
# across the rows.


def df_of_models(models, model_names, flare_class=None):
    """
    Function that will generate a pandas dataframe containing the solar flare
    forecasts of type "flare_class". Expects each model in "models" to be of
    the same size, and to have forecasts over the same range of days.

    Parameters
    ----------
    models : `list of pandas.dataframe objects`
        List of models.
    model_names : `list`
        List of model names to make header for each column of output dataframe.
        Must correspond to models in "models"
    flare_class : `str`
        The flare class to examine. The default is None.

        One of:

        | "C-only" : C class only.
        | "C1+"    : C class exceedence, 0hr latency, 24hr validity.
        | "M-only" : M class only.
        | "M1+"    : M class exceedence, 0hr latency, 24hr validity.

    Returns
    -------
    df_out : `pandas.dataframe`
        pandas dataframe whose columns correspond to the flare forecast of
        type "flare_class" for each model in models, and each row corresponds
        to a date.

    """

    # dates of each forecast, assumed the same for each model.
    dates = models[0].loc[:,'VALID_DATE'].values

    # Dictionary that will store the forecasts for each model.
    # A dictionary can be passed into pd.DataFrame, where the keys represent
    # the column headers, and the values represent the column values.
    df_dict = {}

    df_dict['Date'] = dates # store the dates in the dataframe

    # now need to load the forecast, depending on flare_class parameter.
    for model_name, model_forecast in zip(model_names, models):

        if flare_class == "C-only":
            desired_forecast = model_forecast.loc[:,"C-only(0-24hr)"].values

        elif flare_class == "C1+":
            desired_forecast = model_forecast.loc[:,"C1+(0-24hr)"].values

        elif flare_class == "M-only":
            desired_forecast = model_forecast.loc[:,"M-only(0-24hr)"].values

        elif flare_class == "M1+":
            desired_forecast = model_forecast.loc[:,"M1+(0-24hr)"].values

        else:
            # in this case, an invalid value has been selected.
            raise ValueError("'flare_class' must be one of: 'C-only', 'C1+', "
                              "'M-only', or 'M1+'.")

        # store the model name and its forecast into the dataframe.
        df_dict[model_name] = desired_forecast

    df_out = pd.DataFrame(df_dict) # create dataframe
    
    df_out = df_out.set_index('Date') # set dates as indices
    
    return df_out

# Now that we have the dataframe containing all models, lets use that in a
# function that will calculate a simple average.

def ensemble_average(models, model_names, flare_class=None):
    """
    Function that will compute a simple ensemble average for different space
    weather forecasting models.

    Parameters
    ----------
    models : `list of pandas.dataframe`
        List of models.
    model_names : `list`
        List of model names to make header for each column of output dataframe.
        Must correspond to models in "models"
    flare_class : `str`
        The flare class to examine. The default is None.

        One of:

        | "C-only" : C class, levels bounded on both upper and lower thresholds.
        | "C1+"    : C class exceedence, 0hr latency, 24hr validity.
        | "M-only" : M class, levels bounded on both upper and lower thresholds.
        | "M1+"    : M class exceedence, 0hr latency, 24hr validity.


    Returns
    -------
    ensemble_av : `np.array`
        Average prediction of all models.

    """
    # generate df of models
    all_models = df_of_models(models, model_names, flare_class)

    headers = list(all_models) # the column headers of the df

    all_models[all_models < 0] = 0  # map any negative values to 0
    
    # now compute average of each forecast
    ensemble_av = all_models.sum(axis=1) / len(headers)

    return ensemble_av


# calculate ensemble average using function that has just been defined.
en_av = ensemble_average(model_csv_list, model_names_only, flare_class="M1+")

# let's examine its roc and reliability curves.
plot_roc_curve(M_events,
               en_av,
               title="ROC plot for Ensemble Average for M1+(0-24hr)")

plot_reliability_curve(M_events,
                       en_av,
                       n_bins=BINS,
                       title="Reliability plot for ensemble average for "
                             "M1+(0-24hr)")

# Now let's visualise the bss and tss for each model, including ensemble.
# Firstly, append bss, tss and name to respective lists for ensemble average/
bss_list.append(calculate_bss(M_events, en_av))
tss_list.append(calculate_tss_threshold(M_events, en_av, P_TH))
model_names_only.append("Ensemble Average")

fig = plt.figure(figsize=(20,5)) # create figure

x_ticks = np.arange(len(model_names_only)) # x tick values
BAR_WIDTH = 0.4 # width of bars

score_ax = fig.add_subplot(111) # create axis

# plot the bss values
bss_bar = score_ax.bar(x_ticks - BAR_WIDTH/2, bss_list, BAR_WIDTH,
                        edgecolor='k', linewidth=0.5, label="BSS",
                        color='yellowgreen')
# plot the tss values
tss_bar = score_ax.bar(x_ticks + BAR_WIDTH/2, tss_list, BAR_WIDTH,
                        edgecolor='k', linewidth=0.5, label="TSS",
                        color='khaki')
# draw x axis
score_ax.axhline(0, c='k', lw=0.6)

score_ax.set_ylabel("Score")
score_ax.set_ylim(-0.5, 1.03)
score_ax.set_xlim(-BAR_WIDTH-0.2, len(x_ticks)+BAR_WIDTH-0.8)
score_ax.set_xticks(x_ticks)
score_ax.set_xticklabels(model_names_only, rotation=20, ha="right")

score_ax.bar_label(bss_bar, fmt="%.2f", fontsize=8, label_type="center")
score_ax.bar_label(tss_bar, fmt="%.2f", fontsize=8, label_type="center")

score_ax.legend()
plt.tight_layout()
plt.show()
