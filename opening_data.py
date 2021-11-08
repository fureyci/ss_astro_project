# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 12:41:50 2021

@author: fureyci

Opening data from Leka and Park, to be used for ensemble model.

source: Leka, K. D.; Park, Sung-Hong, 2019,
"A Comparison of Flare Forecasting Methods II: Data and
Supporting Code", https://doi.org/10.7910/DVN/HYP74O, Harvard
Dataverse, V1, UNF:6:yz1noMojlzL7SZM+9flXhQ== [fileUNF]

"""
# Import modules.
import os
import pandas as pd
import numpy as np

def load_benchmark_data():
    """Loads forecasts used in benchmark.    

    Returns
    -------
    events : `pandas.core.frame.DataFrame`
        The events over the testing interval.
    model_csv_list : list of `pandas.core.frame.DataFrame`
        List of forecasts of each model.
    model_names_only : `list`
        The names of each model.
    
    """
    # The directory of the data.
    DATA_DIR = './CSV/'
    
    forecast_dir = os.listdir(DATA_DIR)  # Every file in the directory.

    # Observed C and M events (from 1996 - 2017)
    # These are given as .txt files, and are located at index 7 and 8.
    all_C_events = pd.read_csv(DATA_DIR + forecast_dir[7],
                               names=["Date", "Event"])  # C events.

    all_M_events = pd.read_csv(DATA_DIR + forecast_dir[8],
                               names=["Date", "Event"])  # M evenets.

    # Get the events from 2016 onwards -- the ones that were examined
    # in Leka 2019. First find the index of data frame where the year
    # becomes 2016 for C and M class events, respectively.
    C_events_from_2016_index = np.where(all_C_events["Date"].str.contains("2016"))[0][0]
    M_events_from_2016_index = np.where(all_M_events["Date"].str.contains("2016"))[0][0]

    # Locate the events from 2016 onwards in the original dfs.
    C_events = all_C_events.iloc[C_events_from_2016_index:].loc[:, "Event"].values
    M_events = all_M_events.iloc[M_events_from_2016_index:].loc[:, "Event"].values
    
    events = pd.DataFrame({"Date": model_csv_list[0]["VALID_DATE"],
                           "C": C_events,
                           "M" : M_events})

    events.set_index("Date", inplace=True)
    
    # =================================================================
    # Load the models.

    # They are the CSV files in directory, so to differentiate between
    # the .txt files, load the files ending with ".csv"
    model_name_list = [x for x in forecast_dir if x.endswith(".csv")]

    # Since each file ends with "_release.csv," can get rid of it to
    # have model
    model_names_only = [x[:-12] for x in model_name_list]

    model_csv_list = []  # List to store models.

    # Dataframes for each individual model, stored in all_models.
    for model in model_name_list:
        df = pd.read_csv(DATA_DIR+model)  # Load the model.
        model_csv_list.append(df)  # Store the model.

    return events, model_csv_list, model_names_only
