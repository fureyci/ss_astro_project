# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 12:41:50 2021

@author: fureyci

Opening data from Leka and Park, to be used for ensemble model.

source: Leka, K. D.; Park, Sung-Hong, 2019,
"A Comparison of Flare Forecasting Methods II: Data and
Supporting Code", https://doi.org/10.7910/DVN/HYP74O, Harvard
Dataverse, V1, UNF:6:yz1noMojlzL7SZM+9flXhQ== [fileUNF]

Also produces tables for latex that show the metrics and
bootstrapped uncertainties for each model when this script
is run.
"""
# Import modules.
import metric_utils
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
    
    # =================================================================
    # DATA
    
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

    real_dates = model_csv_list[0]["VALID_DATE"]

    for model in model_csv_list:
        model.loc[:,"VALID_DATE"] = real_dates

    events = pd.DataFrame({"Date": model_csv_list[0]["VALID_DATE"],
                           "C": C_events,
                           "M" : M_events})

    events.set_index("Date", inplace=True)

    return events, model_csv_list, model_names_only

def model_bootstrap(forecast, events, metric):
    """Estimate the uncertainty of a performance metric by performing
    bootstrapping with replacement on the forecasts and events
    for an individual model.

    Parameters
    ----------
    forecast : list
    events : list
    metric : str

    Returns
    -------
    sdev : float
        Standard deviation metric

    """
    valid_metrics = ["tss", "bss", "ets", "apss", "fb", "auc"]

    if metric not in valid_metrics:
        raise TypeError("Invalid metric entered. Chose one of "
                        "'tss', 'bss', 'ets', 'apss', or 'fb'.")


    iterations = 1000 # Number of bootstrap samples to draw.

    # Generate numpy array to store the confidence intervals for
    # each ensemble.
    # sdev = np.zeros(len(ens_list))

    metric_list = []  # List to store bootstrapped metrics.
    for j in range(iterations):
        indices = np.random.choice(len(forecast), forecast.shape)
        bootstrap_forecast = forecast[indices]
        bootstrap_events = events[indices]

        if metric == "tss":
            boot_metric = metric_utils.calculate_tss_threshold(
                                                    bootstrap_events,
                                                    bootstrap_forecast,
                                                    0.5
                                                    )

        elif metric == "bss":
            boot_metric = metric_utils.calculate_bss(bootstrap_events,
                                                      bootstrap_forecast
                                                      )

        elif metric == "fb":
            boot_metric = metric_utils.calculate_fb_threshold(
                                                    bootstrap_events,
                                                    bootstrap_forecast,
                                                    0.5
                                                    )

        elif metric == "ets":
            boot_metric = metric_utils.calculate_ets_threshold(
                                                    bootstrap_events,
                                                    bootstrap_forecast,
                                                    0.5
                                                    )

        elif metric == "apss":
            boot_metric = metric_utils.calculate_apss_threshold(
                                                    bootstrap_events,
                                                    bootstrap_forecast,
                                                    0.5
                                                    )

        elif metric == "auc":
            boot_metric = metric_utils.calculate_roc_area(bootstrap_events,
                                                          bootstrap_forecast)

        metric_list.append(boot_metric)

    sdev = np.std(metric_list, ddof=1)

    return sdev

if __name__ == "__main__":
    """Load models and tabulate the metrics for each model in LaTeX
    form, for thesis.

    """
    events, model_csv_list, models = load_benchmark_data()
    # Metrics for M-class forecasts.
    # Load forecasts
    m_forecasts = np.array([model["M1+(0-24hr)"] for model in model_csv_list])
    m_forecasts[m_forecasts < 0] = 0

    # Calculate metrics and bootstrapped uncertainties.
    models_tss_M = [metric_utils.calculate_tss_threshold(events["M"],
                                                          forecast,
                                                          0.5)
                    for forecast in m_forecasts]

    models_tss_M_err = [model_bootstrap(forecast, events["M"], "tss")
                        for forecast in m_forecasts]

    models_apss_M = [metric_utils.calculate_apss_threshold(events["M"],
                                                          forecast,
                                                          0.5)
                    for forecast in m_forecasts]

    models_apss_M_err = [model_bootstrap(forecast, events["M"], "apss")
                        for forecast in m_forecasts]

    models_ets_M = [metric_utils.calculate_ets_threshold(events["M"],
                                                          forecast,
                                                          0.5)
                    for forecast in m_forecasts]

    models_ets_M_err = [model_bootstrap(forecast, events["M"], "ets")
                        for forecast in m_forecasts]

    models_bss_M = [metric_utils.calculate_bss(events["M"],
                                                forecast)
                    for forecast in m_forecasts]

    models_bss_M_err = [model_bootstrap(forecast, events["M"], "bss")
                        for forecast in m_forecasts]

    models_roc_M = [metric_utils.calculate_roc_area(events["M"],
                                                    forecast)
                    for forecast in m_forecasts]

    models_roc_M_err = [model_bootstrap(forecast, events["M"], "auc")
                        for forecast in m_forecasts]

    # Lists to store metrics of each model in string format.
    tss_column_M = []
    apss_column_M = []
    ets_column_M = []
    bss_column_M = []
    roc_column_M = []

    # Store the scores and uncertainties as strings.
    for i in range(len(models_tss_M)):
        tss_column_M.append("{:.2} ({:.2})".format(models_tss_M[i], models_tss_M_err[i]))
        apss_column_M.append("{:.2} ({:.2})".format(models_apss_M[i], models_apss_M_err[i]))
        ets_column_M.append("{:.2} ({:.2})".format(models_ets_M[i], models_ets_M_err[i]))
        bss_column_M.append("{:.2} ({:.2})".format(models_bss_M[i], models_bss_M_err[i]))
        roc_column_M.append("{:.2} ({:.2})".format(models_roc_M[i], models_roc_M_err[i]))

    # Make dictionary of scores.
    M_metric_dict = {"TSS":tss_column_M, "APSS":apss_column_M, "ETS":ets_column_M,
                      "BSS":bss_column_M, "AUC":roc_column_M}

    # Create table using dictionary.
    M_metric_table = pd.DataFrame(M_metric_dict, index=models)

    M_latex = M_metric_table.to_latex(column_format="lrrrrr")
    print(M_latex)

    # Do same for C-class forecasts.
    C_forecasts = np.array([model["C1+(0-24hr)"] for model in model_csv_list])
    C_forecasts[C_forecasts < 0] = 0

    models_tss_C = [metric_utils.calculate_tss_threshold(events["C"],
                                                          forecast,
                                                          0.5)
                    for forecast in C_forecasts]

    models_tss_C_err = [model_bootstrap(forecast, events["C"], "tss")
                        for forecast in C_forecasts]

    models_apss_C = [metric_utils.calculate_apss_threshold(events["C"],
                                                          forecast,
                                                          0.5)
                    for forecast in C_forecasts]

    models_apss_C_err = [model_bootstrap(forecast, events["C"], "apss")
                        for forecast in C_forecasts]

    models_ets_C = [metric_utils.calculate_ets_threshold(events["C"],
                                                          forecast,
                                                          0.5)
                    for forecast in C_forecasts]

    models_ets_C_err = [model_bootstrap(forecast, events["C"], "ets")
                        for forecast in C_forecasts]

    models_bss_C = [metric_utils.calculate_bss(events["C"],
                                                forecast)
                    for forecast in C_forecasts]

    models_bss_C_err = [model_bootstrap(forecast, events["C"], "bss")
                          for forecast in C_forecasts]

    models_roc_C = [metric_utils.calculate_roc_area(events["C"],
                                                forecast)
                    for forecast in C_forecasts]

    models_roc_C_err = [model_bootstrap(forecast, events["C"], "auc")
                          for forecast in C_forecasts]

    # df_column = np.full(len(models_tss_C), "", dtype=str)
    tss_column_C = []
    apss_column_C = []
    ets_column_C = []
    bss_column_C = []
    roc_column_C = []

    for i in range(len(models_tss_C)):
        tss_column_C.append("{:.2} ({:.2})".format(models_tss_C[i], models_tss_C_err[i]))
        apss_column_C.append("{:.2} ({:.2})".format(models_apss_C[i], models_apss_C_err[i]))
        ets_column_C.append("{:.2} ({:.2})".format(models_ets_C[i], models_ets_C_err[i]))
        bss_column_C.append("{:.2} ({:.2})".format(models_bss_C[i], models_bss_C_err[i]))
        roc_column_C.append("{:.2} ({:.2})".format(models_roc_C[i], models_roc_C_err[i]))

    C_metric_dict = {"TSS":tss_column_C, "APSS":apss_column_C, "ETS":ets_column_C,
                      "BSS":bss_column_C, "AUC":roc_column_C}

    C_metric_table = pd.DataFrame(C_metric_dict, index=models)

    C_latex = C_metric_table.to_latex(column_format="lrrrrr")
    print(C_latex)
