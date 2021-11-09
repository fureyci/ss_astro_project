# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 14:22:48 2021

@author: Ciaran

Opening real time flare forecasts from the Flare Scoreboard, and
opening the GOES event list through the HEK using SunPy.
"""

# import modules
import urllib.request
import json
from datetime import datetime, date, timedelta
import numpy as np
import pandas as pd
from matplotlib import dates as mdates, pyplot as plt
from sunpy.net import Fido
from sunpy.net import attrs as a

# =====================================================================
# Define function that will be used to fill in missing dates.

def fill_missing_dates(df, start, end, date_location):
    """Generate an updated pandas dataframe based on df that will
    account for days where forecasts were excluded. Will fill in the
    missing period between start and end with missing forecasts (values
    of 0). This is to make sure that each model in the ensemble is of
    equal size.

    Parameters
    ----------
    df : `pandas.Dataframe`
        Dataframe of forecast.
    start : `str` or datetime-like
        Start of desired forecasting period.
    end : `str` or datetime-like
        End of desired forecasting period.
    date_location : `str`
        Whether dates are the indices of df, or a column of df. This is
        to match the input into the ensemble class; the observation df
        has the dates as indices, whereas the dates are contained as
        rows for the forecasts. One of "index" or "column".

    Returns
    -------
    df_out : `pandas.Dataframe`
        Updated dataframe that includes forecasts for every day between
        tstart and tend.

    """
    # Check for valid inputs.
    if not isinstance(df, pd.core.frame.DataFrame):
        raise ValueError("'df' must be of type pd.core.frame.DataFrame.")

    if not isinstance(start, str) and not isinstance(start, date):
        raise ValueError("'start' must be of type str or datetime-like.")

    if not isinstance(end, str) and not isinstance(end, date):
        raise ValueError("'tend' must be of type str or datetime-like.")

    valid_date_location = ["index", "column"]
    if date_location not in valid_date_location:
        raise ValueError("Invalid value entered for 'ate_location'. "
                         "Choose one of 'index' or 'column'.")

    # Create range of dates between start and end date.
    date_range = pd.date_range(start, end, freq="d")

    if date_location == "index":
        # Create new df of zeros that has the same columns as the input
        # df, however each row corresponds to each day in the day
        # range, and set the indices to the dates.
        new_df = pd.DataFrame(np.zeros((len(date_range),
                                        len(list(df)))
                                       ),
                              index=date_range,
                              columns=list(df))

        # Set values of observations in the new table.
        new_df.loc[df.index] = df.loc[df.index]

        df_out = new_df.astype(int)

    else:
        # Create new df of zeros that has the same columns as the input
        # df, however each row corresponds to each day in the day
        # range.
        df_out = pd.DataFrame(np.zeros((len(date_range),
                                        len(list(df)))
                                       ),
                              columns=list(df))

        # Set the date column to the dates.
        df_out.loc[:, "VALID_DATE"] = date_range

        # Find the indices of rows where forecast was given.
        events_indices = df_out[
                                df_out["VALID_DATE"].isin(df["VALID_DATE"])
                                ].index

        # Set the df at these indices to the the forecasts.
        # ".values" used to avoid any indexing complications between
        # the new df and the input df, where indices for the same day
        # may not be the same for both dfs.
        df_out.loc[events_indices, list(df)[1:]] = \
            df.loc[:,list(df)[1:]].values

    return df_out

def get_scoreboard_models(model_type="full disk"):
    """Load list of model names from Flare Scoreboard. Different models
    are loaded depending on the value of "model_type".

    Parameters
    ----------
    model_type : `str`, optional
        Desired to models to load. One of "full disk", "regions", or
        "both". The default is "full disk".

    Returns
    -------
    models : `list` of `str`
        Names of Flare Scoreboard models, depending on the value of
        model_type.

    """
    valid_model_types = ["full disk", "regions", "both"]
    if model_type not in valid_model_types:
        raise ValueError("Invalid value entered for 'model type'. Choose "
                         "one of 'full disk', 'regions', or 'both'.")

    # URL of catalogue.
    NAME_URL = "https://iswa.gsfc.nasa.gov/IswaSystemWebApp/flarescoreboard/hapi/catalog"

    # Open the URL. If you don't use json.loads(), will open as a string.
    # This ensures it is opened as a dict.
    with urllib.request.urlopen(NAME_URL) as url:
        model_names_json = json.loads(url.read().decode())

    # List to store models.
    models = []

    if model_type == "full disk":
        desired_model = "FULLDISK"
    elif model_type == "regions":
        desired_model = "REGIONS"
    elif model_type == "both":
        desired_model == ""

    # Loop through each model in the catalogue, check if the name of
    # the model ends with desired model.
    for item in model_names_json["catalog"]:
        if item["id"].endswith(desired_model):
            models.append(item["id"])

    return models


# =====================================================================
# Define function that will fetch real time data.

def get_realtime_data(tstart, tend, verbose=False):
    """Function that loads GOES event list through the Heliophysics
    Event Knowledgebase (HEK) and the forecast of all available models
    from the Flare Scoreboard API between tstart and tend.

    Each model on the Flare Scoreboard have different operational
    periods. Models that are not operational between tstart and three
    days before tend are not loaded.

    Parameters
    ----------
    tstart : `str`, format "YYYY/MM/DD"
        Beginning of testing interval.
    tend : `str`, format "YYYY/MM/DD"
        End of testing interval.
    verbose : `bool`, optional
        Whether or not to display loading process. The default is
        False.

    Returns
    -------
    events_realtime : `pandas.DataFrame`
        Dataframe containing all available events for each day in the
        testing interval. A value of 1 means a flare occurred on that
        day, and a value of 0 means a flare did not occur. Due to how
        rare they are, a column for X class flares will not appear if
        one did not occur over the testing interval.
    list_of_forecasts : `list of pandas.DataFrame`
        Foreast of each available model.
    names : `list` of `str`
        The names of each model from Flare Scoreboard.

    """
    # =================================================================
    # Opening realtime events.

    result = Fido.search(a.Time(tstart, tend),
                          a.hek.EventType("FL"),
                          a.hek.FL.GOESCls > "C1.0",
                          a.hek.OBS.Observatory == "GOES")

    # Make into a "sunpy.net.hek.hek.HEKTable".
    table = np.asarray(result["hek"]["event_starttime",
                                      "fl_goescls"])

    # Convert to pandas dataframe for data handling.
    pd_table = pd.DataFrame(table)

    # Change dates to datetime objects, might not need it.
    pd_table["event_starttime"] = pd.to_datetime(pd_table[
                                        "event_starttime"
                                        ].astype(str).str[:10],
                                        format="%Y-%m-%d")

    # Only get flare class of events, magnitude not necessary.
    pd_table["fl_goescls"] = pd_table["fl_goescls"].astype(str).str[0]

    # Change to one hot encoding, format used for Ensemble class.
    pd_table = pd.get_dummies(pd_table, columns=["fl_goescls"])

    # Change column names to match class.
    # First column header is "Date".
    new_col_names = ["Date"]

    # Dictionary to store aggregation functions when we will be combining
    # rows of the same date.
    aggregation_functions = {}

    # Sometimes, if it is a quiet period, there may be no X or M class
    # flares, so loop through the columns of the original table to avoid
    # manual assignment.
    #
    # Going to loop through the columns from the first column onwards
    # in the HEK table, as these correspond to the available events.
    for fl_cls in list(pd_table)[1:]:
        # Take the last character of the column and append it to the
        # list for new column names. The last character corresponds to
        # the flare class.
        new_col_names.append(fl_cls[-1])
        # Add a new key-value pair to aggregation_functions dict, where
        # the key is the flare class, and the value is "sum".
        aggregation_functions[fl_cls[-1]] = "sum"

    pd_table.columns = new_col_names # Update the columns.

    # If the Sun is active on a given day, it may produce many flares
    # of the same class. However, in this case, all we need is whether
    # or not a flare occurred, so we can drop any duplicate rows.
    pd_table = pd_table.drop_duplicates()

    # Now combine rows of the same date.
    # Since we have dropped duplicates, and used one-hot encoding, can
    # now combine rows of the same date.
    pd_table = pd_table.groupby(
        pd_table["Date"],
        as_index=True
        ).aggregate(aggregation_functions)

    # If a flare did not occur on a certain day, it will not be
    # included in the table, so use the function defined above to
    # ensure each date between tstart and tend is accounted for.
    pd_table = fill_missing_dates(pd_table,
                                  tstart,
                                  tend,
                                  "index")

    # Convert values in the table to integers.
    events_realtime = pd_table.astype(int)
    if verbose:
        print("Obtained events.\n")

    # =================================================================
    # OPENING REAL TIME FORECASTS
    #
    # Load the model names.
    #
    # Bear in mind, want the full disk. Looking at API, ids have either
    # "_REGIONS" or "_FULLDISK" at the end of their name, so I assume
    # I'm looking for models whose names contain the latter.

    # Load full disk model names.
    full_disk_models = get_scoreboard_models()

    # URL that contains all info on a certain model.
    INFO_URL = ("https://iswa.gsfc.nasa.gov/IswaSystemWebApp/flarescoreboard/hapi/"
                "info?id={}&options=fields.all&parameters=C,M")

    # Start of testing period for obtaining forecasts, as datetime
    # object.
    forecast_start = datetime.strptime(tstart,"%Y/%m/%d")

    # End of testing period for obtaining forecasts, as datetime
    # object.
    forecast_end = datetime.strptime(tend,"%Y/%m/%d")

    # 3 days before desired forecast start, to check if model is active
    # in desired forecast period.
    #
    # This is for when forecast end is today's date. Sometimes a model
    # might not update for some reason, so this is done to account for
    # this problem
    three_d_before = forecast_end - timedelta(days=3)

    # Change to date objects.
    three_d_before = datetime.combine(three_d_before, datetime.min.time())
    forecast_end = datetime.combine(forecast_end, datetime.min.time())

    # List to store models that are online for desired period.
    active_models = []

    for i, model in enumerate(full_disk_models):
        # First opten the models info.
        with urllib.request.urlopen(INFO_URL.format(model)) as url:
            data = json.loads(url.read().decode())

        # Store date it went offline (could still be online).
        stop_date = datetime.strptime(data["stopDate"][:10], "%Y-%m-%d")

        # Store date it went online.
        start_date = datetime.strptime(data["startDate"][:10], "%Y-%m-%d")

        # Check if models are active during desired period.
        if stop_date >= three_d_before and start_date <= forecast_start:
            if verbose:
                print("appending:", model,".")
            active_models.append(model)
        else:
            if verbose:
                print(model, "inactive for desired period.")

    # Drop any duplicates from list (MAG4 models appear twice).
    active_models = list(dict.fromkeys(active_models))

    try:
        # This model provides hourly forecasts, not needed for my case.
        active_models.remove("ASSA_1_FULLDISK")
        if verbose:
            print("removing ASSA_1_FULLDISK")
    except:
        pass

    # =================================================================
    # Now we have model names, so load the actual forecasts.

    # Start of testing period for obtaining forecasts, as string.
    forecast_start = forecast_start.strftime("%Y-%m-%d")

    # End of testing period for obtaining forecasts, as string.
    forecast_end = forecast_end.strftime("%Y-%m-%d")

    # Note, formats of times have changed to suit Flare Scoreboard API.

    # URL with forecast.
    URL_TO_LOAD = ("https://iswa.ccmc.gsfc.nasa.gov/IswaSystemWebApp/flarescoreboard/"
    "hapi/data?id={}&"
    "time.min={}T00:00:00.0&time.max={}T00:00:00.0&"
    "format=csv&parameters=start_window,C,M,X&options=fields.all")

    list_of_forecasts = [] # List to store forecasts for ensemble.

    # Load all active models in the desired forecast range.
    for model in active_models:
        # Read URL as CSV, change into pandas dataframe.
        #
        # Using "-only(0-24hr)" since these are threshold forecasts,
        # and not exceedence forecasts.
        csv = pd.read_csv(URL_TO_LOAD.format(model,
                                              forecast_start,
                                              forecast_end),
                          names=["VALID_DATE",
                                  "C-only(0-24hr)",
                                  "M-only(0-24hr)",
                                  "X-only(0-24hr)"])

        # Convert dates to dateteime.
        csv["VALID_DATE"] = pd.to_datetime(csv["VALID_DATE"].astype(str).str[:10],
                                            format="%Y-%m-%d")

        # Drop any duplicates, this will give forecast closest to 00:00UT.
        # (may need to think again about this process)
        csv = csv.drop_duplicates(subset="VALID_DATE")

        # Reset indices to range of numbers.
        csv = csv.set_index(pd.Index(range(len(csv.index))))

        # Now need to set any negative values to 0.
        # Negative values imply a missing foreast.
        # Only do it if there are missing forecasts.

        # First check if any values are negative.
        if (csv.loc[:, list(csv)[1]:].values < 0).any():
            # Make copy of the columns that give the forecasts.
            flare_probs_copy = csv.loc[:, list(csv)[1]:]

            # Set any negative values (missing forecasts) to 0.
            flare_probs_copy[flare_probs_copy < 0] = 0

            # Reassign the updated columns.
            csv.loc[:, list(csv)[1]:] = flare_probs_copy

        # Account for days where forecast is missing.
        csv = fill_missing_dates(csv,
                                  forecast_start,
                                  forecast_end,
                                  "column")

        list_of_forecasts.append(csv)  # Append to list.

    # Names of each model
    names = [i.partition("_")[0] for i in active_models]

    return events_realtime, list_of_forecasts, names


# =====================================================================
# Plot operational period for models.

def plot_operational_periods(model_type="full disk", savename=None):
    """Plot the operational periods of each model from the flare
    scoreboard. The models shown depend on model_type input.

    Provides option to save the figure.

    Parameters
    ----------
    model_type : `str`, optional
        Desired to models to load. One of "full disk", "regions", or
        "both". The default is "full disk".
    savename : `str`, optional
        Path to save the figure. The default is None.

    Returns
    -------
    ax : matplotlib axis
        Axis where the operational periods have been plotted.

    """
    if savename is not None:
        if type(savename) is not str:
            raise ValueError("'savename' must be a string.")

    models = get_scoreboard_models(model_type=model_type)

    # URL that contains all info on a certain model.
    INFO_URL = ("https://iswa.gsfc.nasa.gov/IswaSystemWebApp/flarescoreboard/hapi/"
                "info?id={}&options=fields.all&parameters=C,M")

    # Array to store dates model went online and offline.
    startstop_dates = np.zeros((len(models), 2), dtype=object)

    for i, model in enumerate(models):
        # First opten the models info.
        with urllib.request.urlopen(INFO_URL.format(model)) as url:
            data = json.loads(url.read().decode())

        # Store date it went offline (could still be online).
        stop_date = datetime.strptime(data["stopDate"][:10], "%Y-%m-%d")

        # Store date it went online.
        start_date = datetime.strptime(data["startDate"][:10], "%Y-%m-%d")

        # Append these values to the array.
        startstop_dates[i] = [start_date, stop_date]

    y = np.arange(len(models))  # Ticks for each model.

    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot()
    # Set xticks to dates.
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

    # Only plot the years from today.
    ax.xaxis.set_major_locator(mdates.YearLocator(1,
                                                  month=date.today().month,
                                                  day=date.today().day
                                                  ))

    # Plot each period.
    for i in y:
        ax.plot(startstop_dates[i], [i+1, i+1], "k-")

    # Autoformat the dates, default is 30 degree rotation, horizontal
    # alignment.
    fig.autofmt_xdate()

    plt.yticks(ticks=y+1, labels=models)
    plt.tight_layout()
    if savename is not None:
        plt.savefig(savename, dpi=200)
    plt.show()

    return ax
