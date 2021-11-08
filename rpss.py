# -*- coding: utf-8 -*-
"""
Created on Wed Nov 3 08:47:37 2021

@author: fureyci

Rolling ranked probability skill score, for data of the same format as
ensemble.Ensemble object.

Rolling RPSS is a useful metric for multicategory probabalistic events.
It 'penalises' forecasts when probability is further away from the
outcome. For example, if only an X class flare is forecasted with high
probability, and a 0% chance is forecasted for C and M class flares,
but only a C class flare occurs, the RPSS will account for this
"distance" between forecast and events.

Used in:
    Sharpe, M.A. and Murray, S.A., 2017. Verification of space weather
    forecasts issued by the Met Office Space Weather Operations Centre.
    Space Weather, 15(10), pp.1383-1395.
    https://doi.org/10.1002/2017SW001683

More info:
    https://www.cawcr.gov.au/projects/verification/

"""

import numpy as np
import pandas as pd

from ensemble import Ensemble

def av_rps(forecasts, events):
    """Average ranked probability score (RPS). Used for calculating
    ranked probability skill score (RPSS), and rolling RPSS.

    Each row corresponds to a daily forecast. Each column corresponds
    to the flare class.

    Parameters
    ----------
    forecasts : `np.array`
        Forecast values.
    events : `np.array`
        Events array.

    Returns
    -------
    av_rps : `float`
        Average rps over the forecast and events array.

    """
    # Check if inputs are of equal shape.
    if forecasts.shape != events.shape:
        raise ValueError("'forecasts' and 'events' must be the same "
                         "shape.")

    # Cumulative sum of elements along each row.
    #
    # Demonstration of np.cumsum() (confused me at first):
    # A = np.array([[a1, a2, ... , aN], [b1, b2, ... , bN], ...])
    # A
    # >>> [[a1, a2, ... , aN]
    #      [b1, b2, ... , bN]
    #               ...      ]
    #
    # np.cumsum(A, axis=1)
    # >>> [[a1, (a2 + a1), ... , (aN + aN-1 + ... + a2 + a1)]
    #      [a1, (a2 + a1), ... , (aN + aN-1 + ... + a2 + a1)]
    #                      ...                               ]
    cum_forecasts = np.cumsum(forecasts, axis=1)
    cum_events = np.cumsum(events, axis=1)

    # Array to store RPS for each day.
    rps_for_each_day = np.atleast_2d(np.zeros_like(cum_events[:,0],
                                                   dtype=float)
                                     ).T

    # Now go through every column to perform RPS sum.
    for i in range(1, len(forecasts[0])+1):
        rps_for_each_day += np.atleast_2d(
            np.sum(((cum_forecasts[:,:i] - cum_events[:,:i])**2),
                    axis=1)
            ).T

    rps_for_each_day = rps_for_each_day / float(len(events[0])-1)

    average_rps = np.mean(rps_for_each_day)

    return average_rps

def rpss(forecast, events, bootstrap=False):
    """Ranked probability skill score (RPSS).

    Uses climatology (mean of events list, measure of level of activity
    over the testing interval) as reference forecast for score.

    Parameters
    ----------
    forecasts : `np.array`
        Forecast values.
    events : `np.array`
        Events array.
    bootstrap : `bool`, optional
        Whether to perform bootstrapping with replacement to estimate
        the standard deviation of the RPSS. The default is False.

    Returns
    -------
    rpss : `float`
        The ranked probaility skill score.
    sdev : `float`
        Bootstrapped standard deviation. Only returned when bootstrap
        is True.

    """
    # The number of flare classes forecasted. Depending on the level of
    # activity, could be either [C], [C, M], or [C, M, X].
    available_flare_classes = list(events)

    # Climatology average RPS.
    climatology = np.mean(events, axis=0)

    climatology_values_array = np.zeros((len(forecast.iloc[:,0]),
                                          len(available_flare_classes)
                                          )
                                        )

    for i, fl_class in enumerate(available_flare_classes):
        climatology_values_array[:, i] = np.full(len(forecast.iloc[:,0]),
                                                  climatology[fl_class]
                                                  )

    av_rps_clim = av_rps(climatology_values_array, events.values)

    # Forecast average RPS.
    av_rps_forecast = av_rps(forecast.values, events.values)
    rpss_forecast = 1 - av_rps_forecast/av_rps_clim

    if bootstrap:
        boot_sample = []  # list to store bootstrapped sample.
        for iteration in range(1000):
            ran_indices = np.random.choice(len(events.values[:,0]),
                                            len(events.values[:,0]))

            av_rps_clim_boot = av_rps(climatology_values_array[ran_indices],
                                      events.values[ran_indices])

            av_rps_forecast_boot = av_rps(forecast.values[ran_indices],
                                          events.values[ran_indices])

            rpss_boot = 1 - av_rps_forecast_boot/av_rps_clim_boot

            boot_sample.append(rpss_boot)

        sdev = np.std(boot_sample, ddof=1)

        return rpss_forecast, sdev


    else:
        return rpss_forecast


def rolling_rpss(forecasts, model_names, observations, n_days,
                  forecast_type="exceedance",
                  desired_weighting="constrained",
                  desired_metric="brier",
                  bootstrap=False):
    """The rolling RPSS.

    That is, for an n_days rolling RPSS, for day i in the testing
    interval, calculate the RPSS for the past n_days days.

    Do this from day n_days to the final day of the testing interval.

    To be used with ensemble.Ensemble objects, takes many of the same
    parameters as it.

    Parameters
    ----------
    forecasts : list of pandas.DataFrame
        List of each forecast in the ensemble.
    model_names : list of str
        List of the names of the ensemble members.
    observations : `pandas.core.frame.DataFrame`
        Dataframe containing the events over the testing interval.
    n_days : `int`
        Number of days to calculate the rolling RPSS.
    forecast_type : `str`, optional
        Type of forecast, either "exceedence" or "threshold". The
        default is "exceedance".
    desired_weighting : `str`, optional
        Weighting scheme for ensemble. The default is "constrained".
    desired_metric : `str`, optional
        Metric to be optimised if weighting scheme is either
        "constrained" or "unconstrained". The default is "brier".
    bootstrap : `bool`, optional
        Whether to perform bootstrapping with replacement to estimate
        the standard deviation of the RPSS. The default is False.

    Returns
    -------
    dates : `pandas.core.indexes.datetimes.DatetimeIndex`
        DESCRIPTION.
    rrpss_list : `np.array`
        The rolling rpss over the interval.
    rrpss_sdev_list : `np.array`
        The bootstrapped standard deviation of the rolling rpss over
        the interval.

    """
    # Flare classes that have been forecasted.
    available_flare_classes = list(observations)

    if forecast_type == "threshold":
        end_of_forecast = "-only"
    elif forecast_type == "exceedance":
        end_of_forecast = "1+"

    # Column names of df depending on whether exceedence or threshold
    # classes are forecasted.
    classes_to_forecast = [i+end_of_forecast
                            for i in available_flare_classes]

    # List of ensembles of the same weighting scheme
    # (same desired_weighting) but with different flare classes
    # forecasted.
    ensemble_list = [Ensemble(forecasts, model_names, observations,
                              desired_forecast=flare_class,
                              desired_metric=desired_metric,
                              desired_weighting=desired_weighting)
                      for flare_class in classes_to_forecast]

    dates = pd.to_datetime(observations.index.values[n_days:],
                                    format="%Y-%m-%d")

    df_dict = {} # For eventual pandas dataframe.

    # Create df.
    # Columns correspond to the forecasts of each flare class.
    for i, model in enumerate(ensemble_list):
        df_dict[available_flare_classes[i]] = model.forecast

    ensemble_df = pd.DataFrame(df_dict) # create df

    rrpss_list = [] # List to store rolling RPSS.
    rrpss_sdev_list = [] # List to store bootstrapped sdev if desired.
    for i in range(n_days, len(ensemble_list[0].forecast)):
        rolling_forecast = ensemble_df.iloc[i-n_days:i]
        rolling_events = observations.iloc[i-n_days:i]
        if bootstrap:
            current_rpss, current_rpss_sdev = rpss(rolling_forecast,
                                                    rolling_events,
                                                    bootstrap=True)
            rrpss_list.append(current_rpss)
            rrpss_sdev_list.append(current_rpss_sdev)
        else:
            current_rpss = rpss(rolling_forecast, rolling_events)
            rrpss_list.append(current_rpss)

    return dates, rrpss_list, rrpss_sdev_list
