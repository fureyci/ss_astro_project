# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 16:49:57 2021

@author: fureyci
Ensemble space weather forecast.

Still need to include methods for un/constrained linear
combinations. Might also change how data is input into
the class (sometime in the future).

Example test of the class so far:

from opening_data import observations, model_names_only, model_csv_list
from ensemble import Ensemble

# instantiate
test = Ensemble(model_csv_list, model_names_only, observations)

# visualise performance
test.visualise_performance(which='reliability')
"""

import pandas as pd
import numpy as np
from metric_utils import plot_reliability_curve, plot_roc_curve

class Ensemble:
    """ Ensemble model for solar flare forecasting.
    
    Will be different depending on user needs. That is, depending on
    the desired flare event to be forecasted (desired_forecasts), the
    desired weighting scheme to be used (desired_weighting), and the
    desired metric to be optimised (desired_metric).

    Parameters
    ----------
    forecasts : `list of pandas.DataFrame objects`
        List whose elements correspond to a pandas.DataFrame object
        containing the forecast of a specific model. Column headers
        are:

        - "VALID_DATE" : date of forecast.
        - "C-only(0-24hr)" : forecase of C class only flares.
        - "C1+(0-24hr)" : forecast of C + exceedence class flares.
        - "M-only(0-24hr)" : forecase of M class only flares.
        - "M1+(0-24hr)" : forecast of M + exceedence class flares.

    observations : `pandas.Dataframe`
        pandas.DataFrame object containing the observations for M and C
        class. Indices are:
        - "Date" : date of observation.

        Column headers are:

        - "M" : observation of M class events. 1 if flare occurred, 0
                if not.
        - "C" : observation of C class events. 1 if flare occurred, 0
                if not.

    desired_flare : `str`, The default is "M1+".
        The flare class to examine. One of:

        | "C-only" : C class only.
        | "C1+"    : C class exceedence, 0hr latency, 24hr validity.
        | "M-only" : M class only.
        | "M1+"    : M class exceedence, 0hr latency, 24hr validity.

    desired_weighting : `str`. The default is "average".
        Desired scheme to determine combination weights.  One of:

        | "average" : compute average of all predictions in the
            ensemble. In other words, all weights are equal
            (= 1 / (number of models in ensemble)).
        | "history" : compute weights based off how well models
            have performed in the past.
        | "constrained" : perform constrained linear combination.
        | "unconstrained" : perform unconstrained linear
            combination.

    desired_metric : `str`. The default is None.
        Metric to be optimised. One of:

        | *some metric* : describe
        | *some metric* : describe
        | ...
        
    """

    def __init__(self, forecasts, model_names, observations,
                 desired_forecast="M1+", desired_weighting="average",
                 desired_metric=None):
        """Initialise instance of ensemble forecast."""
        # list of each model predictions
        self.forecasts = forecasts

        # list of the names of each model
        self.model_names = model_names

        # dataframe containing observations of M and C class flares
        self.observations = observations

        self.desired_forecast = desired_forecast
        self.desired_weighting = desired_weighting
        self.desired_metric = desired_metric

        # dataframe containing forecast of each model for the desired
        # flare class AND np.array of the observation of the desired
        # flare
        self.df_of_models, self._obs_vals = self._models_df()

        # calculate ensemble prediction according to desired weighting
        # scheme.
        if self.desired_weighting == "average":
            self.forecast = self._average()

        elif self.desired_weighting == "history":
            self.forecast = self._performance_history()

        elif self.desired_weighting == "constrained":
            self.forecast = self._con_lc()

        elif self.desired_weighting == "unconstrained":
            self.forecast = self._uncon_lc()

        else:
            raise ValueError("desired_weighting must be one of: 'average,'"
                             " 'history,' 'constrained,' or 'unconstrained.'")

    def _models_df(self):
        """ Generate pandas dataframe whose columns represent the
        forecast of the different models for the specific flare class,
        and rows represent the date of the forecast. Due to the nature
        of the for loop used here, will extract the observations using
        this function, also.

        Returns
        -------
        df_of_models : `pandas.Dataframe`
            pandas.DataFrame whose rows contain the forecast of the
            desired flare event for each model.
        observation_values : `np.array`
            The observations of the desired flare events on each day. A
            value of 1 means that a flare occurred on that day; a value
            of 0 means that one did not occur.

        """
        # dates of each forecast, assumed the same for each model.
        dates = self.forecasts[0].loc[:,'VALID_DATE'].values

        # Dictionary that will store the forecasts for each model.
        # A dictionary can be passed into pd.DataFrame, where the keys represent
        # the column headers, and the values represent the column values.
        df_dict = {}

        df_dict['Date'] = dates # store the dates in the dataframe

            # now need to load and forecast, depending on flare_class parameter.
        for name, forecast in zip(self.model_names, self.forecasts):

            if self.desired_forecast == "C-only":
                forecast_values = forecast.loc[:,"C-only(0-24hr)"].values
                observation_values = self.observations.loc[:,"C"].values

            elif self.desired_forecast == "C1+":
                forecast_values = forecast.loc[:,"C1+(0-24hr)"].values
                observation_values = self.observations.loc[:,"C"].values

            elif self.desired_forecast == "M-only":
                forecast_values = forecast.loc[:,"M-only(0-24hr)"].values
                observation_values = self.observations.loc[:,"M"].values

            elif self.desired_forecast == "M1+":
                forecast_values = forecast.loc[:,"M1+(0-24hr)"].values
                observation_values = self.observations.loc[:,"M"].values

            else:
                # in this case, an invalid value has been selected.
                raise ValueError("'desired_forecast' must be one of: "
                                  "'C-only', 'C1+', 'M-only', or 'M1+'.")

            # store the model name and its forecast into the dataframe.
            df_dict[name] = forecast_values

        df_of_models = pd.DataFrame(df_dict) # create dataframe

        df_of_models = df_of_models.set_index('Date') # set dates as indices

        return df_of_models, observation_values

    def _average(self):
        """ Perform simple ensemble average on models_df.

        Returns
        -------
        ensemble_av : `float`
            Ensemble average prediction.

        """
        headers = list(self.df_of_models) # the column headers of the df

        # map any negative values to 0
        self.df_of_models[self.df_of_models < 0] = 0

        # calculate average (weights = 1 / len(headers))
        ensemble_av = self.df_of_models.sum(axis=1) / len(headers)

        return ensemble_av
    
    def _performance_history(self):
        """ Calculate weights based of performance history of the
        models.

        Returns
        -------
        ensemble_prediction : `float`
            Ensemble prediction.

        Reference
        ---------
        Guerra, J.A. et al., 2020. Ensemble forecasting of major solar
        flares: methods for combining models. Journal of Space Weather
        and Space Climate, 10, p.38, Section 3.1:
            - for loop calculates Equation (4)
            - "denominator" is denominator of Equation (3)
            
        """
        # map negative values to 0
        self.df_of_models[self.df_of_models < 0] = 0

        # there is eventually going to be a weight for each model, so
        # instantiate array of 0s same length as number of models
        weights = np.zeros(len(list(self.df_of_models)))

        # loop through each model, calculate weights
        for i, model in enumerate(self.model_names):
            # get the forecast of the model
            forecast_vals = self.df_of_models.loc[:, model].values

            # calculate residaul sum of squares for each observation
            rss = np.sum((forecast_vals-self._obs_vals)**2)

            weights[i] = 1 / rss

        # denominator is sum of all inverse rss sums
        denominator = np.sum(weights)

        # set weights equal to inverse of sum of rss divided by sum of
        # all inverse rss values for each model
        weights = weights / denominator

        # multiply weights by observations
        ensemble_to_be_summed = weights * self.df_of_models.values

        # sum up (predictions of each model) x (weights) for each day
        # to obtain ensemble prediction
        ensemble_prediction = np.sum(ensemble_to_be_summed, axis=1)

        return ensemble_prediction
    
    def con_lc(self):
        """ Perform metric-optimized constrained linear combination of
        weights.

        Returns
        -------
        None.

        """
        # NEED TO UPDATE
        pass

    def uncon_lc(self):
        """ Perform metric-optimized unconstrained linear combination
        of weights.

        Returns
        -------
        None.

        """
        # NEED TO UPDATE
        pass
    
    def visualise_performance(self, which="both", title=None, savename=None):
        """
        Visualise how the ensemble performed through either a
        reiliability diagram, ROC plot, or both, depending on the value
        for "which."

        Parameters
        ----------
        which : `str`, the default is "both".
            Whether to visualise the performance of the ensemble using
            either a reliability diagram, ROC plot, or both. One of:

            | "reliability" : plot reliability diagram only.
            | "ROC" : plot ROC plot only.
            | "both" : plot both reliability diagram and ROC plot.

        Returns
        -------
        Plots the reliability diagram
        ax : ~`matplotlib axes`

        """
        if which == "reliability":
            ax1, ax2 = plot_reliability_curve(self._obs_vals,
                                              self.forecast,
                                              n_bins=20)
            return ax1, ax2
        
        elif which == "ROC":
            ax = plot_roc_curve(self._obs_vals, self.forecast)
            return ax
        
        elif which == "both":
            # NEED TO UPDATE
            pass
        
