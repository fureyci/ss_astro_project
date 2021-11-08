# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 16:49:57 2021

@author: fureyci

Ensemble model for solar flare forecasting.
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize
from metric_utils import plot_reliability_curve, plot_roc_curve

class Ensemble:
    """Ensemble model for solar flare forecasting.

    Will be different depending on user needs. That is, depending on
    the desired flare event to be forecasted (desired_forecasts), the
    desired weighting scheme to be used (desired_weighting), and the
    desired metric to be optimised (desired_metric).

    Parameters
    ----------
    forecasts : `list` of `pandas.core.frame.DataFrame`
        List whose elements correspond to a pandas.DataFrame object
        containing the forecast of a specific model. Column headers
        are:

        - "VALID_DATE" : date of forecast.
        - "C-only(0-24hr)" : forecase of C class only flares.
        - "C1+(0-24hr)" : forecast of C + exceedence class flares.
        - "M-only(0-24hr)" : forecase of M class only flares.
        - "M1+(0-24hr)" : forecast of M + exceedence class flares.

    observations : `pandas.core.frame.DataFrame`
        pandas.DataFrame object containing the observations for M and C
        class. Indices are:
        - "Date" : date of observation.

        Column headers are:

        - "M" : observation of M class events. 1 if flare occurred, 0
                if not.
        - "C" : observation of C class events. 1 if flare occurred, 0
                if not.

    desired_flare : `str`, The default is "M1+"
        The flare class to examine. One of:

        | "C-only" : C class only.
        | "C1+"    : C class exceedence, 0hr latency, 24hr validity.
        | "M-only" : M class only.
        | "M1+"    : M class exceedence, 0hr latency, 24hr validity.
        | "X-only" : X class only.
        | "X1+"    : X class exceedence, 0hr latency, 24hr validity.

    desired_weighting : `str`. The default is "average"
        Desired scheme to determine combination weights.  One of:

        | "average" : compute average of all predictions in the
            ensemble. In other words, all weights are equal
            (= 1 / (number of models in ensemble)).
        | "history" : compute weights based off how well models
            have performed in the past.
        | "constrained" : perform constrained linear combination.
        | "unconstrained" : perform unconstrained linear
            combination.

    desired_metric : `str`. The default is None
        Metric to be optimised. One of:

        | "brier" : Brier Score
        | "LCC" : Linear Correlation Coefficient.
        | "MAE" : Mean Absolute Error
        | "REL" : Reliabililty

    Returns
    -------
    df_of_models : `pandas.core.frame.DataFrame`
        The forecast of each model in the ensemble for the desired
        flare class.

    events : `numpy.ndarray`
        The events of the desired flare class.

    climatology : `numpy.float64`
        The climatology of the desired flare class over the testing
        period.

    forecast : `numpy.ndarray`
        The ensemble forecast.

    weights : `numpy.ndarray`
        The weights of the linear combination that produces the
        ensemble forecast.

    ac_weights : `numpy.ndarray`
        The weights obtained for each optimisation process.

    format : `str`
        Format for plotting. Different depending on desired
        weighting scheme.

    """
    def __init__(self, forecasts, model_names, observations,
                 desired_forecast="M1+", desired_weighting="average",
                 desired_metric=None):
        """Initialise instance of ensemble forecast."""

        # List of each model forecast.
        self.forecasts = forecasts

        # List of the names of each model.
        self.model_names = model_names

        # Dataframe containing M and C class flare events.
        self.observations = observations

        self.desired_forecast = desired_forecast
        self.desired_weighting = desired_weighting
        self.desired_metric = desired_metric

        # Dataframe containing forecast of each model for the desired
        # flare class AND np.array of the events of the desired
        # flare class.
        self.df_of_models, self.events = self._models_df()

        # Map any negative values to 0.
        self.df_of_models[self.df_of_models < 0] = 0

        # Climatology (mean of events).
        self.climatology = np.mean(self.events)

        # Calculate ensemble prediction according to desired weighting
        # scheme.
        if self.desired_weighting == "average":
            self.forecast, self.weights = self._average()

        elif self.desired_weighting == "history":
            self.forecast, self.weights = self._performance_history()

        elif self.desired_weighting == "constrained":
            self.forecast, self.weights, self.ac_weights = self._linear_combo(constrained=True)

        elif self.desired_weighting == "unconstrained":
            # Since unconstrained LC involves climatology in
            # calculation, will append column of climatology to the
            # dataframe of models here, as it is only needed in this
            # case.
            self.df_of_models["Climatology"] = np.full(
                self.events.shape,
                self.climatology
                )
            self.forecast, self.weights, self.ac_weights = self._linear_combo(constrained=False)

        else:
            raise ValueError("desired_weighting must be one of: 'average,'"
                             " 'history,' 'constrained,' or 'unconstrained.'")

        # Use different marker formats depending on ensemble type.
        formats = {
            "average": "o",
            "history": "s",
            "constrained": "^",
            "unconstrained": "P"}

        self.format = formats[self.desired_weighting]

    def _models_df(self):
        """Generate pandas dataframe whose columns represent the
        forecast of the different models for the specific flare class,
        and rows represent the date of the forecast. Due to the nature
        of the for loop used here, will extract the events using
        this function, also.

        Returns
        -------
        df_of_models : `pandas.core.frame.Dataframe`
            pandas.DataFrame whose rows contain the forecast of the
            desired flare event for each model.
        events : `numpy.ndarray`
            The events of the desired flare class on each day. A value
            of 1 means that a flare occurred on that day; a value of 0
            means that one did not occur.

        """

        # Dates of each forecast, same for each model.
        dates = self.forecasts[0].loc[:,"VALID_DATE"].values

        # Dictionary that will store the forecasts for each model.
        # A dictionary can be passed into pd.DataFrame, where the keys
        # represent the column headers, and the values represent the
        # column values.
        df_dict = {}

        df_dict["Date"] = dates  # Store the dates in the dataframe.

        # Now need to load forecast and observations, depending on
        # flare_class parameter.
        for name, forecast in zip(self.model_names, self.forecasts):

            if self.desired_forecast == "C-only":
                forecast_values = forecast.loc[:,"C-only(0-24hr)"].values
                events = self.observations.loc[:,"C"].values

            elif self.desired_forecast == "C1+":
                forecast_values = forecast.loc[:,"C1+(0-24hr)"].values
                events = self.observations.loc[:,"C"].values

            elif self.desired_forecast == "M-only":
                forecast_values = forecast.loc[:,"M-only(0-24hr)"].values
                events = self.observations.loc[:,"M"].values

            elif self.desired_forecast == "M1+":
                forecast_values = forecast.loc[:,"M1+(0-24hr)"].values
                events = self.observations.loc[:,"M"].values

            elif self.desired_forecast == "X-only":
                forecast_values = forecast.loc[:,"X-only(0-24hr)"].values
                events = self.observations.loc[:,"X"].values

            elif self.desired_forecast == "X1+":
                forecast_values = forecast.loc[:,"X1+(0-24hr)"].values
                events = self.observations.loc[:,"X"].values

            else:
                # In this case, an invalid value has been selected.
                raise ValueError("'desired_forecast' must be one of: "
                                  "'C-only', 'C1+', 'M-only', 'M1+', "
                                  "'X-only', or 'X1+'.")

            # Store the model name and its forecast into the dataframe.
            df_dict[name] = forecast_values

        df_of_models = pd.DataFrame(df_dict)  # Create dataframe.

        # Set dates as indices.
        df_of_models = df_of_models.set_index("Date")

        return df_of_models, events

    def _average(self):
        """Perform simple ensemble average on models_df.

        Returns
        -------
        ensemble_av : `numpy.ndarray`
            Ensemble average forecast.

        weights : `numpy.ndarray`
            Weights of each model.

        """
        no_models = len(list(self.df_of_models))

        # Calculate average of each model forecast.
        ensemble_av = self.df_of_models.mean(axis=1).values

        weights = np.full(no_models, 1/float(no_models))

        return ensemble_av, weights

    def _performance_history(self):
        """Calculate weights based of performance history of the
        models. That is, the weight of the model is inversely
        proportional to the residual sum of squares (RSS) between the
        model's forecast and the events.

        Returns
        -------
        ensemble_prediction : `numpy.ndarray`
            Ensemble forecast.

        weights : `numpy.ndarray`
            Weights of each model.

        Reference
        ---------
        Guerra, J.A. et al., 2020. Ensemble forecasting of major solar
        flares: methods for combining models. Journal of Space Weather
        and Space Climate, 10, p.38, Section 3.1:
            - for loop calculates Equation (4)
            - "denominator" is denominator of Equation (3)

        """

        # Array of 0s same length as number of models.
        weights = np.zeros(len(list(self.df_of_models)))

        # Loop through each model, calculate weights.
        for i, model in enumerate(self.model_names):
            # Get the forecast of the model.
            forecast_vals = self.df_of_models.loc[:, model].values

            # Calculate RSS for each event.
            rss = np.sum((forecast_vals-self.events)**2)

            weights[i] = 1 / rss

        # Denominator is sum of all inverse RSS sums.
        denominator = np.sum(weights)

        # Set weights equal to inverse of sum of rss divided by sum of
        # all inverse RSS values for each model.
        weights = weights / denominator
        
        ensemble_prediction = np.sum(weights * self.df_of_models.values, axis=1)

        return ensemble_prediction, weights

    def _linear_combo(self, constrained=True):
        """Perform linear combination to determine weights.

        Value of "constrained" determines weither the combination is
        constrained ("constrained" = True) or unconstrained
        ("constrained" = False).

        Parameters
        ----------
        constrained : `bool`. The default is True.
            Whether the linear combination is constrained (True) or
            unconstrained (False)

        Returns
        -------
        ensemble_prediction : `numpy.ndarray`
            Ensemble forecast.

        weights : `numpy.ndarray`
            Weights of each model.

        Reference
        ---------
        Guerra, J.A. et al., 2020. Ensemble forecasting of major solar
        flares: methods for combining models. Journal of Space Weather
        and Space Climate, 10, p.38, Sections 3.2, 3.3, 4.1.
            - initial weights take value [0,1] for constrained case.
            - initial weights take value [-1,1] for unconstrained case.
            - Solver executed 500 times, results in distribution that
                is normal in shape, mean value used as final optimised
                weight, with std dev used as 10% error

        """

        def _metric_funct(en_forecast_to_optimise):
            """Define the metrics to be used to create the ensemble.
            Depending on self.desired_metric, the options are:

            | "brier" : Brier Score
            | "REL" : Reliability
            | "LCC" : Linear Correlation Coefficient
            | "MAE" : Mean Absolute Error

            Parameters
            ----------
            en_forecast_to_optimise : `numpy.ndarray`
                Forecast of ensemble whose weights have NOT been
                optimised, same shape as self.events.
                Provided by _optimise_funct().

            Returns
            -------
            funct : `numpy.ndarray`
                Metric function.

            """

            valid_metrics = ["brier", "REL", "LCC", "MAE"]
            if self.desired_metric is None:
                raise ValueError("Please choose a desired metric to optimise.")

            if self.desired_metric not in valid_metrics:
                raise ValueError("Invalid metric entered.")

            if self.desired_metric == "brier":
                funct = np.mean(
                    (en_forecast_to_optimise - self.events) ** 2.0
                                )

            if self.desired_metric == "REL":
                # Generate bins of uniform width.
                bins = np.linspace(0., 1. + 1e-8, 21)

                # Assign model predictions to each bin, -1 since it corresponds
                # to the indices.
                binids = np.digitize(en_forecast_to_optimise, bins) - 1

                # Sum up the probabilities in each bin, for eventual average.
                bin_sums = np.bincount(binids, weights=en_forecast_to_optimise,
                                       minlength=len(bins))

                # Sum up number of flare events in each probability bin.
                bin_true = np.bincount(binids, weights=self.events,
                                       minlength=len(bins))

                # Sum up number of events in each bin.
                bin_total = np.bincount(binids, minlength=len(bins))

                # Delete any bins where there are no flare events, since want
                # to replicate benchmark discussed in Leka 2019 paper.
                zeros = np.where(bin_true == 0)

                # Delete these indices from the arrays
                bin_true = np.delete(bin_true, zeros)
                bin_sums = np.delete(bin_sums, zeros)
                bin_total = np.delete(bin_total, zeros)

                # Indices where there are a nonzero number of events in the bins.
                nonzero = bin_total != 0

                mean_predicted_value = bin_sums[nonzero] / bin_total[nonzero]
                mean_observaed_value = bin_true[nonzero] / bin_total[nonzero]

                funct = 1 / float(len(self.events)) * np.sum(
                    bin_total[nonzero] * (mean_predicted_value - mean_observaed_value) ** 2
                    )

            if self.desired_metric == "LCC":
                funct = np.corrcoef(
                    en_forecast_to_optimise,
                    self.events)[0, 1]
                # [0,1] since corrcoef returns 2d covariance matrix -
                # diagonal values = 1, and, if covariance matrix is C,
                # C[i,j] = C[j,i].

            if self.desired_metric == "MAE":
                funct = np.mean(
                    np.abs(en_forecast_to_optimise - self.events)
                    )

            return funct

        def _optimise_funct(ws_ini):
            """Function to be optimised. Calculates linear combination
            using the initial guesses for the weights, passes this into
            _metric_funct(), obtains the desired metric function and
            returns it. This will be used in the
            scipy.optimize.minimize() routine (implemented later)

            Parameters
            ----------
            ws_ini : `numpy.ndarray`
                Array of initial guess for weights.

            Returns
            -------
            ofunct : `numpy.ndarray`
                Function to be optimised.

            """

            # Linear combination.
            # If desired_weighting is "unconstrained", ws_ini includes
            # the weight for climatology (done later in the function),
            # and self.df_of_models.values includes the column for
            # climatology, which was done in __init__.

            combination = np.sum(
                    ws_ini * self.df_of_models.values,
                    axis=1
                    )

            # Function to optimise.
            ofunct = _metric_funct(combination)

            # Functions to maximise.
            to_max = ["LCC"]

            if self.desired_metric in to_max:
                ofunct = -1 * ofunct

            return ofunct

        # Define number of models (used multiple times next)
        number_of_models = len(self.model_names)

        # Number of times to repeat optimisation process, same as
        # Guerra et al. 2020.
        REPITITIONS = 500

        if constrained is True:
            # Jacobian of constraint function, to be used as "jac" in
            # constraints.
            jac = np.ones(number_of_models)

            # Define bounds.
            bounds = np.full((number_of_models,2), (0,1))

            # Set up array of zeros that will store the optimised
            # weights from each repitition..
            accumulated_weights = np.zeros(
                (REPITITIONS, number_of_models)
                )

        else:
            # Jacobian of constraint function, to be used as "jac" in
            # constraints.
            jac = np.ones(number_of_models+1)

            # Set bounds.
            bounds = np.full((number_of_models+1,2), (-1,1))

            accumulated_weights = np.zeros(
                (REPITITIONS, number_of_models+1)
                )

        # Now define the constraints:
        # "type": constraint type, "eq" for equality, meaning that "fun"
        #   must result to zero.
        # "fun" : function defining the constraint. That is, all of the
        #   weights have to sum up to 1, since "type"="eq".
        # "jac" : jacobion of "fun," equals array of 1s.

        # Constraints the same for both unconstrained and constrained
        # cases.
        consts = ({"type": "eq",
                    "fun": lambda ws: np.sum(ws) - 1, # weights add to 0
                    "jac": lambda ws: jac})

        # Repeat optimisation procedure to account for sensitivity of
        # the results of the optimised weights to the choice of the
        # initial guess.

        for i in range(REPITITIONS):

            if constrained is True:
                # Initial weights, random values between 0 and 1
                initial_weights = np.random.rand(number_of_models)

            else:
                # Initial weights, random values between -1 and 1,
                # +1 to account for climatology weight.
                initial_weights = np.random.uniform(-1, 1, number_of_models+1)

            result = minimize(_optimise_funct,
                              initial_weights,
                              constraints=consts,
                              bounds=bounds,
                              method="SLSQP",
                              jac=False,
                              tol=1e-12,
                              options={
                                  "disp": False,
                                  "maxiter": 10000,
                                  "eps": 0.0001
                                  }
                              )

            # Store this set of weights, and repeat.
            accumulated_weights[i] = result.x

        optimised_weights = np.mean(
        accumulated_weights, axis=0
        )

        ensemble_prediction = np.sum(
                optimised_weights * self.df_of_models.values,
                axis=1
                )

        # map any negative values to 0
        ensemble_prediction[ensemble_prediction < 0] = 0

        # Map any values > 1 to 1.
        ensemble_prediction[ensemble_prediction > 1] = 1

        return ensemble_prediction, optimised_weights, accumulated_weights


    def visualise_performance(self, which="both", ax=None, axes=None):
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

        ax : `~matplotlib.axes.Axes`, optional
            If provided the ROC curve will be plotted on the given
            axes.

        axes : `tuple of matplotlib.axes.Axes`, optional
            If provided the reliability diagram and histrogram will be
            plotted on the given axes. Expected form:
                (reliability diagram axis, histogram axis),
            where ratio height of reliability diagram and histogram
            axes is 3:1.

        Returns
        -------
        Plots the desired diagram.
        ax : ~`matplotlib axes` if which=="ROC"

        ax1, ax2 : ~`matplotlib axes` if which=="reliability"

        rel_ax, hist_ax, roc_ax : ~`matplotlib axes` if which=="both"

        """

        if which == "reliability":
            if axes is None:
                ax1, ax2 = plot_reliability_curve(self.events,
                                                  self.forecast,
                                                  n_bins=20,
                                                  fmt = self.format
                                                  )
                return ax1, ax2

            else:
                ax1, ax2 = plot_reliability_curve(self.events,
                                                  self.forecast,
                                                  n_bins=20,
                                                  fmt = self.format,
                                                  axes=axes
                                                  )
                return ax1, ax2

        elif which == "ROC":
            if ax is None:
                ax = plot_roc_curve(self.events, self.forecast,
                                    fmt = self.format)
            else:
                ax = plot_roc_curve(self.events, self.forecast,
                                    fmt = self.format,
                                    ax=ax)
            return ax


        elif which == "both":
            rel_ax, hist_ax = plot_reliability_curve(self.events,
                                              self.forecast,
                                              n_bins=20,
                                              fmt = self.format
                                              )
            roc_ax = plot_roc_curve(self.events, self.forecast,
                                 fmt = self.format)

            return rel_ax, hist_ax, roc_ax
