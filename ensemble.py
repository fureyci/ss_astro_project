# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 16:49:57 2021

@author: Ciaran
Ensemble space weather forecast.
"""

class Ensemble:
    """ Ensemble model for solar flare forecasting."""

    def __init__(self):
        pass

    def models_df(self):
        """ Generate pandas dataframe whose columns represent the
        forecast of the different models for the specific flare class,
        and rows represent the date of the forecast.


        Returns
        -------
        None.

        """
        return None

    def average(self):
        """ Perform simple ensemble average on models_df.



        Returns
        -------
        None.

        """
        pass

    def con_lc(self):
        """ Perform metric-optimized constrained linear combination of
        weights.


        Returns
        -------
        None.

        """
        pass

    def uncon_lc(self):
        """ Perform metric-optimized unconstrained linear combination
        of weights.


        Returns
        -------
        None.

        """
        pass
