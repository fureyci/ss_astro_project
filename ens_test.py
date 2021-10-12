# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 15:57:22 2021

@author: fureyci

Code to test functionality of ensemble.py.

Prompts user for input into the console to chose desired flare
forecast, desired weighting scheme, and desired metric to
optimise, if the weighting scheme is constrained linear
combination.
"""

from opening_data import observations, model_names_only, model_csv_list
from ensemble import Ensemble

def user_input():
    valid_forecasts = ["C-only", "C1+", "M-only", "M1+"]
    valid_methods = ["average", "history", "constrained", "unconstrained"]
    valid_weights = ["brier", "LCC", "MAE"]

    print("What flare class would you like to forecast?\n"
          "Choose one of \"C-only\", \"C1+\", \"M-only\", "
          " or \"M1+\". ")

    forecast = input("Enter desired forecast: ")

    while True:
        if forecast in valid_forecasts:
            break
        else:
            print("\nPlease choose one of \"C-only\", \"C1+\", \"M-only\", "
                  " or \"M1+\". ")
            forecast = input("Enter desired forecast: ")

    print("\nWhat weighting scheme would you like to use?\n"
          "Choose one of \"average\", \"history\", "
          "\"constrained\" or \"unconstrained\".")

    weighting = input("Enter desired scheme: ")

    while True: # loop until all inputs are valid
        if weighting in valid_methods:
            if weighting in valid_methods[:2]:
                metric = None
                # dont need to optimise metric if using average or
                # performance history ensemble
            else:
                print("\nWhat metric would you like to optimise?\n"
                      "Choose one of \"brier\", \"LCC\", or \"MAE\".")
                metric = input("Enter desired metric: ")
                while True:
                    if metric in valid_weights:
                        break
                    else:
                        # ensure metric is valid
                        print("\n Please choose one of \"brier\", \"LCC\", "
                              "or \"MAE\".")
                        metric = input("Enter desired metric: ")
            break

        else:
            # ensure weighting is valid
            print("\nPlease choose one of \"average\", \"history\", "
                  "\"constrained\" or \"unconstrained\".")
            weighting = input("Enter desired scheme: ")


    return forecast, weighting, metric

forecast, weighting, metric = user_input()

print("\nBuilding ensemble...")
test = Ensemble(model_csv_list, model_names_only, observations,
                desired_forecast=forecast,
                desired_metric=metric,
                desired_weighting=weighting)
print("Done. Now plotting reliability diagram and ROC curve...")

test.visualise_performance(which="both")
