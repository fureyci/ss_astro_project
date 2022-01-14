# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 15:57:22 2021

@author: fureyci

Code to test functionality of ensemble.py.

Prompts user for input into the console to chose desired flare
forecast, desired weighting scheme, and desired metric to
optimise, if the weighting scheme is CLC/ULC.

As of 03/11/2021, contains the plots that will be used in my thesis.

"""
import numpy as np
import pandas as pd
import time
from matplotlib import pyplot as plt, patches as mpatches, ticker as mticker
from matplotlib.legend_handler import HandlerTuple
import metric_utils
from opening_data import load_benchmark_data
from ensemble import Ensemble

# WD_TO_SAVE = _____ # define a directory to save files if desired.

plt.rcParams['font.family'] = 'Palatino Linotype'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 11

# ===========================================================================
# Load data.
events, model_csv_list, model_names_only = load_benchmark_data()

# ===========================================================================
# User input function and how it can be used to load an ensemble.
# Pretty tedious...
def user_input():
    """
    Prompts user to input their desired ensemble parameters

    Returns
    -------
    forecast : str
        Flare class to forecast.
    weighting : str
        Weighting scheme to use.
    metric : str
        Metric to optimise, only when weighting = {"unconstrained",
        "constrained"}.

    """
    valid_forecasts = ["C-only", "C1+", "M-only", "M1+"]
    valid_methods = ["Average", "EV", "CLC", "ULC"]
    valid_weights = ["brier", "LCC", "MAE", "REL"]

    print("What flare class would you like to forecast?\n"
          "Choose one of 'C-only', 'C1+', 'M-only', "
          " or 'M1+'. ")

    forecast = input("Enter desired forecast: ")

    while True:
        if forecast in valid_forecasts:
            break
        else:
            print("\nPlease choose one of 'C-only', 'C1+', 'M-only', "
                  " or 'M1+'. ")
            forecast = input("Enter desired forecast: ")

    print("\nWhat weighting scheme would you like to use?\n"
          "Choose one of 'Average', 'EV', "
          "'CLC' or 'ULC'.")

    weighting = input("Enter desired scheme: ")

    while True: # Loop until all inputs are valid.
        if weighting in valid_methods:
            if weighting in valid_methods[:2]:
                metric = None
                # Dont need to optimise metric if using average or
                # performance history ensemble.
            else:
                print("\nWhat metric would you like to optimise?\n"
                      "Choose one of 'BS', 'LCC', or 'MAE'.")
                metric = input("Enter desired metric: ")
                while True:
                    if metric in valid_weights:
                        break
                    else:
                        # Ensure metric is valid.
                        print("\n Please choose one of 'BS', 'LCC', "
                              "or 'MAE'.")
                        metric = input("Enter desired metric: ")
            break

        else:
            # Ensure weighting is valid.
            print("\nPlease choose one of 'Average', 'EV', "
                  "'CLC' or 'ULC'.")
            weighting = input("Enter desired scheme: ")

    return forecast, weighting, metric

forecast, weighting, metric = user_input()

# print("\nBuilding ensemble...")
# forecast, weighting, metric = user_input()

# test = Ensemble(model_csv_list, model_names_only, events,
#                 desired_forecast=forecast,
#                 desired_metric=metric,
#                 desired_weighting=weighting)
# print("Done. Now plotting reliability diagram and ROC curve...")
# test.visualise_performance()

# # ===========================================================================
# # Example of how models are loaded

# FORECAST = "M1+" # flare class to forecast
# WEIGHTING = "ULC" # weighting scheme to use
# METRIC = "BS" # metric to optimise, if weighting is either CLC or ULC

# test_ensemble = Ensemble(model_csv_list, model_names_only, events,
#                          desired_forecast=FORECAST,
#                          desired_weighting=WEIGHTING,
#                          desired_metric=METRIC)

# ===========================================================================
# Plots for thesis

now = time.time()  # let's see how long it takes.

# Instantiate models.
print("average")
# Average weighting, M class.
average_M = Ensemble(model_csv_list, model_names_only, events,
                desired_forecast="M1+",
                # desired_metric=METRIC,
                desired_weighting="Average")

# Average weighting, C class.
average_C = Ensemble(model_csv_list, model_names_only, events,
                desired_forecast="C1+",
                # desired_metric=METRIC,
                desired_weighting="Average")

# Performance history weighting, M class.
print("history")
history_M = Ensemble(model_csv_list, model_names_only, events,
                desired_forecast="M1+",
                # desired_metric=METRIC,
                desired_weighting="EV")

# Performance history weighting, C class.
history_C = Ensemble(model_csv_list, model_names_only, events,
                desired_forecast="C1+",
                # desired_metric=METRIC,
                desired_weighting="EV")

# CLC weighting, BS score optimised, M class.
print("clc m BS")
CLC_M_BS = Ensemble(model_csv_list, model_names_only, events,
                                desired_forecast="M1+",
                                desired_metric="BS",
                                desired_weighting="CLC")

# ULC weighting, BS optimised, M class.
print("ulc m BS")
ULC_M_BS = Ensemble(model_csv_list, model_names_only, events,
                                desired_forecast="M1+",
                                desired_metric="BS",
                                desired_weighting="ULC")

# CLC weighting, BS optimised, C class.
print("clc c BS")
CLC_C_BS = Ensemble(model_csv_list, model_names_only, events,
                                desired_forecast="C1+",
                                desired_metric="BS",
                                desired_weighting="CLC")

# ULC weighting, BS optimised, C class.
print("ulc c BS")
ULC_C_BS = Ensemble(model_csv_list, model_names_only, events,
                                desired_forecast="C1+",
                                desired_metric="BS",
                                desired_weighting="ULC")

# CLC weighting, MAE optimised, M class.
print("clc m MAE")
CLC_M_MAE = Ensemble(model_csv_list, model_names_only, events,
                                desired_forecast="M1+",
                                desired_metric="MAE",
                                desired_weighting="CLC")

# ULC weighting, MAE optimised, M class.
print("ulc m MAE")
ULC_M_MAE = Ensemble(model_csv_list, model_names_only, events,
                                desired_forecast="M1+",
                                desired_metric="MAE",
                                desired_weighting="ULC")

# CLC weighting, MAE optimised, C class.
print("clc c MAE")
CLC_C_MAE = Ensemble(model_csv_list, model_names_only, events,
                                desired_forecast="C1+",
                                desired_metric="MAE",
                                desired_weighting="CLC")

# ULC weighting, MAE optimised, C class.
print("ulc c MAE")
ULC_C_MAE = Ensemble(model_csv_list, model_names_only, events,
                                desired_forecast="C1+",
                                desired_metric="MAE",
                                desired_weighting="ULC")

# CLC weighting, REL optimised, M class.
print("clc m REL")
CLC_M_REL = Ensemble(model_csv_list, model_names_only, events,
                                desired_forecast="M1+",
                                desired_metric="REL",
                                desired_weighting="CLC")

# ULC weighting, REL optimised, M class.
print("ulc m REL")
ULC_M_REL = Ensemble(model_csv_list, model_names_only, events,
                                desired_forecast="M1+",
                                desired_metric="REL",
                                desired_weighting="ULC")

# CLC weighting, REL optimised, C class.
print("clc c REL")
CLC_C_REL = Ensemble(model_csv_list, model_names_only, events,
                                desired_forecast="C1+",
                                desired_metric="REL",
                                desired_weighting="CLC")

# ULC weighting, REL optimised, C class.
print("ulc c REL")
ULC_C_REL = Ensemble(model_csv_list, model_names_only, events,
                                desired_forecast="C1+",
                                desired_metric="REL",
                                desired_weighting="ULC")

# CLC weighting, LCC optimised, M class.
print("clc m LCC")
CLC_M_LCC = Ensemble(model_csv_list, model_names_only, events,
                                desired_forecast="M1+",
                                desired_metric="LCC",
                                desired_weighting="CLC")

# ULC weighting, LCC optimised, M class.
print("ulc m LCC")
ULC_M_LCC = Ensemble(model_csv_list, model_names_only, events,
                                desired_forecast="M1+",
                                desired_metric="LCC",
                                desired_weighting="ULC")

# CLC weighting, LCC optimised, C class.
print("clc C LCC")
CLC_C_LCC = Ensemble(model_csv_list, model_names_only, events,
                                desired_forecast="C1+",
                                desired_metric="LCC",
                                desired_weighting="CLC")

# ULC weighting, LCC optimised, C class.
print("ulc c LCC")
ULC_C_LCC = Ensemble(model_csv_list, model_names_only, events,
                                desired_forecast="C1+",
                                desired_metric="LCC",
                                desired_weighting="ULC")

# ===========================================================================
# The layout of each plot.
layout = np.array([[average_C, history_C, average_M, history_M],
              [CLC_C_BS, ULC_C_BS, CLC_M_BS, ULC_M_BS],
              [CLC_C_MAE, ULC_C_MAE, CLC_M_MAE, ULC_M_MAE],
              [CLC_C_REL, ULC_C_REL, CLC_M_REL, ULC_M_REL],
              [CLC_C_LCC, ULC_C_LCC, CLC_M_LCC, ULC_M_LCC]
              ], dtype=object)

# ---------------------------------------------------------------------------
# RELIABILITY DIAGRAMS of each model.

rel_fig = plt.figure(figsize=(10,12))
rel_gs = rel_fig.add_gridspec(20, 4)

# Plot each reliability diagram.
for i in range(len(layout)):
    for j in range(len(layout[i])):
        model = layout[i,j] # Access model.

        # Create axes for each plot.
        rel_ax = rel_fig.add_subplot(rel_gs[4*i:4*i+3, j])
        hist_ax = rel_fig.add_subplot(rel_gs[4*i+3, j])

        # Now need to get rid of space between top and bottom axes.
        # First access the position of the axes.
        rel_ax_pos = rel_ax.get_position()
        hist_ax_pos = hist_ax.get_position()

        # Get 2x2 numpy array of the form [[x0, y0], [x1, y1]], where
        # x0, y0, x1, y1 are the coordinates of the axes on the main
        # figure.
        rel_ax_points = rel_ax_pos.get_points()
        hist_ax_points = hist_ax_pos.get_points()

        # Set y coord of top of histogram axis to y coord of bottom of
        # reliability diagram axis.
        hist_ax_points[1][1] = rel_ax_points[0][1]

        hist_ax_pos.set_points(hist_ax_points)

        hist_ax.set_position(hist_ax_pos)

        model.visualise_performance(plot="reliability",
                                    axes=(rel_ax, hist_ax))

        rel_ax.tick_params(direction="inout",
                            which="both",
                            top=True,
                            bottom=False,
                            right=True)

        hist_ax.tick_params(direction="inout",
                            which="both",
                            top=True,
                            right=True)

        if j != 0:
            hist_ax.tick_params(axis="y",
                                pad=0.1)
        else:
            rel_ax.tick_params(axis="y",
                                pad=1)
            hist_ax.tick_params(axis="y",
                                pad=1)

        rel_ax.xaxis.set_minor_locator(mticker.MultipleLocator(0.1))
        rel_ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.1))

        hist_ax.xaxis.set_minor_locator(mticker.MultipleLocator(0.1))
        hist_ax.yaxis.set_minor_locator(mticker.MultipleLocator(50))


        # Remove labels of inner plots.
        # Can't be done with label_outer(), since want to keep yaxis
        # labels of the distributions, since not always the same.
        if i < 4 and j > 0:
            rel_ax.set_ylabel("")
            rel_ax.set_yticklabels([])

            hist_ax.set_ylabel("")

        # Remove ylabels for bottom row.
        if i == 4 and j > 0:
            rel_ax.set_ylabel("")
            rel_ax.set_yticklabels([])

            hist_ax.set_ylabel("")

        # Set titles of columns.
        if i == 0 and j == 0:
            rel_ax.text(1.02, 1.1, "C+",
                        transform = rel_ax.transAxes)

        if i == 0 and j == 2:
            rel_ax.text(1.02, 1.1, "M+",
                        transform = rel_ax.transAxes)

# plt.savefig(WD_TO_SAVE+"rel.jpg", dpi=300, bbox_inches="tight")
plt.show()

# =====================================================================
# PLOT METRICS
# This plot is trying to replicate the one from Leka et al. 2019. Since
# this paper defines the benchmark for comparing flare forecasts, I
# figured that making the plots as similar as possible would keep it
# consistent.

# M CLASS FORECASTS.
ensemble_list_M = [average_M,
                    history_M,
                    CLC_M_BS,
                    ULC_M_BS,
                    CLC_M_MAE,
                    ULC_M_MAE,
                    CLC_M_REL,
                    ULC_M_REL,
                    CLC_M_LCC,
                    ULC_M_LCC]

# C CLASS FORECASTS.
ensemble_list_C = [average_C,
                    history_C,
                    CLC_C_BS,
                    ULC_C_BS,
                    CLC_C_MAE,
                    ULC_C_MAE,
                    CLC_C_REL,
                    ULC_C_REL,
                    CLC_C_LCC,
                    ULC_C_LCC]

# Threshold to convert probabilistic forecasts into dichotomous ones.
PTH = 0.5

def bootstrap(ens_list, metric, verbose=False):
    """Estimate the uncertainty of a performance metric by performing
    bootstrapping with replacement on the forecasts and obsercations
    for each Ensemble in ens_list.

    Parameters
    ----------
    ens_list : list of Ensemble objects
        List of trained ensemble objects.
    metric : str
        Desired mertic to bootstrap.
    verbose : bool
        Whether to display progress of bootstrapping or not

    Returns
    -------
    sdev : np.array, shape(len(ens_list),)
        Standard deviation of the desired metric for each model in
        ens_list.

    """
    # Check if inputs are valid.
    if (not isinstance(ens_list, list)
            or not isinstance(ens_list[0], Ensemble)):
        raise TypeError(" 'ens_list' must be either type ensemble.Ensemble "
                        "object, or a list of type ensemble.Ensemble objects.")

    valid_metrics = ["tss", "bss", "ets", "apss", "fb", "auc"]

    if metric not in valid_metrics:
        raise TypeError("Invalid metric entered. Chose one of "
                        "'tss', 'bss', 'ets', 'apss', 'fb', or 'auc'.")

    iterations = 1000  # Number of bootstrap samples to draw.

    # Generate numpy array to store the confidence intervals for
    # each ensemble.
    sdev = np.zeros(len(ens_list))

    for i, model in enumerate(ens_list):
        if verbose:
            print(f"Bootstrapping {model.desired_weighting} ensemble.")
        forecast = model.forecast
        events = model.events
        metric_list = []  # List to store bootstrapped metrics.
        for j in range(iterations):
            indices = np.random.choice(len(forecast), forecast.shape)
            bootstrap_forecast = forecast[indices]
            bootstrap_events = events[indices]

            if metric == "tss":
                boot_metric = metric_utils.calculate_tss_threshold(
                                                        bootstrap_events,
                                                        bootstrap_forecast,
                                                        PTH
                                                        )

            elif metric == "bss":
                boot_metric = metric_utils.calculate_bss(bootstrap_events,
                                                          bootstrap_forecast
                                                          )

            elif metric == "fb":
                boot_metric = metric_utils.calculate_fb_threshold(
                                                        bootstrap_events,
                                                        bootstrap_forecast,
                                                        PTH
                                                        )

            elif metric == "ets":
                boot_metric = metric_utils.calculate_ets_threshold(
                                                        bootstrap_events,
                                                        bootstrap_forecast,
                                                        PTH
                                                        )

            elif metric == "apss":
                boot_metric = metric_utils.calculate_apss_threshold(
                                                        bootstrap_events,
                                                        bootstrap_forecast,
                                                        PTH
                                                        )

            elif metric == "auc":
                boot_metric = metric_utils.calculate_roc_area(bootstrap_events,
                                                              bootstrap_forecast)

            metric_list.append(boot_metric)

        sdev[i] = np.std(metric_list, ddof=1)

    return sdev

# Make lists containing metric and its bootstrapped uncertainty for
# each ensemble, for each metric.
bss_list_M = [metric_utils.calculate_bss(
    x.events, x.forecast
    ) for x in ensemble_list_M]

bss_err_list_M = bootstrap(ensemble_list_M, "bss")

tss_list_M = [metric_utils.calculate_tss_threshold(
    x.events,
    x.forecast,
    PTH) for x in ensemble_list_M]

tss_err_list_M = bootstrap(ensemble_list_M, "tss")

ets_list_M = [metric_utils.calculate_ets_threshold(
    x.events,
    x.forecast,
    PTH) for x in ensemble_list_M]

ets_err_list_M = bootstrap(ensemble_list_M, "ets")

apss_list_M = [metric_utils.calculate_apss_threshold(
    x.events,
    x.forecast,
    PTH) for x in ensemble_list_M]

apss_err_list_M = bootstrap(ensemble_list_M, "apss")

fb_list_M = [metric_utils.calculate_fb_threshold(
    x.events,
    x.forecast,
    PTH) for x in ensemble_list_M]

fb_err_list_M = bootstrap(ensemble_list_M, "fb")


bss_list_C = [metric_utils.calculate_bss(
    x.events, x.forecast
    ) for x in ensemble_list_C]

bss_err_list_C = bootstrap(ensemble_list_C, "bss")

tss_list_C = [metric_utils.calculate_tss_threshold(
    x.events,
    x.forecast,
    PTH) for x in ensemble_list_C]

tss_err_list_C = bootstrap(ensemble_list_C, "tss")

ets_list_C = [metric_utils.calculate_ets_threshold(
    x.events,
    x.forecast,
    PTH) for x in ensemble_list_C]

ets_err_list_C = bootstrap(ensemble_list_C, "ets")

apss_list_C = [metric_utils.calculate_apss_threshold(
    x.events,
    x.forecast,
    PTH) for x in ensemble_list_C]

apss_err_list_C = bootstrap(ensemble_list_C, "apss")

fb_list_C = [metric_utils.calculate_fb_threshold(
    x.events,
    x.forecast,
    PTH) for x in ensemble_list_C]

fb_err_list_C = bootstrap(ensemble_list_C, "fb")

M_metric_array = np.array((tss_list_M,
                            apss_list_M,
                            ets_list_M,
                            bss_list_M,
                            fb_list_M))

C_metric_array = np.array((tss_list_C,
                            apss_list_C,
                            ets_list_C,
                            bss_list_C,
                            fb_list_C))

M_err_metric_array = np.array((tss_err_list_M,
                                apss_err_list_M,
                                ets_err_list_M,
                                bss_err_list_M,
                                fb_err_list_M))

C_err_metric_array = np.array((tss_err_list_C,
                                apss_err_list_C,
                                ets_err_list_C,
                                bss_err_list_C,
                                fb_err_list_C))

# Names for labelling axes.
score_names = ["TSS", "APSS", "ETS", "BSS", "FB"]

# Define number of metrics, used to position plots, might add more
# so better to have it in general. Also need the rightmost plot for FB,
# where y limits are [0,2], while the other scores' limits are [-1,1]
NO_METRICS = len(score_names)
NO_MODELS = len(ensemble_list_M)

# Create figure.
metric_plot = plt.figure(figsize=(9,6))

# Add gridspec for M metrics.
M_metrics = metric_plot.add_gridspec(nrows=1, ncols=NO_METRICS,
                                      left=0.06, right = 0.95,
                                      top=0.99, bottom = 0.56,
                                      wspace=0)

# Same for C metrics.
C_metrics = metric_plot.add_gridspec(nrows=1, ncols=NO_METRICS,
                                      left=0.06, right = 0.95,
                                      top=0.48, bottom = 0.05,
                                      wspace=0)

# Axes for all scores apart from FB.
M_no_fb = metric_plot.add_subplot(M_metrics[0,:NO_METRICS-1])
C_no_fb = metric_plot.add_subplot(C_metrics[0,:NO_METRICS-1])

# Axes for FB (rightmost axis).
M_fb = metric_plot.add_subplot(M_metrics[0,NO_METRICS-1])
C_fb = metric_plot.add_subplot(C_metrics[0,NO_METRICS-1])

# Alter y axes, plot x axis.
# No fb plots.
M_no_fb.set_ylim(-1,1)
M_no_fb.yaxis.set_ticks([-1,-0.5,0,0.5,1])
M_no_fb.yaxis.set_minor_locator(mticker.MultipleLocator(0.1))
M_no_fb.tick_params(which="both", left=True, direction="inout")
M_no_fb.axhline(0, c="k", ls="--", lw=1)
M_no_fb.text(0.02, 1.01, "M+", va="bottom", ha="left",
              transform=M_no_fb.transAxes)

C_no_fb.set_ylim(-1,1)
C_no_fb.yaxis.set_ticks([-1,-0.5,0,0.5,1])
C_no_fb.yaxis.set_minor_locator(mticker.MultipleLocator(0.1))
C_no_fb.tick_params(which="both", left=True, direction="inout")
C_no_fb.axhline(0, c="k", ls="--", lw=1)
C_no_fb.text(0.02, 1.01, "C+", va="bottom", ha="left",
              transform=C_no_fb.transAxes)

# fb plots.
M_fb.set_ylim(0,2)
M_fb.yaxis.set_ticks([0,0.5,1,1.5,2])
M_fb.yaxis.tick_right()
M_fb.yaxis.set_minor_locator(mticker.MultipleLocator(0.1))
M_fb.tick_params(which="both", left=True, direction="inout")

C_fb.set_ylim(0,2)
C_fb.yaxis.set_ticks([0,0.5,1,1.5,2])
C_fb.yaxis.tick_right()
C_fb.yaxis.set_minor_locator(mticker.MultipleLocator(0.1))
C_fb.tick_params(which="both", left=True, direction="inout")

# Positions for plotting each metric.
x_no_fb = np.arange(1, NO_METRICS)

# Setting x limits, plotting lines to separate each metric.
M_no_fb.set_xlim(0.5, NO_METRICS-0.5)
M_no_fb.xaxis.set_ticks(x_no_fb)
M_no_fb.set_xticklabels(score_names[:-1])
M_no_fb.vlines(np.linspace(1.5, NO_METRICS-1.5, NO_METRICS-2), ymin=-1, ymax=1,
                colors="grey", linestyles="dotted")

C_no_fb.set_xlim(0.5, NO_METRICS-0.5)
C_no_fb.xaxis.set_ticks(x_no_fb)
C_no_fb.set_xticklabels(score_names[:-1])
C_no_fb.vlines(np.linspace(1.5, NO_METRICS-1.5, NO_METRICS-2), ymin=-1, ymax=1,
                colors="grey", linestyles="dotted")

M_fb.set_xlim([0.5,1.5])
M_fb.xaxis.set_ticks([1])
M_fb.set_xticklabels([score_names[-1]])

C_fb.set_xlim([0.5,1.5])
C_fb.xaxis.set_ticks([1])
C_fb.set_xticklabels([score_names[-1]])

W = 0.7  # Width of region points will be plotted.

artists = []
labels = []

for i in range(NO_MODELS):
    x_pos_no_fb = x_no_fb - (W/2.0) + i*(W/float(NO_MODELS - 1))
    x_pos_fb = 1 - (W/2.0) + i*(W/float(NO_MODELS - 1))

    # Plot points for unambiguity of position.
    M_no_fb_point = M_no_fb.errorbar(x_pos_no_fb, M_metric_array[:-1,i],
                                      yerr=M_err_metric_array[:-1,i],
                                      elinewidth=0.5, capsize=3, capthick=0.5,
                                      color=ensemble_list_M[i].colour, fmt=".", ms=5)
    C_no_fb_point = C_no_fb.errorbar(x_pos_no_fb, C_metric_array[:-1,i],
                                      yerr=C_err_metric_array[:-1,i],
                                      elinewidth=0.5, capsize=3, capthick=0.5,
                                      color=ensemble_list_M[i].colour, fmt=".", ms=5)

    M_fb_point = M_fb.errorbar(x_pos_fb, M_metric_array[-1,i],
                                yerr=M_err_metric_array[-1,i],
                                elinewidth=0.5, capsize=3, capthick=0.5,
                                color=ensemble_list_M[i].colour, fmt=".", ms=5)
    C_fb_point = C_fb.errorbar(x_pos_fb, C_metric_array[-1,i],
                                yerr=C_err_metric_array[-1,i],
                                elinewidth=0.5, capsize=3, capthick=0.5,
                                color=ensemble_list_M[i].colour, fmt=".", ms=5)

    M_no_fb_point_test, = M_no_fb.plot(x_pos_no_fb, M_metric_array[:-1,i],".",
                                      color=ensemble_list_M[i].colour,ms=5)
    # Plot the symbols.
    M_no_fb_marker, = M_no_fb.plot(x_pos_no_fb, M_metric_array[:-1,i],
                                  ls="",fillstyle="none",ms=9,
                                      color=ensemble_list_M[i].colour,
                                      marker=ensemble_list_M[i].format)
                                      # facecolors="none", s=90)
    C_no_fb_marker, = C_no_fb.plot(x_pos_no_fb, C_metric_array[:-1,i],
                                  ls="",fillstyle="none",ms=9,
                                      color=ensemble_list_M[i].colour,
                                      marker=ensemble_list_M[i].format)
    M_fb_marker, = M_fb.plot(x_pos_fb, M_metric_array[-1,i],
                                  ls="",fillstyle="none",ms=9,
                                      color=ensemble_list_M[i].colour,
                                      marker=ensemble_list_M[i].format)
    C_fb_marker, = C_fb.plot(x_pos_fb, C_metric_array[-1,i],
                                  ls="",fillstyle="none",ms=9,
                                      color=ensemble_list_M[i].colour,
                                      marker=ensemble_list_M[i].format)

    if i < 2:
        artists.append(M_no_fb_marker)
        labels.append(ensemble_list_M[i].desired_weighting)

    if i == 2:
        artists.append(mpatches.Patch(color="white"))
        labels.append("")

    if i == 3:
        artists.append(mpatches.Patch(color=ensemble_list_M[i+1].colour,
                                      ec="black", lw=0.5))
        labels.append(ensemble_list_M[i+1].desired_weighting)
        artists.append(mpatches.Patch(color=ensemble_list_M[i].colour,
                                      ec="black", lw=0.5))
        labels.append(ensemble_list_M[i].desired_weighting)

    if i >= 2:
        if i % 2 == 0:
            double_handle = []
        double_handle.append(M_no_fb_marker)

        if i % 2 == 1:
            artists.append(tuple(double_handle))
            labels.append(ensemble_list_M[i].desired_metric)

M_fb.legend(artists, labels,
            bbox_to_anchor=(1.95, 1.035),frameon=False, handlelength=2,
              handler_map={tuple: HandlerTuple(ndivide=None)})
# plt.savefig(WD_TO_SAVE+"metric_plot.jpg", dpi=300, bbox_inches="tight")
plt.show()

# ---------------------------------------------------------------------
# Create table in latex form with all scores and uncertainties.

table_headers = ["TSS", "ApSS", "ETS", "BSS", "AUC"]
model_column = ["Average", "EV", "CLC, BS", "ULC, BS", "CLC, MAE", "ULC, MAE",
                "CLC, REL", "ULC, REL", "CLC, LCC", "ULC, LCC"]

# Calcuate AUC of ROC curve.
roc_list_M = [metric_utils.calculate_roc_area(
    x.events, x.forecast
    ) for x in ensemble_list_M]

roc_err_list_M = bootstrap(ensemble_list_M, "auc")

roc_list_C = [metric_utils.calculate_roc_area(
    x.events, x.forecast
    ) for x in ensemble_list_C]

roc_err_list_C = bootstrap(ensemble_list_C, "auc")

# Get rid of FB (index -1) and add in roc area.
M_metrics_latex = np.append(M_metric_array[:-1], [roc_list_M],axis=0).T
M_err_latex = np.append(M_err_metric_array[:-1], [roc_err_list_M], axis=0).T

# Empty array to store metrics and bootstrapped uncertainties in string
# format.
M_latex_table = np.empty(M_metrics_latex.shape, dtype=object)

for i in range(len(M_latex_table)):
    for j in range(len(M_latex_table[i])):
        M_latex_table[i][j] = "{:.2} ({:.2})".format(M_metrics_latex[i][j],
                                                      M_err_latex[i][j])

# Create latex table of pandas dataframe.
ensemble_M_metrics_latex = pd.DataFrame(M_latex_table,
                                        columns=table_headers,
                                        index=model_column
                                        ).to_latex(column_format="lrrrr")

C_metrics_latex = np.append(C_metric_array[:-1], [roc_list_C],axis=0).T
C_err_latex = np.append(C_err_metric_array[:-1], [roc_err_list_C], axis=0).T
C_latex_table = np.empty(C_metrics_latex.shape, dtype=object)

for i in range(len(M_latex_table)):
    for j in range(len(M_latex_table[i])):
        C_latex_table[i][j] = "{:.2} ({:.2})".format(C_metrics_latex[i][j],
                                                      C_err_latex[i][j])

ensemble_C_metrics_latex = pd.DataFrame(C_latex_table,
                                        columns=table_headers,
                                        index=model_column
                                        ).to_latex(column_format="lrrrr")

print(ensemble_M_metrics_latex)
print(ensemble_C_metrics_latex)

# ---------------------------------------------------------------------------
# EXAMPLE PLOTS FOR INTRODUCTION OF THESIS.
# Will plot ULC C BS figures.

example_index = np.where(np.array(model_column) == "ULC, BS")[0][0]

rocfig = plt.figure(figsize=(5,5))
rocax = rocfig.add_subplot(111)
ULC_C_BS.visualise_performance(plot="ROC", ax=rocax, display_auc=False)
rocax.tick_params(axis="both", which="major", top=True, right=True,
                  direction="inout")
rocax.text(0.98, 0.03,
            r"AUC = $\mathdefault{%.2f \pm %.2f}$" %(roc_list_C[example_index],
                                                    roc_err_list_C[example_index]),
        horizontalalignment="right",
        verticalalignment="bottom",
        transform = rocax.transAxes)
# plt.savefig(WD_TO_SAVE+"testroc.jpg", dpi=300, bbox_inches="tight")
plt.show()

test_relfig = plt.figure(figsize=(5,5))
relgs1 = test_relfig.add_gridspec(nrows=4, ncols=1, hspace=0)
relax1 = test_relfig.add_subplot(relgs1[0:3, 0])
relax2 = test_relfig.add_subplot(relgs1[3, 0], sharex=relax1)
ULC_C_BS.visualise_performance(plot="reliability", axes=(relax1, relax2))
relax1.tick_params(axis="both", which="major", top=True, right=True,
                    direction="inout")
relax2.tick_params(axis="both", which="major", top=True, right=True,
                    direction="inout")
# plt.savefig(WD_TO_SAVE+"testrel.jpg", dpi=300, bbox_inches="tight")
plt.show()

# ---------------------------------------------------------------------------
# ROC PLOTS.

# Reformat auc arrays in order to print out values on ROC plots.
auc_vals = np.zeros((5,4))
auc_errs = np.zeros((5,4))

# C models AUC values and erros
auc_vals[:,:2] = np.reshape(roc_list_C, (5,2))
auc_errs[:,:2] = np.reshape(roc_err_list_C, (5,2))

# M models AUC values and erros
auc_vals[:,2:] = np.reshape(roc_list_M, (5,2))
auc_errs[:,2:] = np.reshape(roc_err_list_M, (5,2))

roc_fig, roc_axs = plt.subplots(5,4, figsize=(9,10),
                                gridspec_kw={"wspace":0.05,"hspace":0.05})

for i, model in enumerate(layout.flatten()):
    model.visualise_performance(plot="ROC", ax=roc_axs.flatten()[i],
                                display_auc=False)


# Hide inner x and y labels, adjust ticks.
for i, ax in enumerate(roc_axs.flatten()):
    ax.text(0.98, 0.03,r"AUC = $\mathdefault{%.2f \pm %.2f}$" %(auc_vals.flatten()[i],
                                                          auc_errs.flatten()[i]),
            horizontalalignment="right",
            verticalalignment="bottom",
            transform = ax.transAxes)
    ax.label_outer()
    ax.tick_params(direction="inout", which="both", top=True, right=True)
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.1))

roc_axs.flatten()[0].text(1.05, 1.1, "C+",
                          ha="center", va="center",
                          transform=roc_axs.flatten()[0].transAxes)

roc_axs.flatten()[2].text(1.05, 1.1, "M+",
                          ha="center", va="center",
                          transform=roc_axs.flatten()[2].transAxes)

# plt.savefig(WD_TO_SAVE+"ROC.jpg", dpi=300, bbox_inches="tight")
plt.show()

# =====================================================================
# PLOT WEIGHTS.

no_colours = len(ULC_M_BS.weights)  # Number of colours.

# Range of indices 0 to number of weights, to be shuffled.
ran_cols = np.linspace(0, no_colours-1, no_colours, dtype=int)

np.random.seed(100)  # Set seed for reproducibility.
np.random.shuffle(ran_cols)  # Shuffle indices.

# Now get twighlight colormap that has been divided evenly into
# no_colours colours. This will ensure good colour scheme.
cmap = plt.cm.get_cmap("twilight", no_colours)

# Empty arrays to store weights and their standard deviations.
weights = np.empty_like(layout)
weight_stds = np.empty_like(layout)

for i in range(len(weights)):
    for j in range(len(weights[0])):
        current_model = layout[i][j]
        if current_model.desired_weighting == "ULC":
            ws = current_model.weights
        else:
            # Need to account for fact that all but ULC models dont
            # include climatology in final prediction, so add a 0 to
            # the weights array. This will ensure neat layout in the
            # figure.
            ws = np.append(current_model.weights,0)
        if i == 0:
            # Since average and EV schemes dont provide sdevs, set
            # to none. This is dealt with in the plotting function/
            weight_stds[i][j] = None
        else:
            # Sdev of weight distribution.
            weight_std = np.std(current_model.ac_weights, axis=0)

            if current_model.desired_weighting != "ULC":
                # Again, account for models that dont use climatology.
                weight_std = np.append(weight_std, 0)

            weight_stds[i][j] = weight_std

        weights[i][j] = ws

# Width of each individual bar.
width = (W/float(len(weights[0][0]) - 1))

def plot_weights(weights, axis, yerr=None, return_bar=False):
    """Plot each weight on ax. Will return list of bar artists if
    return_bar is True, which will be used for legend.

    Parameters
    ----------
    weights : np.array, shape (number of weights,)
        Weights of model.
    axis : matplotlib axis
        Axis to plot the weights.
    yerr : np.array
        Standard deviations of weights. The default is None.
    return_bar : bool, optional
        Whether or not to return the bar artists. The default is False.

    Returns
    -------
    bar_artists : list of matplotlib.patches.Patch objects
        Bar artists for legend, only returned when return_bar is True.

    """
    # Set y limits of axis, uses all_weights from global scope.
    axis.set_ylim(weights.flatten().min()-0.1, 1)

    bar_artists = []  # List to store bar artists.

    for i in range(len(weights)):
        # Position of the bar.
        pos = 1 - (W/2.0) + i*(W/float(len(weights) - 1))
        # print(ran_cols[i])
        # Plot the bar.
        # print(i, ran_cols[i])
        if yerr is not None:
            axis.bar(pos, weights[i], width=width, align="center",
                      linewidth=0.5, edgecolor="black", yerr=yerr[i],
                      color=cmap(ran_cols[i]), error_kw={"elinewidth":0.2,
                                                          "capsize":1.5,
                                                          "capthick":0.2,
                                                          "alpha":0.8})

        else:
            axis.bar(pos, weights[i], width=width, align="center",
                  linewidth=0.5, edgecolor="black",
                  color=cmap(ran_cols[i]))
        if return_bar:
            # Create patch object whose colour corresponds to colour of
            # current bar.
            patch = mpatches.Patch(color=cmap(ran_cols[i]),
                                    ec="black", lw=0.5)
            bar_artists.append(patch)  # Append patch to list.
    # print("\n\n")
    if return_bar:
        return bar_artists

weight_fig = plt.figure(figsize=(10,12))

weight_gs = weight_fig.add_gridspec(5, 4, wspace=0.25)

# Plot weights of each model.
for i in range(len(layout)):
    for j in range(len(layout[i])):

        # Create axes for each plot.
        ax = weight_fig.add_subplot(weight_gs[i, j])

        # Clear xticks, since x axis provides no info.
        ax.set_xticks([])

        # Plot the weights.
        plot_weights(weights[i][j], ax, yerr=weight_stds[i][j])

        if i == len(layout)-1 and j == len(layout[0])-1:
            # Return artists of ULC M1+, includes climatology.
            bar = plot_weights(weights[i][j], ax, weight_stds[i][j],
                                return_bar=True)

        # x limits, also for plotting horizontal axis.
        X_MIN = 1 - (W/2.0) - 0.05
        X_MAX = 1 + (W/2.0) + 0.05

        ax.plot([X_MIN, X_MAX], [0, 0], "k-", lw=0.9)
        ax.set_xlim(X_MIN, X_MAX)

        current_model = layout[i][j]

        if (current_model.desired_weighting == "Average" or
            current_model.desired_weighting == "EV"):
            axis_text = current_model.desired_weighting
        else:
            axis_text = "{}, {}".format(current_model.desired_weighting,
                                        current_model.desired_metric)

        ax.text(0.0, 1.02, axis_text, va="bottom", ha="left",
                transform=ax.transAxes)

        ax.yaxis.set_major_locator(mticker.MultipleLocator(0.5))
        ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.25))
        ax.tick_params(which="both", length=4)

        # Plot horizontal grid for visual aid.
        ax.grid(which="both", axis="y", ls="--", lw = 0.9)

        # Include ticks on right hand side.
        ax.tick_params(right=True, which="both", direction="inout")


        if i == 0:
            ax.set_ylim(0.-0.1, 0.27)
            ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.05))
            ax.yaxis.set_major_locator(mticker.MultipleLocator(0.25))
            ax.grid(which="minor", axis="y", ls=":", lw = 0.5)
            ax.tick_params(which="minor", length=2)

all_axes = weight_fig.get_axes()
all_axes[0].text(1.02, 1.15, "C+",
                transform = all_axes[0].transAxes)
all_axes[2].text(1.02, 1.15, "M+",
                transform = all_axes[2].transAxes)
all_axes[3].legend(handles=bar,
                    labels=list(ULC_M_BS.df_of_models),
                    bbox_to_anchor=(1, 1.09),frameon=False)
all_axes[8].set_ylabel("Weight value", size="x-large")

# plt.savefig(WD_TO_SAVE+"weights.jpg", dpi=300, bbox_inches="tight")
plt.show()

done = time.time() - now
print("Entire process done in %.3f seconds, or %.3f minutes" % (done, done/60.0))
