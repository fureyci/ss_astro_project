# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 15:57:22 2021

@author: fureyci

Code to test functionality of ensemble.py.

Prompts user for input into the console to chose desired flare
forecast, desired weighting scheme, and desired metric to
optimise, if the weighting scheme is constrained linear
combination.

As of 03/11/2021, contains the plots that will be used in my thesis.

"""
import numpy as np
from matplotlib import pyplot as plt, patches as mpatches, ticker as mticker
from opening_data import load_benchmark_data
import metric_utils
from ensemble import Ensemble

# Load data.
events, model_csv_list, model_names_only = load_benchmark_data()

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
    valid_methods = ["average", "history", "constrained", "unconstrained"]
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
          "Choose one of 'average', 'history', "
          "'constrained' or 'unconstrained'.")

    weighting = input("Enter desired scheme: ")

    while True: # Loop until all inputs are valid.
        if weighting in valid_methods:
            if weighting in valid_methods[:2]:
                metric = None
                # Dont need to optimise metric if using average or
                # performance history ensemble.
            else:
                print("\nWhat metric would you like to optimise?\n"
                      "Choose one of 'brier', 'LCC', or 'MAE'.")
                metric = input("Enter desired metric: ")
                while True:
                    if metric in valid_weights:
                        break
                    else:
                        # Ensure metric is valid.
                        print("\n Please choose one of 'brier', 'LCC', "
                              "or 'MAE'.")
                        metric = input("Enter desired metric: ")
            break

        else:
            # Ensure weighting is valid.
            print("\nPlease choose one of 'average', 'history', "
                  "'constrained' or 'unconstrained'.")
            weighting = input("Enter desired scheme: ")

    return forecast, weighting, metric

forecast, weighting, metric = user_input()

print("\nBuilding ensemble...")
test = Ensemble(model_csv_list, model_names_only, events,
                desired_forecast=forecast,
                desired_metric=metric,
                desired_weighting=weighting)
print("Done. Now plotting reliability diagram and ROC curve...")

# # Alternatively, enter manually.
#
# FORECAST = "C1+"
# METRIC = "brier"
# WEIGHTING = "constrained"

# test = Ensemble(model_csv_list, model_names_only, events,
#                 desired_forecast=FORECAST,
#                 desired_metric=METRIC,
#                 desired_weighting=WEIGHTING)

test.visualise_performance(which="both")

# ===========================================================================
# VISUALISE PERFORMANCE ENSEMBLES.

# Instantiate models.

# Average weighting, M class.
average_M = Ensemble(model_csv_list, model_names_only, events,
                desired_forecast="M1+",
                desired_weighting="average")

# Average weighting, C class.
average_C = Ensemble(model_csv_list, model_names_only, events,
                desired_forecast="C1+",
                desired_weighting="average")

# Performance history weighting, M class.
history_M = Ensemble(model_csv_list, model_names_only, events,
                desired_forecast="M1+",
                desired_weighting="history")

# Performance history weighting, C class.
history_C = Ensemble(model_csv_list, model_names_only, events,
                desired_forecast="C1+",
                desired_weighting="history")

# CLC weighting, brier score optimised, M class.
constrained_M_brier = Ensemble(model_csv_list, model_names_only, events,
                                desired_forecast="M1+",
                                desired_metric="brier",
                                desired_weighting="constrained")

# ULC weighting, brier score optimised, M class.
unconstrained_M_brier = Ensemble(model_csv_list, model_names_only, events,
                                desired_forecast="M1+",
                                desired_metric="brier",
                                desired_weighting="unconstrained")

# CLC weighting, brier score optimised, C class.
constrained_C_brier = Ensemble(model_csv_list, model_names_only, events,
                                desired_forecast="C1+",
                                desired_metric="brier",
                                desired_weighting="constrained")

# ULC weighting, brier score optimised, C class.
unconstrained_C_brier = Ensemble(model_csv_list, model_names_only, events,
                                desired_forecast="C1+",
                                desired_metric="brier",
                                desired_weighting="unconstrained")
# ---------------------------------------------------------------------------
# ROC PLOTS.

# Build figure and axes.
ROC_fig, \
    ((avM_ROC_ax, avC_ROC_ax),(hiM_ROC_ax, hiC_ROC_ax), \
      (ccM_ROC_ax, ccC_ROC_ax),(ucM_ROC_ax, ucC_ROC_ax)) \
    = plt.subplots(4,2, figsize=(13,16.4),sharex=True,sharey=True)

# Use "visualise_performance()" method, set the kwarg "plot" to "ROC".
average_M.visualise_performance(plot="ROC", ax=avM_ROC_ax)
average_C.visualise_performance(plot="ROC", ax=avC_ROC_ax)
history_M.visualise_performance(plot="ROC", ax=hiM_ROC_ax)
history_C.visualise_performance(plot="ROC", ax=hiC_ROC_ax)
constrained_M_brier.visualise_performance(plot="ROC", ax=ccM_ROC_ax)
constrained_C_brier.visualise_performance(plot="ROC", ax=ccC_ROC_ax)
unconstrained_M_brier.visualise_performance(plot="ROC", ax=ucM_ROC_ax)
unconstrained_C_brier.visualise_performance(plot="ROC", ax=ucC_ROC_ax)

# Hide inner x and y labels.
for ax in ROC_fig.get_axes():
    ax.label_outer()

# Set titles.
avM_ROC_ax.set_title("M1+", fontsize="x-large")
avC_ROC_ax.set_title("C1+", fontsize="x-large")

plt.tight_layout()
# plt.savefig("ROC.jpg", dpi=250)
plt.show()

# ---------------------------------------------------------------------------
# RELIABILITY DIAGRAMS.

rel_fig = plt.figure(figsize=(12,16))


avM_gs = rel_fig.add_gridspec(nrows=4, ncols=1,
                              left=0.055, right=0.5075,
                              top=0.985, bottom = 0.7575,
                              hspace=0)

avC_gs = rel_fig.add_gridspec(nrows=4, ncols=1,
                              left=0.5375, right=0.99,
                              top=0.985, bottom = 0.7575,
                              hspace=0)

hiM_gs = rel_fig.add_gridspec(nrows=4, ncols=1,
                              left=0.055, right=0.5075,
                              top=0.7425, bottom = 0.515,
                              hspace=0)

hiC_gs = rel_fig.add_gridspec(nrows=4, ncols=1,
                              left=0.5375, right=0.99,
                              top=0.7425, bottom=0.515,
                              hspace=0)

ccM_gs = rel_fig.add_gridspec(nrows=4, ncols=1,
                              left=0.055, right=0.5075,
                              top=0.50, bottom = 0.2725,
                              hspace=0)

ccC_gs = rel_fig.add_gridspec(nrows=4, ncols=1,
                              left=0.5375, right=0.99,
                              top=0.50, bottom=0.2725,
                              hspace=0)

ucM_gs = rel_fig.add_gridspec(nrows=4, ncols=1,
                              left=0.055, right=0.5075,
                              top=0.2575, bottom = 0.03,
                              hspace=0)

ucC_gs = rel_fig.add_gridspec(nrows=4, ncols=1,
                              left=0.5375, right=0.99,
                              top=0.2575, bottom=0.03,
                              hspace=0)


# Top left corner.
avM_rel_ax = rel_fig.add_subplot(avM_gs[0:3, 0])
avM_hist_ax = rel_fig.add_subplot(avM_gs[3, 0])

# Top right corner.
avC_rel_ax = rel_fig.add_subplot(avC_gs[0:3, 0])
avC_hist_ax = rel_fig.add_subplot(avC_gs[3, 0])

# 2nd row, left.
hiM_rel_ax = rel_fig.add_subplot(hiM_gs[0:3, 0])
hiM_hist_ax = rel_fig.add_subplot(hiM_gs[3, 0])

# 2nd row, right.
hiC_rel_ax = rel_fig.add_subplot(hiC_gs[0:3, 0])
hiC_hist_ax = rel_fig.add_subplot(hiC_gs[3, 0])

# 3rd row, left.
ccM_rel_ax = rel_fig.add_subplot(ccM_gs[0:3, 0])
ccM_hist_ax = rel_fig.add_subplot(ccM_gs[3, 0])

# 3rd row, right.
ccC_rel_ax = rel_fig.add_subplot(ccC_gs[0:3, 0])
ccC_hist_ax = rel_fig.add_subplot(ccC_gs[3, 0])

# Bottom row, left.
ucM_rel_ax = rel_fig.add_subplot(ucM_gs[0:3, 0])
ucM_hist_ax = rel_fig.add_subplot(ucM_gs[3, 0])

# Bottom row, right.
ucC_rel_ax = rel_fig.add_subplot(ucC_gs[0:3, 0])
ucC_hist_ax = rel_fig.add_subplot(ucC_gs[3, 0])

average_M.visualise_performance(plot="reliability",
                                axes=(avM_rel_ax, avM_hist_ax))
average_C.visualise_performance(plot="reliability",
                                axes=(avC_rel_ax, avC_hist_ax))
history_M.visualise_performance(plot="reliability",
                                axes=(hiM_rel_ax, hiM_hist_ax))
history_C.visualise_performance(plot="reliability",
                                axes=(hiC_rel_ax, hiC_hist_ax))
constrained_M_brier.visualise_performance(plot="reliability",
                                axes=(ccM_rel_ax, ccM_hist_ax))
constrained_C_brier.visualise_performance(plot="reliability",
                                axes=(ccC_rel_ax, ccC_hist_ax))
unconstrained_M_brier.visualise_performance(plot="reliability",
                                axes=(ucM_rel_ax, ucM_hist_ax))
unconstrained_C_brier.visualise_performance(plot="reliability",
                                axes=(ucC_rel_ax, ucC_hist_ax))

# Adjust ticks and tick labels.
# Top left.
avM_rel_ax.tick_params(top=True, right=True, direction="inout")
avM_hist_ax.set_xticklabels([])
avM_hist_ax.set_xlabel("")
avM_hist_ax.tick_params(top=True, direction="inout")

# Top right.
avC_rel_ax.tick_params(top=True, right=True, direction="inout")
avC_rel_ax.set_yticklabels("")
avC_rel_ax.set_ylabel("")
avC_rel_ax.get_legend().remove()
avC_hist_ax.set_xticklabels([])
avC_hist_ax.set_xlabel("")
avC_hist_ax.set_ylabel("")
avC_hist_ax.tick_params(top=True, direction="inout")

# 2nd row, left.
hiM_rel_ax.tick_params(top=True, right=True, direction="inout")
hiM_rel_ax.get_legend().remove()
hiM_hist_ax.set_xticklabels([])
hiM_hist_ax.set_xlabel("")
hiM_hist_ax.tick_params(top=True, direction="inout")


# 2nd row, right.
hiC_rel_ax.tick_params(top=True, right=True, direction="inout")
hiC_rel_ax.set_yticklabels("")
hiC_rel_ax.set_ylabel("")
hiC_rel_ax.get_legend().remove()
hiC_hist_ax.set_xticklabels([])
hiC_hist_ax.set_xlabel("")
hiC_hist_ax.set_ylabel("")
hiC_hist_ax.tick_params(top=True, direction="inout")

# 3rd row, left.
ccM_rel_ax.tick_params(top=True, right=True, direction="inout")
ccM_rel_ax.get_legend().remove()
ccM_hist_ax.set_xticklabels([])
ccM_hist_ax.set_xlabel("")
ccM_hist_ax.tick_params(top=True, direction="inout")

# 3rd row, right.
ccC_rel_ax.tick_params(top=True, right=True, direction="inout")
ccC_rel_ax.set_yticklabels("")
ccC_rel_ax.set_ylabel("")
ccC_rel_ax.get_legend().remove()
ccC_hist_ax.set_xticklabels([])
ccC_hist_ax.set_xlabel("")
ccC_hist_ax.set_ylabel("")
ccC_hist_ax.tick_params(top=True, direction="inout")

# Bottom row, left.
ucM_rel_ax.tick_params(top=True, right=True, direction="inout")
ucM_rel_ax.get_legend().remove()
ucM_hist_ax.tick_params(top=True, direction="inout")

# Bottom row, right.
ucC_rel_ax.tick_params(top=True, right=True, direction="inout")
ucC_rel_ax.set_yticklabels("")
ucC_rel_ax.set_ylabel("")
ucC_rel_ax.get_legend().remove()
ucC_hist_ax.set_ylabel("")
ucC_hist_ax.tick_params(top=True, direction="inout")

# Label flare class.
avM_rel_ax.set_title("M1+")
avC_rel_ax.set_title("C1+")

# plt.savefig("rel.jpg", dpi=250)
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
                    constrained_M_brier,
                    unconstrained_M_brier]

# C CLASS FORECASTS.
ensemble_list_C = [average_C,
                    history_C,
                    constrained_C_brier,
                    unconstrained_C_brier]

# Threshold to convert probabilistic forecasts into dichotomous ones.
# If p > PTH, it will be registered as a flare occured, whereas if
# p < PTH, this will imply a flare did not occur.
PTH = 0.5

def bootstrap(ens_list, metric, verbose=False):
    """Estimate the uncertainty of a performance metric by performing
    bootstrapping with replacement on the forecasts and events
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

    valid_metrics = ["tss", "bss", "ets", "apss", "fb"]

    if metric not in valid_metrics:
        raise TypeError("Invalid metric entered. Chose one of "
                        "'tss', 'bss', 'ets', 'apss', or 'fb'.")

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

M_metric_array = np.array((bss_list_M,
                            tss_list_M,
                            ets_list_M,
                            apss_list_M,
                            fb_list_M))

C_metric_array = np.array((bss_list_C,
                            tss_list_C,
                            ets_list_C,
                            apss_list_C,
                            fb_list_C))

M_err_metric_array = np.array((bss_err_list_M,
                                tss_err_list_M,
                                ets_err_list_M,
                                apss_err_list_M,
                                fb_err_list_M))

C_err_metric_array = np.array((bss_err_list_C,
                                tss_err_list_C,
                                ets_err_list_C,
                                apss_err_list_C,
                                fb_err_list_C))

# Names for labelling axes.
score_names = ["BSS", "TSS", "ETS", "APSS", "FB"]

# Define number of metrics, used to position plots, might add more
# so better to have it in general. Also need the rightmost plot for FB,
# where y limits are [0,2], while the other scores' limits are [-1,1]
NO_METRICS = len(score_names)
NO_MODELS = len(ensemble_list_M)

# Create figure.
metric_plot = plt.figure(figsize=(2*(NO_METRICS+1), 6))

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

C_no_fb.set_ylim(-1,1)
C_no_fb.yaxis.set_ticks([-1,-0.5,0,0.5,1])
C_no_fb.yaxis.set_minor_locator(mticker.MultipleLocator(0.1))
C_no_fb.tick_params(which="both", left=True, direction="inout")
C_no_fb.axhline(0, c="k", ls="--", lw=1)

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

W = 0.4  # width of region points will be plotted

for i in range(NO_MODELS):
    x_pos_no_fb = x_no_fb - (W/2.0) + i*(W/float(NO_MODELS - 1))
    x_pos_fb = 1 - (W/2.0) + i*(W/float(NO_MODELS - 1))

    # Plot points for unambiguity of position.
    M_no_fb.errorbar(x_pos_no_fb, M_metric_array[:-1,i],
                      yerr=M_err_metric_array[:-1,i],
                  elinewidth=0.5, capsize=3, capthick=0.5,
                      color="#1f77b4", fmt=".", ms=5)
    C_no_fb.errorbar(x_pos_no_fb, C_metric_array[:-1,i],
                      yerr=C_err_metric_array[:-1,i],
                  elinewidth=0.5, capsize=3, capthick=0.5,
                      color="#1f77b4", fmt=".", ms=5)

    M_fb.errorbar(x_pos_fb, M_metric_array[-1,i],
                  yerr=M_err_metric_array[-1,i],
                  elinewidth=0.5, capsize=3, capthick=0.5,
                  color="#1f77b4", fmt=".", ms=5)
    C_fb.errorbar(x_pos_fb, C_metric_array[-1,i],
                  yerr=C_err_metric_array[-1,i],
                  elinewidth=0.5, capsize=3, capthick=0.5,
                  color="#1f77b4", fmt=".", ms=5)

    # Plot the symbols.
    M_no_fb.scatter(x_pos_no_fb, M_metric_array[:-1,i],
                    color="#1f77b4", marker=ensemble_list_M[i].format,
                    facecolors="none", s=90)
    C_no_fb.scatter(x_pos_no_fb, C_metric_array[:-1,i],
                    color="#1f77b4", marker=ensemble_list_M[i].format,
                    facecolors="none", s=90)
    M_fb.scatter(x_pos_fb, M_metric_array[-1,i],
                    color="#1f77b4", marker=ensemble_list_M[i].format,
                    facecolors="none", s=90)
    C_fb.scatter(x_pos_fb, C_metric_array[-1,i],
                    color="#1f77b4", marker=ensemble_list_M[i].format,
                    facecolors="none", s=90)

# plt.savefig("metric_plot.jpg", dpi=300)
plt.show()

# =====================================================================
# PLOT WEIGHTS.

no_colours = len(unconstrained_M_brier.weights)  # Number of colours.

# Range of indices 0 to number of weights, to be shuffled.
ran_cols = np.linspace(0, no_colours-1, no_colours, dtype=int)

np.random.seed(100)  # Set seed for reproducibility.
np.random.shuffle(ran_cols)  # Shuffle indices.

# Now get twighlight colormap that has been divided evenly into
# no_colours colours. This will ensure good colour scheme.
cmap = plt.cm.get_cmap("twilight", no_colours)

# Create array to store all the weights.
# Appending 0 to all but the unconstrained weights to ensure that
# bars of the same model are directly below each other.
# Since climatology used in ULC, that means there is one more weight
# than the CLC or performance history weighting scheme.
all_weights = np.array((np.append(history_C.weights, 0),
                    np.append(history_M.weights, 0),
                    np.append(constrained_C_brier.weights, 0),
                    np.append(constrained_M_brier.weights, 0),
                    unconstrained_C_brier.weights,
                    unconstrained_M_brier.weights),
                    dtype=object)

# Width of each individual bar.
width = (W/float(len(all_weights[0]) - 1))

def plot_weights(weights, axis, return_bar=False):
    """Plot each weight on ax. Will return list of bar artists if
    return_bar is True, which will be used for legend.

    Parameters
    ----------
    weights : np.array, shape (number of weights,)
        Weights of model.
    axis : matplotlib axis
        Axis to plot the weights.
    return_bar : bool, optional
        Whether or not to return the bar artists. The default is False.

    Returns
    -------
    bar_artists : list of matplotlib.patches.Patch objects
        Bar artists for legend, only returned when return_bar is True.

    """
    # Set y limits of axis, uses all_weights from global scope.
    axis.set_ylim(all_weights.flatten().min()-0.1, 1)

    bar_artists = []  # List to store bar artists.

    for i in range(len(weights)):
        # Oosition of the bar.
        pos = 1 - (W/2.0) + i*(W/float(len(weights) - 1))

        # Plot the bar.
        axis.bar(pos, weights[i], width=width, align="center",
                  linewidth=0.5, edgecolor="black",
                  color=cmap(ran_cols[i]))
        if return_bar:
            # Create patch object whose colour corresponds to colour of
            # current bar.
            patch = mpatches.Patch(color=cmap(ran_cols[i]),
                                    ec="black", lw=0.5)
            bar_artists.append(patch)  # Append patch to list.

    if return_bar:
        return bar_artists

# ylables for each plot.
labels = ["History", "Constrained", "Unconstrained"]

# Create figure and subplots.
weight_fig, axes = plt.subplots(3, 2, figsize=(9,9),
                                sharex="all",
                                gridspec_kw={"wspace":0.05,
                                              "hspace":0.1})

# Loop through each axis. Use .flatten() since axes is array shape
# (nrows, ncols).
for i, axis in enumerate(axes.flatten()):
    axis.set_xticks([]) # Clear xticks, since x axis provides no info.
    axis.label_outer() # Remove yticks from inner plots.
    plot_weights(all_weights[i], axis) # Plot the weights.

    if i == len(axes.flatten())-1:
        # Return artists of ULC M1+, includes climatology.
        bar = plot_weights(all_weights[i], axis, return_bar=True)

    # x limits, also for plotting horizontal axis.
    X_MIN = 1 - (W/2.0) - 0.05
    X_MAX = 1 + (W/2.0) + 0.05

    axis.plot([X_MIN, X_MAX], [0, 0], "k-", lw=0.9)
    axis.set_xlim(X_MIN, X_MAX)

    # Set titles.
    if i == 0:
        axis.set_title("C1+")
    if i == 1:
        axis.set_title("M1+")

    # Set y labels.
    if i % 2 == 0:
        axis.set_ylabel(labels[i//2])

    # Adjust y limits for performance history weighting scheme, since
    # values are so small.
    if i < 2:
        axis.set_ylim(0.-0.1, 0.27)
        axis.yaxis.set_minor_locator(mticker.MultipleLocator(0.05))
        axis.yaxis.set_major_locator(mticker.MultipleLocator(0.25))
        axis.grid(which="minor", axis="y", ls=":", lw = 0.5)
    # Plot horizontal grid for visual aid.
    axis.grid(which="major", axis="y", ls="--", lw = 0.9)
    # Include ticks on right hand side.
    axis.tick_params(right=True, which="both", direction="inout")

# Place legend on performance history M1+ axis including the
# climatology.
axes.flatten()[1].legend(handles=bar,
                          labels=list(unconstrained_M_brier.df_of_models),
                          bbox_to_anchor=(1, 1.035),frameon=False, )

# plt.savefig("weights.jpg", dpi=300, bbox_inches="tight")
plt.show()
