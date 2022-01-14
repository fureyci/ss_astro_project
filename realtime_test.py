# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 09:34:41 2021

@author: fureyci

Initially, the purpose of this script was to test the realtime
ensembles, but it ended up becoming a script to test rpss.py, however
there is an example of a realtime ensemble model at the end of the
script.
"""

from datetime import date, timedelta
from matplotlib import (pyplot as plt,
                        dates as mdates,
                        ticker as mticker)
import numpy as np
import metric_utils
from opening_realtime import get_realtime_data, plot_operational_periods
from ensemble import Ensemble
from opening_data import load_benchmark_data
import rpss
import time
import matplotlib.patheffects as pe

#W D_TO_SAVE = ____

# =====================================================================
# Define start and end times.
# Start time, 90 weeks before 11/11/2021y.
tstart = "2020/02/20"

# End time of testing interval.
tend = "2021/11/11"

now = time.time()

# =====================================================================
# Rolling RPSS
#
# Benchmark Ensemble.

events, forecasts, models = load_benchmark_data()

# Number of days to calculate benchmark rolling RPSS.
N_BENCHMARK = 365

# Need to remove models that dont provide C forecasts.
desired_cols=[]

for col in list(forecasts[0]):
    if "1+" in col:
        desired_cols.append(col)

models_with_all_forecasts=[]

for model, forecast in zip(models, forecasts):
    for col in desired_cols:
        try:
            if np.unique(forecast.loc[:,col]) == [-1.]:
                # if this is the case, model doesn't provide C
                # forecast.
                # if models do provide c forecast, this line will
                # throw exception and will move on to next block.
                break
        except:
            models_with_all_forecasts.append(model)
            pass

# find models that dont have forecasts (invert=True)
indices_to_remove = np.where(np.isin(models,
                                      models_with_all_forecasts,
                                      invert=True)
                              )[0]

# make copies of original files
forecasts_to_use = forecasts.copy()
models_to_use = models.copy()

# remove models that don't provide C forecasts
for index in indices_to_remove[::-1]:
    models_to_use.pop(index)
    forecasts_to_use.pop(index)

# RRPSS of each model
print("clc brier")
con_rrpss = rpss.rolling_rpss(forecasts_to_use, events, N_BENCHMARK,
                              model_names=models_to_use, desired_weighting="CLC",
                              bootstrap=True)

print("ulc brier")
uncon_rrpss = rpss.rolling_rpss(forecasts_to_use, events, N_BENCHMARK,
                              model_names=models_to_use, desired_weighting="ULC",
                                bootstrap=True)

print("av")
av_rrpss = rpss.rolling_rpss(forecasts_to_use, events, N_BENCHMARK,
                              model_names=models_to_use, desired_weighting="Average",
                              bootstrap=True)
print("ev")
his_rrpss = rpss.rolling_rpss(forecasts_to_use, events, N_BENCHMARK,
                              model_names=models_to_use,desired_weighting="EV",
                              bootstrap=True)

print("clc mae")
mae_con_rrpss = rpss.rolling_rpss(forecasts_to_use, events, N_BENCHMARK,
                              model_names=models_to_use,desired_weighting="CLC",
                              desired_metric="MAE", bootstrap=True)

print("ulc mae")
mae_uncon_rpss = rpss.rolling_rpss(forecasts_to_use, events, N_BENCHMARK,
                              model_names=models_to_use,desired_weighting="ULC",
                              desired_metric="MAE", bootstrap=True)

print("clc lcc")
lcc_con_rrpss = rpss.rolling_rpss(forecasts_to_use, events, N_BENCHMARK,
                              model_names=models_to_use,desired_weighting="CLC",
                              desired_metric="LCC", bootstrap=True)

print("ulc lcc")
lcc_uncon_rpss = rpss.rolling_rpss(forecasts_to_use, events, N_BENCHMARK,
                              model_names=models_to_use,desired_weighting="ULC",
                              desired_metric="LCC", bootstrap=True)

print("clc rel")
rel_con_rrpss = rpss.rolling_rpss(forecasts_to_use, events, N_BENCHMARK,
                              model_names=models_to_use,desired_weighting="CLC",
                              desired_metric="REL", bootstrap=True)

print("ulc rel")
rel_uncon_rpss = rpss.rolling_rpss(forecasts_to_use, events, N_BENCHMARK,
                              model_names=models_to_use,desired_weighting="ULC",
                              desired_metric="REL", bootstrap=True)


rrpss_list = [con_rrpss, uncon_rrpss, mae_con_rrpss, mae_uncon_rpss,
              lcc_con_rrpss, lcc_uncon_rpss, rel_con_rrpss, rel_uncon_rpss,
              av_rrpss, his_rrpss]
rrpss_labels = ["CLC, BS", "ULC, BS", "CLC, MAE", "ULC, MAE",
                "CLC, LCC", "ULC, LCC","CLC, REL", "ULC, REL",
                "Average", "EV"]

# for clarity, find average uncertainty to plot on graph
all_uncertainties = np.zeros((len(rrpss_list), len(rrpss_list[0][2])))
for i, model in enumerate(rrpss_list):
    all_uncertainties[i] = model[2]

av_uncertainty = np.mean(all_uncertainties)

plt.figure()
ax = plt.gca()
for i, model_rrpss in enumerate(rrpss_list):
    # model_rrpss[0] = dates
    # model_rrpss[1] = rrpss scores
    # model_rrpss[2] = bootstrapped uncertainties
    # model_rrpss[3] = color of ensemble (weighting scheme)
    # model_rrpss[4] = marker of ensemble (metric optimised)
    line = plt.plot(model_rrpss[0], model_rrpss[1],
                    lw=1, color=model_rrpss[3])
    # plt.fill_between(model_rrpss[0],
    #                    np.array(model_rrpss[1])+np.array(model_rrpss[2]),
    #                    np.array(model_rrpss[1])-np.array(model_rrpss[2]),
    #                    color=line[0].get_color(), alpha=0.2)

    for j in range(len(rrpss_list[0][0])):
        if j % 20 == 0:
          point = plt.scatter(model_rrpss[0][j], model_rrpss[1][j],
                              color=model_rrpss[3], marker=model_rrpss[4],
                              facecolor="none")
          if j == 20:
              point = plt.scatter(model_rrpss[0][j], model_rrpss[1][j],
                                  color=model_rrpss[3],marker=model_rrpss[4],
                                  label=rrpss_labels[i],facecolor="none")

# plot average uncertainty
plt.errorbar(con_rrpss[0][-20], 0.15, yerr=av_uncertainty,
              fmt='.', ms=3, elinewidth=0.8, capsize=3, capthick=0.8, c='k')
plt.text(con_rrpss[0][-60],0.145, r"$\mathdefault{\pm1\sigma}\simeq$", va="center")

plt.plot([con_rrpss[0][0], con_rrpss[0][-1]], [0, 0], 'k-', lw=0.6)

plt.plot(con_rrpss[0][events.iloc[N_BENCHMARK:]["C"].values == 1],
          events.iloc[N_BENCHMARK:]["C"].values[events.iloc[N_BENCHMARK:]["C"].values == 1]-0.35,
          'k.',
          label="C events", alpha=0.9)
plt.plot(con_rrpss[0][events.iloc[N_BENCHMARK:]["M"].values == 1],
          events.iloc[N_BENCHMARK:]["M"].values[events.iloc[N_BENCHMARK:]["M"].values == 1]-0.3,
          'k1',
          label="M events", alpha=0.9)

# Set xtick formats to dates.
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

# Set xticks as every quarter of a year.
ax.xaxis.set_major_locator(mdates.MonthLocator((1,4,7,10)))
ax.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.05))
ax.tick_params(which="both", top=True, right=True, direction="out")
plt.ylabel(f"{N_BENCHMARK} Day Rolling RPSS")
# plt.legend(bbox_to_anchor=(1, 1.035),frameon=False)
# plt.savefig(WD_TO_SAVE+"rpss.jpg", dpi=300, bbox_inches="tight")
plt.show()

# ---------------------------------------------------------------------
# RPSS of ensemble members.

member_rpss = []
rpss_names = []
for i,model in enumerate(forecasts_to_use):
    # calculate rrpss of each model in ensemble taht provides both
    # forecasts.
    #
    # Note ensemble = False, so outputs are slightly different
    rrpss = rpss.rolling_rpss(model, events, N_BENCHMARK, ensemble=False,
                              bootstrap=False)
    if rrpss is not None:
        member_rpss.append(rrpss)
        rpss_names.append(models_to_use[i])

# # find average uncertainty for all models
# all_member_uncertainties = np.zeros((len(member_rpss), len(member_rpss[0][2])))
# for i, model in enumerate(member_rpss):
#     all_member_uncertainties[i] = model[2]

# av_member_uncertainty = np.std(all_member_uncertainties)

# use same colour scheme as weight figure to represent models
no_colours = len(forecasts)+1  # Number of colours.

# Range of indices 0 to number of weights, to be shuffled.
ran_cols = np.linspace(0, no_colours-1, no_colours, dtype=int)

np.random.seed(100)  # Set seed for reproducibility.
np.random.shuffle(ran_cols)  # Shuffle indices.

# Find models that provide forecasts
colour_indices = np.where(np.isin(models,
                                  models_with_all_forecasts)
                          )[0]

# Use same colourmap.
cmap = plt.cm.get_cmap("twilight", no_colours)

plt.figure()
model_ax = plt.gca()
for i, rrpss in enumerate(member_rpss):
    if i % 2 == 0:
        ls="--"
    elif i % 3 == 0:
        ls="-."
    elif i == 1:
        ls=":"
    else:
        ls="-"
    plt.plot(rrpss[0], rrpss[1], label=rpss_names[i],ls=ls,
              c=cmap(ran_cols[colour_indices[i]]),
              path_effects=[pe.Stroke(linewidth=1.5, foreground='k'), pe.Normal()])

# Set xtick formats to dates.
model_ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

model_ax.plot([con_rrpss[0][0], con_rrpss[0][-1]], [0, 0], 'k-', lw=0.6)

# plot events
model_ax.plot(con_rrpss[0][events.iloc[N_BENCHMARK:]["C"].values == 1],
          events.iloc[N_BENCHMARK:]["C"].values[events.iloc[N_BENCHMARK:]["C"].values == 1]-0.35,
          'k.',
          label="C events", alpha=0.9)
model_ax.plot(con_rrpss[0][events.iloc[N_BENCHMARK:]["M"].values == 1],
          events.iloc[N_BENCHMARK:]["M"].values[events.iloc[N_BENCHMARK:]["M"].values == 1]-0.3,
          'k1',
          label="M events", alpha=0.9)

# Set xticks as every quarter of a year.
model_ax.xaxis.set_major_locator(mdates.MonthLocator((1,4,7,10)))
model_ax.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
model_ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.05))
model_ax.set_ylim(-0.1,0.73)
model_ax.tick_params(which="both", top=True, right=True, direction="out")
model_ax.set_ylabel(f"{N_BENCHMARK} Day Rolling RPSS")
plt.legend(bbox_to_anchor=(1, 1.035),frameon=False)
# plt.savefig(WD_TO_SAVE+"individual_rpss1.jpg", dpi=300, bbox_inches="tight")
plt.show()

# ---------------------------------------------------------------------
# Realtime Ensemble test.

# Load realtime data.
goes_events, realtime_forecasts, realtime_models = get_realtime_data(tstart,
                                                                     tend)

# # MAG4 is now online, but wasn't at time of project. This code just
# # removes it from models.
# a = np.where(["MAG4" in i for i in realtime_models])[0]
# copy_models = realtime_models.copy()
# copy_forecasts = realtime_forecasts.copy()
# # # print(copy_forecasts)
# # # print(a)
# for index in a[::-1]:
#     copy_models.pop(index)
#     copy_forecasts.pop(index)

# A test realtime ensemble.

realtime_ulc_bs = Ensemble(realtime_forecasts, realtime_models, goes_events,
                            desired_forecast="M-only", desired_weighting="ULC",
                            desired_metric="BS")

# Let's see some parameters of model
print("Ensemble members: ", realtime_ulc_bs.model_names)
print("Member forecasts: ", realtime_ulc_bs.df_of_models)
print("BSS of ensemble: ", metric_utils.calculate_bss(realtime_ulc_bs.events,
                                  realtime_ulc_bs.forecast))
# plot both ROC curve and reliability diagram
realtime_ulc_bs.visualise_performance()

done = time.time() - now
print("Entire process done in %.3f seconds, or %.3f minutes" % (done, done/60.0))
