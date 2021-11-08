# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 09:34:41 2021

@author: fureyci

"""

from datetime import date, timedelta
from matplotlib import (pyplot as plt,
                        dates as mdates,
                        ticker as mticker)
import numpy as np

from opening_realtime import get_realtime_data
from opening_data import load_benchmark_data
import rpss

# WD_TO_SAVE = 'C:/Users/Ciaran/OneDrive/Desktop/SS/Project/Thesis/images/'

# =====================================================================
# Define start and end times.
# Start time, one year ago today.
tstart = (date.today() - timedelta(weeks=52)).strftime("%Y/%m/%d")

# End time of testing interval, today's date.
tend = date.today().strftime("%Y/%m/%d")

# =====================================================================
# Rolling RPSS
#
# Benchmark Ensemble.

events, forecasts, models = load_benchmark_data()

# Number of days to calculate benchmark rolling RPSS.
N_BENCHMARK = 365

con_rrpss = rpss.rolling_rpss(forecasts, models, events,
                              N_BENCHMARK, desired_weighting="constrained",
                              bootstrap=True)

uncon_rrpss = rpss.rolling_rpss(forecasts, models, events,
                                N_BENCHMARK, desired_weighting="unconstrained",
                                bootstrap=True)

av_rrpss = rpss.rolling_rpss(forecasts, models, events,
                              N_BENCHMARK, desired_weighting="average",
                              bootstrap=True)

his_rrpss = rpss.rolling_rpss(forecasts, models, events,
                              N_BENCHMARK, desired_weighting="history",
                              bootstrap=True)

rrpss_list = [con_rrpss, uncon_rrpss, av_rrpss, his_rrpss]
rrpss_labels = ["Constrained", "Unconstrained", "Average", "History"]

plt.figure()
ax = plt.gca()
for i, model_rrpss in enumerate(rrpss_list):
    line = plt.plot(model_rrpss[0], model_rrpss[1], label=rrpss_labels[i],
                    lw=1)
    plt.fill_between(model_rrpss[0],
                      np.array(model_rrpss[1])+np.array(model_rrpss[2]),
                      np.array(model_rrpss[1])-np.array(model_rrpss[2]),
                      color=line[0].get_color(), alpha=0.2)

plt.plot([con_rrpss[0][0], con_rrpss[0][-1]], [0, 0], 'k-', lw=0.6)

# Set xtick formats to dates.
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

# Set xticks as every quarter of a year.
ax.xaxis.set_major_locator(mdates.MonthLocator((1,4,7,10)))
ax.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.02))

plt.ylabel(f"{N_BENCHMARK} Day Rolling RPSS")
plt.legend()
# plt.savefig(WD_TO_SAVE+"rpss.jpg", dpi=250)
plt.show()

# ---------------------------------------------------------------------
# Realtime Ensemble RPSS.

# Load realtime data.
goes_events, realtime_forecasts, realtime_models = get_realtime_data(tstart,
                                                                      tend)

# Number of days to calculate realtime rolling RPSS.
N_REALTIME = 50

realtime_con = rpss.rolling_rpss(realtime_forecasts, realtime_models,
                                 goes_events, N_REALTIME,
                                 forecast_type="threshold",
                                 desired_weighting="constrained",
                                 desired_metric="brier",
                                 bootstrap=True)

plt.figure()
real_ax = plt.gca()
line1 = plt.plot(realtime_con[0], realtime_con[1], label="Real-Time",
                lw=1)
plt.fill_between(realtime_con[0],
                  np.array(realtime_con[1])+np.array(realtime_con[2]),
                  np.array(realtime_con[1])-np.array(realtime_con[2]),
                  color=line1[0].get_color(), alpha=0.2)

# Set xtick formats to dates.
real_ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
real_ax.set_ylim(-2, 1.05)
# set xticks as every quearter of a year
real_ax.xaxis.set_major_locator(mdates.MonthLocator((1,4,7,10)))
real_ax.yaxis.set_major_locator(mticker.MultipleLocator(0.5))
real_ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.1))
plt.plot([realtime_con[0][0], realtime_con[0][-1]], [0, 0], 'k-', lw=0.6)

plt.ylabel(F"{N_REALTIME} Day Rolling RPSS")
# plt.savefig(WD_TO_SAVE+"realrpss.jpg", dpi=250)
plt.show()
