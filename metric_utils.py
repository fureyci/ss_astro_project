# -*- coding: utf-8 -*-
"""
@author: hayesla

This script holds some useful functions for calculating forcast
verification metrics, plots reliability diagramand  ROC curve to
visualise performance of models, and also plots feature importance.

original script:
https://github.com/hayesla/flare_forecast_proj/blob/main/forecast_tests/metric_utils.py

"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.calibration import calibration_curve

############## SKILL SCORES #################

def calculate_tss(true_vals, pred_vals):
    """Calculate the True Skill Score (TSS) to test the overall
    predictive abilities of a given forecast.

    Parameters
    ----------
    true_vals: `~np.array`
        the events.
    pred_vals : `~np.array`
        The predicted Y values from the model.

    Returns
    -------
    TSS : ~`float`
        calculated TSS value.

    """
    tn, fp, fn, tp = metrics.confusion_matrix(true_vals, pred_vals).ravel()

    TSS = (tp / (tp + fn)) - (fp / (fp + tn))
    return TSS


def calculate_tss_threshold(true_vals, prob_vals, thresh):
    """Calculate the TSS for a given threshold. This should be
    used when the forecast gives a probability.

    Parameters
    ---------
    true_vals: `~np.array`
        the events.
    prob_vals: `~np.array`
        the predicted value probabilities.
    thresh : `~float`
        the threshold value to take for binary event (i.e. values above
        this threshold are taken to be 1 (flare) and those below as 0
        (no flare)).

    Returns
    -------
    TSS : `float`
        calculated TSS.

    """
    # Convert to dichotomous forecast.
    pred_thresh = [1 if x>=thresh else 0 for x in prob_vals]

    tn, fp, fn, tp = metrics.confusion_matrix(true_vals, pred_thresh).ravel()
    TSS = (tp / (tp + fn)) - (fp / (fp + tn))
    return TSS


def calculate_bss(true_vals, prob_vals):
    """ Calculate the Brier Skill Score (BSS) of a predictive
    model.

    Parameters
    ----------
    true_vals: `~np.array`
        the events.
    prob_vals : `~np.array`
        The predicted probability values of the model.

    Returns
    -------
    BSS : `float`
        calculated BSS.

    """
    bs = metrics.brier_score_loss(true_vals, prob_vals)
    clim = np.full(true_vals.shape,np.mean(true_vals))
    bs_clim = metrics.brier_score_loss(true_vals, clim)
    BSS = 1 - bs/bs_clim
    return BSS

def calculate_fb_threshold(true_vals, prob_vals, thresh):
    """Calculate frequency bias (fb) for a predictive model where
    dichotomous (yes/no) forecasts are produced for a probabilistic
    model if the prediction is either above or below thresh.

    Parameters
    ----------
    true_vals: `~np.array`
        the true values.
    prob_vals: `~np.array`
        the predicted value probabilities.
    thresh : `~float`
        the threshold value to take for binary event (i.e. values above
        this threshold are taken to be 1 (flare) and those below as 0
        (no flare)).

    Returns
    -------
    fb: `float`
        the frequency bias.

    """
    pred_thresh = [1 if x>=thresh else 0 for x in prob_vals]
    tn, fp, fn, tp = metrics.confusion_matrix(true_vals, pred_thresh).ravel()

    fb = (tp + fp) / (tp + fn) # Frequency bias.
    return fb

def calculate_ets_threshold(true_vals, prob_vals, thresh):
    """Calculate equitable threat score (ets) for a predictive model
    where dichotomous (yes/no) forecasts are produced for a
    probabilistic model if the prediction is either above or below
    thresh.

    Parameters
    ----------
    true_values: `~np.array`
        the true values.
    prob_vals: `~np.array`
        the predicted value probabilities.
    thresh : `~float`
        the threshold value to take for binary event (i.e. values above
        this threshold are taken to be 1 (flare) and those below as 0
        (no flare)).

    Returns
    -------
    ets: `float`
        the equitable threat score.

    """
    pred_thresh = [1 if x>=thresh else 0 for x in prob_vals]
    tn, fp, fn, tp = metrics.confusion_matrix(true_vals, pred_thresh).ravel()
    ar = ((tp + fp) * (tp + fn)) / (tn+tp+fn+fp)
    ets = (tp - ar) / (tp - ar + fp + fn) # frequency bias
    return ets

def calculate_apss_threshold(true_vals, prob_vals, thresh):
    """
    Calculate Appleman's skill score (ApSS) for a predictive model
    where dichotomous (yes/no) forecasts are produced for a
    probabilistic model if the prediction is either above or below
    thresh.

    Parameters
    ----------
    true_vals: `~np.array`
        the true values.
    prob_vals: `~np.array`
        the predicted value probabilities.
    thresh : `~float`
        the threshold value to take for binary event (i.e. values above
        this threshold are taken to be 1 (flare) and those below as 0
        (no flare)).

    Returns
    -------
    apss: `float`
        the Appleman's skill score.

    """
    pred_thresh = [1 if x>=thresh else 0 for x in prob_vals]
    tn, fp, fn, tp = metrics.confusion_matrix(true_vals, pred_thresh).ravel()
    n = float(tn + fp + fn + tp)
    forecast_acc = (tn + tp) / n

    # if number of non-events greater than number of events
    if (tn + fp) > (tp + fn):
        reference_acc = (tn + fp) / n
    # if number of non-events less than number of events
    elif (tn + fp) < (tp + fn):
        reference_acc = (tp + fn) / n

    apss = (forecast_acc - reference_acc) / (1 - reference_acc)

    return apss

def calculate_roc_area(true_vals, prob_vals):
    """Calculate the area under the roc curve (AUC).

    Parameters
    ----------
    true_values: `~np.array`
        the true values.
    prob_vals: `~np.array`
        the predicted value probabilities.

    Returns
    -------
    auc : `float`
        The AUC score.

    """
    fpr, tpr, _ = metrics.roc_curve(true_vals,
                                    prob_vals)
    auc = metrics.auc(fpr, tpr)

    return auc

############## FORECAST METRIC PLOTS #################

def plot_roc_curve(true_vals, prob_vals, ax=None,
                   mformat=None, mcolour=None, display_auc=True):
    """Plot the receiver operating characteristic (ROC) curve.

    Parameters
    ----------
    true_vals : `~np.array`
        The true Y (label) values. The test outputs to compare with the
        predicted values.
    prob_vals : `~np.array`
        The predicted probability values of the model.
    ax : `~matplotlib.axes.Axes`, optional
        If provided the image will be plotted on the given axes.
    mformat : `str`, optional, default is None.
        Marker format of plot. If None, plots square. Used for
        ensemble.py, plots a different format depending on desired
        weighting scheme.
    mcolour : `str`, optional, default is None
        Colour of plot, depending on weighting scheme and metric
        optimised of Ensemble.

    Returns
    -------
    Plots the ROC curve
    ax : ~`matplotlib axes`

    """
    fpr, tpr, _ = metrics.roc_curve(true_vals, prob_vals)
    auc = metrics.auc(fpr, tpr)

    # fs = 'x-large' # Font size

    if ax is None:
        fig, ax = plt.subplots()

    if mformat is None:
        fmt = "-"
    else:
        fmt = mformat+"-"
    if mcolour is None:
        c="#1f77b4"
    else:
        c=mcolour

    ax.plot(fpr, tpr, fmt, lw=0.7, mew=0.7, ms=6, mfc='none', c=c)

    ax.plot([0, 1], [0, 1], color='grey', linestyle='--')
    ax.set_xlabel("FPR",size=13)
    ax.set_ylabel("TPR",size=13)
    if display_auc is True:
        ax.text(0.98, 0.03,"AUC = {:.2f}".format(auc),
                horizontalalignment="right",
                verticalalignment="bottom",
                transform = ax.transAxes)
    ax.tick_params(top=True, right=True)
    return ax


def plot_reliability_curve(true_vals, pred_vals, n_bins=10, laplace=True,
                           mformat=None, mcolour=None, axes=None):
    """ Plot the reliability curve (also known as a calibration curve).
    Adjusted to replicate plots from Leka et al 2019 [1].

    Parameters
    ----------
    true_vals : `~np.array`
        The true Y (label) values. The test outputs to compare with the
        predicted values
    pred_vals : `~np.array`
        The predicted Y values from the model.
    n_bins : `int`, optional, default 10.
        Number of bins to discretize the [0, 1] interval (input to
        sklearn `calibration_curve` function).
    laplace : `bool`, optional, default is True.
        Whether or not to use Laplace's rule of succession when
        converting observed frequency into probability [2].
    mformat : `str`, optional, default is None.
        Marker format of plot. If None, plots square. Used for
        ensemble.py, plots a different format depending on desired
        weighting scheme.
    mcolour : `str`, optional, default is None
        Colour of plot, depending on weighting scheme and metric
        optimised of Ensemble.
    axes : `tuple of matplotlib.axes.Axes`, optional
        If provided the image will be plotted on the given axes.
        Expected form:
            (reliability diagram axis, histogram axis),
        where ratio height of reliability diagram and histogram
        axes is 3:1.

    Returns
    -------
    Plots the reliability diagram
    ax1, ax2 : ~`matplotlib axes`
        reliability diagram axis and histogram axis, respectively.

    References
    ----------
    [1] Leka, K.D. et al., 2019. A comparison of flare forecasting
        methods. II. Benchmarks, metrics, and performance results for
        operational solar flare forecasting systems. The Astrophysical
        Journal Supplement Series, 243(2), p.36.
    [2] Wheatland, M.S., 2005. A statistical solar flare forecast
        method. Space Weather, 3(7).

    """
    if laplace:
        # Do what sklearn.calibration.calibration_curve() does,
        # but implement laplace's rule of succession when calculating
        # fraction of positives.

        # Generate bins of uniform width.
        bins = np.linspace(0., 1. + 1e-8, n_bins + 1)

        # Assign model predictions to each bin.
        # -1 since it corresponds to the indices.
        binids = np.digitize(pred_vals, bins) - 1

        # Sum up the probabilities in each bin.
        bin_sums = np.bincount(binids, weights=pred_vals, minlength=len(bins))

        # Sum up number of flare events in each probability bin.
        bin_true = np.bincount(binids, weights=true_vals, minlength=len(bins))

        # Sum up number of events in each bin.
        bin_total = np.bincount(binids, minlength=len(bins))

        # Delete any bins where there are no flare events, since want
        # to replicate plots inbenchmark discussed in Leka 2019 paper.
        zeros = np.where(bin_true == 0)

        # Delete these bins.
        bin_true = np.delete(bin_true, zeros)
        bin_sums = np.delete(bin_sums, zeros)
        bin_total = np.delete(bin_total, zeros)

        # Indices where there are a nonzero number of events in the
        # bins.
        nonzero = bin_total != 0

        # Laplace's rule of succession for the probability.
        fraction_of_positives = (bin_true[nonzero] + 1) / (bin_total[nonzero] + 2)

        mean_predicted_value = bin_sums[nonzero] / bin_total[nonzero]

        # Uncertainty in the probability.
        true_err = np.sqrt(
            (fraction_of_positives*(1-fraction_of_positives)) / (bin_total[nonzero] + 3)
            )

    else:
        fraction_of_positives, mean_predicted_value = calibration_curve(true_vals,
                                                                    pred_vals,
                                                                    n_bins=n_bins)
        # dont plot 0, to replicate plots benchmark from Leka 2019
        zeros = np.where(fraction_of_positives == 0)

        fraction_of_positives = np.delete(fraction_of_positives, zeros)
        mean_predicted_value = np.delete(mean_predicted_value, zeros)

    climatology = np.mean(true_vals)

    # No skill line
    x = [0,1]
    no_skill = 0.5 * (x-climatology) + climatology

    if axes is None:
        fig = plt.figure(figsize=(6,6))
        gs1 = fig.add_gridspec(nrows=4, ncols=1)
        ax1 = fig.add_subplot(gs1[0:3, 0])
        ax2 = fig.add_subplot(gs1[3, 0], sharex=ax1)
    else:
        ax1 = axes[0]
        ax2 = axes[1]

    ax1.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated", lw=0.7)

    if mformat == None:
        marker_format = "s"
    else:
        marker_format = mformat

    if mcolour == None:
        c = "#1f77b4" # Default blue
        ecolor = "#1f77b4"
        mfc = "#1f77b4"
    else:
        c = mcolour
        ecolor = mcolour
        mfc = "none"

    if laplace:
        ax1.errorbar(mean_predicted_value, fraction_of_positives,
                     c=c, fmt=marker_format, linestyle='-', ms = 7, lw=0.7,
                     mfc=mfc, mew=0.7, yerr=true_err, ecolor=ecolor, capthick=0.7,
                     elinewidth=0.7, capsize=2)
    else:
        ax1.plot(mean_predicted_value, fraction_of_positives,
                 c=c, fmt=marker_format, ls='-', mfc=mfc)

    ax1.plot(x, no_skill, "k",
             ls=(0, (5, 10)),
             lw=0.7, label="No-skill")
    ax1.tick_params(which="both", labelbottom=False)
    ax1.plot([0,1], [climatology,climatology],
             color="grey", label="Climatology", lw=0.7)
    # ax1.legend(loc="upper left")

    ax2.hist(pred_vals, range=(0, 1), bins=n_bins,
             histtype="step", lw=1, color=c)

    if axes is None:
        fs = 'medium'
    else:
        fs = 15
    ax1.set_ylabel(r"$\mathdefault{P_k^{\rm obs}}$", size=fs)
    ax2.set_xlabel(r"$\mathdefault{P_k}$", size=fs)
    ax2.set_ylabel(r"$\mathdefault{n_k}$", size=fs)

    # plt.subplots_adjust(hspace=0)

    if axes is None:
        plt.tight_layout()

    return ax1, ax2

def plot_feature_importance(mdl, features, top=None, title="Feature importance"):
    """
    Function to plot the importance of features from a sklearn model. 
    
    Parameters
    ----------
    mdl : sklearn model that has been already fit
    features : `pd.DataFrame` of features. 
    top : `int`, number of top features to plot, optional.
          default is to plot all. 
    
    """

    if not hasattr(mdl, "feature_importances_"):
        print("{:s} doesn't have feature importance attribute".format(str(mdl)))
        return

    feature_importance = mdl.feature_importances_
    if top is not None:
        sorted_idx = np.argsort(feature_importance)[::-1][0:top]
    else:
        sorted_idx = np.argsort(feature_importance)[::-1]
    np.array(features.columns)[sorted_idx]
    
    pos = np.arange(0, sorted_idx.shape[0]*2, 2)
    
    fig = plt.figure(figsize=(10, 8))
    plt.barh(pos, feature_importance[sorted_idx], 2, align='center', edgecolor="k")
    plt.gca().invert_yaxis()
    plt.yticks(pos, np.array(features.columns)[sorted_idx])
    plt.title(title)
