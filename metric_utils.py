# -*- coding: utf-8 -*-


import numpy as np 
import matplotlib.pyplot as plt 
from sklearn import metrics
from sklearn.calibration import calibration_curve


"""
This script holds some useful functions for calculating forcast verification metrics.
"""

############## SKILL SCORES #################

def calculate_tss(true_vals, pred_vals):
    """
    Calculate the True Skill Score (TSS) to test the overall predictive 
    abilities of a given forecast.
    Parameters
    ----------
    true_vals : `~np.array`
        The true Y (label) values. The test outputs to compare with the predicted values
    pred_vals : `~np.array`
        The predicted Y values from the model.
    Returns
    -------
    TSS : ~`float`
        calculated TSS value
    Notes
    -----
    See Bloomfield et al. 2012.
    """
    tn, fp, fn, tp = metrics.confusion_matrix(true_vals, pred_vals).ravel()
    
    TSS = (tp / (tp + fn)) - (fp / (fp + tn))
    return TSS


def calculate_tss_threshold(true_vals, prob_vals, thresh):
    """
    Calculate the TSS for a given threshold. This should be 
    used when the forecast gives a probability.
    Parameters
    ---------
    true_values: `~np.array`
        the true values.
    prob_vals: `~np.array`
        the predicted value probabilities.
    thresh : `~float`
        the threshold value to take for binary event (i.e. values above this 
        threshold are taken to be 1 (flare) and those below as 0 (no flare)).
    """
    pred_thresh = [1 if x>thresh else 0 for x in prob_vals]
    tn, fp, fn, tp = metrics.confusion_matrix(true_vals, pred_thresh).ravel()
    TSS = (tp / (tp + fn)) - (fp / (fp + tn))
    return TSS


def calculate_bss(true_vals, prob_vals):
    """
    Calculate the Brier Skill Score (BSS) of a predictive model.
    Parameters
    ----------
    Y_test : `~np.array`
        The true Y (label) values. The test outputs to compare with the predicted values.
    prob_vals : `~np.array`
        The predicted probability values of the model. If using sklearn model - use mdl.predict_proba() function.
    Returns
    -------
    BSS : `float`
        calculated BSS
    """
    bs = metrics.brier_score_loss(true_vals, prob_vals)
    bs_clim = np.mean(true_vals)
    BSS = 1 - bs/bs_clim
    return BSS

def calculate_fb_threshold(true_vals, prob_vals, thresh):
    """
    Calculate frequency bias (fb) for a predictive model where
    dichotomous (yes/no) forecasts are produced for a probabilistic
    model if the prediction is either above or below thresh.

    Parameters
    ----------
    true_values: `~np.array`
        the true values.
    prob_vals: `~np.array`
        the predicted value probabilities.
    thresh : `~float`
        the threshold value to take for binary event (i.e. values above this
        threshold are taken to be 1 (flare) and those below as 0 (no flare)).

    Returns
    -------
    fb: `float`
        the frequency bias.

    """
    pred_thresh = [1 if x>=thresh else 0 for x in prob_vals]
    tn, fp, fn, tp = metrics.confusion_matrix(true_vals, pred_thresh).ravel()
    fb = (tp + fp) / (tp + fn) # frequency bias
    return fb

############## FORECAST METRIC PLOTS #################

def plot_roc_curve(true_vals, prob_vals, ax=None, fmt=None):
    """
    Plot the receiver operating characteristic (ROC) curve
    Parameters
    ----------
    true_vals : `~np.array`
        The true Y (label) values. The test outputs to compare with the predicted values.
    prob_vals : `~np.array`
        The predicted probability values of the model.
    ax : `~matplotlib.axes.Axes`, optional
        If provided the image will be plotted on the given axes.
    fmt : `str`, optional, default is None.
        Marker format of plot. If None, plots square. Used for
        ensemble.py, plots a different format depending on desired
        weighting scheme.
    
    Returns
    -------
    Plots the ROC curve
    ax : ~`matplotlib axes`
    """
    
    fpr, tpr, _ = metrics.roc_curve(true_vals, prob_vals)
    auc_mcstat = metrics.auc(fpr, tpr)
    
    fs = 'x-large' # fontsize of axis labels
    
    if ax is None:
        fig, ax = plt.subplots()
        fs = 'medium' # fontsize of axis labels

    if fmt == None:
        ax.plot(fpr, tpr, label="(AUC = {:.3f})".format(auc_mcstat))
    else:
        ax.plot(fpr, tpr, fmt+"-", ms=10, mfc='none',
                label="(AUC = {:.3f})".format(auc_mcstat))
        
    ax.plot([0, 1], [0, 1], color='grey', linestyle='--')
    ax.set_xlabel("False Positive Rate", size=fs)
    ax.set_ylabel("True Positive Rate", size=fs)
    ax.legend(loc="lower right")
    plt.tight_layout()
    return ax

def plot_reliability_curve(true_vals, pred_vals, n_bins=10, laplace=True,
                           fmt=None, axes=None):
    """
    Plot the reliability curve (also known as a calibration curve).
    Adjusted to replicate plots from Leka et al 2019 [1].
    
    Parameters
    ----------
    true_vals : `~np.array`
        The true Y (label) values. The test outputs to compare with the predicted values
    pred_vals : `~np.array`
        The predicted Y values from the model.
    n_bins : `int`, optional, default 10.
        Number of bins to discretize the [0, 1] interval (input to sklearn
        `calibration_curve` function).
    laplace : `bool`, optional, default is True.
        Whether or not to use Laplace's rule of succession when considering
        fraction of positives [1]. Useful for when there is a small amount of
        data.
    fmt : `str`, optional, default is None.
        Marker format of plot. If None, plots square. Used for
        ensemble.py, plots a different format depending on desired
        weighting scheme.
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
        # do what sklearn.calibration.calibration_curve() does,
        # but implement laplace's rule of succession when calculating
        # fraction of positives.

        # generate bins of uniform width
        bins = np.linspace(0., 1. + 1e-8, n_bins + 1)

        # assign model predictions to each bin, -1 since it corresponds
        # to the indices.
        binids = np.digitize(pred_vals, bins) - 1

        # sum up the probabilities in each bin, for eventual average
        bin_sums = np.bincount(binids, weights=pred_vals, minlength=len(bins))

        # sum up number of flare events in each probability bin
        bin_true = np.bincount(binids, weights=true_vals, minlength=len(bins))

        # sum up number of events in each bin
        bin_total = np.bincount(binids, minlength=len(bins))

        # delete any bins where there are no flare events, since want
        # to replicate benchmark discussed in Leka 2019 paper
        zeros = np.where(bin_true == 0)

        # delete these indices from the arrays
        bin_true = np.delete(bin_true, zeros)
        bin_sums = np.delete(bin_sums, zeros)
        bin_total = np.delete(bin_total, zeros)

        # indices where there are a nonzero number of events in the bins
        nonzero = bin_total != 0

        # laplace's rule of succession for the probability
        fraction_of_positives = (bin_true[nonzero] + 1) / (bin_total[nonzero] + 2)
        
        # mean bin probability
        mean_predicted_value = bin_sums[nonzero] / bin_total[nonzero]

        # uncertainty in the fraction of positives
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

    # no skill line
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

    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    
    if fmt == None:
        if laplace:
            ax1.errorbar(mean_predicted_value, fraction_of_positives, fmt="s-",
                         yerr=true_err, capthick=1, elinewidth=1, capsize=3)
        else:
            ax1.plot(mean_predicted_value, fraction_of_positives, "s-")
    else:
        if laplace:
            ax1.errorbar(mean_predicted_value, fraction_of_positives,
                         fmt=fmt, ls='-', ms = 10, mfc='none',
                         yerr=true_err, capthick=1, elinewidth=1, capsize=3)
        else:
            ax1.plot(mean_predicted_value, fraction_of_positives,
                     fmt=fmt, ls='-', mfc='none')

    ax1.plot(x, no_skill, "k",
             ls=(0, (5, 10)),
             lw=0.5, label="No-skill")
    
    ax1.tick_params(which="both", labelbottom=False)
    ax1.plot([0,1], [climatology,climatology],
             color="grey", label="climatology")
    ax1.legend(loc="upper left")

    ax2.hist(pred_vals, range=(0, 1), bins=n_bins,
                    histtype="step", lw=2)
    
    if axes is None:
        fs = 'medium' # font size for labels
    else:
        fs = 'x-large'
    ax1.set_ylabel("Observed Probability", size=fs)
    ax2.set_xlabel("Forecast Probability", size=fs)
    ax2.set_ylabel("# events", size=fs)

    plt.subplots_adjust(hspace=0.05)
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
