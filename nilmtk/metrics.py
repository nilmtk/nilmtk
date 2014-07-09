from __future__ import print_function, division
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from .metergroup import MeterGroup

# For some reason, importing sklearn causes PyTables to raise lots
# of DepreciatedWarnings for Pandas code.
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


'''Metrics to compare disaggregation performance of various algorithms

Notation
--------

Below is the notation used to mathematically define each metric. 

:math:`T` - number of time slices.

:math:`t` - a time slice.

:math:`N` - number of appliances.

:math:`n` - an appliance.

:math:`y^{(n)}_t` -  ground truth power of appliance :math:`n` in time slice :math:`t`.

:math:`\\hat{y}^{(n)}_t` -  estimated power of appliance :math:`n` in time slice :math:`t`.

:math:`x^{(n)}_t` - ground truth state of appliance :math:`n` in time slice :math:`t`.

:math:`\\hat{x}^{(n)}_t` - estimated state of appliance :math:`n` in time slice :math:`t`.
'''


def error_in_assigned_energy(predictions, ground_truth):
    """Compute error in assigned energy.
    
    .. math::
        error^{(n)} = 
        \\left | \\sum_t y^{(n)}_t - \\sum_t \\hat{y}^{(n)}_t \\right |

    Parameters
    ----------
    predictions, ground_truth : nilmtk.MeterGroup

    Returns
    -------
    errors : pd.Series
        Each index is an meter instance int (or tuple for MeterGroups).
        Each value is the absolute error in assigned energy for that appliance,
            in kWh.
    """
    errors = {}
    for meter in predictions.submeters():
        ground_truth_meter_identifier = meter.identifier._replace(
            dataset=ground_truth.dataset())
        ground_truth_meter = ground_truth[ground_truth_meter_identifier]
        sections = meter.good_sections()

        ground_truth_energy = ground_truth_meter.total_energy(periods=sections)
        predicted_energy = meter.total_energy(periods=sections)
        errors[meter.instance()] = np.abs(predicted_energy - ground_truth_energy)
    return pd.Series(errors)


def fraction_energy_assigned_correctly(predictions, ground_truth):
    '''Compute fraction of energy assigned correctly
    
    .. math::
        fraction = 
        \\sum_n min \\left ( 
        \\frac{\\sum_n y}{\\sum_{n,t} y}, 
        \\frac{\\sum_n \\hat{y}}{\\sum_{n,t} \\hat{y}} 
        \\right )

    Ignores distinction between different AC types, instead if there are 
    multiple AC types for each meter then we just take the max value across
    the AC types.

    Parameters
    ----------
    predictions, ground_truth : nilmtk.MeterGroup

    Returns
    -------
    float in the range [0,1] representing Fraction of Energy Correctly Assigned
    '''

    predictions_submeters = MeterGroup(meters=predictions.submeters())
    ground_truth_submeters = MeterGroup(meters=ground_truth.submeters())

    fraction_per_meter_predictions = predictions_submeters.fraction_per_meter()
    fraction_per_meter_ground_truth = ground_truth_submeters.fraction_per_meter()

    fractions = []
    for meter_instance in predictions_submeters.instance():
        fraction = min(fraction_per_meter_predictions[meter_instance],
                       fraction_per_meter_ground_truth[meter_instance])
        fractions.append(fraction)

    return sum(fractions)


def mean_normalized_error_power(predictions, ground_truth):
    '''Compute mean normalized error in assigned power
        
    .. math::
        error^{(n)} = 
        \\frac
        { \\sum_t {\\left | y_t^{(n)} - \\hat{y}_t^{(n)} \\right |} }
        { \\sum_t y_t^{(n)} }

    Parameters
    ----------
    predictions, ground_truth : nilmtk.MeterGroup

    Returns
    -------
    pd.Series
        Each index is an meter instance int (or tuple for MeterGroups).
        Each value is the MNE for that appliance.
    '''

    # TODO: need to resample to keep things in step
    mne = {}
    for meter in predictions.submeters():
        ground_truth_meter_identifier = meter.identifier._replace(
            dataset=ground_truth.dataset())
        ground_truth_meter = ground_truth[ground_truth_meter_identifier]
        sections = meter.good_sections()
        sample_period = meter.sample_period()
        period_alias = '{:d}S'.format(sample_period)

        # TODO: preprocessing=[Resample(sample_period)])
        pred_generator = meter.power_series(periods=sections)
        total_diff = 0
        sum_of_ground_truth_power = 0
        while True:
            try:
                pred_chunk = next(pred_generator)
            except StopIteration:
                break
            else:
                truth_generator = ground_truth_meter.power_series(
                    periods=[pred_chunk.timeframe], chunksize=1E9)
                truth_chunk = next(truth_generator)
                
                # TODO: do this resampling in the pipeline?
                truth_chunk = truth_chunk.resample(period_alias)
                pred_chunk = pred_chunk.resample(period_alias)

                diff = (pred_chunk.icol(0) - truth_chunk.icol(0)).dropna() 
                total_diff += sum(abs(diff))
                sum_of_ground_truth_power += truth_chunk.icol(0).dropna().sum()

        mne[meter.instance()] = total_diff / sum_of_ground_truth_power

    return pd.Series(mne)

########## FUNCTIONS BELOW THIS LINE HAVE NOT YET CONVERTED TO NILMTK v0.2 #####

def rms_error_power(predicted_power, df_appliances_ground_truth):
    '''Compute RMS error in assigned power
    
    # TODO: Give a vanilla example
    
    .. math::
            error^{(n)} = \\sqrt{ \\frac{1}{T} \\sum_t{ \\left ( y_t - \\hat{y}_t \\right )^2 } }

    Parameters
    ----------

    predicted_power: Pandas DataFrame of type {appliance :
         [array of predicted power]}

    df_appliances_ground_truth: Pandas DataFrame of type {appliance :
        [array of ground truth power]}

    Returns
    -------
    re: dict of type {appliance : RMS error in predicted power}
    '''

    re = {}

    for appliance in predicted_power:
        re[appliance] = np.std(predicted_power[appliance] -
                               df_appliances_ground_truth[appliance].values)

    return re


def powers_to_states(powers):
    '''Converts power demands into binary states
    
    # TODO: Give a vanilla example

    Parameters
    ----------

    powers: Pandas DataFrame of type {appliance :
         [array of power]}

    Returns
    -------
    states: Pandas DataFrame of type {appliance :
         [array of states]}
    '''

    on_power_threshold = 50

    states = pd.DataFrame(np.zeros(power.shape))
    states[power > on_power_threshold] = 1

    return states


def confusion_matrices(predicted_states, ground_truth_states):
    '''Compute confusion matrix between appliance states for each appliance

    # TODO: Give a vanilla example

    Parameters
    ----------

    predicted_state: Pandas DataFrame of type {appliance :
         [array of predicted states]}

    ground_truth_state: Pandas DataFrame of type {appliance :
        [array of ground truth states]}

    Returns
    -------
    dict of type {appliance : confusion matrix}
    '''

    re = {}

    for appliance in predicted_states:
        matrix = np.zeros([np.max(ground_truth_states[appliance]) + 1,
                           np.max(ground_truth_states[appliance]) + 1])
        for time in predicted_states[appliance]:
            matrix[predicted_states.values[time, appliance],
                   ground_truth_states.values[time, appliance]] += 1
        re[appliance] = matrix

    return re


def tp_fp_fn_tn(predicted_states, ground_truth_states):
    '''Compute counts of True Positives, False Positives, False Negatives, True Negatives

    # TODO: Give a vanilla example
    
    .. math::
        TP^{(n)} = 
        \\sum_{t}
        and \\left ( x^{(n)}_t = on, \\hat{x}^{(n)}_t = on \\right )
        
        FP^{(n)} = 
        \\sum_{t}
        and \\left ( x^{(n)}_t = off, \\hat{x}^{(n)}_t = on \\right )
        
        FN^{(n)} = 
        \\sum_{t}
        and \\left ( x^{(n)}_t = on, \\hat{x}^{(n)}_t = off \\right )
        
        TN^{(n)} = 
        \\sum_{t}
        and \\left ( x^{(n)}_t = off, \\hat{x}^{(n)}_t = off \\right )

    Parameters
    ----------

    predicted_state: Pandas DataFrame of type {appliance :
         [array of predicted states]}

    ground_truth_state: Pandas DataFrame of type {appliance :
        [array of ground truth states]}

    Returns
    -------
    numpy array where columns represent appliances and rows represent: [TP, FP, FN, TN]
    '''

    # assumes state 0 = off, all other states = on
    predicted_states_on = predicted_states > 0
    ground_truth_states_on = ground_truth_states > 0

    tp = np.sum(np.logical_and(predicted_states_on.values == True,
                ground_truth_states_on.values == True), axis=0)
    fp = np.sum(np.logical_and(predicted_states_on.values == True,
                ground_truth_states_on.values == False), axis=0)
    fn = np.sum(np.logical_and(predicted_states_on.values == False,
                ground_truth_states_on.values == True), axis=0)
    tn = np.sum(np.logical_and(predicted_states_on.values == False,
                ground_truth_states_on.values == False), axis=0)

    return np.array([tp, fp, fn, tn]).astype(float)


def tpr_fpr(predicted_states, ground_truth_states):
    '''Compute True Positive Rate and False Negative Rate

    # TODO: Give a vanilla example
    
    .. math::
        TPR^{(n)} = \\frac{TP}{\\left ( TP + FN \\right )}
        
        FPR^{(n)} = \\frac{FP}{\\left ( FP + TN \\right )}

    Parameters
    ----------

    predicted_state: Pandas DataFrame of type {appliance :
         [array of predicted states]}

    ground_truth_state: Pandas DataFrame of type {appliance :
        [array of ground truth states]}

    Returns
    -------
    numpy array where columns represent appliances and rows represent: [TPR, FPR]
    '''

    tfpn = tp_fp_fn_tn(predicted_states, ground_truth_states)

    tpr = tfpn[0, :] / (tfpn[0, :] + tfpn[2, :])
    fpr = tfpn[1, :] / (tfpn[1, :] + tfpn[3, :])

    return np.array([tpr, fpr])


def precision_recall(predicted_states, ground_truth_states):
    '''Compute Precision and Recall

    # TODO: Give a vanilla example
    
    .. math::
        Precision^{(n)} = \\frac{TP}{\\left ( TP + FP \\right )}
        
        Recall^{(n)} = \\frac{TP}{\\left ( TP + FN \\right )}

    Parameters
    ----------

    predicted_state: Pandas DataFrame of type {appliance :
         [array of predicted states]}

    ground_truth_state: Pandas DataFrame of type {appliance :
        [array of ground truth states]}

    Returns
    -------
    numpy array where columns represent appliances and rows represent: [Precision, Recall]
    '''

    tfpn = tp_fp_fn_tn(predicted_states, ground_truth_states)

    prec = tfpn[0, :] / (tfpn[0, :] + tfpn[1, :])
    rec = tfpn[0, :] / (tfpn[0, :] + tfpn[2, :])

    return np.array([prec, rec])


def f_score(predicted_power, ground_truth_power):
    '''Compute F1 score

    # TODO: Give a vanilla example
    
    .. math::
        F_score^{(n)} = \\frac
            {2 * Precision * Recall}
            {Precision + Recall}

    Parameters
    ----------

    predicted_state: Pandas DataFrame of type {appliance :
         [array of predicted states]}

    ground_truth_state: Pandas DataFrame of type {appliance :
        [array of ground truth states]}

    Returns
    -------
    numpy array where columns represent appliances and rows represent F score
    '''
    threshold = 30
    predicted_states = (predicted_power > threshold).astype(int)
    ground_truth_states = (ground_truth_power > threshold).astype(int)
    f_score_out = {}
    for appliance in predicted_states.columns:
        f_score_out[appliance] = f1_score(
            ground_truth_states[[appliance]], predicted_states[[appliance]])
    return f_score_out

    #prec_rec = precision_recall(predicted_states, ground_truth_states)
    # return (2 * prec_rec[0, :] * prec_rec[1,:]) / (prec_rec[0,:] +
    # prec_rec[1,:])
    # return f1_score(ground_truth_states, predicted_states)


def hamming_loss(predicted_state, ground_truth_state):
    '''Compute Hamming loss

    # TODO: Give a vanilla example
    
    .. math::
        HammingLoss = 
        \\frac{1}{T} \\sum_{t}
        \\frac{1}{N} \\sum_{n}
        xor \\left ( x^{(n)}_t, \\hat{x}^{(n)}_t \\right )

    Parameters
    ----------

    predicted_state: Pandas DataFrame of type {appliance :
         [array of predicted states]}

    ground_truth_state: Pandas DataFrame of type {appliance :
        [array of ground truth states]}

    Returns
    -------
    float of hamming_loss
    '''

    num_appliances = np.size(ground_truth_state.values, axis=1)

    xors = np.sum((predicted_state.values != ground_truth_state.values),
                  axis=1) / num_appliances

    return np.mean(xors)
