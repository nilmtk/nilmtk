'''Metrics to compare disaggregation performance of various algorithms

Notation
-----------

Below is the notation used to mathematically define each metric. 

:math:`T` - number of time slices.

:math:`t` - a time slice.

:math:`N` - number of appliances.

:math:`n` - an appliance.

:math:`y^{(n)}_t` -  ground truth power of appliance :math:`n` in time slice :math:`t`.

:math:`\\hat{y}^{(n)}_t` -  estimated power of appliance :math:`n` in time slice :math:`t`.

:math:`z^{(n)}_t` - ground truth state of appliance :math:`n` in time slice :math:`t`.

:math:`\\hat{z}^{(n)}_t` - estimated state of appliance :math:`n` in time slice :math:`t`.

Functions
-----------

'''


import numpy as np
import pandas as pd


def error_energy(predicted_power, df_appliances_ground_truth):
    '''Compute error in assigned energy

    # TODO: Give a vanilla example
    
    .. math::
        error^{(n)} = 
        \\left | \\sum_t y^{(n)}_t - \\sum_t \\hat{y}^{(n)}_t \\right |

    Attributes
    ----------

    predicted_power: Pandas DataFrame of type {appliance :
         [array of predictd power]}

    df_appliances_ground_truth: Pandas DataFrame of type {appliance :
        [array of ground truth power]}

    Returns
    -------
    error: dict of type {appliance : error in assigned energy}
    '''
    error = {}

    for appliance in predicted_power:
        ground_truth_energy = np.sum(
            df_appliances_ground_truth[appliance].values)
        predicted_energy = np.sum(predicted_power[appliance].values)
        error[appliance] = np.abs(predicted_energy - ground_truth_energy)
    return error


def fraction_energy_assigned_correctly(predicted_power, df_appliances_ground_truth):
    '''Compute fraction of energy assigned correctly

    # TODO: Give a vanilla example
    
    .. math::
        fraction = 
        \\sum_n min \\left ( 
        \\frac{\\sum_n y}{\\sum_{n,t} y}, 
        \\frac{\\sum_n \\hat{y}}{\\sum_{n,t} \\hat{y}} 
        \\right )

    Attributes
    ----------

    predicted_power: Pandas DataFrame of type {appliance :
         [array of predictd power]}

    df_appliances_ground_truth: Pandas DataFrame of type {appliance :
        [array of ground truth power]}

    Returns
    -------
    re: float representing Fraction of Energy Correctly Assigned
    '''

    fraction = np.array([])

    for appliance in predicted_power:

        appliance_energy_predicted = np.sum(predicted_power[appliance].values)
        total_energy_predicted = np.sum(predicted_power.values)

        appliance_energy_ground_truth = np.sum(
            df_appliances_ground_truth[appliance].values)
        total_energy_ground_truth = np.sum(df_appliances_ground_truth.values)

        fraction = np.append(fraction, np.min(
                                              appliance_energy_predicted/total_energy_predicted,
                                              appliance_energy_ground_truth/total_energy_ground_truth
                                              ))
    return np.sum(fraction)


def hamming_loss(predicted_state, ground_truth_state):
    '''Compute Hamming loss

    # TODO: Give a vanilla example
    
    .. math::
        HammingLoss^{(n)} = 
        \\frac{1}{T} \\sum_{t}
        xor \\left ( z^{(n)}_t, \\hat{z}^{(n)}_t \\right )

    Attributes
    ----------

    predicted_state: Pandas DataFrame of type {appliance :
         [array of predicted states]}

    ground_truth_state: Pandas DataFrame of type {appliance :
        [array of ground truth states]}

    Returns
    -------
    re: dict of type {appliance : HammingLoss}
    '''

    loss = {}

    for appliance in predicted_state:
        loss[appliance] = np.sum(predicted_state[appliance].values ==
                                 ground_truth_state[appliance].values)
    return loss

def mean_normalized_error_power(predicted_power, df_appliances_ground_truth):
    '''Compute mean normalized error in assigned power

    # TODO: Give a vanilla example
        
    .. math::
        error^{(n)} = 
        \\frac
        { \\sum_t {\\left | y_t^{(n)} - \\hat{y}_t^{(n)} \\right |} }
        { \\sum_t y_t^{(n)} }

    Attributes
    ----------

    predicted_power: Pandas DataFrame of type {appliance :
         [array of predicted power]}

    df_appliances_ground_truth: Pandas DataFrame of type {appliance :
        [array of ground truth power]}

    Returns
    -------
    mne: dict of type {appliance : MNE error}
    '''

    mne = {}
    numerator = {}
    denominator = {}

    for appliance in predicted_power:
        numerator[appliance] = np.sum(np.abs(predicted_power[appliance] -
                                             df_appliances_ground_truth[appliance].values))
        denominator[appliance] = np.sum(
            df_appliances_ground_truth[appliance].values)
        mne[appliance] = numerator[appliance] * 1.0 / denominator[appliance]
    return mne


def rms_error_power(predicted_power, df_appliances_ground_truth):
    '''Compute RMS error in assigned power
    
    # TODO: Give a vanilla example
    
    .. math::
            error^{(n)} = \\sqrt{ \\frac{1}{T} \\sum_t{ \\left ( y_t - \\hat{y}_t \\right )^2 } }

    Attributes
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

def confusion_matrices(predicted_states, ground_truth_states):
    '''Compute confusion matrix between appliance states for each appliance

    # TODO: Give a vanilla example

    Attributes
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
        matrix = np.zeros([np.max(ground_truth_states[appliance])+1,
                           np.max(ground_truth_states[appliance])+1])
        for time in predicted_states[appliance]:
            matrix[predicted_states.values[time,appliance],ground_truth_states.values[time,appliance]] += 1
        re[appliance] = matrix
        
    return re

def tp_fp_fn_tn(predicted_states, ground_truth_states):
    
    '''Compute counts of True Positives, False Positives, False Negatives, True Negatives

    # TODO: Give a vanilla example
    
    .. math::
        TP^{(n)} = 
        \\sum_{t}
        and \\left ( z^{(n)}_t = on, \\hat{z}^{(n)}_t = on \\right )
        
        FP^{(n)} = 
        \\sum_{t}
        and \\left ( z^{(n)}_t = off, \\hat{z}^{(n)}_t = on \\right )
        
        FN^{(n)} = 
        \\sum_{t}
        and \\left ( z^{(n)}_t = on, \\hat{z}^{(n)}_t = off \\right )
        
        TN^{(n)} = 
        \\sum_{t}
        and \\left ( z^{(n)}_t = off, \\hat{z}^{(n)}_t = off \\right )

    Attributes
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
    
    tp = np.sum(np.logical_and(predicted_states_on.values==True, ground_truth_states_on.values==True), axis=0)
    fp = np.sum(np.logical_and(predicted_states_on.values==True, ground_truth_states_on.values==False), axis=0)
    fn = np.sum(np.logical_and(predicted_states_on.values==False, ground_truth_states_on.values==True), axis=0)
    tn = np.sum(np.logical_and(predicted_states_on.values==False, ground_truth_states_on.values==False), axis=0)
    
    return np.array([tp, fp, fn, tn]).astype(float)

def tpr_fpr(predicted_states, ground_truth_states):
    
    '''Compute True Positive Rate and False Negative Rate

    # TODO: Give a vanilla example
    
    .. math::
        TPR^{(n)} = \\frac{TP}{\\left ( TP + FN \\right )}
        
        FPR^{(n)} = \\frac{FP}{\\left ( FP + TN \\right )}

    Attributes
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
    
    tpr = tfpn[0,:] / (tfpn[0,:] + tfpn[2,:])
    fpr = tfpn[1,:] / (tfpn[1,:] + tfpn[3,:])
    
    return np.array([tpr, fpr])

def precision_recall(predicted_states, ground_truth_states):
    
    '''Compute Precision and Recall

    # TODO: Give a vanilla example
    
    .. math::
        Precision^{(n)} = \\frac{TP}{\\left ( TP + FP \\right )}
        
        Recall^{(n)} = \\frac{TP}{\\left ( TP + FN \\right )}

    Attributes
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
    
    prec = tfpn[0,:] / (tfpn[0,:] + tfpn[1,:])
    rec = tfpn[0,:] / (tfpn[0,:] + tfpn[2,:])
    
    return np.array([prec, rec])

def f_score(predicted_states, ground_truth_states):
    
    '''Compute F1 score

    # TODO: Give a vanilla example
    
    .. math::
        F_score^{(n)} = \\frac
            {2 * Precision * Recall}
            {Precision + Recall}

    Attributes
    ----------

    predicted_state: Pandas DataFrame of type {appliance :
         [array of predicted states]}

    ground_truth_state: Pandas DataFrame of type {appliance :
        [array of ground truth states]}

    Returns
    -------
    numpy array where columns represent appliances and rows represent F score
    '''
    
    prec_rec = precision_recall(predicted_states, ground_truth_states)
    
    return (2 * prec_rec[0,:] * prec_rec[1,:]) / (prec_rec[0,:] + prec_rec[1,:])

def hamming_loss(predicted_state, ground_truth_state):
    '''Compute Hamming loss

    # TODO: Give a vanilla example
    
    .. math::
        HammingLoss = 
        \\frac{1}{T} \\sum_{t}
        \\frac{1}{N} \\sum_{n}
        xor \\left ( z^{(n)}_t, \\hat{z}^{(n)}_t \\right )

    Attributes
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
    
    xors = np.sum((predicted_state.values != ground_truth_state.values), axis=1) / num_appliances
    
    return np.mean(xors)
