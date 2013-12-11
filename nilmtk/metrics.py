'''Metrics to compare disaggregation performance of various algorithms

Notation
-----------

Below is the notation used to mathematically define each metric. 

:math:`T` - number of time slices.

:math:`t` - a time slice.

:math:`N` - number of appliances.

:math:`n` - an appliance.

:math:`y^{(n)}_t` -  ground truth power of feed :math:`n` in time slice :math:`t`.

'''


import numpy as np

def feca(predicted_power, df_appliances_ground_truth):
    '''Compute Fraction of Energy Correctly Assigned

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
        
        appliance_energy_ground_truth = np.sum(df_appliances_ground_truth[appliance].values)
        total_energy_ground_truth = np.sum(df_appliances_ground_truth.values)
        
        print appliance_energy_predicted
        print total_energy_predicted
        print appliance_energy_ground_truth
        print total_energy_ground_truth
        
        fraction = np.append(fraction, np.min(
                                              appliance_energy_predicted/total_energy_predicted,
                                              appliance_energy_ground_truth/total_energy_ground_truth
                                              ))
    return fraction

def mne(predicted_power, df_appliances_ground_truth):
    '''Compute Mean Normalized Error

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


def re(predicted_power, df_appliances_ground_truth):
    '''Compute RMS Error
    
    # TODO: Give a vanilla example
    
    .. math::
        error^{(n)} = \\sqrt{ \\frac{1}{T} \\sum_t{ \\left ( y_t - \\hat{y}_t \\right )^2 } }

    Attributes
    ----------

    predicted_power: Pandas DataFrame of type {appliance :
         [array of predictd power]}

    df_appliances_ground_truth: Pandas DataFrame of type {appliance :
        [array of ground truth power]}

    Returns
    -------
    re: dict of type {appliance : MNE error}
    '''

    re = {}

    for appliance in predicted_power:
        re[appliance] = np.std(predicted_power[appliance] -
            df_appliances_ground_truth[appliance].values)
    return re



