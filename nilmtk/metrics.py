'''Metrics to compare disaggregation performance of various algorithms

Following is the convention used consistently throughout all the metrics

#TODO: Add all conventions

#TODO: Add more metrics

'''


import numpy as np


def mne(predicted_power, df_appliances_ground_truth):
    '''Compute Mean Normalized Error

    # TODO: Put the formula in terms of conventions and give a vanilla example
    explaining the same

    Attributes
    ----------

    predicted_power: Pandas DataFrame of type {appliance :
         [array of predictd power]}

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

    # TODO: Put the formula in terms of conventions and give a vanilla example
    explaining the same

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
    numerator = {}
    denominator = {}

    for appliance in predicted_power:
        numerator[appliance] = np.sum(np.abs(predicted_power[appliance] -
           df_appliances_ground_truth[appliance].values))
        denominator[appliance] = np.sum(
            df_appliances_ground_truth[appliance].values)
        re[appliance] = np.std(predicted_power[appliance] -
        df_appliances_ground_truth[appliance].values)
    return re



