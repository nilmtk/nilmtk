"""Preprocessing functions for a single appliance / mains / circuit DataFrame"""

from __future__ import print_function, division
import numpy as np
import pandas as pd
from copy import deepcopy

from nilmtk.stats.electricity.single import get_sample_period
from nilmtk.sensors.electricity import Measurement


def insert_zeros(single_appliance_dataframe, max_sample_period=None):
    """There are two possible reasons for lost samples in individual
    appliance data: 

    1) a broken IAM (hence we do not have any information about the appliance)
    2) the IAM and appliance have been unplugged (hence we can infer that the
       appliance is off)

    Only the user who can decide which of these two assumptions best
    fits their data.  insert_zeros is applicable only in case 2.

    Some individual appliance monitors (IAMs) get turned off.
    This might happen, for example, in the case where a hoover's IAM is 
    permanently attached to the hoover's power cord, even when the hoover is
    unplugged and put away in the cupboard.

    Say the hoover was switched on and then both the hoover and the hoover's IAM
    were unplugged.  This would result in the dataset having a gap immediately
    after an on-segment.  This combination of an on-segment followed (without
    any zeros) by a gap might confuse downstream statistics and
    disaggregation functions which assume that the power drawn by an appliance
    between reading[i] and reading[i+1] is held constant at reading[i] watts.

    If, after any reading > 0, there is a gap in the dataset of more than 
    `max_sample_period` seconds then assume the appliance (and 
    individual appliance monitor) have been turned off from the
    mains and hence insert a zero max_sample_period seconds after 
    the last sample of the on-segment.

    TODO: a smarter version of this function might use information from
    the aggregate data to do a better job of estimating exactly when
    the appliance was turned off.

    Parameters
    ----------
    single_appliance_dataframe : pandas.DataFrame
        Data from a single appliance.

    max_sample_period : float or int, optional
        The maximum sample permissible period (in seconds). Any gap longer
        than `max_sample_period` is assumed to imply that the IAM 
        and appliance are off.  If None then will default to
        4 x the sample period of `single_appliance_dataframe`.

    Returns
    -------
    df_with_zeros : pandas.DataFrame
        A copy of `single_appliance_dataframe` with zeros inserted 
        `max_sample_period` seconds after the last sample of each on-segment.

    """
    if max_sample_period is None:
        max_sample_period = get_sample_period(single_appliance_dataframe) * 4

    # Make a copy (the copied dataframe is what we return, after inserting
    # zeros)
    df_with_zeros = deepcopy(single_appliance_dataframe)

    # Get the length of time between each pair of consecutive samples. Seconds.
    timedeltas = np.diff(df_with_zeros.index.values) / np.timedelta64(1, 's')
    readings_before_gaps = df_with_zeros[:-1][timedeltas > max_sample_period]

    # we only add a 0 if the recorded value just before the gap is > 0
    readings_before_gaps = readings_before_gaps[
        readings_before_gaps.sum(axis=1) > 0]

    # Make a DataFrame of zeros, ready for insertion
    dates_to_insert_zeros = (readings_before_gaps.index +
                             pd.DateOffset(seconds=max_sample_period))
    zeros = pd.DataFrame(data=0, index=dates_to_insert_zeros,
                         columns=df_with_zeros.columns, dtype=np.float32)

    # Insert the dataframe of zeros into the data.
    df_with_zeros = df_with_zeros.append(zeros)
    df_with_zeros = df_with_zeros.sort_index()
    return df_with_zeros


def replace_nans_with_zeros(multiple_appliances_dataframe, max_sample_period):
    """For a single column, find gaps of > max_sample_period and replace
    all NaNs in these gaps with zeros.  
    But leave NaNs is gaps <= max_sample_period."""
    raise NotImplementedError


def normalise_power(power, voltage, nominal_voltage):
    """ Uses Hart's formula to calculate:
     "admittance in the guise of 'normalized power':
        
    P_{Norm}(t) = 230 ^ 2 x Y(t) = (230 / V(t)) ^ 2 x P(t)

    This is just the admittance adjusted by a constant scale
    factor, resulting in the power normalized to 120 V, i.e.,
    what the power would be if the utility provided a steady
    120 V and the load obeyed a linear model. It is a far more
    consistent signature than power... All of our prototype
    NALMs use step changes in the normalized power as the
    signature."
    (equation 4, page 8 of Hart 1992)

    Parameters
    ----------
    power : pd.Series
    voltage : pd.Series
    nominal_voltage :float
        Rated voltage supply in the country

    Returns
    -------
    power_normalized : pd.Series
    """
    power_normalized = ((nominal_voltage / voltage) ** 2) * power
    return power_normalized


def remove_implausible_entries(channel_df, measurement,
                               min_threshold=None, max_threshold=None):
    """
    It is sometimes observed that sometimes sensors may give implausible
    entries. eg. Voltage of 0.0 makes no sense or power of an appliance 
    to be 1MW makes no sense. These records must be filtered out. This
    method filters out values outside a given range.

    Parameters
    ----------
    channel_df : pandas.DataFrame
        Corresponds to either an appliance or mains or circuit which 
        contains `measurement` as one of the columns

    measurement : nilmtk.sensors.electricity.Measurement
    min_threshold: float
    max_threshold: float

    Returns
    -------
    implausible_entries_dropped : pandas.DataFrame
    """

    assert(measurement in channel_df.columns)
    # Atleast one of min_threshold or max_threshold must be there
    assert((min_threshold is not None) or (max_threshold is not None))

    if min_threshold is None:
        min_threshold = channel_df[measurement].min()
    if max_threshold is None:
        max_threshold = channel_df[measurement].max()

    s = channel_df[measurement]
    implausible_entries_dropped = channel_df[
        (s > min_threshold) & (s < max_threshold)]
    return implausible_entries_dropped


def filter_datetime_single(channel_df, start_datetime=None, end_datetime=None):
    """
    Filtering out a channel outside certain datetimes

    Parameters
    ----------
    channel_df : pandas.DataFrame
        Corresponds to either an appliance or mains or circuit         
    
    start_datetime :string, 'mm-dd-yyyy hh:mm:ss'
    end_datetime : string, 'mm-dd-yyyy hh:mm:ss'

    Returns
    -------
    pandas.DataFrame
    """

    # Atleast one of start_datetime or end_datetime must be there
    assert((start_datetime is not None) or (end_datetime is not None))

    if start_datetime is None:
        start_datetime = channel_df.index.values[0]
    if end_datetime is None:
        end_datetime = channel_df.index.values[0]

    return channel_df[pd.Timestamp(start_datetime):pd.Timestamp(end_datetime)]
