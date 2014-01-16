"""Preprocessing functions for a single appliance / mains / circuit DataFrame"""

from __future__ import print_function, division
import numpy as np
import pandas as pd
from copy import deepcopy

from nilmtk.stats.electricity.single import get_sample_period
from nilmtk.sensors.electricity import Measurement
from nilmtk.utils import secs_per_period_alias, timedelta64_to_secs


def insert_zeros(single_appliance_dataframe, max_sample_period=None,
                 sample_period_multiplier=4,
                 round_sample_period=True):
    """Find all gaps in `single_appliance_dataframe` longer than
    `max_sample_period` and insert a zero 1 sample period after
    the start of the gap and insert a second zero 1 sample period
    before the end of the gap.

    In other words: "book-end" the gap with a zero at each end.

    Zeros are only inserted at the start of the gap if the gap
    starts with a reading above zero; and likewise for insertion
    of zeros at the end of the gap.

    Note that this function does not fill the entire gap with zeros,
    if you want that then try pandas.DataFrame.fillna

    What is `insert_zeros` useful for?

    There are two possible reasons for lost samples in individual
    appliance data: 

    1) a broken IAM (hence we do not have any information about the appliance)
    2) the IAM and appliance have been unplugged (hence we can infer that the
       appliance is off)

    Only the user who can decide which of these two assumptions best
    fits their data.  insert_zeros is applicable only in case 2.

    For example, say a hoover's IAM is permanently attached to the
    hoover's power cord, even when the hoover is unplugged and put
    away in the cupboard.

    Say the hoover was switched on when both the hoover and the
    hoover's IAM were unplugged.  This would result in the dataset
    having a gap immediately after an on-segment.  This combination of
    an on-segment followed (without any zeros) by a gap might confuse
    downstream statistics and disaggregation functions which assume
    that the power drawn by an appliance between reading[i] and
    reading[i+1] is held constant at reading[i] watts.

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
        `sample_period_multiplier` x the sample period of 
        `single_appliance_dataframe`.

    sample_period_multiplier : float or int, optional
        default = 4
    
    Returns
    -------
    df_with_zeros : pandas.DataFrame
        A copy of `single_appliance_dataframe` with zeros inserted 
        `max_sample_period` seconds after the last sample of each on-segment.

    """
    sample_period = get_sample_period(single_appliance_dataframe)
    if round_sample_period:
        sample_period = int(round(sample_period))
    if max_sample_period is None:
        max_sample_period = sample_period * sample_period_multiplier

    # Drop NaNs (because we want those to be gaps in the index)
    df = single_appliance_dataframe.dropna()

    # Get the length of time between each pair of consecutive samples. Seconds.
    timedeltas = np.diff(df.index.values) / np.timedelta64(1, 's')
    gaps_mask = timedeltas > max_sample_period
    readings_before_gaps = df[:-1][gaps_mask]
    readings_after_gaps = df[1:][gaps_mask]

    # we only add a 0 if the recorded value just before the gap is > 0
    readings_before_gaps = readings_before_gaps[
        readings_before_gaps.sum(axis=1) > 0]

    readings_after_gaps = readings_after_gaps[
        readings_after_gaps.sum(axis=1) > 0]

    # Find dates to insert zeros
    dates_to_insert_zeros_before_gaps = (
        readings_before_gaps.index + pd.DateOffset(seconds=sample_period))

    dates_to_insert_zeros_after_gaps = (
        readings_after_gaps.index - pd.DateOffset(seconds=sample_period))

    dates_to_insert_zeros = dates_to_insert_zeros_before_gaps.append(
        dates_to_insert_zeros_after_gaps)

    # Columns containing power_energy
    power_columns = [x for x in df.columns if x.physical_quantity in ['power']]
    non_power_columns = [x for x in df.columns if x not in power_columns]

    # Don't insert duplicate indicies
    assert((dates_to_insert_zeros & df.index).size == 0)

    # Create new dataframe of zeros at new indicies ready for insertion
    zeros = pd.DataFrame(data=0,
                         index=dates_to_insert_zeros,
                         columns=power_columns,
                         dtype=np.float32)

    # Now, take median of non-power columns (like voltage)
    for measurement in non_power_columns:
        zeros[measurement] = single_appliance_dataframe[measurement].median()

    # Insert the dataframe of zeros into the data.
    df_with_zeros = deepcopy(single_appliance_dataframe)
    df_with_zeros = df_with_zeros.append(zeros)
    df_with_zeros = df_with_zeros.sort_index()

    # If input data had a regular frequency then resample
    # because appending turns off the regular frequency.
    original_freq = single_appliance_dataframe.index.freq
    if original_freq is not None:
        df_with_zeros = df_with_zeros.resample(rule=original_freq)

    return df_with_zeros


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


def reframe_index(index, window_start=None, window_end=None):
    """
    Parameters
    ----------
    index : pd.DatetimeIndex

    window_start, window_end : pd.Timestamp
        The start and end of the window of interest.  If this window
        is larger than the duration of `data` then a single zero will be
        inserted at `window_start` or `window_end` as necessary.  If this window
        is shorter than the duration of `data` data will be cropped.

    Returns
    -------
    index : pd.DatetimeIndex
    """
    # TODO: can this function be merged with
    # preprocessing.building.building.prepend_append_zeros ?

    tz = index.tzinfo

    # Handle window...
    if window_start is not None:
        if window_start >= index[0]:
            index = index[index >= window_start]
        else:
            index = index.insert(0, window_start).tz_localize(
                'UTC').tz_convert(tz)

    if window_end is not None:
        if window_end <= index[-1]:
            index = index[index <= window_end]
        else:
            index = index.insert(len(index), window_end).tz_localize(
                'UTC').tz_convert(tz)

    return index


def contiguous_blocks(datetimeindex):
    sample_period = get_sample_period(datetimeindex)
    time_delta = timedelta64_to_secs(np.diff(datetimeindex.values))
    breaks = time_delta > sample_period
    if np.sum(breaks) == 0:
        # All contiguous data
        contiguous_time_tuples = [(datetimeindex[0], datetimeindex[-1])]
    # Data has breaks
    else:
        break_indices_int = np.where(breaks)[0]
        contiguous_time_tuples = []
        start = 0
        for end in break_indices_int:
            contiguous_time_tuples.append(
                (datetimeindex[start], datetimeindex[end]))
            start = end + 1
        # Appending last block
        contiguous_time_tuples.append(
            (datetimeindex[start], datetimeindex[-1]))
    return contiguous_time_tuples
