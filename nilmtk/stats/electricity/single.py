"""Statistics applicable to a single appliance / circuit / mains split.

In general, these functions each take a DataFrame representing a single
appliance / circuit / mains split.
"""
from __future__ import print_function, division
import scipy.stats as stats
import numpy as np
import pandas as pd
from matplotlib.dates import SEC_PER_HOUR


def sample_period(df):
    """Estimate the sample period in seconds.

    Find the sample period by finding the stats.mode of the 
    forward difference.  Only use the first 100 samples (for speed).

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    period : float
        Sample period in seconds.
    """
    fwd_diff = np.diff(df.index.values[:100]).astype(np.float)
    mode_fwd_diff = stats.mode(fwd_diff)[0][0]
    period = mode_fwd_diff / 1E9
    return period
    

def dropout_rate(df):
    """The proportion of samples that have been lost.

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    rate : float [0,1]
        The proportion of samples that have been lost; where 
        1 means that all samples have been lost and 
        0 means that no samples have been lost.
    """
    duration = df.index[-1] - df.index[0]        
    n_expected_samples = duration.total_seconds() / sample_period(df)
    return 1 - (df.index.size / n_expected_samples)


def hours_on(series, on_power_threshold=5, max_sample_period=None):
    """Returns a float representing the number of hours this channel
    has been above threshold.

    Parameters
    ----------
    series : pandas.Series

    on_power_threshold : float or int, optional, default = 5
        Threshold which defines the distinction between "on" and "off".  Watts.

    max_sample_period : float or int, optional 
        The maximum allowed sample period in seconds.  This is used
        where, for example, we have a wireless meter which is supposed
        to report every `K` seconds and we assume that if we don't
        hear from it for more than `max_sample_period=K*3` seconds
        then the sensor (and appliance) have been turned off from the
        wall. If we find a sample above `on_power_threshold` at time
        `t` and there are more than `max_sample_period` seconds until
        the next sample then we assume that the appliance has only
        been on for `max_sample_period` seconds after time `t`.

    Returns
    -------
    hours_above_threshold : float


    See Also
    --------
    kwh
    joules
    """

    i_above_threshold = np.where(series[:-1] >= on_power_threshold)[0]
    td_above_thresh = (series.index[i_above_threshold+1].values -
                       series.index[i_above_threshold].values)
    if max_sample_period is not None:
        td_above_thresh[td_above_thresh > max_sample_period] = max_sample_period

    secs_on = td_above_thresh.sum().astype('timedelta64[s]').astype(np.int64)
    return secs_on / SEC_PER_HOUR


def energy(series, max_sample_period=None, unit='kwh'):
    """Returns a float representing the quantity of energy this 
    channel consumed.

    Parameters
    ----------
    series : pd.Series

    max_sample_period : float or int, optional 
        The maximum allowed sample period in seconds.  If we find a
        sample above `on_power_threshold` at time `t` and there are
        more than `max_sample_period` seconds until the next sample
        then we assume that the appliance has only been on for
        `max_sample_period` seconds after time `t`.  This is used where,
        for example, we have a wireless meter which is supposed to
        report every `K` seconds and we assume that if we don't hear
        from it for more than `max_sample_period=K*3` seconds then the
        sensor (and appliance) have been turned off from the wall.

    unit : {'kwh', 'joules'}

    Returns
    -------
    _energy : float

    See Also
    --------
    hours_on
    """
    td = np.diff(series.index.values)
    if max_sample_period is not None:
        td = np.where(td > max_sample_period, max_sample_period, td)
    td_secs = td / np.timedelta64(1, 's')
    joules = (td_secs * series.values[:-1]).sum()

    if unit == 'kwh':
        JOULES_PER_KWH = 3600000
        _energy = joules / JOULES_PER_KWH
    elif unit == 'joules':
        _energy = joules
    else:
        raise ValueError('unrecognised value for `unit`.')

    return _energy


def usage_per_period(series, freq, tz_convert=None, on_power_threshold=5, 
                     max_dropout_rate=0.2, verbose=False, 
                     energy_unit='kwh', max_sample_period=None):
    """Calculate the usage (hours on and kwh) per time period.

    Parameters
    ----------
    series : pd.Series

    freq : str
        see _indicies_of_periods() for acceptable values.

    on_power_threshold : float or int, optional, default = 5
        Threshold which defines the distinction between "on" and "off".  Watts.

    max_dropout_rate : float (0,1), optional, default = 0.2
        Remove any row which has a worse (larger) dropout rate.
    
    verbose : boolean, optional, default = False
        if True then print more information
    
    energy_unit : {'kwh', 'joules'}, optional

    max_sample_period : float or int, optional 
        The maximum allowed sample period in seconds.  If we find a
        sample above `on_power_threshold` at time `t` and there are
        more than `max_sample_period` seconds until the next sample
        then we assume that the appliance has only been on for
        `max_sample_period` seconds after time `t`.  This is used where,
        for example, we have a wireless meter which is supposed to
        report every `K` seconds and we assume that if we don't hear
        from it for more than `max_sample_period=K*3` seconds then the
        sensor (and appliance) have been turned off from the wall.

    Returns
    -------
    usage : pd.DataFrame
        One row per period (as defined by `freq`).  
        Index is PeriodIndex (UTC).
        Columns:
            hours_on
            <`energy_unit`>
    """

    assert(0 <= max_dropout_rate <= 1)

    date_range, boundaries = _indicies_of_periods(series.index,freq)
    period_range = date_range.to_period(freq=freq)
    name = str(series.name)
    hours_on_series = pd.Series(index=period_range, dtype=np.float, 
                                name=name+' hours on')
    energy_series = pd.Series(index=period_range, dtype=np.float, 
                              name=name+' '+energy_unit)

    MAX_SAMPLES_PER_PERIOD = _secs_per_period_alias(freq) / sample_period(series)
    MIN_SAMPLES_PER_PERIOD = (MAX_SAMPLES_PER_PERIOD *
                              (1-max_dropout_rate))

    for period_i, period in enumerate(period_range):
        try:
            period_start_i, period_end_i = boundaries[period_i]
        except IndexError:
            if verbose:
                print("No data available for   ",
                      period.strftime('%Y-%m-%d'))
            continue

        data_for_period = series[period_start_i:period_end_i]
        if data_for_period.size < MIN_SAMPLES_PER_PERIOD:
            if verbose:
                dropout_rate = (1 - (data_for_period.size / 
                                     MAX_SAMPLES_PER_PERIOD))
                print("Insufficient samples for ",
                      period.strftime('%Y-%m-%d'),
                      "; n samples = ", data_for_period.size,
                      "; dropout_rate = {:.2%}".format(dropout_rate), sep='')
                print("                 start =", data_for_period.index[0])
                print("                   end =", data_for_period.index[-1])
            continue

        hours_on_series[period] = hours_on(data_for_period, 
                                           on_power_threshold=on_power_threshold,
                                           max_sample_period=max_sample_period)
        energy_series[period] = energy(data_for_period, 
                                       max_sample_period=max_sample_period, 
                                       unit=energy_unit)

    return pd.DataFrame({'hours_on': hours_on_series,
                         energy_unit: energy_series})


#------------------------ HELPER FUNCTIONS -------------------------

def _secs_per_period_alias(alias):
    """The duration of a period alias in seconds."""
    dr = pd.date_range('00:00', periods=2, freq=alias)
    return (dr[-1] - dr[0]).total_seconds()


def _indicies_of_periods(datetime_index, freq):
    """Find which elements of `datetime_index` fall into each period
    of a regular date_range with frequency `freq`.

    Parameters
    ----------
    datetime_index : pd.tseries.index.DatetimeIndex

    freq : str
        one of the following:
        'A' for yearly
        'M' for monthly
        'D' for daily
        'H' for hourly
        'T' for minutely

    Returns
    -------
    date_range, boundaries:

        date_range : pd.tseries.index.DateTimeIndex

        boundaries : list
            Each list element represents a single period and is a tuple of ints:
            (<start index into `datetime_index` for period>, <end index>)
    """
    date_range = pd.date_range(datetime_index[0], datetime_index[-1], freq=freq)

    # Declare and initialise some constants and variables used
    # during the loop...

    # Find the minimum sample period.
    # For the sake of speed, only use the first 100 samples.
    FWD_DIFF = np.diff(datetime_index.values[:100]).astype(np.float)
    MIN_SAMPLE_PERIOD = FWD_DIFF.min() / 1E9
    MAX_SAMPLES_PER_PERIOD = _secs_per_period_alias(freq) / MIN_SAMPLE_PERIOD
    MAX_SAMPLES_PER_2_PERIODS = MAX_SAMPLES_PER_PERIOD * 2
    n_rows_processed = 0
    boundaries = []
    for end_time in date_range[1:]:
        # The simplest way to get data for just a single period is to use
        # data_for_day = datetime_index[period.strftime('%Y-%m-%d')]
        # but this takes about 300ms per call on my machine.
        # So we take advantage of several features of the data to achieve
        # a 300x speedup:
        # 1. We use the fact that the data is sorted in order, hence 
        #    we can chomp through it in order.
        # 2. MAX_SAMPLES_PER_PERIOD sets an upper bound on the number of
        #    datapoints per period.  The code is conservative and uses 
        #    MAX_SAMPLES_PER_2_PERIODS. We only search through a small subset
        #    of the available data.
        end_index = n_rows_processed+MAX_SAMPLES_PER_2_PERIODS
        rows_to_process = datetime_index[n_rows_processed:end_index]

        indicies_for_period = np.where(rows_to_process < end_time)[0]
        if indicies_for_period.size > 0:
            first_i_for_period = indicies_for_period[0] + n_rows_processed
            last_i_for_period = indicies_for_period[-1] + n_rows_processed + 1
            boundaries.append((first_i_for_period, last_i_for_period))
            n_rows_processed += last_i_for_period - first_i_for_period

    date_range = date_range[:-1] # so as not to exceed size of boundaries
    return date_range, boundaries
