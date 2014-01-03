"""Statistics applicable to a single appliance / circuit / mains split.

In general, these functions each take a DataFrame representing a single
appliance / circuit / mains split.
"""
from __future__ import print_function, division
import scipy.stats as stats
import numpy as np
import pandas as pd
from matplotlib.dates import SEC_PER_HOUR
import copy

DEFAULT_MAX_DROPOUT_RATE = 0.4  # [0,1]
DEFAULT_ON_POWER_THRESHOLD = 5  # watts


def get_sample_period(data):
    """Estimate the sample period in seconds.

    Find the sample period by finding the stats.mode of the 
    forward difference.  Only use the first 100 samples (for speed).

    Parameters
    ----------
    data : pandas.DataFrame or Series or DatetimeIndex

    Returns
    -------
    period : float
        Sample period in seconds.
    """
    index = _get_index(data)
    fwd_diff = np.diff(index.values[:100]).astype(np.float)
    mode_fwd_diff = stats.mode(fwd_diff)[0][0]
    period = mode_fwd_diff / 1E9
    return period


def get_dropout_rate(data, sample_period=None):
    """The proportion of samples that have been lost.

    Parameters
    ----------
    data : pandas.DataFrame or Series or DatetimeIndex

    sample_period : int or float, optional
        Sample period in seconds.  If not provided then will
        calculate it.

    Returns
    -------
    rate : float [0,1]
        The proportion of samples that have been lost; where 
        1 means that all samples have been lost and 
        0 means that no samples have been lost.
    """
    if sample_period is None:
        sample_period = get_sample_period(data)

    index = _get_index(data)
    duration = index[-1] - index[0]
    n_expected_samples = duration.total_seconds() / sample_period
    return 1 - (index.size / n_expected_samples)


def hours_on(series, on_power_threshold=DEFAULT_ON_POWER_THRESHOLD,
             max_sample_period=None):
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
    # now calculate timedelta ('td') above threshold...
    td_above_thresh = (series.index[i_above_threshold + 1].values -
                       series.index[i_above_threshold].values)
    if max_sample_period is not None:
        td_above_thresh[td_above_thresh >
                        max_sample_period] = max_sample_period

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
    timedelta = np.diff(series.index.values)
    if max_sample_period is not None:
        timedelta = np.where(timedelta > max_sample_period,
                             max_sample_period, timedelta)
    timedelta_secs = timedelta / np.timedelta64(1, 's')  # convert to seconds
    joules = (timedelta_secs * series.values[:-1]).sum()

    if unit == 'kwh':
        JOULES_PER_KWH = 3600000
        _energy = joules / JOULES_PER_KWH
    elif unit == 'joules':
        _energy = joules
    else:
        raise ValueError('unrecognised value for `unit`.')

    return _energy


def usage_per_period(series, freq,
                     on_power_threshold=DEFAULT_ON_POWER_THRESHOLD,
                     max_dropout_rate=DEFAULT_MAX_DROPOUT_RATE,
                     verbose=False,
                     energy_unit='kwh', max_sample_period=None):
    """Calculate the usage (hours on and kwh) per time period.

    Parameters
    ----------
    series : pd.Series

    freq : str
        see _indicies_of_periods() for acceptable values.

    on_power_threshold : float or int, optional, default = 5
        Threshold which defines the distinction between "on" and "off".  Watts.

    max_dropout_rate : float (0,1), optional, default = 0.4
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

    Examples
    --------
    Say we have loaded fridge data from house_1 in REDD into `fridge` and we
    want to see how it was used each day:

    >>> usage_per_period(fridge, 'D')

                 hours_on       kwh
    2011-04-18        NaN       NaN
    2011-04-19  23.999444  1.104083
    2011-04-20  23.998889  1.293223
    2011-04-21  23.998889  1.138540
    ...
    2011-05-22  23.832500  2.042271
    2011-05-23  23.931111  1.394619
    2011-05-24        NaN       NaN 

    Hmmm... why does the fridge appear to be on for 24 hours per day?
    Inspecting the fridge.plot(), we find that the fridge rarely ever
    gets below this function's default on_power_threshold of 5 Watts,
    so let's specify a larger threshold:

    >>> usage_per_period(fridge, 'D', on_power_threshold=100)

                hours_on       kwh
    2011-04-18       NaN       NaN
    2011-04-19  5.036111  1.104083
    2011-04-20  5.756667  1.293223
    2011-04-21  4.931667  1.138540
    2011-04-22  4.926111  1.076958
    2011-04-23  6.099167  1.357812
    2011-04-24  6.373056  1.361579
    2011-04-25  6.496667  1.441966
    2011-04-26  6.381389  1.404637
    2011-04-27  5.558611  1.196464
    2011-04-28  6.668611  1.478141
    2011-04-29  6.493056  1.446713
    2011-04-30  5.885278  1.263918
    2011-05-01  5.983611  1.351419
    2011-05-02  5.398333  1.167111
    2011-05-03       NaN       NaN
    2011-05-04       NaN       NaN
    2011-05-05       NaN       NaN
    2011-05-06       NaN       NaN
    2011-05-07  5.112222  1.120848
    2011-05-08  6.349722  1.413897
    2011-05-09  7.270833  1.573199
    2011-05-10  5.997778  1.249120
    2011-05-11  5.685556  1.264841
    2011-05-12  7.153333  1.478244
    2011-05-13  5.949444  1.306350
    2011-05-14  6.446944  1.415302
    2011-05-15  5.958333  1.275853
    2011-05-16  6.801944  1.501816
    2011-05-17  5.836389  1.342787
    2011-05-18  5.254444  1.164683
    2011-05-19  6.234444  1.397851
    2011-05-20  5.814444  1.265143
    2011-05-21  6.738333  1.498687
    2011-05-22  9.308056  2.042271
    2011-05-23  6.127778  1.394619
    2011-05-24       NaN       NaN

    That looks sensible!  Now, let's find out why the cause of the NaNs by 
    setting verbose=True:
    
    >>> usage_per_period(fridge, 'D', on_power_threshold=100, verbose=True)

    Insufficient samples for 2011-04-18; n samples = 13652; dropout_rate = 52.60%
                     start = 2011-04-18 09:22:13-04:00
                       end = 2011-04-18 23:59:57-04:00
    Insufficient samples for 2011-05-03; n samples = 16502; dropout_rate = 42.70%
                     start = 2011-05-03 00:00:03-04:00
                       end = 2011-05-03 17:33:17-04:00
    No data available for    2011-05-04
    No data available for    2011-05-05
    Insufficient samples for 2011-05-06; n samples = 12465; dropout_rate = 56.72%
                     start = 2011-05-06 10:51:50-04:00
                       end = 2011-05-06 23:59:58-04:00
    Insufficient samples for 2011-05-24; n samples = 13518; dropout_rate = 53.06%
                     start = 2011-05-24 00:00:02-04:00
                       end = 2011-05-24 15:56:34-04:00
    Out[209]: 
                hours_on       kwh
    2011-04-18       NaN       NaN
    2011-04-19  5.036111  1.104083
    2011-04-20  5.756667  1.293223
    ...

    Ah, OK, there are insufficient samples for the periods with NaNs.  We could
    set max_dropout_rate to a number closer to 1, but that would give us data
    for days where there isn't much data for that day.

    """

    assert(0 <= max_dropout_rate <= 1)

    period_range, boundaries = _indicies_of_periods(series.index, freq)
    name = str(series.name)
    hours_on_series = pd.Series(index=period_range, dtype=np.float,
                                name=name + ' hours on')
    energy_series = pd.Series(index=period_range, dtype=np.float,
                              name=name + ' ' + energy_unit)

    MAX_SAMPLES_PER_PERIOD = _secs_per_period_alias(
        freq) / get_sample_period(series)
    MIN_SAMPLES_PER_PERIOD = (MAX_SAMPLES_PER_PERIOD *
                              (1 - max_dropout_rate))

    for period in period_range:
        try:
            period_start_i, period_end_i = boundaries[period]
        except KeyError:
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


def activity_distribution(
    series, on_power_threshold=DEFAULT_ON_POWER_THRESHOLD,
        bin_size='T', timespan='D'):
    """Returns a distribution describing when this appliance was turned
    on over repeating timespans.  For example, if you want to see
    which times of day this appliance was used, on average, then use 
    bin_size='T' (minutely) or bin_size='H' (hourly) and
    timespan='D' (daily).

    Parameters
    ----------
    series : pandas.Series

    on_power_threshold : float, optional, default=5
        Threshold which defines the difference between 'on' and 'off'. Watts.

    bin_size, timespan : str
        offset alias (e.g. 'T' or 'D')
        For valid offset aliases, see:
        http://pandas.pydata.org/pandas-docs/dev/timeseries.html#offset-aliases

    Returns
    -------
    pandas.Series
        One row for each bin in a timespan.
        The values count the number of times this appliance has been on at
        that particular time of the timespan.
        Times are handled in local time.
        The index uses specific dates. For example, if `timespan='D'` then
        the index might be from '2012/1/1 00:00' to '2012/1/1 59:59'. In this
        example, ignore the '2012/1/1'.
    """

    # Create a pd.Series with PeriodIndex
    binned_data = series.resample(bin_size, how='max').to_period()
    binned_data = binned_data > on_power_threshold

    timespans, boundaries = _indicies_of_periods(
        binned_data.index.to_timestamp(),
        freq=timespan)

    first_timespan = timespans[0]
    bins = pd.period_range(first_timespan.start_time,
                           first_timespan.end_time,
                           freq=bin_size)
    distribution = pd.Series(0, index=bins)

    bins_per_timespan = int(round(_secs_per_period_alias(timespan) /
                                  _secs_per_period_alias(bin_size)))

    for span in timespans:
        try:
            start_index, end_index = boundaries[span]
        except KeyError:
            print("No data for", span)
            continue
        else:
            data_for_timespan = binned_data[start_index:end_index]

        bins_since_first_timespan = (first_timespan - span) * bins_per_timespan
        data_shifted = data_for_timespan.shift(bins_since_first_timespan,
                                               bin_size)
        distribution = distribution.add(data_shifted, fill_value=0)

    return distribution


def on(series, max_sample_period=None,
       on_power_threshold=DEFAULT_ON_POWER_THRESHOLD):
    """Returns pd.Series with Boolean values indicating whether the
    appliance is on (True) or off (False).  Adds an 'off' entry if data
    is lost for more than self.max_sample_period.

    Parameters
    ----------
    series : pandas.Series

    on_power_threshold : float, optional, default=5
        Threshold which defines the difference between 'on' and 'off'. Watts.

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
    pandas.Series
        index is the same as for input `series`
        values are booleans
    """
    when_on = series >= on_power_threshold

    if max_sample_period is not None:
        # add an 'off' entry whenever data is lost for > self.max_sample_period
        time_delta = np.diff(series.index.values) / np.timedelta64(1, 's')
        dropout_dates = series.index[:-1][time_delta > max_sample_period]
        insert_offs = pd.Series(False,
                                index=dropout_dates +
                                pd.DateOffset(seconds=max_sample_period))
        when_on = when_on.append(insert_offs)
        when_on = when_on.sort_index()

    return when_on


def on_off_events(on_series, ignore_n_off_samples=None):
    """Detects on/off switch events.

    Parameters
    ----------
    on_series : pd.Series
        Series of booleans indicating if the appliance is on or off.
        Produced, for example, by `on()`.

    ignore_n_off_samples : int, optional
        Ignore this number of off samples.  For example, if the input
        is [0, 100, 0, 100, 100, 0, 0] then the single zero with 100
        either side could be ignored if ignore_n_off_samples = 1,
        hence only one on-event and one off-event would be reported.

    Returns
    -------
    events : pd.Series
        Index is the same as the input `on_series`.
        Values are np.int8:
         1 == turn-on event
        -1 == turn-off event

    Example
    -------
    >>> series = pd.Series([0, 0, 100, 100, 100, 0])
    >>> on_off_events(series)
    2:  1
    5: -1

    """
    if ignore_n_off_samples is not None:
        on_smoothed = pd.rolling_max(
            on_series, window=ignore_n_off_samples + 1)
        on_smoothed.iloc[:ignore_n_off_samples] = on_series.iloc[
            :ignore_n_off_samples].values
        on_series = on_smoothed.dropna()

    on_series = on_series.astype(np.int8)
    events = on_series[1:] - on_series.shift(1)[1:]

    if ignore_n_off_samples is not None:
        i_off_events = np.where(events == -1)[0]  # indicies of off-events
        for i in range(ignore_n_off_samples):
            events.iloc[i_off_events - i] = 0
        events.iloc[i_off_events - ignore_n_off_samples] = -1

    events = events[events != 0]
    events.name = 'on/off events for ' + str(events.name)
    return events


def durations(on_series, on_or_off, ignore_n_off_samples=None,
              sample_period=None):
    """The length of every on or off duration (in seconds).

    Parameters
    ----------
    on_series : pd.Series
        Series of booleans indicating if the appliance is on or off.
        Produced, for example, by `on()`.

    on_or_off : {'on', 'off'}

    ignore_n_off_samples : int, optional
        Ignore this number of off samples.  For example, if the input
        is [0, 100, 0, 100, 100, 0, 0] then the single zero with 100
        either side could be ignored if ignore_n_off_samples = 1,
        hence only one on-event and one off-event would be reported.

    sample_period : int, optional
        Required if `ignore_n_off_samples` is not None. Sample period
        in seconds.  See `sample_period()`.

    Returns
    -------
    pd.Series
        Index is the datetime at which the event starts.
        Values are the length of every on or off duration (in seconds).

    See also
    --------
    on_off_events()
    on()
    sample_period()

    """
    events = on_off_events(on_series, ignore_n_off_samples)
    delta_time_array = np.diff(events.index.values).astype(int) / 1E9
    delta_time = pd.Series(delta_time_array, index=events.index[:-1])
    diff_for_mode = 1 if on_or_off == 'on' else -1
    events_for_mode = events == diff_for_mode
    durations = delta_time[events_for_mode]
    if ignore_n_off_samples is not None:
        durations = durations[durations > sample_period * ignore_n_off_samples]

    durations.name = 'seconds ' + on_or_off + ' for ' + str(on_series.name)
    return durations


def periods_with_sufficient_samples(datetime_index, freq,
                                    max_dropout_rate=DEFAULT_MAX_DROPOUT_RATE,
                                    use_local_time=True):
    """Find periods where the dropout rate is less than max_dropout_rate.

    Returns
    -------
    set of Periods
    """
    good_periods = set()
    periods, boundaries = _indicies_of_periods(datetime_index, freq=freq,
                                               use_local_time=use_local_time)
    sample_period = get_sample_period(datetime_index)
    for period in periods:
        try:
            start_i, end_i = boundaries[period]
        except KeyError:
            continue
        index_for_period = datetime_index[start_i:end_i]
        dropout_rate = get_dropout_rate(index_for_period, sample_period)
        if dropout_rate < max_dropout_rate:
            good_periods.add(period)

    return good_periods


def start_end_datetime(data):
    """Returns the first and the last time for which data is recorded

    Parameters
    ----------
    data : pandas.DataFrame or Series or DatetimeIndex

    Returns
    -------
    start_end_datetimes : List
        Contains two elements- start and end timestamps 

    """
    index = _get_index(data)
    return [index[0], index[-1]]


#------------------------ HELPER FUNCTIONS -------------------------
def _secs_per_period_alias(alias):
    """The duration of a period alias in seconds."""
    dr = pd.date_range('00:00', periods=2, freq=alias)
    return (dr[-1] - dr[0]).total_seconds()


def _indicies_of_periods(datetime_index, freq, use_local_time=True):
    """Find which elements of `datetime_index` fall into each period
    of a regular periods with frequency `freq`.  Uses some tricks to do
    this more efficiently that appears possible with native Pandas tools.

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

    use_local_time : boolean, optional, default=True
        If True then start and end each time period at appropriate local times.
        e.g. if `freq='D'` and:
            `use_local_time=True` then divide at midnight *local time* or if
            `use_local_time=False` then divide at midnight UTC

    Returns
    -------
    periods : pd.tseries.period.PeriodIndex

    boundaries : dict
        Each key is a pd.tseries.period.Period
        Each value is a tuple of ints:
        (<start index into `datetime_index` for period>, <end index>)
        Periods for which no data exists will not have a key.

    Examples
    --------
    Say you have a pd.Series with data covering a month:

    >>> series.index
    <class 'pandas.tseries.index.DatetimeIndex'>
    [2011-04-18 09:22:13, ..., 2011-05-24 15:56:34]
    Length: 745878, Freq: None, Timezone: US/Eastern

    You want to divide it up into day-sized chunks, starting and ending each
    chunk at midnight local time:

    >>> periods, boundaries = _indicies_of_periods(series.index, freq='D')

    >>> periods
    <class 'pandas.tseries.period.PeriodIndex'>
    freq: D
    [2011-04-18, ..., 2011-05-24]
    length: 37

    >>> boundaries
    {Period('2011-04-18', 'D'): (0, 13652),
     Period('2011-04-19', 'D'): (13652, 34926),
     Period('2011-04-20', 'D'): (34926, 57310),
     ...
     Period('2011-05-23', 'D'): (710750, 732360),
     Period('2011-05-24', 'D'): (732360, 745878)}

    Now, say that we want chomp though our data a day at a time:

    >>> for period in periods:
    >>>     start_i, end_i = boundaries[period]
    >>>     data_for_day = series.iloc[start_i:end_i]
    >>>     # do something with data_for_day

    """

    if use_local_time:
        datetime_index = _tz_to_naive(datetime_index)

    periods = pd.period_range(datetime_index[0], datetime_index[-1], freq=freq)

    # Declare and initialise some constants and variables used
    # during the loop...

    # Find the minimum sample period.
    MIN_SAMPLE_PERIOD = int(get_sample_period(datetime_index))
    MAX_SAMPLES_PER_PERIOD = int(
        _secs_per_period_alias(freq) / MIN_SAMPLE_PERIOD)
    MAX_SAMPLES_PER_2_PERIODS = MAX_SAMPLES_PER_PERIOD * 2
    n_rows_processed = 0
    boundaries = {}
    for period in periods:
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

        end_index = n_rows_processed + MAX_SAMPLES_PER_2_PERIODS
        rows_to_process = datetime_index[n_rows_processed:end_index]
        indicies_for_period = np.where(rows_to_process < period.end_time)[0]
        if indicies_for_period.size > 0:
            first_i_for_period = indicies_for_period[0] + n_rows_processed
            last_i_for_period = indicies_for_period[-1] + n_rows_processed + 1
            boundaries[period] = (first_i_for_period, last_i_for_period)
            n_rows_processed += last_i_for_period - first_i_for_period

    return periods, boundaries


def _tz_to_naive(datetime_index):
    """Converts a tz-aware DatetimeIndex into a tz-naive DatetimeIndex,
    effectively baking the timezone into the internal representation.

    Parameters
    ----------
    datetime_index : pandas.DatetimeIndex, tz-aware

    Returns
    -------
    pandas.DatetimeIndex, tz-naive
    """
    # See http://stackoverflow.com/q/16628819/732596

    if datetime_index.tzinfo is None:
        return datetime_index

    # Calculate timezone offset relative to UTC
    timestamp = datetime_index[0]
    tz_offset = (timestamp.replace(tzinfo=None) -
                 timestamp.tz_convert('UTC').replace(tzinfo=None))
    tz_offset_td64 = np.timedelta64(tz_offset)

    # Now convert to naive DatetimeIndex
    return pd.DatetimeIndex(datetime_index.values + tz_offset_td64)


def _get_index(data):
    """
    Parameters
    ----------
    data : pandas.DataFrame or Series or DatetimeIndex
    
    Returns
    -------
    index : the index for the DataFrame or Series
    """

    if isinstance(data, (pd.DataFrame, pd.Series)):
        index = data.index
    elif isinstance(data, pd.DatetimeIndex):
        index = data
    else:
        raise TypeError('wrong type for `data`.')
    return index
