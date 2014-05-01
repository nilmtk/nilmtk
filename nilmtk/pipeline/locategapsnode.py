from __future__ import print_function, division
from node import Node
import numpy as np
from nilmtk import TimeFrame
from node import UnsatisfiedRequirementsError
from nilmtk.utils import timedelta64_to_secs
# from locategapresults import LocateGapResults #TODO

def reframe_index(index, window_start=None, window_end=None):
    """
    Parameters
    ----------
    index : pd.DatetimeIndex

    window_start, window_end : pd.Timestamp
        The start and end of the window of interest. If this window
        is larger than the duration of `data` then a single zero will be
        inserted at `window_start` or `window_end` as necessary. If this window
        is shorter than the duration of `data` data will be cropped.

    Returns
    -------
    index : pd.DatetimeIndex
    """

    tz = index.tzinfo
    reset_tz = False

    # Handle window...
    if window_start is not None:
        if window_start >= index[0]:
            index = index[index >= window_start]
        else:
            index = index.insert(0, window_start)
            reset_tz = True

    if window_end is not None:
        if window_end <= index[-1]:
            index = index[index <= window_end]
        else:
            index = index.insert(len(index), window_end)
            reset_tz = True

    if reset_tz:
        # index.insert breaks timezone.
        # TODO: check if this is still true in Pandas 0.13.1
        index = index.tz_localize('UTC').tz_convert(tz)

    return index

class LocateGapsNode(Node):

    requirements = {'device': {
        'max_sample_period': 'ANY VALUE'}}
    postconditions =  {'preprocessing': {'gaps_located':True}}

    def __init__(self, name='locate_gaps'):
        super(LocateGapsNode, self).__init__(name)

    def process(self, df, metadata=None):
        assert metadata is not None
        assert hasattr(df, 'timeframe')

        max_sample_period = metadata['device']['max_sample_period']    
        index = df.dropna().index
        index = reframe_index(index, df.timeframe.start, df.timeframe.end)
        timedeltas_sec = timedelta64_to_secs(np.diff(index.values))
        overlong_timedeltas = timedeltas_sec > max_sample_period
        gap_starts = index[:-1][overlong_timedeltas]
        gap_ends = index[1:][overlong_timedeltas] 

        results = getattr(df, 'results', {})
#        locate_gaps_results = LocateGapsResults()
        # TODO: populate
        results[self.name] = 'TODO' # locate_gaps_results
        df.results = results
        return df
