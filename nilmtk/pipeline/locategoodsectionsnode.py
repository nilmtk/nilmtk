from __future__ import print_function, division
from node import Node, UnsatisfiedRequirementsError
import numpy as np
from nilmtk import TimeFrame
from nilmtk.utils import timedelta64_to_secs
# from locategapresults import LocateGapResults #TODO

def reframe_index(index, window_start=None, window_end=None):
    """
    Parameters
    ----------
    index : pd.DatetimeIndex

    window_start, window_end : pd.Timestamp
        The start and end of the window of interest. If this window
        is larger than the duration of `data` then a single timestamp will be
        inserted at `window_start` or `window_end` as necessary. If this window
        is shorter than the duration of `data` then `data` will be cropped.

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
        index = index.tz_localize('UTC').tz_convert(tz)

    return index

class LocateGoodSectionsNode(Node):

    requirements = {'device': {'max_sample_period': 'ANY VALUE'}}
    postconditions =  {'preprocessing': {'good_sections_located':True}}

    def __init__(self, name='locate_good_sections'):
        super(LocateGoodSectionsNode, self).__init__(name)

    def process(self, df, metadata=None):
        assert metadata is not None
        assert hasattr(df, 'timeframe')

        max_sample_period = metadata['device']['max_sample_period']    
        index = df.dropna().index
        index = reframe_index(index, df.timeframe.start, df.timeframe.end)
        timedeltas_sec = timedelta64_to_secs(np.diff(index.values))
        overlong_timedeltas = timedeltas_sec > max_sample_period
        good_sect_starts = index[1:][overlong_timedeltas] 
        good_sect_starts = good_sect_starts.insert(0, index[0])
        good_sect_ends = index[:-1][overlong_timedeltas]
        good_sect_ends = good_sect_ends.insert(len(good_sect_ends), index[-1])
        
        assert len(good_sect_starts) == len(good_sect_ends)

        mask = []
        for start, end in zip(good_sect_starts, good_sect_ends):
            try:
                mask.append(TimeFrame(start, end))
            except ValueError:
                pass # silently ignore good sections of zero length

        results = getattr(df, 'results', {})
#        locate_gaps_results = LocateGapsResults()
        # TODO: populate
        results[self.name] = 'TODO' # locate_gaps_results
        df.results = results
        return df
