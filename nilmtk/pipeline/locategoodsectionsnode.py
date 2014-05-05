from __future__ import print_function, division
from node import Node, UnsatisfiedRequirementsError
import numpy as np
from nilmtk import TimeFrame
from nilmtk.utils import timedelta64_to_secs
from locategoodsectionsresults import LocateGoodSectionsResults

# reframe_index is perhaps not needed any more.  Might be 
# safe to remove it.  Leaving it here for now as it might
# come in handy, and is well tested at the moment.
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
        self.previous_chunk_ended_with_open_ended_good_section = False

    def process(self, df, metadata=None):
        assert metadata is not None
        assert hasattr(df, 'timeframe')

        max_sample_period = metadata['device']['max_sample_period']    
        index = df.dropna().index
        timedeltas_sec = timedelta64_to_secs(np.diff(index.values))
        ok_timedeltas = timedeltas_sec <= max_sample_period
        ok_timedeltas = np.concatenate(
            [[self.previous_chunk_ended_with_open_ended_good_section], 
             ok_timedeltas])
        changes = np.diff(ok_timedeltas.astype(np.int))
        good_sect_starts = index[:-1][changes==1]
        good_sect_ends = index[:-1][changes==-1]

        # Fix up last good_sect_end using look ahead
        if not df.look_ahead.empty:
            look_ahead_gap = (df.look_ahead.dropna().index[0] - index[-1]).total_seconds()
        if ok_timedeltas[-1]: # current chunk ends with a good section
            if df.look_ahead.empty or look_ahead_gap > max_sample_period:
                # current chunk ends with a good section which needs to 
                # be closed because next chunk either does not exist
                # or starts with a sample which is more than max_sample_period
                # away from df.index[-1]
                good_sect_ends = good_sect_ends.insert(
                    len(good_sect_ends), index[-1])
        elif not df.look_ahead.empty and look_ahead_gap <= max_sample_period:
            # Current chunk appears to end with a bad section
            # but last sample is the start of a good section
            good_sect_starts = good_sect_starts.insert(
                len(good_sect_starts), index[-1])

        if len(good_sect_ends) == 0:
            ends_with_open_ended_good_section = (len(good_sect_starts) > 0 or 
                self.previous_chunk_ended_with_open_ended_good_section)
        elif len(good_sect_starts) > 0:
            ends_with_open_ended_good_section = (
                good_sect_ends[-1] < good_sect_starts[-1])
        else:
            ends_with_open_ended_good_section = False

        good_sect_ends = list(good_sect_ends)
        good_sect_starts = list(good_sect_starts)
        if self.previous_chunk_ended_with_open_ended_good_section:
            good_sect_starts = [None] + good_sect_starts
        if ends_with_open_ended_good_section:
            good_sect_ends = good_sect_ends + [None]
            
        assert len(good_sect_starts) == len(good_sect_ends)

        sections = [TimeFrame(start, end) 
                    for start, end in zip(good_sect_starts, good_sect_ends)]

        self.previous_chunk_ended_with_open_ended_good_section = (
            ends_with_open_ended_good_section)
        
        good_section_results = LocateGoodSectionsResults()
        good_section_results.append(df.timeframe, {'sections': [sections]})
        results = getattr(df, 'results', {})
        results[self.name] = good_section_results
        df.results = results
        return df
