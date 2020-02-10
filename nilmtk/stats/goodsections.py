import numpy as np
from numpy import diff, concatenate
import gc
from .goodsectionsresults import GoodSectionsResults
from ..timeframe import TimeFrame
from ..utils import timedelta64_to_secs
from ..node import Node
from ..timeframe import list_of_timeframes_from_list_of_dicts, timeframe_from_dict


class GoodSections(Node):
    """Locate sections of data where the sample period is <= max_sample_period.

    Attributes
    ----------
    previous_chunk_ended_with_open_ended_good_section : bool
    """

    requirements = {'device': {'max_sample_period': 'ANY VALUE'}}
    postconditions =  {'statistics': {'good_sections': []}}
    results_class = GoodSectionsResults
        
    def reset(self):
        self.previous_chunk_ended_with_open_ended_good_section = False

    def process(self):
        metadata = self.upstream.get_metadata()
        self.check_requirements()
        self.results = GoodSectionsResults(
            metadata['device']['max_sample_period'])
        for chunk in self.upstream.process():
            self._process_chunk(chunk, metadata)
            yield chunk

    def _process_chunk(self, df, metadata):
        """
        Parameters
        ----------
        df : pd.DataFrame
            with attributes:
            - look_ahead : pd.DataFrame
            - timeframe : nilmtk.TimeFrame
        metadata : dict
            with ['device']['max_sample_period'] attribute

        Returns
        -------
        None

        Notes
        -----
        Updates `self.results`
            Each good section in `df` is marked with a TimeFrame.
            If this df ends with an open-ended good section (assessed by
            examining df.look_ahead) then the last TimeFrame will have
            `end=None`. If this df starts with an open-ended good section
            then the first TimeFrame will have `start=None`.
        """
        # Retrieve relevant metadata
        max_sample_period = metadata['device']['max_sample_period']
        look_ahead = getattr(df, 'look_ahead', None)
        timeframe = df.timeframe

        # Process dataframe
        good_sections = get_good_sections(
            df, max_sample_period, look_ahead,
            self.previous_chunk_ended_with_open_ended_good_section)

        # Set self.previous_chunk_ended_with_open_ended_good_section
        if good_sections:
            self.previous_chunk_ended_with_open_ended_good_section = (
                good_sections[-1].end is None)

            # Update self.results
            self.results.append(timeframe, {'sections': [good_sections]})


def get_good_sections(df, max_sample_period, look_ahead=None,
                      previous_chunk_ended_with_open_ended_good_section=False):
    """
    Parameters
    ----------
    df : pd.DataFrame
    look_ahead : pd.DataFrame
    max_sample_period : number

    Returns
    -------
    sections : list of TimeFrame objects
        Each good section in `df` is marked with a TimeFrame.
        If this df ends with an open-ended good section (assessed by
        examining `look_ahead`) then the last TimeFrame will have
        `end=None`.  If this df starts with an open-ended good section
        then the first TimeFrame will have `start=None`.
    """
    index = df.dropna().sort_index().index
    del df

    if len(index) < 2:
        return []

    timedeltas_sec = timedelta64_to_secs(diff(index.values))
    timedeltas_check = timedeltas_sec <= max_sample_period

    # Memory management
    del timedeltas_sec
    gc.collect()

    timedeltas_check = concatenate(
        [[previous_chunk_ended_with_open_ended_good_section],
         timedeltas_check])
    transitions = diff(timedeltas_check.astype(np.int))

    # Memory management
    last_timedeltas_check = timedeltas_check[-1]
    del timedeltas_check
    gc.collect()

    good_sect_starts = list(index[:-1][transitions ==  1])
    good_sect_ends   = list(index[:-1][transitions == -1])

    # Memory management
    last_index = index[-1]
    del index
    gc.collect()

    # Use look_ahead to see if we need to append a 
    # good sect start or good sect end.
    look_ahead_valid = look_ahead is not None and not look_ahead.empty
    if look_ahead_valid:
        look_ahead_timedelta = look_ahead.dropna().index[0] - last_index
        look_ahead_gap = look_ahead_timedelta.total_seconds()
    if last_timedeltas_check: # current chunk ends with a good section
        if not look_ahead_valid or look_ahead_gap > max_sample_period:
            # current chunk ends with a good section which needs to 
            # be closed because next chunk either does not exist
            # or starts with a sample which is more than max_sample_period
            # away from df.index[-1]
            good_sect_ends += [last_index]
    elif look_ahead_valid and look_ahead_gap <= max_sample_period:
        # Current chunk appears to end with a bad section
        # but last sample is the start of a good section
        good_sect_starts += [last_index]

    # Work out if this chunk ends with an open ended good section
    if len(good_sect_ends) == 0:
        ends_with_open_ended_good_section = (
            len(good_sect_starts) > 0 or 
            previous_chunk_ended_with_open_ended_good_section)
    elif len(good_sect_starts) > 0:
        # We have good_sect_ends and good_sect_starts
        ends_with_open_ended_good_section = (
            good_sect_ends[-1] < good_sect_starts[-1])
    else:
        # We have good_sect_ends but no good_sect_starts
        ends_with_open_ended_good_section = False

    # If this chunk starts or ends with an open-ended
    # good section then the relevant TimeFrame needs to have
    # a None as the start or end.
    if previous_chunk_ended_with_open_ended_good_section:
        good_sect_starts = [None] + good_sect_starts
    if ends_with_open_ended_good_section:
        good_sect_ends += [None]

    assert len(good_sect_starts) == len(good_sect_ends)

    sections = [TimeFrame(start, end)
                for start, end in zip(good_sect_starts, good_sect_ends)
                if not (start == end and start is not None)]

    # Memory management
    del good_sect_starts
    del good_sect_ends
    gc.collect()

    return sections
