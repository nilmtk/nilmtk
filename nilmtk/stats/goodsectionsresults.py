import pandas as pd
from datetime import timedelta
import matplotlib.pyplot as plt
from ..results import Results
from nilmtk.timeframe import TimeFrame, convert_none_to_nat, convert_nat_to_none
from nilmtk.utils import get_tz, tz_localize_naive
from nilmtk.timeframegroup import TimeFrameGroup

class GoodSectionsResults(Results):
    """
    Attributes
    ----------
    max_sample_period_td : timedelta
    _data : pd.DataFrame
        index is start date for the whole chunk
        `end` is end date for the whole chunk
        `sections` is a TimeFrameGroups object (a list of nilmtk.TimeFrame objects)
    """
    
    name = "good_sections"

    def __init__(self, max_sample_period):
        self.max_sample_period_td = timedelta(seconds=max_sample_period)
        super(GoodSectionsResults, self).__init__()

    def append(self, timeframe, new_results):
        """Append a single result.

        Parameters
        ----------
        timeframe : nilmtk.TimeFrame
        new_results : {'sections': list of TimeFrame objects}
        """
        new_results['sections'] = [TimeFrameGroup(new_results['sections'][0])]
        super(GoodSectionsResults, self).append(timeframe, new_results)

    def combined(self):
        """Merges together any good sections which span multiple segments,
        as long as those segments are adjacent 
        (previous.end - max_sample_period <= next.start <= previous.end).

        Returns
        -------
        sections : TimeFrameGroup (a subclass of Python's list class)
        """
        sections = TimeFrameGroup()
        end_date_of_prev_row = None
        for index, row in self._data.iterrows():
            row_sections = row['sections']

            # Check if first TimeFrame of row_sections needs to be merged with
            # last TimeFrame of previous section
            if (end_date_of_prev_row is not None):

                rows_are_adjacent = (
                    (end_date_of_prev_row - self.max_sample_period_td)
                    <= index <=
                    end_date_of_prev_row)

                if rows_are_adjacent and row_sections[0].start is None:
                    assert sections[-1].end is None
                    sections[-1].end = row_sections[0].end
                    row_sections.pop(0)
                else:
                    # row_sections[0] and sections[-1] were not in adjacent chunks
                    # so check if they are both open-ended and close them...
                    if sections and sections[-1].end is None:
                        try:
                            sections[-1].end = end_date_of_prev_row
                        except ValueError: # end_date_of_prev_row before sections[-1].start
                            pass
                    if row_sections and row_sections[0].start is None:
                        try:
                            row_sections[0].start = index
                        except ValueError:
                            pass
                
            end_date_of_prev_row = row['end']
            sections.extend(row_sections)

        if sections:
            sections[-1].include_end = True
            if sections[-1].end is None:
                sections[-1].end = end_date_of_prev_row

        return sections

    def unify(self, other):
        super(GoodSectionsResults, self).unify(other)
        for start, row in self._data.iterrows():
            other_sections = other._data['sections'].loc[start]
            intersection = row['sections'].intersection(other_sections)
            self._data['sections'].loc[start] = intersection

    def to_dict(self):
        good_sections = self.combined()
        good_sections_list_of_dicts = [timeframe.to_dict() 
                                       for timeframe in good_sections]
        return {'statistics': {'good_sections': good_sections_list_of_dicts}}

    def plot(self, **kwargs):
        timeframes = self.combined()
        return timeframes.plot(**kwargs)
        
    def import_from_cache(self, cached_stat, sections):
        # we (deliberately) use duplicate indices to cache GoodSectionResults
        grouped_by_index = cached_stat.groupby(level=0)
        tz = get_tz(cached_stat)
        for tf_start, df_grouped_by_index in grouped_by_index:
            grouped_by_end = df_grouped_by_index.groupby('end')
            for tf_end, sections_df in grouped_by_end:
                end = tz_localize_naive(tf_end, tz)
                timeframe = TimeFrame(tf_start, end)
                if timeframe in sections:
                    timeframes = []
                    for _, row in sections_df.iterrows():
                        section_start = tz_localize_naive(row['section_start'], tz)
                        section_end = tz_localize_naive(row['section_end'], tz)
                        timeframes.append(TimeFrame(section_start, section_end))
                    self.append(timeframe, {'sections': [timeframes]})

    def export_to_cache(self):
        """
        Returns
        -------
        DataFrame with three columns: 'end', 'section_end', 'section_start'.
            Instead of storing a list of TimeFrames on each row,
            we store one TimeFrame per row.  This is because pd.HDFStore cannot
            save a DataFrame where one column is a list if using 'table' format'.
            We also need to strip the timezone information from the data columns.
            When we import from cache, we assume the timezone for the data 
            columns is the same as the tz for the index.
        """
        index_for_cache = []
        data_for_cache = [] # list of dicts with keys 'end', 'section_end', 'section_start'
        for index, row in self._data.iterrows():
            for section in row['sections']:
                index_for_cache.append(index)
                data_for_cache.append(
                    {'end': row['end'], 
                     'section_start': convert_none_to_nat(section.start),
                     'section_end': convert_none_to_nat(section.end)})
        df = pd.DataFrame(data_for_cache, index=index_for_cache)
        return df.apply(pd.to_numeric, errors='ignore')
