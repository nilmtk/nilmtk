from __future__ import print_function, division
import matplotlib.pyplot as plt
import pandas as pd

# NILMTK imports
from nilmtk.consts import SECS_PER_DAY
from nilmtk.timeframe import TimeFrame

class TimeFrameGroup(list):
    """A collection of nilmtk.TimeFrame objects."""

    def __init__(self, timeframes=None):
        if isinstance(timeframes, pd.tseries.period.PeriodIndex):
            periods = timeframes
            timeframes = [TimeFrame(period.start_time, period.end_time) 
                          for period in periods]
        args = [timeframes] if timeframes else []
        super(TimeFrameGroup, self).__init__(*args)

    def plot(self, ax=None, y=0, height=1, gap=0.05, color='b', **kwargs):
        if ax is None:
            ax = plt.gca()
        ax.xaxis.axis_date()
        height -= gap * 2
        for timeframe in self:
            length = timeframe.timedelta.total_seconds() / SECS_PER_DAY
            bottom_left_corner = (timeframe.start, y + gap)
            rect = plt.Rectangle(bottom_left_corner, length, height, 
                                 color=color, **kwargs)
            ax.add_patch(rect)

        ax.autoscale_view()
        return ax

    def intersection(self, other):
        """Returns a new TimeFrameGroup of self masked by other.

        Illustrated example:

         self.good_sections():  |######----#####-----######|
        other.good_sections():  |---##---####----##-----###|
               intersection():  |---##-----##-----------###|
        """
        assert isinstance(other, (TimeFrameGroup, list))
        new_tfg = TimeFrameGroup()
        for self_timeframe in self:
            for other_timeframe in other:
                intersect = self_timeframe.intersection(other_timeframe)
                if not intersect.empty:
                    new_tfg.append(intersect)
        return new_tfg

 
