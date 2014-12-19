from __future__ import print_function, division
import matplotlib.pyplot as plt
from nilmtk.consts import SECS_PER_DAY

class TimeFrameGroup(list):
    """A collection of nilmtk.TimeFrame objects."""

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
