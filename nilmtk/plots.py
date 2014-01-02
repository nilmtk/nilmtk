from __future__ import print_function, division
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

_to_ordinalf_np_vectorized = np.vectorize(mdates._to_ordinalf)

def plot_series(series, ax=None, label=None, date_format='%d/%m/%y %H:%M:%S', **kwargs):
    """Plot function for series which is about 5 times faster than
    pd.Series.plot().

    Parameters
    ----------
    ax : matplotlib Axes, optional
        If not provided then will generate our own axes.

    label : str, optional
        The label for the plotted line. The
        caller is responsible for enabling the legend.

    date_format : str, optional, default='%d/%m/%y %H:%M:%S'
    """
    if ax is None:
        fig, ax = plt.subplots(1)
    x = _to_ordinalf_np_vectorized(series.index.to_pydatetime())
    ax.plot(x, series, label=label, **kwargs)
    ax.xaxis.set_major_formatter(mdates.DateFormatter(date_format, 
                                                      tz=series.index.tzinfo))
    ax.set_ylabel('watts')
    fig.autofmt_xdate()
    plt.draw()
    return ax
