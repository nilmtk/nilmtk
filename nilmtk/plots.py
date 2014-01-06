from __future__ import print_function, division
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

_to_ordinalf_np_vectorized = np.vectorize(mdates._to_ordinalf)

def plot_series(series, **kwargs):
    """Plot function for series which is about 5 times faster than
    pd.Series.plot().

    Parameters
    ----------
    ax : matplotlib Axes, optional
        If not provided then will generate our own axes.

    date_format : str, optional, default='%d/%m/%y %H:%M:%S'

    Can also use all **kwargs expected by `ax.plot`
    """
    ax = kwargs.get('ax')
    date_format = kwargs.get('date_format', '%d/%m/%y %H:%M:%S')

    if ax is None:
        fig, ax = plt.subplots(1)
    x = _to_ordinalf_np_vectorized(series.index.to_pydatetime())
    ax.plot(x, series, **kwargs)
    ax.xaxis.set_major_formatter(mdates.DateFormatter(date_format, 
                                                      tz=series.index.tzinfo))
    ax.set_ylabel('watts')
    fig.autofmt_xdate()
    return ax
