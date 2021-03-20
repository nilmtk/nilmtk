import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from math import sqrt

try:
    _to_ordinalf_np_vectorized = np.vectorize(mdates._to_ordinalf)
except:
    try:
        _to_ordinalf_np_vectorized = np.vectorize(mdates._dt64_to_ordinalf)
    except:
        raise RuntimeError('This matplotlib version is not supported.')


def plot_series(series, ax=None, fig=None, 
                date_format='%d/%m/%y %H:%M:%S', tz_localize=True, **kwargs):
    """Plot function for series which is about 5 times faster than
    pd.Series.plot().

    Parameters
    ----------
    series : pd.Series
    ax : matplotlib Axes, optional
        If not provided then will generate our own axes.
    fig : matplotlib Figure
    date_format : str, optional, default='%d/%m/%y %H:%M:%S'
    tz_localize : boolean, optional, default is True
        if False then display UTC times.

    Can also use all **kwargs expected by `ax.plot`
    """
    if series is None or len(series) == 0:
        return ax

    if ax is None:
        ax = plt.gca()

    if fig is None:
        fig = plt.gcf()

    #TODO: we probably don't need this anymore
    x = _to_ordinalf_np_vectorized(series.index.to_pydatetime())
    ax.plot(x, series, **kwargs)
    tz = series.index.tzinfo if tz_localize else None
    ax.xaxis.set_major_formatter(
        mdates.DateFormatter(date_format, tz=tz))
    ax.set_ylabel('watts')
    fig.autofmt_xdate()
    return ax


def plot_pairwise_heatmap(df, labels, edgecolors='w',
                          cmap=matplotlib.cm.RdYlBu_r, log=False):
    """
    Plots a heatmap of a 'square' df
    Rows and columns are same and the values in this dataframe
    correspond to the computation b/w row,column.
    This plot can be used for plotting pairwise_correlation
    or pairwise_mutual_information or any method which works
    similarly
    """
    width = len(df.columns) / 4
    height = len(df.index) / 4

    fig, ax = plt.subplots(figsize=(width, height))

    heatmap = ax.pcolor(
        df,
        edgecolors=edgecolors,  # put white lines between squares in heatmap
        cmap=cmap,
        norm=matplotlib.colors.LogNorm() if log else None)

    ax.autoscale(tight=True)  # get rid of whitespace in margins of heatmap
    ax.set_aspect('equal')  # ensure heatmap cells are square
    ax.xaxis.set_ticks_position('top')  # put column labels at the top
    # turn off ticks:
    ax.tick_params(bottom='off', top='off', left='off', right='off')

    plt.yticks(np.arange(len(df.index)) + 0.5, labels)
    plt.xticks(np.arange(len(df.columns)) + 0.5, labels, rotation=90)

    # ugliness from http://matplotlib.org/users/tight_layout_guide.html
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", "3%", pad="1%")
    plt.colorbar(heatmap, cax=cax)


def latexify(fig_width=None, fig_height=None, columns=1, fontsize=8):
    """Set up matplotlib's RC params for LaTeX plotting.
    Call this before plotting a figure.

    Parameters
    ----------
    fig_width : float, optional, inches
    fig_height : float,  optional, inches
    columns : {1, 2}
    """

    # code adapted from http://www.scipy.org/Cookbook/Matplotlib/LaTeX_Examples

    # Width and max height in inches for IEEE journals taken from
    # computer.org/cms/Computer.org/Journal%20templates/transactions_art_guide.pdf

    assert columns in [1, 2]

    if fig_width is None:
        fig_width = 3.39 if columns == 1 else 6.9  # width in inches

    if fig_height is None:
        golden_mean = (sqrt(5)-1.0)/2.0    # Aesthetic ratio
        fig_height = fig_width * golden_mean  # height in inches

    MAX_HEIGHT_INCHES = 8.0
    if fig_height > MAX_HEIGHT_INCHES:
        print("WARNING: fig_height too large:", fig_height,
              "so will reduce to", MAX_HEIGHT_INCHES, "inches.")
        fig_height = MAX_HEIGHT_INCHES

    params = {
        'backend': 'ps',
        'text.latex.preamble': ['\\usepackage{gensymb}'],
        'axes.labelsize': fontsize,  # fontsize for x and y labels (was 10)
        'axes.titlesize': fontsize,
        'font.size': fontsize,
        'legend.fontsize': fontsize,
        'xtick.labelsize': fontsize,
        'ytick.labelsize': fontsize,
        'text.usetex': True,
        'figure.figsize': [fig_width, fig_height],
        'font.family': 'serif'
    }
    matplotlib.rcParams.update(params)


def format_axes(ax, spine_color='gray'):

    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color(spine_color)
        ax.spines[spine].set_linewidth(0.5)

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_tick_params(direction='out', color=spine_color)

#    matplotlib.pyplot.tight_layout()

    return ax
