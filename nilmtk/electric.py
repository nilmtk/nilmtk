import pandas as pd
import numpy as np
from collections import Counter
from warnings import warn
import scipy.spatial as ss
from scipy import fft
from pandas.plotting import lag_plot, autocorrelation_plot
from scipy.special import digamma,gamma
from math import log,pi
import numpy.random as nr
import matplotlib.pyplot as plt
import numpy as np
from datetime import timedelta
import gc
import pytz

from .timeframe import TimeFrame
from .measurement import select_best_ac_type
from .utils import (offset_alias_to_seconds, convert_to_timestamp,
                    flatten_2d_list, append_or_extend_list,
                    timedelta64_to_secs, safe_resample)
from .plots import plot_series
from .preprocessing import Apply
from nilmtk.stats.histogram import histogram_from_generator
from nilmtk.appliance import DEFAULT_ON_POWER_THRESHOLD

MAX_SIZE_ENTROPY = 10000

class Electric(object):
    """Common implementations of methods shared by ElecMeter and MeterGroup.
    """
    def when_on(self, on_power_threshold=None, **load_kwargs):
        """Are the connected appliances appliance is on (True) or off (False)?

        Uses `self.on_power_threshold()` if `on_power_threshold` not provided.

        Parameters
        ----------
        on_power_threshold : number, optional
            Defaults to self.on_power_threshold()
        **load_kwargs : key word arguments
            Passed to self.power_series()

        Returns
        -------
        generator of pd.Series
            index is the same as for chunk returned by `self.power_series()`
            values are booleans
        """
        if on_power_threshold is None:
            on_power_threshold = self.on_power_threshold()
        for chunk in self.power_series(**load_kwargs):
            yield chunk >= on_power_threshold

    def on_power_threshold(self):
        """Returns the minimum `on_power_threshold` across all appliances
        immediately downstream of this meter.  If any appliance
        does not have an `on_power_threshold` then default to 10 watts."""
        if not self.appliances:
            return DEFAULT_ON_POWER_THRESHOLD
        on_power_thresholds = [a.on_power_threshold() for a in self.appliances]
        return min(on_power_thresholds)

    def min_on_duration(self):
        return self._aggregate_metadata_attribute('min_on_duration')

    def min_off_duration(self):
        return self._aggregate_metadata_attribute('min_off_duration')

    def _aggregate_metadata_attribute(self, attr, agg_func=np.max,
                                      default_value=0,
                                      from_type_metadata=True):
        attr_values = []
        for a in self.appliances:
            if from_type_metadata:
                attr_value = a.type.get(attr)
            else:
                attr_value = a.metadata.get(attr)
            if attr_value is not None:
                attr_values.append(attr_value)

        if len(attr_values) == 0:
            return default_value
        else:
            return agg_func(attr_values)

    def matches_appliances(self, key):
        """
        Parameters
        ----------
        key : dict

        Returns
        -------
        True if all key:value pairs in `key` match any appliance
        in `self.appliances`.
        """
        for appliance in self.appliances:
            if appliance.matches(key):
                return True
        return False

    def power_series_all_data(self, **kwargs):
        chunks = []
        for series in self.power_series(**kwargs):
            if len(series) > 0:
                chunks.append(series)
        if chunks:
            # Get rid of overlapping indicies
            prev_end = None
            for i, chunk in enumerate(chunks):
                if i > 0:
                    if chunk.index[0] <= prev_end:
                        chunks[i] = chunk.iloc[1:]
                prev_end = chunk.index[-1]

            all_data = pd.concat(chunks)
        else:
            all_data = None
        return all_data

    def _prep_kwargs_for_sample_period_and_resample(self, sample_period=None,
                                                    resample=False,
                                                    resample_kwargs=None,
                                                    **kwargs):
        if 'preprocessing' in kwargs:
            warn("If you are using `preprocessing` to resample then please"
                 " do not!  Instead, please use the `sample_period` parameter"
                 " and set `resample=True`.")

        if sample_period is None:
            sample_period = self.sample_period()
        elif resample != False:
            resample = True
        sample_period = int(round(sample_period))

        if resample:
            if resample_kwargs is None:
                resample_kwargs = {}

            def resample_func(df):
                resample_kwargs['rule'] = '{:d}S'.format(sample_period)
                return safe_resample(df, **resample_kwargs)

            kwargs.setdefault('preprocessing', []).append(
                Apply(func=resample_func))

        return kwargs

    def _replace_none_with_meter_timeframe(self, start=None, end=None):
        if start is None or end is None:
            timeframe_for_meter = self.get_timeframe()
            if start is None:
                start = timeframe_for_meter.start
            if end is None:
                end = timeframe_for_meter.end
        return start, end

    def plot(self, ax=None, timeframe=None, plot_legend=True, unit='W',
             plot_kwargs=None, **kwargs):
        """
        Parameters
        ----------
        width : int, optional
            Number of points on the x axis required
        ax : matplotlib.axes, optional
        plot_legend : boolean, optional
            Defaults to True.  Set to False to not plot legend.
        unit : {'W', 'kW'}
        **kwargs
        """
        # Get start and end times for the plot
        timeframe = self.get_timeframe() if timeframe is None else timeframe
        if not timeframe or timeframe.empty:
            return ax

        kwargs['sections'] = [timeframe]
        kwargs = self._set_sample_period(timeframe, **kwargs)
        power_series = self.power_series_all_data(**kwargs)
        if power_series is None or power_series.empty:
            return ax

        if unit == 'kW':
            power_series /= 1000

        if plot_kwargs is None:
            plot_kwargs = {}
        plot_kwargs.setdefault('label', self.label())
        # Pandas 0.16.1 has a bug where 'label' isn't respected. See:
        # https://github.com/nilmtk/nilmtk/issues/407
        # The following line is a work around for this bug.
        power_series.name = self.label()
        ax = power_series.plot(ax=ax, **plot_kwargs)
        ax.set_ylabel('Power ({})'.format(unit))
        if plot_legend:
            plt.legend()

        return ax

    def _set_sample_period(self, timeframe, width=800, **kwargs):
        # Calculate the resolution for the x axis
        duration = timeframe.timedelta.total_seconds()
        secs_per_pixel = int(round(duration / width))
        kwargs.update({'sample_period': secs_per_pixel, 'resample': True})
        return kwargs

    def proportion_of_upstream(self, **load_kwargs):
        """Returns a value in the range [0,1] specifying the proportion of
        the upstream meter's total energy used by this meter.
        """
        upstream = self.upstream_meter()
        upstream_good_sects = upstream.good_sections(**load_kwargs)
        proportion_of_energy = (self.total_energy(sections=upstream_good_sects) /
                                upstream.total_energy(sections=upstream_good_sects))
        if isinstance(proportion_of_energy, pd.Series):
            best_ac_type = select_best_ac_type(proportion_of_energy.keys())
            return proportion_of_energy[best_ac_type]
        else:
            return proportion_of_energy

    def vampire_power(self, **load_kwargs):
        # TODO: this might be a naive approach to calculating vampire power.
        power_series = self.power_series_all_data(**load_kwargs)
        return get_vampire_power(power_series)

    def uptime(self, **load_kwargs):
        """
        Returns
        -------
        timedelta: total duration of all good sections.
        """
        good_sections = self.good_sections(**load_kwargs)
        return good_sections.uptime()

    def average_energy_per_period(self, offset_alias='D', use_uptime=True, **load_kwargs):
        """Calculate the average energy per period.  e.g. the average 
        energy per day.

        Parameters
        ----------
        offset_alias : str
            A Pandas `offset alias`.  See:
            pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases
        use_uptime : bool

        Returns
        -------
        pd.Series
            Keys are AC types.
            Values are energy in kWh per period.
        """
        if 'sections' in load_kwargs:
            raise RuntimeError("Please do not pass in 'sections' into"
                               " 'average_energy_per_period'.  Instead"
                               " use 'use_uptime' param.")
        if use_uptime:
            td = self.uptime(**load_kwargs)
        else:
            td = self.get_timeframe().timedelta
        if not td:
            return np.NaN
        uptime_secs = td.total_seconds()
        periods = uptime_secs / offset_alias_to_seconds(offset_alias)
        energy = self.total_energy(**load_kwargs)
        return energy / periods
        
    def proportion_of_energy(self, other, **loader_kwargs):
        """Compute the proportion of energy of self compared to `other`.

        By default, only uses other.good_sections().  You may want to set 
        `sections=self.good_sections().intersection(other.good_sections())`

        Parameters
        ----------
        other : nilmtk.MeteGroup or ElecMeter
            Typically this will be mains.

        Returns
        -------
        float [0,1] or NaN if other.total_energy == 0
        """
        good_other_sections = other.good_sections(**loader_kwargs)
        loader_kwargs.setdefault('sections', good_other_sections)

        # TODO test effect of setting `sections` for other
        other_total_energy = other.total_energy(**loader_kwargs)
        if other_total_energy.sum() == 0:
            return np.NaN

        total_energy = self.total_energy(**loader_kwargs)
        if total_energy.empty:
            return 0.0

        other_ac_types = other_total_energy.keys()
        self_ac_types = total_energy.keys()
        shared_ac_types = set(other_ac_types).intersection(self_ac_types)
        n_shared_ac_types = len(shared_ac_types)
        if n_shared_ac_types > 1:
            return (total_energy[shared_ac_types] / 
                    other_total_energy[shared_ac_types]).mean()
        elif n_shared_ac_types == 0:
            ac_type = select_best_ac_type(self_ac_types)
            other_ac_type = select_best_ac_type(other_ac_types)
            warn("No shared AC types.  Using '{:s}' for submeter"
                 " and '{:s}' for other.".format(ac_type, other_ac_type))
        elif n_shared_ac_types == 1:
            ac_type = list(shared_ac_types)[0]
            other_ac_type = ac_type
        return total_energy[ac_type] / other_total_energy[other_ac_type]

    def correlation(self, other, **load_kwargs):
        """
        Finds the correlation between the two ElecMeters. Both the ElecMeters 
        should be perfectly aligned
        Adapted from: 
        http://www.johndcook.com/blog/2008/11/05/how-to-calculate-pearson-correlation-accurately/

        Parameters
        ----------
        other : an ElecMeter or MeterGroup object

        Returns
        -------
        float : [-1, 1]
        """
        sample_period = max(self.sample_period(), other.sample_period())
        load_kwargs.setdefault('sample_period', sample_period)

        def sum_and_count(electric):
            n = 0
            cumulator = 0.0
            for power in electric.power_series(**load_kwargs):
                n += len(power.index)
                cumulator += power.sum()
            return n, cumulator

        x_n, x_sum = sum_and_count(self)
        if x_n <= 1:
            return np.NaN

        y_n, y_sum = sum_and_count(other)
        if y_n <= 1:
            return np.NaN

        x_bar = x_sum / x_n 
        y_bar = y_sum / y_n

        # Second pass is used to find x_s and y_s (std.devs)
        def stdev(electric, mean, n):
            s_square_sum = 0.0
            for power in electric.power_series(**load_kwargs):
                s_square_sum += ((power - mean) * (power - mean)).sum()
            s_square = s_square_sum / (n - 1)
            return np.sqrt(s_square)

        x_s = stdev(self, x_bar, x_n)
        y_s = stdev(other, y_bar, y_n)

        numerator = 0.0
        for (x_power, y_power) in zip(self.power_series(**load_kwargs), 
                                       other.power_series(**load_kwargs)):
            xi_minus_xbar = x_power - x_bar
            del x_power
            gc.collect()
            yi_minus_ybar = y_power - y_bar
            del y_power
            gc.collect()
            numerator += (xi_minus_xbar * yi_minus_ybar).sum()
            del xi_minus_xbar
            del yi_minus_ybar
            gc.collect()
        denominator = (x_n - 1) * x_s * y_s
        corr = numerator / denominator
        return corr

    def plot_lag(self, lag=1, ax=None):
        """
        Plots a lag plot of power data
        http://www.itl.nist.gov/div898/handbook/eda/section3/lagplot.htm

        Returns
        -------
        matplotlib.axis
        """
        if ax is None:
            ax = plt.gca()
        for power in self.power_series():
            lag_plot(power, lag, ax=ax)
        return ax

    def plot_spectrum(self, ax=None):
        """
        Plots spectral plot of power data
        http://www.itl.nist.gov/div898/handbook/eda/section3/spectrum.htm

        Code borrowed from:
        http://glowingpython.blogspot.com/2011/08/how-to-plot-frequency-spectrum-with.html

        Returns
        -------
        matplotlib.axis
        """ 
        if ax is None:
            ax = plt.gca()
        Fs = 1.0/self.sample_period()
        for power in self.power_series():
            n = len(power.values) # length of the signal
            k = np.arange(n)
            T = n/Fs
            frq = k/T # two sides frequency range
            frq = frq[range(n//2)] # one side frequency range

            Y = fft(power)/n # fft computing and normalization
            Y = Y[range(n//2)]

            ax.plot(frq,abs(Y)) # plotting the spectrum

        ax.set_xlabel('Freq (Hz)')
        ax.set_ylabel('|Y(freq)|')
        return ax
          
    def plot_autocorrelation(self, ax=None):
        """
        Plots autocorrelation of power data 
        Reference: 
        http://www.itl.nist.gov/div898/handbook/eda/section3/autocopl.htm

        Returns
        -------
        matplotlib.axis 
        """
        if ax is None:
            ax = plt.gca()
        for power in self.power_series():
            autocorrelation_plot(power, ax=ax)
        return ax

    def plot_power_histogram(self, ax=None, load_kwargs=None, 
                             plot_kwargs=None, range=None, **hist_kwargs):
        """
        Parameters
        ----------
        ax : axes
        load_kwargs : dict
        plot_kwargs : dict
        range : None or tuple
            if range=(None, x) then on_power_threshold will be used as minimum.
        **hist_kwargs

        Returns
        -------
        ax
        """
        if ax is None:
            ax = plt.gca()
        if load_kwargs is None:
            load_kwargs = {}
        if plot_kwargs is None:
            plot_kwargs = {}
        generator = self.power_series(**load_kwargs)

        # set range
        if range is None or range[0] is None:
            maximum = None if range is None else range[1]
            range = (self.on_power_threshold(), maximum)

        hist, bins = histogram_from_generator(generator, range=range, 
                                              **hist_kwargs)

        # Plot
        plot_kwargs.setdefault('linewidth', 0.1)
        ax.fill_between(bins[:-1], 0, hist, **plot_kwargs)
        first_bin_width = bins[1] - bins[0]
        ax.set_xlim([bins[0]-(first_bin_width/2), bins[-1]])
        ax.set_xlabel('Power (watts)')
        ax.set_ylabel('Count')
        return ax

    def switch_times(self, threshold=40):
        """
        Returns an array of pd.DateTime when a switch occurs as defined by threshold

        Parameters
        ----------
        threshold: int, threshold in Watts between succcessive readings 
        to amount for an appliance state change
        """

        datetime_switches = []
        for power in self.power_series():
            delta_power = power.diff()
            delta_power_absolute = delta_power.abs()
            datetime_switches.append(delta_power_absolute[(delta_power_absolute>threshold)].index.values.tolist())
        return flatten_2d_list(datetime_switches)

    def entropy(self, k=3, base=2):
        """ 
        This implementation is provided courtesy NPEET toolbox,
        the authors kindly allowed us to directly use their code.
        As a courtesy procedure, you may wish to cite their paper, 
        in case you use this function.
        This fails if there is a large number of records. Need to
        ask the authors what to do about the same! 
        The classic K-L k-nearest neighbor continuous entropy estimator
        x should be a list of vectors, e.g. x = [[1.3],[3.7],[5.1],[2.4]]
        if x is a one-dimensional scalar and we have four samples
        """
        def kdtree_entropy(z):
            assert k <= len(z)-1, "Set k smaller than num. samples - 1"
            d = len(z[0])
            N = len(z)
            #small noise to break degeneracy, see doc.
            intens = 1e-10
            z = [list(p + intens*nr.rand(len(z[0]))) for p in z]
            tree = ss.cKDTree(z)
            nn = [tree.query(point, k+1, p=float('inf'))[0][k] for point in z]
            const = digamma(N)-digamma(k) + d*log(2)
            return (const + d*np.mean(map(log, nn)))/log(base)

        out = []
        for power in self.power_series():
            x = power.values
            num_elements = len(x)
            x = x.reshape((num_elements, 1))            
            if num_elements > MAX_SIZE_ENTROPY:

                splits = num_elements/MAX_SIZE_ENTROPY + 1
                y = np.array_split(x, splits)
                for z in y:            
                    out.append(kdtree_entropy(z))                    
            else:
                out.append(kdtree_entropy(x))
        return sum(out)/len(out)

    def mutual_information(self, other, k=3, base=2):
        """ 
        Mutual information of two ElecMeters
        x,y should be a list of vectors, e.g. x = [[1.3],[3.7],[5.1],[2.4]]
        if x is a one-dimensional scalar and we have four samples

        Parameters
        ----------
        other : ElecMeter or MeterGroup
        """
        def kdtree_mi(x, y, k, base):
            intens = 1e-10 #small noise to break degeneracy, see doc.
            x = [list(p + intens*nr.rand(len(x[0]))) for p in x]
            y = [list(p + intens*nr.rand(len(y[0]))) for p in y]
            points = zip2(x,y)
            # Find nearest neighbors in joint space, p=inf means max-norm
            tree = ss.cKDTree(points)
            dvec = [tree.query(point, k+1, p=float('inf'))[0][k] for point in points]
            a, b, c, d = avgdigamma(x, dvec), avgdigamma(y, dvec), digamma(k), digamma(len(x))
            return (-a-b+c+d)/log(base)
            
        def zip2(*args):
            # zip2(x,y) takes the lists of vectors and makes it a list of vectors in a joint space
            # E.g. zip2([[1],[2],[3]],[[4],[5],[6]]) = [[1,4],[2,5],[3,6]]
            return [sum(sublist, []) for sublist in zip(*args)]

        def avgdigamma(points, dvec):
            #This part finds number of neighbors in some radius in the marginal space
            #returns expectation value of <psi(nx)>
            N = len(points)
            tree = ss.cKDTree(points)
            avg = 0.
            for i in range(N):
                dist = dvec[i]
                #subtlety, we don't include the boundary point, 
                #but we are implicitly adding 1 to kraskov def bc center point is included
                num_points = len(tree.query_ball_point(points[i],dist-1e-15,p=float('inf'))) 
                avg += digamma(num_points)/N
            return avg

        out = []
        for power_x, power_y in zip(self.power_series(), other.power_series()):
            power_x_val = power_x.values
            power_y_val = power_y.values 
            num_elements = len(power_x_val)
            power_x_val = power_x_val.reshape((num_elements, 1))
            power_y_val = power_y_val.reshape((num_elements, 1))            
            if num_elements > MAX_SIZE_ENTROPY:
                splits = num_elements/MAX_SIZE_ENTROPY + 1
                x_split = np.array_split(power_x_val, splits)
                y_split = np.array_split(power_y_val, splits)
                for x, y  in zip(x_split, y_split):
                    out.append(kdtree_mi(x, y, k, base))
            else:
                out.append(kdtree_mi(power_x_val, power_y_val, k, base))
        return sum(out)/len(out)

    def available_power_ac_types(self):
        """Finds available alternating current types from power measurements.

        Returns
        -------
        list of strings e.g. ['apparent', 'active']

        .. note:: Deprecated in NILMTK v0.3
                  `available_power_ac_types` should not be used.  Instead please
                  use `available_ac_types('power').`
        """
        warn("`available_power_ac_types` is deprecated.  Please use"
             " `available_ac_types('power')` instead.", DeprecationWarning)
        return self.available_ac_types('power')

    def load_series(self, **kwargs):
        """
        Parameters
        ----------
        ac_type : str
        physical_quantity : str
            We sum across ac_types of this physical quantity.
        **kwargs : passed through to load().

        Returns
        -------
        generator of pd.Series.  If a single ac_type is found for the
        physical_quantity then the series.name will be a normal tuple.
        If more than 1 ac_type is found then the ac_type will be a string
        of the ac_types with '+' in between.  e.g. 'active+apparent'.
        """
        # Pull data through preprocessing pipeline
        physical_quantity = kwargs['physical_quantity']
        generator = self.load(**kwargs)
        for chunk in generator:
            if chunk.empty:
                yield chunk
                continue
            chunk_to_yield = chunk[physical_quantity].sum(axis=1, skipna=False)
            ac_types = '+'.join(chunk[physical_quantity].columns)
            chunk_to_yield.name = (physical_quantity, ac_types)
            chunk_to_yield.timeframe = getattr(chunk, 'timeframe', None)
            chunk_to_yield.look_ahead = getattr(chunk, 'look_ahead', None)
            yield chunk_to_yield

    def power_series(self, **kwargs):
        """Get power Series.

        Parameters
        ----------
        ac_type : str, defaults to 'best'
        **kwargs :
            Any other key word arguments are passed to self.load()

        Returns
        -------
        generator of pd.Series of power measurements.
        """
        # Select power column:
        kwargs['physical_quantity'] = 'power'
        kwargs.setdefault('ac_type', 'best')
        return self.load_series(**kwargs)

    def activity_histogram(self, period='D', bin_duration='H', **kwargs):
        """Return a histogram vector showing when activity occurs.

        e.g. to see when, over the course of an average day, activity occurs
        then use `bin_duration='H'` and `period='D'`.

        Parameters
        ----------
        period : str. Pandas period alias.
        bin_duration : str. Pandas period alias e.g. 'H' = hourly; 'D' = daily.
            Width of each bin of the histogram.  `bin_duration` must exactly
            divide the chosen `period`.
        Returns
        -------
        hist : np.ndarray
            length will be `period / bin_duration`
        """
        n_bins = (offset_alias_to_seconds(period) / 
                  offset_alias_to_seconds(bin_duration))
        if n_bins != int(n_bins):
            raise ValueError('`bin_duration` must exactly divide the'
                             ' chosen `period`')
        n_bins = int(n_bins)

        # Resample to `bin_duration` and load
        kwargs['sample_period'] = offset_alias_to_seconds(bin_duration)
        kwargs['resample_kwargs'] = {'how': 'max'}
        when_on = self.when_on(**kwargs)

        # Calculate histogram...
        hist = np.zeros(n_bins, dtype=int)
        for on_chunk in when_on:
            if len(on_chunk) < 5:
                continue

            on_chunk = on_chunk.fillna(0).astype(int)

            # Trick using resample to 'normalise' start and end dates
            # to natural boundaries of 'period'.
            resampled_to_period = on_chunk.iloc[[0, -1]].resample(period).mean().index
            start = resampled_to_period[0]
            end = resampled_to_period[-1] + 1 * resampled_to_period.freq

            # Reindex chunk
            new_index = pd.date_range(start, end, freq=bin_duration, closed='left')
            on_chunk = on_chunk.reindex(new_index, fill_value=0, copy=False)

            # reshape
            matrix = on_chunk.values.reshape((-1, n_bins))

            # histogram
            hist += matrix.sum(axis=0)

        return hist

    def plot_activity_histogram(self, ax=None, period='D', bin_duration='H',
                                plot_kwargs=None, **kwargs):
        
        # Some common labels
        freq_labels = {
            'H': 'hour',
            'D': 'day',
            'M': 'month',
            'T': 'minute',
            'min': 'minute',
            'W': 'week'
        }

        if ax is None:
            ax = plt.gca()
        hist = self.activity_histogram(bin_duration=bin_duration,
                                       period=period, **kwargs)
        if plot_kwargs is None:
            plot_kwargs = {}
        n_bins = len(hist)
        plot_kwargs.setdefault('align', 'center')
        plot_kwargs.setdefault('linewidth', 0)
        ax.bar(range(n_bins), hist, np.ones(n_bins), **plot_kwargs)
        ax.set_xlim([-0.5, n_bins])
        ax.set_title('Activity distribution')
        ax.set_xlabel(
            freq_labels.get(bin_duration, bin_duration).title() + 
            ' of ' + 
            freq_labels.get(period, period)
        )
        ax.set_ylabel('Count')
        return ax


    def activation_series(self, *args, **kwargs):
        """Returns runs of an appliance.

        Most appliances spend a lot of their time off.  This function finds
        periods when the appliance is on.

        Parameters
        ----------
        min_off_duration : int
            If min_off_duration > 0 then ignore 'off' periods less than
            min_off_duration seconds of sub-threshold power consumption
            (e.g. a washing machine might draw no power for a short
            period while the clothes soak.)  Defaults value from metadata or,
            if metadata absent, defaults to 0.
        min_on_duration : int
            Any activation lasting less seconds than min_on_duration will be
            ignored.  Defaults value from metadata or, if metadata absent,
            defaults to 0.
        border : int
            Number of rows to include before and after the detected activation
        on_power_threshold : int or float
            Defaults to self.on_power_threshold()
        **kwargs : kwargs for self.power_series()

        Returns
        -------
        list of pd.Series.  Each series contains one activation.

        .. note:: Deprecated
          `activation_series` will be removed in NILMTK v0.3.
          Please use `get_activations` instead.
        """
        warn("`activation_series()` is deprecated."
             "  Please use `get_activations()` instead!", DeprecationWarning)
        return self.get_activations(*args, **kwargs)

    def get_activations(self, min_off_duration=None, min_on_duration=None,
                        border=1, on_power_threshold=None, **kwargs):
        """Returns runs of an appliance.

        Most appliances spend a lot of their time off.  This function finds
        periods when the appliance is on.

        Parameters
        ----------
        min_off_duration : int
            If min_off_duration > 0 then ignore 'off' periods less than
            min_off_duration seconds of sub-threshold power consumption
            (e.g. a washing machine might draw no power for a short
            period while the clothes soak.)  Defaults value from metadata or,
            if metadata absent, defaults to 0.
        min_on_duration : int
            Any activation lasting less seconds than min_on_duration will be
            ignored.  Defaults value from metadata or, if metadata absent,
            defaults to 0.
        border : int
            Number of rows to include before and after the detected activation
        on_power_threshold : int or float
            Defaults to self.on_power_threshold()
        **kwargs : kwargs for self.power_series()

        Returns
        -------
        list of pd.Series.  Each series contains one activation.
        """
        if on_power_threshold is None:
            on_power_threshold = self.on_power_threshold()

        if min_off_duration is None:
            min_off_duration = self.min_off_duration()

        if min_on_duration is None:
            min_on_duration = self.min_on_duration()

        activations = []
        kwargs.setdefault('resample', True)
        for chunk in self.power_series(**kwargs):
            activations_for_chunk = get_activations(
                chunk=chunk, min_off_duration=min_off_duration,
                min_on_duration=min_on_duration, border=border,
                on_power_threshold=on_power_threshold)
            activations.extend(activations_for_chunk)

        return activations


def align_two_meters(master, slave, func='power_series'):
    """Returns a generator of 2-column pd.DataFrames.  The first column is from
    `master`, the second from `slave`.

    Takes the sample rate and good_periods of `master` and applies to `slave`.

    Parameters
    ----------
    master, slave : ElecMeter or MeterGroup instances
    """
    sample_period = master.sample_period()
    period_alias = '{:d}S'.format(sample_period)
    sections = master.good_sections()
    master_generator = getattr(master, func)(sections=sections)
    for master_chunk in master_generator:
        if len(master_chunk) < 2:
            return
        chunk_timeframe = TimeFrame(master_chunk.index[0],
                                    master_chunk.index[-1])
        slave_generator = getattr(slave, func)(sections=[chunk_timeframe])
        slave_chunk = next(slave_generator)

        # TODO: do this resampling in the pipeline?
        if not slave_chunk.empty:
            slave_chunk = slave_chunk.resample(period_alias).mean()
            
        if slave_chunk.empty:
            continue
            
        master_chunk = master_chunk.resample(period_alias).mean()

        yield pd.DataFrame({'master': master_chunk, 'slave': slave_chunk})


def activation_series_for_chunk(*args, **kwargs):
    """Returns runs of an appliance.

    Most appliances spend a lot of their time off.  This function finds
    periods when the appliance is on.

    Parameters
    ----------
    chunk : pd.Series
    min_off_duration : int
        If min_off_duration > 0 then ignore 'off' periods less than
        min_off_duration seconds of sub-threshold power consumption
        (e.g. a washing machine might draw no power for a short
        period while the clothes soak.)  Defaults to 0.
    min_on_duration : int
        Any activation lasting less seconds than min_on_duration will be
        ignored.  Defaults to 0.
    border : int
        Number of rows to include before and after the detected activation
    on_power_threshold : int or float
        Watts

    Returns
    -------
    list of pd.Series.  Each series contains one activation.

    .. note:: Deprecated
      `activation_series` will be removed in NILMTK v0.3.
      Please use `get_activations` instead.
    """
    warn("`activation_series_for_chunk()` is deprecated."
         "  Please use `get_activations()` instead!", DeprecationWarning)
    return get_activations(*args, **kwargs)


def get_activations(chunk, min_off_duration=0, min_on_duration=0,
                    border=1, on_power_threshold=5):
    """Returns runs of an appliance.

    Most appliances spend a lot of their time off.  This function finds
    periods when the appliance is on.

    Parameters
    ----------
    chunk : pd.Series
    min_off_duration : int
        If min_off_duration > 0 then ignore 'off' periods less than
        min_off_duration seconds of sub-threshold power consumption
        (e.g. a washing machine might draw no power for a short
        period while the clothes soak.)  Defaults to 0.
    min_on_duration : int
        Any activation lasting less seconds than min_on_duration will be
        ignored.  Defaults to 0.
    border : int
        Number of rows to include before and after the detected activation
    on_power_threshold : int or float
        Watts

    Returns
    -------
    list of pd.Series.  Each series contains one activation.
    """
    when_on = chunk >= on_power_threshold

    # Find state changes
    state_changes = when_on.astype(np.int8).diff()
    del when_on
    switch_on_events = np.where(state_changes == 1)[0]
    switch_off_events = np.where(state_changes == -1)[0]
    del state_changes

    if len(switch_on_events) == 0 or len(switch_off_events) == 0:
        return []

    # Make sure events align
    if switch_off_events[0] < switch_on_events[0]:
        switch_off_events = switch_off_events[1:]
        if len(switch_off_events) == 0:
            return []
    if switch_on_events[-1] > switch_off_events[-1]:
        switch_on_events = switch_on_events[:-1]
        if len(switch_on_events) == 0:
            return []
    assert len(switch_on_events) == len(switch_off_events)

    # Smooth over off-durations less than min_off_duration
    if min_off_duration > 0:
        off_durations = (chunk.index[switch_on_events[1:]].values -
                         chunk.index[switch_off_events[:-1]].values)

        off_durations = timedelta64_to_secs(off_durations)

        above_threshold_off_durations = np.where(
            off_durations >= min_off_duration)[0]

        # Now remove off_events and on_events
        switch_off_events = switch_off_events[
            np.concatenate([above_threshold_off_durations,
                            [len(switch_off_events)-1]])]
        switch_on_events = switch_on_events[
            np.concatenate([[0], above_threshold_off_durations+1])]
    assert len(switch_on_events) == len(switch_off_events)

    activations = []
    for on, off in zip(switch_on_events, switch_off_events):
        duration = (chunk.index[off] - chunk.index[on]).total_seconds()
        if duration < min_on_duration:
            continue
        on -= 1 + border
        if on < 0:
            on = 0
        off += border
        activation = chunk.iloc[on:off]
        # throw away any activation with any NaN values
        if not activation.isnull().values.any():
            activations.append(activation)

    return activations


def get_vampire_power(power_series):
    # This is a very simple function at the moment but
    # in the future we may want to implement a more
    # sophisticated vampire power function, so it is worth
    # calling `get_vampire_power` instead of just doing
    # power_series.min() in case we get round to building
    # a better vampire power function!
    return power_series.min()
