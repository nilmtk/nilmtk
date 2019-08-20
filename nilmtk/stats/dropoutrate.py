import numpy as np
from ..node import Node
from ..exceptions import TooFewSamplesError
from ..utils import get_index 
from .dropoutrateresults import DropoutRateResults

class DropoutRate(Node):

    requirements = {'device': {'sample_period': 'ANY VALUE'}}
    postconditions =  {'statistics': {'dropout_rate': None}}
    results_class = DropoutRateResults

    def process(self):
        self.check_requirements()
        metadata = self.upstream.get_metadata()
        sample_period = metadata['device']['sample_period']
        for chunk in self.upstream.process():
            dropout_rate = get_dropout_rate(chunk, sample_period)
            self.results.append(chunk.timeframe, 
                                {'dropout_rate': dropout_rate,
                                 'n_samples': len(chunk)})
            yield chunk


def get_dropout_rate(data, sample_period):
    """
    Parameters
    ----------
    data : pd.DataFrame or pd.Series
    sample_period : number, seconds

    Returns
    -------
    dropout_rate : float [0,1]
        The proportion of samples that have been lost; where 
        1 means that all samples have been lost and 
        0 means that no samples have been lost.
        NaN means too few samples.
    """
    MIN_N_SAMPLES = 5
    if len(data) < MIN_N_SAMPLES:
        return np.NaN

    index = get_index(data)
    assert(index[-1] > index[0])
    duration = index[-1] - index[0]
    n_expected_samples = round(duration.total_seconds() / sample_period) + 1
    dropout_rate = 1 - (index.size / n_expected_samples)
    if dropout_rate < 0:
        dropout_rate = 0.0
    assert(1 >= dropout_rate >= 0)
    return dropout_rate
