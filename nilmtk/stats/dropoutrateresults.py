import matplotlib.pyplot as plt
from ..results import Results
from ..consts import SECS_PER_DAY

class DropoutRateResults(Results):
    """
    Attributes
    ----------
    _data : pd.DataFrame
        index is start date for the whole chunk
        `end` is end date for the whole chunk
        `dropout_rate` is float [0,1]
        `n_samples` is int, used for calculating weighted mean
    """
    name = "dropout_rate"
    
    def combined(self):
        """Calculates weighted average.

        Returns
        -------
        dropout_rate : float, [0,1]
        """
        tot_samples = self._data['n_samples'].sum()
        proportion = self._data['n_samples'] / tot_samples
        dropout_rate = (self._data['dropout_rate'] * proportion).sum()
        return dropout_rate

    def unify(self, other):
        super(DropoutRateResults, self).unify(other)
        for i, row in self._data.iterrows():
            # store mean of dropout rate
            self._data['dropout_rate'].loc[i] += other._data['dropout_rate'].loc[i]
            self._data['dropout_rate'].loc[i] /= 2
            
            self._data['n_samples'].loc[i] += other._data['n_samples'].loc[i]

    def to_dict(self):
        return {'statistics': {'dropout_rate': self.combined()}}

    def plot(self, ax=None):
        if ax is None:
            ax = plt.gca()
        ax.xaxis.axis_date()
        for index, row in self._data.iterrows():
            length = (row['end'] - index).total_seconds() / SECS_PER_DAY
            rect = plt.Rectangle((index, 0), # bottom left corner
                                 length,
                                 row['dropout_rate'], # width
                                 color='b') 
            ax.add_patch(rect)            
        ax.autoscale_view()
