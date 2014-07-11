from ..results import Results

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
        # TODO!
        super(DropoutRateResults, self).unify(other)

    def to_dict(self):
        return {'statistics': {'dropout_rate': self.combined()}}
