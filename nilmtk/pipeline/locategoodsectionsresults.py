from results import Results

class LocateGoodSectionsResults(Results):
    """
    Attributes
    ----------
    _data : pd.DataFrame
        index is start date for the whole chunk
        `end` is end date for the whole chunk
        `sections` is a list of nilmtk.TimeFrame objects
    """
    
    def append(self, timeframe, new_results):
        """Append a single result.

        Parameters
        ----------
        timeframe : nilmtk.TimeFrame
        new_results : dict with one key: `sections`
        """
        assert new_results.keys() == ['sections']
        super(LocateGoodSectionsResults, self).append(timeframe, new_results)


    @property
    def combined(self):
        """Merges together any good sections which span multiple segments.

        Returns
        -------
        mask : list of nilmtk.TimeFrame objects
        """
        assert hasattr(self, 'max_sample_period')
        mask = []
        for index, row in self._data.iterrows():
            if mask and mask[-1].adjacent(row['sections'][0], 
                                          self.max_sample_period):
                mask[-1] = mask[-1].union(row['sections'][0])
                mask.extend(row['sections'][1:])
            else:
                mask.extend(row['sections'])

        return mask
