from .dataset import DataSet

class HES(DataSet):
    """Load data from UK Government's Household Electricity Survey 
    (the cleaned version of the dataset released in summer 2013).
    """

    # TODO: re-use code from 
    # https://github.com/JackKelly/pda/blob/master/scripts/hes/load_hes.py

    # TODO: before we can implement this, we need to decide
    # how we're going to represent Buildings:
    # https://github.com/nilmtk/nilmtk/issues/12

    def load_building(self, filename, building_name):
        raise NotImplementedError

    def load_building_names(self, filename):
        raise NotImplementedError
