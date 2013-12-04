"""Base class for all datasets."""

class DataSet(object):
    """Base class for all datasets.  This class can be used
    for loading nilmtk's REDD+ data format."""

    # TODO: before we can implement this, we need to decide
    # how we're going to represent Buildings:
    # https://github.com/nilmtk/nilmtk/issues/12

    def __init__(self):
        buildings = {}

    def load(self, filename):
        """Load entire dataset into memory"""
        building_names = self.load_building_names(filename)
        for building in building_names:
            self.load_building(building, filename)

    def export(self):
        """Export dataset to disk as REDD+"""
        raise NotImplementedError

    def print_summary_stats(self):
        raise NotImplementedError

    # This will be overridden by subclasses
    def load_building_names(self, filename):
        # return list of building names
        raise NotImplementedError

    # This will be overridden by subclasses
    def load_building(self, building, filename):
        # convert units
        # convert to standard appliance names
        # self.buildings[building] = DataFrame storing building data
        raise NotImplementedError
