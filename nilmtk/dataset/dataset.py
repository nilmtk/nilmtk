import os

"""Base class for all datasets."""


def load_labels(data_dir):
    """Loads data from labels.dat file.

    Arguments
    ---------
    data_dir : str

    Returns
    -------
    labels : dict
        mapping channel numbers (ints) to appliance names (str)
    """
    filename = os.path.join(data_dir, 'labels.dat')
    with open(filename) as labels_file:
        lines = labels_file.readlines()
    
    labels = {}
    for line in lines:
        line = line.split(' ')
        labels[int(line[0])] = line[1].strip() # TODO add error handling if line[0] not an int

    return labels


class DataSet(object):
    """Base class for all datasets.  This class can be used
    for loading nilmtk's REDD+ data format.

    Attributes
    ----------
    buildings : dict
        Each key is a string representing the name of the building and is 
        preserved from the original dataset.  Each value is a 
        nilmtk.building.Building object.

    urls : list of strings, optional
        The URL(s) for more information about this dataset

    citations : list of strings, optional
        Academic citation(s) for this dataset
    
    """

    # TODO: before we can implement this, we need to decide
    # how we're going to represent Buildings:
    # https://github.com/nilmtk/nilmtk/issues/12

    def __init__(self):
        self.buildings = {}
        self.urls = []
        self.citations = []

    def load(self, root_directory):
        """Load entire dataset into memory"""
        building_names = self.load_building_names(root_directory)
        for building in building_names:
            self.load_building(root_directory, building)

    def export(self, directory, format='REDD+', compact=False):
        """Export dataset to disk as REDD+.

        Arguments
        ---------
        directory : str
            Output directory

        format : str, optional
            `REDD+` or `HDF5`

        compact : boolean, optional
            Defaults to false.  If True then only save change points.
        """
        raise NotImplementedError

    def print_summary_stats(self):
        raise NotImplementedError

    # This will be overridden by each subclass
    def load_building_names(self, root_directory):
        """return list of building names"""
        raise NotImplementedError

    # This will be overridden by each subclass
    def load_building(self, root_directory, building_name):
        # convert units
        # convert to standard appliance names
        # self.buildings[building] = DataFrame storing building data
        raise NotImplementedError
