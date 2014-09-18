from __future__ import print_function, division
from nilmtk.dataset_converters.redd.convert_redd import _convert 

def convert_ukdale(ukdale_path, hdf_filename):
    """
    Parameters
    ----------
    ukdale_path : str
        The root path of the UK-DALE dataset.
    hdf_filename : str
        The destination HDF5 filename (including path and suffix).
    """

    def _ukdale_measurement_mapping_func(house_id, chan_id):
        # TODO: This needs updating.  It's wrong!
        ac_type = 'apparent' if chan_id <= 2 else 'active'
        return [('power', ac_type)]

    _convert(ukdale_path, hdf_filename, _ukdale_measurement_mapping_func, 
             'Europe/London')
