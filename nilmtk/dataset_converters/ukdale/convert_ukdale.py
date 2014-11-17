from __future__ import print_function, division
from os.path import join
from nilmtk.dataset_converters.redd.convert_redd import _convert
from nilmtk.utils import get_module_directory
from nilm_metadata import convert_yaml_to_hdf5


def convert_ukdale(ukdale_path, hdf_filename):
    """
    Parameters
    ----------
    ukdale_path : str
        The root path of the UK-DALE dataset.  It is assumed that the YAML
        metadata is in 'ukdale_path/metadata'.
    hdf_filename : str
        The destination HDF5 filename (including path and suffix).
    """

    def _ukdale_measurement_mapping_func(house_id, chan_id):
        # TODO: This needs updating.  It's wrong!
        ac_type = 'apparent' if chan_id <= 2 else 'active'
        return [('power', ac_type)]

    _convert(ukdale_path, hdf_filename, _ukdale_measurement_mapping_func, 
             'Europe/London')

    # Add metadata
    convert_yaml_to_hdf5(join(ukdale_path, 'metadata'), hdf_filename)

    print("Done converting UK-DALE to HDF5!")
