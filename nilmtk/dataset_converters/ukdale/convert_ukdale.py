from __future__ import print_function, division
from os import remove
from os.path import join
from nilmtk.dataset_converters.redd.convert_redd import _convert
from nilmtk.utils import get_module_directory
from nilm_metadata import convert_yaml_to_hdf5
from nilmtk import DataSet


def convert_ukdale(ukdale_path, hdf_filename):
    """Converts the UK-DALE dataset to NILMTK HDF5 format.

    For more information about the UK-DALE dataset, and to download
    it, please see http://www.doc.ic.ac.uk/~dk3810/data/

    Parameters
    ----------
    ukdale_path : str
        The root path of the UK-DALE dataset.  It is assumed that the YAML
        metadata is in 'ukdale_path/metadata'.
    hdf_filename : str
        The destination HDF5 filename (including path and suffix).
    """
    ac_type_map = _get_ac_type_map(ukdale_path)

    def _ukdale_measurement_mapping_func(house_id, chan_id):
        ac_type = ac_type_map[(house_id, chan_id)][0]
        return [('power', ac_type)]

    _convert(ukdale_path, hdf_filename, _ukdale_measurement_mapping_func, 
             'Europe/London')

    # Add metadata
    convert_yaml_to_hdf5(join(ukdale_path, 'metadata'), hdf_filename)

    print("Done converting UK-DALE to HDF5!")


def _get_ac_type_map(ukdale_path):
    """First we need to convert the YAML metadata to HDF5
    so we can load the metadata into NILMTK to allow
    us to use NILMTK to find the ac_type for each channel."""

    hdf5_just_metadata = join(ukdale_path, 'metadata', 'ukdale_metadata.h5')
    convert_yaml_to_hdf5(join(ukdale_path, 'metadata'), hdf5_just_metadata)
    ukdale_dataset = DataSet(hdf5_just_metadata)
    ac_type_map = {}
    for building_i, building in ukdale_dataset.buildings.iteritems():
        for meter in building.elec.meters:
            key = (building_i, meter.instance())
            ac_type_map[key] = meter.available_power_ac_types()
    ukdale_dataset.store.close()
    remove(hdf5_just_metadata)
    return ac_type_map
