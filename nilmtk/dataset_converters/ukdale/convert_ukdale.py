from __future__ import print_function, division
from os import remove
from os.path import join
import pandas as pd
from nilmtk.dataset_converters.redd.convert_redd import (_convert, _load_csv, 
                                                         _store_put)
from nilmtk.utils import get_module_directory
from nilmtk import DataSet
from nilmtk.datastore import Key
from nilm_metadata import convert_yaml_to_hdf5


ONE_SEC_COLUMNS = [('power', 'active'), ('power', 'apparent'), ('voltage', '')]
TZ = 'Europe/London'


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

    # Convert 6-second data
    _convert(ukdale_path, hdf_filename, _ukdale_measurement_mapping_func, TZ,
             sort_index=False)

    # Add metadata
    convert_yaml_to_hdf5(join(ukdale_path, 'metadata'), hdf_filename)

    # Convert 1-second data
    _convert_one_sec_data(ukdale_path, hdf_filename, ac_type_map)

    print("Done converting UK-DALE to HDF5!")


def _get_ac_type_map(ukdale_path):
    """First we need to convert the YAML metadata to HDF5
    so we can load the metadata into NILMTK to allow
    us to use NILMTK to find the ac_type for each channel.
    
    Parameters
    ----------
    ukdale_path : str

    Returns
    -------
    ac_type_map : dict.  
        Keys are pairs of ints: (<house_instance>, <meter_instance>)
        Values are list of available power ac type for that meter.
    """

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


def _convert_one_sec_data(ukdale_path, hdf_filename, ac_type_map):
    ids_of_one_sec_data = [
        identifier for identifier, ac_types in ac_type_map.iteritems()
        if ac_types == ['active', 'apparent']]

    if not ids_of_one_sec_data:
        return

    store = pd.HDFStore(hdf_filename, 'a')
    for identifier in ids_of_one_sec_data:
        key = Key(building=identifier[0], meter=identifier[1])
        print("Loading 1-second data for", key, "...")
        house_path = 'house_{:d}'.format(key.building)
        filename = join(ukdale_path, house_path, 'mains.dat')
        df = _load_csv(filename, ONE_SEC_COLUMNS, TZ)
        _store_put(store, str(key), df)       
        
        # Set 'disabled' metadata attributes
        group = store._handle.get_node('/building{:d}'.format(key.building))
        metadata = group._f_getattr('metadata')
        metadata['elec_meters'][key.meter]['disabled'] = True
        group._f_setattr('metadata', metadata)
        store.flush()

    store.close()
