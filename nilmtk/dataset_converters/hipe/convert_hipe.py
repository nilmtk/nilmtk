import pandas as pd
import numpy as np
from copy import deepcopy
from os.path import join, isdir, isfile
from os import listdir
import re
import glob
from sys import stdout
from nilmtk.utils import get_datastore
from nilmtk.datastore import Key
from nilmtk.timeframe import TimeFrame
from nilmtk.measurement import LEVEL_NAMES
from nilmtk.utils import get_module_directory, check_directory_exists
from nilm_metadata import convert_yaml_to_hdf5, save_yaml_to_datastore
import sys


def convert_hipe(hipe_path, output_filename, format="HDF"):
    """Convert the HIPE data set to the NILMTK-format. This method works
    with the 1 week and the 3 month data.

    Parameters
    ----------
    hipe_path : str
        The root path of the HIPE dataset.
    output_filename : str
        The destination filename (including path and suffix).
    format : str
        format of output. Either "HDF" or "CSV". Defaults to "HDF".

    """

    datastore = get_datastore(output_filename, format, mode="w")

    _convert(hipe_path, datastore,
             _hipe_measurement_mapping_func, "Europe/Berlin")

    metadata_path = "metadata"

    save_yaml_to_datastore(metadata_path, datastore)

    datastore.close()

    print("Done converting HIPE!")


def _hipe_measurement_mapping_func(chan_id):
    return 'apparent' if chan_id < 2 else 'active'


def _convert(input_path,
             data_store,
             measurement_mapping_func,
             sort_index=True,
             drop_duplicates=False):
    meter_to_machine = {
        1: "MainTerminal",
        2: "ChipPress",
        3: "ChipSaw",
        4: "HighTemperatureOven",
        5: "PickAndPlaceUnit",
        6: "ScreenPrinter",
        7: "SolderingOven",
        8: "VacuumOven",
        9: "VacuumPump1",
        10: "VacuumPump2",
        11: "WashingMachine",
    }

    check_directory_exists(input_path)

    print("Loading factory 1...", end="... ")
    chans = _find_all_channels(input_path, meter_to_machine)
    for chan_id, filename in chans.items():
        print(chan_id, end=" ")
        stdout.flush()
        key = Key(building=1, meter=chan_id)
        measurements = measurement_mapping_func(chan_id)
        df = _load_csv(filename,
                       measurements,
                       sort_index=sort_index,
                       drop_duplicates=drop_duplicates)

        data_store.put(str(key), df)
    print()


def _find_all_channels(input_path, names):
    return {
        key: glob.glob(input_path + "/" + value + "*.csv")[0]
        for (key, value) in names.items()
    }


def _load_csv(filename, measurements, drop_duplicates=False, sort_index=False):
    print(f"Processing {filename}")
    df = pd.read_csv(filename, usecols=["SensorDateTime", "P_kW"], index_col=0)

    df.index = pd.to_datetime(df.index, utc=True)
    df = df.tz_convert("Europe/Berlin")
    df["P_kW"] *= 1000  # convert kW to W

    df.index.name = ""
    df.columns = pd.MultiIndex.from_arrays(
        [["power"], [measurements]]
    )  # weird hack to have the same datastructure as NILMTK with REDD data set
    df.columns.set_names(LEVEL_NAMES, inplace=True)

    if sort_index:
        df = df.sort_index()

    if drop_duplicates:
        dups_in_index = df.index.duplicated(keep='first')
        if dups_in_index.any():
            df = df[~dups_in_index]
    return df.abs()  # only positive loads


def main(file_in, file_out):
    print(f"Output {file_out}")
    convert_hipe(file_in, file_out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
