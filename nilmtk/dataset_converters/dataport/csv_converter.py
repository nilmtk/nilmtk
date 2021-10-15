"""
This module should be used when you have no access to the SQL database of Pecan
Street, e.g. for university access.
To use this script, you have to first extract the CSV files from their TAR
archives and to download the `metadata.csv` file.

NB: If you have direct access to the SQL database, please prefer using the other
converter named `download_dataport.py`.
"""

import nilmtk.datastore
import nilmtk.measurement
import numpy as np
import os
import os.path
import pandas as pd
import re
import shutil
import tempfile
import yaml

from collections import OrderedDict
from nilm_metadata import convert_yaml_to_hdf5
from nilmtk.utils import get_module_directory


DATA_AVAILABILITIES = [ "97%", "98%", "99%", "100%" ]
METADATA_COLS = [
        "dataid", "building_type", "city", "state", "house_construction_year",
]
DATABASE_TZ = "US/Central"
METER_COLS = [ ("power", "active") ]
COL_MAPPING = {
        "air1": {"type": "air conditioner"},
        "air2": {"type": "air conditioner"},
        "air3": {"type": "air conditioner"},
        "airwindowunit1": {"type": "air conditioner"},
        "aquarium1": {"type": "appliance"},
        "bathroom1": {"type": "sockets", "room": "bathroom"},
        "bathroom2": {"type": "sockets", "room": "bathroom"},
        "bedroom1": {"type": "sockets", "room": "bedroom"},
        "bedroom2": {"type": "sockets", "room": "bedroom"},
        "bedroom3": {"type": "sockets", "room": "bedroom"},
        "bedroom4": {"type": "sockets", "room": "bedroom"},
        "bedroom5": {"type": "sockets", "room": "bedroom"},
        "car1": {"type": "electric vehicle"},
        "car2": {"type": "electric vehicle"},
        "clotheswasher1": {"type": "washing machine"},
        "clotheswasher_dryg1": {"type": "washer dryer"},
        "diningroom1": {"type": "sockets", "room": "dining room"},
        "diningroom2": {"type": "sockets", "room": "dining room"},
        "dishwasher1": {"type": "dish washer"},
        "disposal1": {"type": "waste disposal unit"},
        "drye1": {"type": "spin dryer"},
        "dryg1": {"type": "spin dryer"},
        "freezer1": {"type": "freezer"},
        "furnace1": {"type": "electric furnace"},
        "furnace2": {"type": "electric furnace"},
        "garage1": {"type": "sockets", "room": "garage"},
        "garage2": {"type": "sockets", "room": "garage"},
        "grid": {},
        "heater1": {"type": "electric space heater"},
        "heater2": {"type": "electric space heater"},
        "heater3": {"type": "electric space heater"},
        "housefan1": {"type": "fan"},
        "jacuzzi1": {"type": "electric hot tub heater"},
        "kitchen1": {"type": "sockets", "room": "kitchen"},
        "kitchen2": {"type": "sockets", "room": "kitchen"},
        "kitchenapp1": {"type": "sockets", "room": "kitchen"},
        "kitchenapp2": {"type": "sockets", "room": "kitchen"},
        "lights_plugs1": {"type": "light"},
        "lights_plugs2": {"type": "light"},
        "lights_plugs3": {"type": "light"},
        "lights_plugs4": {"type": "light"},
        "lights_plugs5": {"type": "light"},
        "lights_plugs6": {"type": "light"},
        "livingroom1": {"type": "sockets", "room": "living room"},
        "livingroom2": {"type": "sockets", "room": "living room"},
        "microwave1": {"type": "microwave"},
        "office1": {"type": "sockets", "room": "office"},
        "outsidelights_plugs1": {"type": "sockets", "room": "outside"},
        "outsidelights_plugs2": {"type": "sockets", "room": "outside"},
        "oven1": {"type": "oven"},
        "oven2": {"type": "oven"},
        "pool1": {"type": "electric swimming pool heater"},
        "pool2": {"type": "electric swimming pool heater"},
        "poollight1": {"type": "light"},
        "poolpump1": {"type": "swimming pool pump"},
        "pump1": {"type": "water pump"},
        "range1": {"type": "stove"},
        "refrigerator1": {"type": "fridge"},
        "refrigerator2": {"type": "fridge"},
        "security1": {"type": "security alarm"},
        "sewerpump1": {"type": "water pump"},
        "shed1": {"type": "sockets", "room": "shed"},
        "solar": {},
        "solar2": {},
        "sprinkler1": {"type": "garden sprinkler"},
        "sumppump1": {"type": "water pump", "room": "basement"},
        "utilityroom1": {"type": "sockets", "room": "utility room"},
        "venthood1": {"type": "stove"},
        "waterheater1": {"type": "electric water heating appliance"},
        "waterheater2": {"type": "electric water heating appliance"},
        "winecooler1": {"type": "cold appliance"},
        "wellpump1": {"type": "water pump"},
}
# Unused variable, all those fields need a mapping in nilm_metadata.
NEED_MAPPING = [ "battery1", "circpump1", "icemaker1", "solar", "solar2", ]


def load_dataport_metadata(csv_metadata_path):
    """
    Read Dataport static metadata.csv and parse building metadata.
    This function filters buildings with:
        1. sufficient data availability (see DATA_AVAILABILITIES variable),
        2. site metering,
        3. in the given state(s).

    Parameters
    ----------
    csv_metadata_path: str
        Path to the metadata.csv file.

    Returns
    -------
    pandas.DataFrame
        Matrix of the buildings and their characteristics,
        see METADATA_COLS variable.
    """
    metadata = pd.read_csv(csv_metadata_path,
                           engine="c", encoding="ISO-8859-1",
                           skiprows=[1]) # Skip header description
    buildings = metadata[
            metadata.egauge_1s_data_availability.isin(DATA_AVAILABILITIES) &
            metadata.grid.eq("yes")
    ].sort_values(by="dataid")
    buildings.reset_index(drop=True, inplace=True)
    return buildings[[*METADATA_COLS]]


def create_tmp_metadata_dir():
    """
    Create an OS-aware temporary metadata directory.
    dataset.yaml and meter_devices.yaml are static metadata contained by NILMTK.
    building<x>.yaml are dynamic, however and must be procedurally generated.

    Returns
    -------
    str
        Path to the temporary metadata directory.
    """
    nilmtk_static_metadata = os.path.join(
            get_module_directory(), 'dataset_converters', 'dataport', 'metadata')
    tmp_dir = tempfile.mkdtemp()
    metadata_dir = os.path.join(tmp_dir, "metadata")
    shutil.copytree(nilmtk_static_metadata, metadata_dir)
    print("Using temporary dir for metadata:", metadata_dir)
    # Clear dynamic metadata (if any)
    for f in os.listdir(metadata_dir):
        if re.search('^building', f):
            os.remove(join(metadata_dir, f))

    return metadata_dir


def preprocess_chunk(chunk_data, b_metadata):
    """
    Clean a chunk of data coming from a Dataport CSV.
    Drop NaN columns and all columns that are not in the COL_MAPPING variable.
    Compute the electrical consumption based on grid and photovoltaic readings.
    Dataport readings are in kW. They are converted in W.

    Parameters
    ----------
    chunk_data: pandas.DataFrame
        Matrix of readings with one appliance per column and the "grid" column
        for the total consumption and production of the building.
    b_metadata: dict
        Building metdata, compatible with nilmtk_metadata.
        This variable is modified by the present function.

    Returns
    -------
    pandas.DataFrame
        Matrix of the reading in Watts with a new column "use" replacing the
        "grid" and "solar" ones.
    """
    cleaned = chunk_data.dropna(axis=1, how="all")
    cols_to_drop = [ c for c in cleaned.columns if c not in COL_MAPPING ]
    cleaned = cleaned.drop(axis=1, labels=cols_to_drop)
    if "solar" in cleaned.columns or "solar2" in cleaned.columns:
        gen = pd.Series(data=np.zeros(cleaned["grid"].shape), index=cleaned.index)
        b_metadata["energy_improvements"] = [ "photovoltaics" ]

        if "solar" in cleaned.columns:
            gen += cleaned["solar"]
            cleaned.drop("solar", axis=1, inplace=True)

        if "solar2" in cleaned.columns:
            gen += cleaned["solar2"]
            cleaned.drop("solar2", axis=1, inplace=True)

        use = cleaned["grid"] + gen
        use.name = "use"
        cleaned = cleaned.join(use)
        cleaned.drop("grid", axis=1, inplace=True)
    else:
        cleaned.rename(columns={ "grid": "use" }, inplace=True)

    return cleaned * 1000


def create_nilmtk_metadata(building_id, dataport_metadata):
    """
    Create metadata for a single building based on the nilm_metadata format.

    Parameters
    ----------
    building_id: int
        Dataport identifier for the building ("dataid" field).
    dataport_metadata: pandas.DataFrame
        Output of `load_dataport_metadata`. Handy, isn't it?
    """
    locality = ", ".join(
            dataport_metadata.loc[dataport_metadata.dataid==building_id, "city"].tolist() \
            + dataport_metadata.loc[dataport_metadata.dataid==building_id, "state"].tolist()
    )
    building_metadata = {
            "original_name": int(building_id),
            "elec_meters": {},
            "appliances": [],
            "geo_location": {
                "locality": locality,
                "country": "US",
            },
    }
    return building_metadata


def extract_building_data(csv_filename, b_id, b_metadata, chunksize, csv_b_cache):
    """
    Extract and clean the data from a single CSV file for a single building.

    Parameters
    ----------
    csv_filename: str
        Path to a single Dataport CSV data file.
    b_id: int
        Dataport building identifier ("dataid" field).
    b_metadata: dict
        Building metdata, compatible with nilmtk_metadata.
        This variable is modified by the present function.
    chunksize: int
        Number of CSV rows to load in memory for processing.
    csv_b_cache: dict of set
        Cache of the unique buildings per CSV file.

    Returns
    -------
    pandas.DataFrame
        A matrix of measurement indexed with increasing timestamp and with each
        meter as a column.
    """
    csv_data_gen = pd.read_csv(
            csv_filename,
            engine="c", chunksize=chunksize,
            index_col=[ "localminute" ]
    )
    b_isnew = True
    b_data = pd.DataFrame()
    csv_isnew = csv_filename not in csv_b_cache
    if csv_isnew:
        csv_b_cache[csv_filename] = set()

    for i, csv_chunk in enumerate(csv_data_gen):
        # Update cache if needed.
        b_list = set(csv_chunk["dataid"].values)
        if csv_isnew:
            csv_b_cache[csv_filename].update(b_list)

        msg = "\tChunk {}, extracting...".format(i)
        if b_id in b_list:
            if b_isnew:
                print("Building {} found in {}!".format(b_id, csv_filename))
                b_isnew = False

            print(msg, end='\r')
            b_chunk = csv_chunk.loc[csv_chunk.dataid.eq(b_id)].copy()
            b_chunk.index = pd.to_datetime(b_chunk.index, utc=True, infer_datetime_format=True)
            b_chunk = b_chunk.tz_convert(DATABASE_TZ)
            b_chunk = preprocess_chunk(b_chunk, b_metadata)
            if b_data.empty:
                b_data = b_chunk
            else:
                b_data = b_data.append(b_chunk)

    print("\t" + " " * len(msg), end='\r')
    if not b_data.empty:
        msg = "\tSorting data..."
        print(msg, end='\r')
        # mergesort is 30% faster than quicksort in this case.
        b_data.sort_index(kind="mergesort", inplace=True)

    print("\t" + " " * len(msg), end='\r')
    return b_data


def write_meter_data(hdf_store, b_id, b_data, nilmtk_metadata):
    """
    Write meter data and generate meter and appliance data from a measurement
    matrix.

    Note:   pandas.HDFStore.put is not thread-safe.
            (see https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html#caveats)
            Parallel implementation should include a single writer process.

    Parameters
    ----------
    hdf_store: pandas.HDFStore
        HDF file descriptor.
        Opening and closing the descriptor is left to the caller function.
    b_id: int
        Dataport building identifier ("dataid" field).
    b_data: pandas.DataFrame
        Matrix of measurements indexed by timestamp with column-wise meter data.
    nilmtk_metadata: collections.OrderedDict
        Already initialized dictionary to store NILM metadata associated with
        buildings, meters and appliances.
        This variable is changed by the present function!
    """
    if "instance" in nilmtk_metadata[b_id]:
        nilmtk_b_id = nilmtk_metadata[b_id]["instance"]
    else:
        nilmtk_b_id = tuple(nilmtk_metadata).index(b_id) + 1
        nilmtk_metadata[b_id]["instance"] = nilmtk_b_id

    for m_id, meter in enumerate(b_data.columns):
        key = nilmtk.datastore.Key(building=nilmtk_b_id, meter=m_id + 1)
        msg = "\tWriting {}".format(str(key))
        print(msg, end='\r')
        # Meter data.
        m_data = pd.DataFrame(b_data[meter])
        m_data.columns = pd.MultiIndex.from_tuples(METER_COLS)
        m_data.columns.set_names(nilmtk.measurement.LEVEL_NAMES, inplace=True)
        hdf_store.put(str(key), m_data, format="table", append=True)
        hdf_store.flush()
        # Meter & appliance metadata
        if meter == 'use':
            meter_metadata = { "device_model": "eGauge", "site_meter": True }
        else:
            meter_metadata = { "device_model": "eGauge", "submeter_of": 0 }
            app_metadata = { "original_name": meter, "meters": [ m_id + 1, ] }
            app_metadata.update(COL_MAPPING[meter])
            app_id = 1
            app_type = app_metadata["type"]
            for other in nilmtk_metadata[b_id]["appliances"]:
                if other["type"] == app_type:
                    app_id += 1

            app_metadata["instance"] = app_id
            nilmtk_metadata[b_id]["appliances"].append(app_metadata)

        nilmtk_metadata[b_id]["elec_meters"][m_id + 1] = meter_metadata
        print("\t" + " " * len(msg), end='\r')

    print("\tWriting done.")


def convert_dataport(csv_filenames, metadata_path, hdf_filename, chunksize=1e6):
    """
    Sequentially convert Dataport data and metadata to a NILMTK-compatible HDF
    file.
    All CSV files given as input will be converted in a single HDF file.
    This function uses a cache to speed up the loading.
    Expect the first pass to be slow, as the cache is not yet effective.
    The convertion of the data from Austin and New York lasts one day.

    Note:   It would have been more computationally efficient to load each CSV
            file once and convert its data.
            However, the CSV rows are not sorted by increasing timestamp.
            Besides, the CSV are too big to fit in memory.
            Filenames follow a chronological order, though.
            Conclusion: we have to load the CSV files as many times as there
            are buildings.

    Parameters
    ----------
    csv_filenames: list of str
        Array of path to the Dataport CSV data files to include in the dataset.
    metadata_path: str
        Path to the Dataport "metadata.csv" containing the building info.
    hdf_filename: str
        Path of the HDF dataset file that will be created.
    chunksize: int
        Number of CSV rows to load in memory for processing.
        Adapt according to the available memory.
        Defaults to 1 million rows.
    """
    csv_filenames.sort()
    csv_building_cache = {} # cache of the building list per file to speed up the search
    store = pd.HDFStore(hdf_filename, "w", complevel=9, complib="zlib")
    metadata_dir = create_tmp_metadata_dir()
    nilmtk_metadata = OrderedDict()
    dataport_metadata = load_dataport_metadata(metadata_path)
    dataids = [ int(i) for i in dataport_metadata["dataid"].values ]
    for b_id in dataids:
        b_metadata = create_nilmtk_metadata(b_id, dataport_metadata)
        for csv_file in csv_filenames:
            print("Looking for building {}".format(b_id), end='\r')
            if csv_file not in csv_building_cache or b_id in csv_building_cache[csv_file]:
                b_data = extract_building_data(
                         csv_file, b_id, b_metadata, chunksize, csv_building_cache)

                if not b_data.empty:
                    nilmtk_metadata[b_id] = b_metadata
                    write_meter_data(store, b_id, b_data, nilmtk_metadata)

    store.close()
    for b_metadata in nilmtk_metadata.values():
        b_name = "building{:d}.yaml".format(b_metadata["instance"])
        yml_filename = os.path.join(metadata_dir, b_name)
        with open(yml_filename, "w") as yml_file:
            yml_file.write(yaml.dump(b_metadata))

    convert_yaml_to_hdf5(metadata_dir, hdf_filename)


if __name__ == "__main__":
    csvs = [
            "1s_data_austin_file1/1s_data_austin_file1.csv",
            "1s_data_austin_file2/1s_data_austin_file2.csv",
            "1s_data_austin_file3/1s_data_austin_file3.csv",
            "1s_data_austin_file4/1s_data_austin_file4.csv",
            "1s_data_newyork_file1/1s_data_newyork_file1.csv",
            "1s_data_newyork_file2/1s_data_newyork_file2.csv",
            "1s_data_newyork_file3/1s_data_newyork_file3.csv",
            "1s_data_newyork_file4/1s_data_newyork_file4.csv",
    ]
    convert_dataport(csvs, "metadata.csv", "dataport.h5", 3e6)

