import os
import shutil

from nilmtk.datastore import HDFDataStore, CSVDataStore
from nilmtk.datastore.datastore import convert_datastore


def test_convert_random_dataset():
    input_filepath = 'data/random.h5'
    output_filepath = 'data/random_csv'

    if os.path.isdir(output_filepath):
        shutil.rmtree(output_filepath)

    input_store = HDFDataStore(input_filepath)
    output_store = CSVDataStore(output_filepath)

    convert_datastore(input_store, output_store)

    input_store.close()
    output_store.close()
