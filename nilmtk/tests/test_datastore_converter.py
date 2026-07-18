from nilmtk.datastore import CSVDataStore, HDFDataStore
from nilmtk.datastore.datastore import convert_datastore
from nilmtk.tests.testingtools import data_dir


def test_convert_random_dataset(tmp_path):
    input_filepath = data_dir() + "/random.h5"
    output_filepath = tmp_path / "random_csv"

    input_store = HDFDataStore(input_filepath)
    output_store = CSVDataStore(str(output_filepath))
    try:
        convert_datastore(input_store, output_store)
    finally:
        input_store.close()
        output_store.close()

    assert (output_filepath / "metadata" / "dataset.yaml").is_file()
    assert (output_filepath / "building1" / "elec" / "meter1.csv").is_file()
