# re-enable deprecation warnings
import warnings
warnings.simplefilter('default')

from nilmtk import *
from nilmtk.version import version as __version__
from nilmtk.timeframe import TimeFrame
from nilmtk.elecmeter import ElecMeter
from nilmtk.datastore import DataStore, HDFDataStore, CSVDataStore, Key
from nilmtk.metergroup import MeterGroup
from nilmtk.appliance import Appliance
from nilmtk.building import Building
from nilmtk.dataset import DataSet

global_meter_group = MeterGroup()


def setup_package():
    """Nosetests package setup function (run before any tests are run).
    See http://nose.readthedocs.org/en/latest/writing_tests.html#test-packages
    """
    from nilmtk.tests.testingtools import data_dir
    from os.path import join
    from glob import glob
    import shutil

    read_only_data = join(data_dir(), 'read_only', '*.h5')
    for filename in glob(read_only_data):
        print "copying", filename, "to", data_dir()
        shutil.copy(filename, data_dir())


def teardown_package():
    """Nosetests package teardown function (run when tests are done).
    See http://nose.readthedocs.org/en/latest/writing_tests.html#test-packages
    """
    from nilmtk.tests.testingtools import data_dir
    from os.path import join
    from glob import glob
    import shutil, os

    data = join(data_dir(), '*.h5')
    for filename in glob(data):
        print "removing", filename
        os.remove(filename)
