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

def teardown_package():
    """Nosetests package teardown function (run when tests are done).
    See http://nose.readthedocs.org/en/latest/writing_tests.html#test-packages

    Uses git to reset data_dir after tests have run.
    """
    from nilmtk.tests.testingtools import data_dir
    import subprocess
    cmd = "cd {data_dir};git checkout -- {data_dir}".format(data_dir=data_dir())
    output = subprocess.check_output(cmd, shell=True)
    if output:
        raise RuntimeError("Attempt to run '{}' failed with this output: '{}'"
                           .format(cmd, output))
    else:
        print "Succeeded in running '{}'".format(cmd)
