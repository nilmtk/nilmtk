# re-enable deprecation warnings
import warnings
warnings.simplefilter('default')

# Silence ImportWarnings for the time being
warnings.filterwarnings('ignore', category=ImportWarning)

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
    
    #Workaround for open .h5 files on Windows
    from tables.file import _open_files
    _open_files.close_all()
    
    cmd = "git checkout -- {}".format(data_dir())
    try:
        subprocess.check_output(cmd, shell=True, cwd=data_dir())
    except Exception:
        print("Failed to run '{}'".format(cmd))
        raise
    else:
        print("Succeeded in running '{}'".format(cmd))
