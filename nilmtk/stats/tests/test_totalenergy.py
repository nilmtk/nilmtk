import unittest
from ..totalenergy import TotalEnergy, _energy_for_power_series
from ...preprocessing import Clip
from ... import TimeFrame, ElecMeter, HDFDataStore
from ...elecmeter import ElecMeterID
from ...consts import JOULES_PER_KWH
from ...tests.testingtools import data_dir
from os.path import join
import numpy as np
import pandas as pd
from datetime import timedelta
from copy import deepcopy

METER_ID = ElecMeterID(instance=1, building=1, dataset='REDD')

def check_energy_numbers(testcase, energy):
    true_active_kwh =  0.0163888888889
    testcase.assertAlmostEqual(energy['active'], true_active_kwh)
    testcase.assertAlmostEqual(energy['reactive'], true_active_kwh*0.9)
    testcase.assertAlmostEqual(energy['apparent'], true_active_kwh*1.1)


class TestEnergy(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        filename = join(data_dir(), 'energy.h5')
        cls.datastore = HDFDataStore(filename)
        ElecMeter.load_meter_devices(cls.datastore)
        cls.meter_meta = cls.datastore.load_metadata('building1')['elec_meters'][METER_ID.instance]
        
    @classmethod
    def tearDownClass(cls):
        cls.datastore.close()

    def test_energy_per_power_series(self):
        data = np.array([0,  0,  0, 100, 100, 100, 150, 150, 200,   0,   0, 100, 5000,    0])
        secs = np.arange(start=0, stop=len(data)*10, step=10)
        true_kwh = ((data[:-1] * np.diff(secs)) / JOULES_PER_KWH).sum()
        index = [pd.Timestamp('2010-01-01') + timedelta(seconds=int(sec)) for sec in secs]
        df = pd.Series(data=data, index=index)
        kwh = _energy_for_power_series(df, max_sample_period=15)
        self.assertAlmostEqual(true_kwh, kwh)

    def test_pipeline(self):
        meter = ElecMeter(store=self.datastore, 
                          metadata=self.meter_meta, 
                          meter_id=METER_ID)
        source_node = meter.get_source_node()
        clipped = Clip(source_node)
        energy = TotalEnergy(clipped)
        energy.run()
        energy_results = deepcopy(energy.results)
        check_energy_numbers(self, energy_results.combined())

if __name__ == '__main__':
    unittest.main()



