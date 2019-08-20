import unittest
from os.path import join
import numpy as np
import pandas as pd
from datetime import timedelta
from .. import DropoutRate
from ..goodsectionsresults import GoodSectionsResults
from ..totalenergy import _energy_for_power_series
from ... import TimeFrame, ElecMeter, HDFDataStore, MeterGroup
from ...elecmeter import ElecMeterID
from ...consts import JOULES_PER_KWH
from ...tests.testingtools import data_dir

METER_ID = ElecMeterID(instance=1, building=1, dataset='REDD')
METER_ID2 = ElecMeterID(instance=2, building=1, dataset='REDD')

class TestLocateGoodSections(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        filename = join(data_dir(), 'energy_complex.h5')
        cls.datastore = HDFDataStore(filename)
        ElecMeter.load_meter_devices(cls.datastore)
        cls.meter_meta = cls.datastore.load_metadata('building1')['elec_meters'][METER_ID.instance]
        
    @classmethod
    def tearDownClass(cls):
        cls.datastore.close()

    def test_pipeline(self):
        meter = ElecMeter(store=self.datastore, metadata=self.meter_meta,
                          meter_id=METER_ID)
        source_node = meter.get_source_node()
        dropout_rate = DropoutRate(source_node)
        dropout_rate.run()

        # TODO: remove prints and actually test value of dropout rate.
        print(dropout_rate.results)
        print(next(meter.power_series()))

        # Now test metergroup
        meter2 = ElecMeter(store=self.datastore, metadata=self.meter_meta, 
                           meter_id=METER_ID2)
        metergroup = MeterGroup([meter, meter2])
        dr = metergroup.dropout_rate(ignore_gaps=False)        
        print("dr =", dr) # dr = 0.861386138614

        # Test a second time to make sure cache works
        dr_from_cache = metergroup.dropout_rate(ignore_gaps=False)
        self.assertEqual(dr, dr_from_cache)

        metergroup.clear_cache()

if __name__ == '__main__':
    unittest.main()
