import unittest
from os.path import join
import pandas as pd
from datetime import timedelta
from .testingtools import data_dir, WarningTestMixin
from ..datastore import HDFDataStore
from ..elecmeter import ElecMeter, ElecMeterID
from ..stats.tests.test_totalenergy import check_energy_numbers

METER_ID = ElecMeterID(instance=1, building=1, dataset='REDD')
METER_ID2 = ElecMeterID(instance=2, building=1, dataset='REDD')
METER_ID3 = ElecMeterID(instance=3, building=1, dataset='REDD')

class TestElecMeter(WarningTestMixin, unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        filename = join(data_dir(), 'energy.h5')
        cls.datastore = HDFDataStore(filename)
        ElecMeter.load_meter_devices(cls.datastore)
        cls.meter_meta = cls.datastore.load_metadata('building1')['elec_meters'][METER_ID.instance]

    @classmethod
    def tearDownClass(cls):
        cls.datastore.close()

    def test_load(self):
        meter = ElecMeter(store=self.datastore, metadata=self.meter_meta, 
                          meter_id=METER_ID)
        self.assertEqual(meter.metadata['device_model'], 'Energy Meter')
        self.assertEqual(meter.device['sample_period'], 10)

    def test_total_energy(self):
        meter = ElecMeter(meter_id=METER_ID)
        with self.assertRaises(RuntimeError):
            meter.total_energy()
        meter = ElecMeter(store=self.datastore, metadata=self.meter_meta, 
                          meter_id=METER_ID)
        energy = meter.total_energy()
        check_energy_numbers(self, energy)

        # Check a second time to check cache
        energy = meter.total_energy()
        check_energy_numbers(self, energy)

        meter.clear_cache()

        # Check period_range
        period_index = pd.period_range(start=meter.get_timeframe().start, 
                                       periods=5, freq='D')
        meter.total_energy(sections=period_index, full_results=True)
        
    def test_upstream_meter(self):
        meter1 = ElecMeter(metadata={'site_meter': True}, meter_id=METER_ID)
        self.assertIsNone(meter1.upstream_meter())
        meter2 = ElecMeter(metadata={'submeter_of': 1}, meter_id=METER_ID2)
        self.assertEquals(meter2.upstream_meter(), meter1)
        meter3 = ElecMeter(metadata={'submeter_of': 2}, meter_id=METER_ID3)
        self.assertEquals(meter3.upstream_meter(), meter2)

    def test_proportion_of_energy(self):
        meter = ElecMeter(store=self.datastore, metadata=self.meter_meta, 
                          meter_id=METER_ID)
        self.assertEquals(meter.proportion_of_energy(meter), 1.0)

    def correlation(self):
        meter_1 = ElecMeter(store=self.datastore, metadata=self.meter_meta, 
                          meter_id=METER_ID)
        meter_2 = ElecMeter(store=self.datastore, metadata=self.meter_meta, 
                          meter_id=METER_ID2)

        # Let us get raw DataFrames 
        df1 = self.datastore.store.get('/building1/elec/meter1')
        df2 = self.datastore.store.get('/building1/elec/meter2')

        # Let us compute the value using nilmtk
        corr12_nilmtk = meter_1.correlation(meter_2)
        print("Correlation using nilmtk:", corr12_nilmtk)

        # Let us now compute the value using Pandas functions
        corr12_pandas = df1.corr(df2)
        print("Correlation using pandas:", corr12_pandas)
        from pandas.util.testing import assert_frame_equal
        assert_frame_equal(corr12_nilmtk, corr12_pandas)

        #self.assertEqual(corr12_nilmtk, corr12_pandas)
        #print("Correlation using pandas:", corr12_pandas)


if __name__ == '__main__':
    unittest.main()
