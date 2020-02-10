import unittest
from os.path import join
import numpy as np
import pandas as pd
from datetime import timedelta
from nilmtk.stats import GoodSections
from nilmtk.stats.goodsectionsresults import GoodSectionsResults
from nilmtk import TimeFrame, ElecMeter, DataSet
from nilmtk.datastore import HDFDataStore
from nilmtk.elecmeter import ElecMeterID
from nilmtk.tests.testingtools import data_dir

METER_ID = ElecMeterID(instance=1, building=1, dataset='REDD')


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
        meter1 = ElecMeter(store=self.datastore, metadata=self.meter_meta,
                           meter_id=METER_ID)

        # load co_test.h5
        dataset = DataSet(join(data_dir(), 'co_test.h5'))
        meter2 = dataset.buildings[1].elec.mains()

        for meter in [meter1, meter2]:
            for chunksize in [None, 2**10, 2**29]:
                if chunksize is None:
                    kwargs = {}
                else:
                    kwargs = {'chunksize': chunksize}

                source_node = meter.get_source_node(**kwargs)
                good_sections = GoodSections(source_node)
                good_sections.run()
                combined = good_sections.results.simple()
                meter.clear_cache()
                meter.good_sections(**kwargs)
                meter.good_sections(**kwargs)
                meter.clear_cache()

        dataset.store.close()

    def test_process_chunk(self):
        MAX_SAMPLE_PERIOD = 10
        metadata = {'device': {'max_sample_period': MAX_SAMPLE_PERIOD}}
        #       0  1  2  3    4  5     6     7
        secs = [0,10,20,30,  50,60,  100,  200,

        #         8   9  10  11  12  13    14  15  16
                250,260,270,280,290,300,  350,360,370]
        index = pd.DatetimeIndex([pd.Timestamp('2011-01-01 00:00:00') +
                                  timedelta(seconds=sec) for sec in secs])
        df = pd.DataFrame(data=np.random.randn(len(index), 3), index=index,
                          columns=['a', 'b', 'c'])
        df.timeframe = TimeFrame(index[0], index[-1])
        df.look_ahead = pd.DataFrame()

        locate = GoodSections()
        locate.results = GoodSectionsResults(MAX_SAMPLE_PERIOD)
        locate._process_chunk(df, metadata)
        results = locate.results.combined()
        self.assertEqual(len(results), 4)
        self.assertAlmostEqual(results[0].timedelta.total_seconds(), 30)
        self.assertEqual(results[1].timedelta.total_seconds(), 10)
        self.assertEqual(results[2].timedelta.total_seconds(), 50)
        self.assertEqual(results[3].timedelta.total_seconds(), 20)

        # Now try splitting data into multiple chunks
        timestamps = [
            pd.Timestamp("2011-01-01 00:00:00"),
            pd.Timestamp("2011-01-01 00:00:40"),
            pd.Timestamp("2011-01-01 00:01:20"),
            pd.Timestamp("2011-01-01 00:04:20"),
            pd.Timestamp("2011-01-01 00:06:20")
        ]
        for split_point in [[4, 6, 9, 17], [4, 10, 12, 17]]:
            locate = GoodSections()
            locate.results = GoodSectionsResults(MAX_SAMPLE_PERIOD)
            df.results = {}
            prev_i = 0
            for j, i in enumerate(split_point):
                cropped_df = df.iloc[prev_i:i]
                cropped_df.timeframe = TimeFrame(timestamps[j],
                                                 timestamps[j+1])
                try:
                    cropped_df.look_ahead = df.iloc[i:]
                except IndexError:
                    cropped_df.look_ahead = pd.DataFrame()
                prev_i = i
                locate._process_chunk(cropped_df, metadata)

            results = locate.results.combined()
            self.assertEqual(len(results), 4)
            self.assertAlmostEqual(results[0].timedelta.total_seconds(), 30)
            self.assertEqual(results[1].timedelta.total_seconds(), 10)
            self.assertEqual(results[2].timedelta.total_seconds(), 50)
            self.assertEqual(results[3].timedelta.total_seconds(), 20)


if __name__ == '__main__':
    unittest.main()
