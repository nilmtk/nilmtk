import unittest
from os.path import join
from nilmtk.tests.testingtools import data_dir
from nilmtk import (Appliance, MeterGroup, ElecMeter, HDFDataStore, 
                    global_meter_group, TimeFrame, DataSet)
from nilmtk.utils import tree_root, nodes_adjacent_to_root
from nilmtk.elecmeter import ElecMeterID
from nilmtk.building import BuildingID
from nilmtk.legacy.disaggregate import CombinatorialOptimisation
from nilmtk.metrics import f1_score

class TestMetrics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        filename = join(data_dir(), 'co_test.h5')
        cls.dataset = DataSet(filename)
        
    @classmethod
    def tearDownClass(cls):
        cls.dataset.store.close()

    def test_f1(self):
        pass
        # The code below doesn't work yet because it complains that
        # AttributeError: Attribute 'metadata' does not exist in node: '/'
        """
        co = CombinatorialOptimisation()
        co.train(self.dataset.buildings[1].elec)
        disag_filename = join(data_dir(), 'co-disag.h5')
        output = HDFDataStore(disag_filename, 'w')
        co.disaggregate(self.dataset.buildings[1].elec.mains(), output)
        output.close()
        disag = DataSet(disag_filename)
        disag_elec = disag.buildings[1].elec
        f1 = f1_score(disag_elec, self.dataset.buildings[1].elec)
        """

if __name__ == '__main__':
    unittest.main()
