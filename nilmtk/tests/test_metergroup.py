#!/usr/bin/python
from __future__ import print_function, division
import unittest
from os.path import join
from .testingtools import data_dir
from nilmtk import Appliance, MeterGroup, ElecMeter, HDFDataStore
from nilmtk.utils import tree_root, nodes_adjacent_to_root
from nilmtk.elecmeter import ElecMeterID

class TestMeterGroup(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        filename = join(data_dir(), 'energy.h5')
        cls.datastore = HDFDataStore(filename)
        ElecMeter.load_meter_devices(cls.datastore)

    def test_getitem(self):
        fridge_meter = ElecMeter()
        fridge = Appliance({'type':'fridge', 'instance':1})
        fridge_meter.appliances = [fridge]
        mg = MeterGroup([fridge_meter])

        # test good keys
        for key in ['fridge', ('fridge', 1), {'type':'fridge'}, 
                    {'type':'fridge', 'instance': 1}]:
            self.assertEqual(mg[key], fridge_meter)
        
        # test bad key values
        for key in ['foo', ('foo', 2), ('fridge', 2), 
                    {'type':'fridge', 'instance': -12}]:
            with self.assertRaises(KeyError):
                mg[key]

        # test bad key types
        for key in [True, False, (1,2,3), ['fridge']]:
            with self.assertRaises(TypeError):
                mg[key]

    def test_select(self):
        fridge_meter = ElecMeter()
        fridge = Appliance({'type':'fridge', 'instance':1})
        fridge_meter.appliances = [fridge]
        mg = MeterGroup([fridge_meter])

        self.assertEqual(mg.select(category='cold'), mg)
        # TODO: make this test more rigorous!
        
    def test_wiring_graph(self):
        meter1 = ElecMeter(metadata={'site_meter': True}, 
                           meter_id=ElecMeterID(1,1,'REDD'))
        meter2 = ElecMeter(metadata={'submeter_of': 1},
                           meter_id=ElecMeterID(2,1,'REDD'))
        meter3 = ElecMeter(metadata={'submeter_of': 2},
                           meter_id=ElecMeterID(3,1,'REDD'))
        mg = MeterGroup([meter1, meter2, meter3])
        wiring_graph = mg.wiring_graph()
        self.assertEqual(wiring_graph.nodes(), [meter2, meter3, meter1])

    def test_wiring_graph(self):
        meter1 = ElecMeter(metadata={'site_meter': True}, 
                           meter_id=ElecMeterID(1,1,'REDD'))
        meter2 = ElecMeter(metadata={'submeter_of': 1}, 
                           meter_id=ElecMeterID(2,1,'REDD'))
        meter3 = ElecMeter(metadata={'submeter_of': 2},
                           meter_id=ElecMeterID(3,1,'REDD'))
        mg = MeterGroup([meter1, meter2, meter3])
        wiring_graph = mg.wiring_graph()
        
        self.assertIs(mg.mains(), meter1)
        self.assertEqual(mg.meters_directly_downstream_of_mains(), [meter2])

    def test_proportion_of_energy_submetered(self):
        meters = []
        for i in [1,2,3]:
            meter_meta = self.datastore.load_metadata('building1')['elec_meters'][i]
            meter_id = ElecMeterID(i, 1, 'REDD')
            meter = ElecMeter(self.datastore, meter_meta, meter_id)
            meters.append(meter)

        mains = meters[0]
        mg = MeterGroup(meters)
        self.assertEqual(mg.proportion_of_energy_submetered(), 1.0) 


if __name__ == '__main__':
    unittest.main()
