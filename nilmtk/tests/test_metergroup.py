#!/usr/bin/python
from __future__ import print_function, division
import unittest
from nilmtk import Appliance, MeterGroup, ElectricityMeter, Mains
from nilmtk.utils import tree_root, nodes_adjacent_to_root

class TestMeterGroup(unittest.TestCase):
    def test_getitem(self):
        fridge_meter = ElectricityMeter(1, 1, 'REDD')
        fridge = Appliance(type='fridge', instance=1)
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
        for key in [True, False, 3, (1,2,3), (1), ['fridge']]:
            with self.assertRaises(TypeError):
                mg[key]

    def test_select(self):
        fridge_meter = ElectricityMeter(1, 1, 'REDD')
        fridge = Appliance(type='fridge', instance=1)
        fridge_meter.appliances = [fridge]
        mg = MeterGroup([fridge_meter])

        self.assertEqual(mg.select(category='cold'), mg)
        # TODO: make this test more rigorous!
        
    def test_wiring_graph(self):
        meter1 = ElectricityMeter(1,1,'REDD',metadata={'site_meter': True})
        mains = Mains(1, 'REDD', meters=[meter1])
        meter2 = ElectricityMeter(2,1,'REDD',metadata={'submeter_of': 1})
        meter2.mains = mains
        meter3 = ElectricityMeter(3,1,'REDD',metadata={'submeter_of': 2})
        mg = MeterGroup([meter1, meter2, meter3])
        wiring_graph = mg.wiring_graph()
        self.assertEqual(wiring_graph.nodes(), [meter2, meter3, mains])
        
        self.assertIs(mg.mains(), mains)
        self.assertEqual(mg.meters_directly_downstream_of_mains(), [meter2])
        

if __name__ == '__main__':
    unittest.main()
