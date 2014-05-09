#!/usr/bin/python
from __future__ import print_function, division
import unittest
from nilmtk import Appliance, MeterGroup, EMeter

class TestMeterGroup(unittest.TestCase):
    def test_getitem(self):
        fridge_meter = EMeter()
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
        fridge_meter = EMeter()
        fridge = Appliance(type='fridge', instance=1)
        fridge_meter.appliances = [fridge]
        mg = MeterGroup([fridge_meter])

        self.assertEqual(mg.select(category='cold'), mg)
        # TODO: make this test more rigorous!
        

if __name__ == '__main__':
    unittest.main()
