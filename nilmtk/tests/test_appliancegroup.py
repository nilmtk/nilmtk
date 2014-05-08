#!/usr/bin/python
from __future__ import print_function, division
import unittest
from nilmtk import Appliance, ApplianceGroup

class TestApplianceGroup(unittest.TestCase):
    def test_getitem(self):
        fridge = Appliance(type='fridge', instance=1)
        ag = ApplianceGroup([fridge])

        # test good keys
        for key in ['fridge', ('fridge', 1), {'type':'fridge'}, 
                    {'type':'fridge', 'instance': 1}]:
            self.assertEqual(ag[key], fridge)
        
        # test bad key values
        for key in ['foo', ('foo', 2), ('fridge', 2), 
                    {'type':'fridge', 'instance': -12}]:
            with self.assertRaises(KeyError):
                ag[key]

        # test bad key types
        for key in [True, False, 3, (1,2,3), (1), ['fridge']]:
            with self.assertRaises(TypeError):
                ag[key]

    def test_select(self):
        fridge = Appliance(type='fridge', instance=1)
        ag = ApplianceGroup([fridge])

        self.assertEqual(ag.select(category='cold'), ag)
        # TODO: make this test more rigorous!
        

if __name__ == '__main__':
    unittest.main()
