#!/usr/bin/python
from __future__ import print_function, division
import unittest
from nilmtk import Electricity, ElectricityMeter, Mains

class TestElectricity(unittest.TestCase):
        
    def test_wiring_graph(self):
        meter1 = ElectricityMeter(1,1,'REDD',metadata={'site_meter': True})
        mains = Mains(1, 'REDD', meters=[meter1])
        meter2 = ElectricityMeter(2,1,'REDD',metadata={'submeter_of': 1})
        meter2.mains = mains
        meter3 = ElectricityMeter(3,1,'REDD',metadata={'submeter_of': 2})
        elec = Electricity([meter1, meter2, meter3])
        wiring_graph = elec.wiring_graph()
        
        self.assertIs(elec.mains(), mains)
        self.assertEqual(elec.meters_directly_downstream_of_mains(), [meter2])
        

    

if __name__ == '__main__':
    unittest.main()
