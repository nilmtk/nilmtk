#!/usr/bin/python
from __future__ import print_function, division
import unittest
from os.path import join
from .testingtools import data_dir
from nilmtk import Electricity, ElecMeter, HDFDataStore
from nilmtk.elecmeter import ElecMeterID


class TestElectricity(unittest.TestCase):
        
    @classmethod
    def setUpClass(cls):
        filename = join(data_dir(), 'energy.h5')
        cls.datastore = HDFDataStore(filename)
        ElecMeter.load_meter_devices(cls.datastore)

    def test_wiring_graph(self):
        meter1 = ElecMeter(metadata={'site_meter': True}, 
                           meter_id=ElecMeterID(1,1,'REDD'))
        meter2 = ElecMeter(metadata={'submeter_of': 1}, 
                           meter_id=ElecMeterID(2,1,'REDD'))
        meter3 = ElecMeter(metadata={'submeter_of': 2},
                           meter_id=ElecMeterID(3,1,'REDD'))
        elec = Electricity([meter1, meter2, meter3])
        wiring_graph = elec.wiring_graph()
        
        self.assertIs(elec.mains(), meter1)
        self.assertEqual(elec.meters_directly_downstream_of_mains(), [meter2])

    def test_proportion_of_energy_submetered(self):
        meters = []
        for i in [1,2,3]:
            meter_meta = self.datastore.load_metadata('building1')['elec_meters'][i]
            meter_id = ElecMeterID(i, 1, 'REDD')
            meter = ElecMeter(self.datastore, meter_meta, meter_id)
            meters.append(meter)

        mains = meters[0]
        elec = Electricity(meters)
        self.assertEqual(elec.proportion_of_energy_submetered(), 1.0) 
    

if __name__ == '__main__':
    unittest.main()
