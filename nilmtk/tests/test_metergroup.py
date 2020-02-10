import unittest
from os.path import join
from nilmtk.tests.testingtools import data_dir
from nilmtk import (Appliance, MeterGroup, ElecMeter, HDFDataStore, 
                    global_meter_group, TimeFrame, DataSet)
from nilmtk.utils import tree_root, nodes_adjacent_to_root
from nilmtk.elecmeter import ElecMeterID
from nilmtk.building import BuildingID

class TestMeterGroup(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        filename = join(data_dir(), 'energy.h5')
        cls.datastore = HDFDataStore(filename)
        ElecMeter.load_meter_devices(cls.datastore)
        
    @classmethod
    def tearDownClass(cls):
        cls.datastore.close()

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
        for key in [True, False, ['fridge']]:
            with self.assertRaises(TypeError):
                mg[key]

    def test_select(self):
        fridge_meter = ElecMeter()
        fridge = Appliance({'type':'fridge', 'instance':1})
        fridge_meter.appliances = [fridge]
        mg = MeterGroup([fridge_meter])

        self.assertEqual(mg.select_using_appliances(category='cold'), mg)
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
        
        self.assertIs(mg.mains(), meter1)
        self.assertEqual(mg.meters_directly_downstream_of_mains().meters, [meter2])
        self.assertEqual(list(wiring_graph.nodes()), [meter1, meter2, meter3])

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
        # Check a second time to check cache works
        self.assertEqual(mg.proportion_of_energy_submetered(), 1.0) 
        mg.clear_cache()

    def test_dual_supply(self):
        elec_meters = {1: {'data_location': '/building1/elec/meter1',
                           'device_model': 'Energy Meter'},
                       2: {'data_location': '/building1/elec/meter1',
                           'device_model': 'Energy Meter'},
                       3: {'data_location': '/building1/elec/meter1',
                           'device_model': 'Energy Meter'}}

        appliances = [{'type': 'washer dryer', 'instance': 1, 'meters': [1,2]},
                      {'type': 'fridge', 'instance': 1, 'meters': [3]}]
        mg = MeterGroup()
        mg.import_metadata(self.datastore, elec_meters, appliances, BuildingID(1, 'REDD'))
        self.assertEqual(mg['washer dryer'].total_energy()['active'], 
                         mg['fridge'].total_energy()['active'] * 2)

        # Test total_energy a second time to check cache works
        self.assertEqual(mg['washer dryer'].total_energy()['active'], 
                         mg['fridge'].total_energy()['active'] * 2)

        self.assertIsInstance(mg['washer dryer'], MeterGroup)
        self.assertIsInstance(mg['fridge'], ElecMeter)
        mg.clear_cache()

    def test_from_list(self):
        meters = []
        for i in range(1,6):
            meters.append(ElecMeter(meter_id=ElecMeterID(i, 1, None)))
            
        mg = global_meter_group.from_list([
            ElecMeterID(1,1,None),
            (
                ElecMeterID(2,1,None), 
                (ElecMeterID(3,1,None), ElecMeterID(4,1,None), ElecMeterID(5,1,None))
            )
        ])
        self.assertEqual(mg.meters[0], meters[0])
        self.assertEqual(mg.meters[1].meters[0], meters[1])
        self.assertEqual(len(mg.meters[1].meters[1].meters), 3)
        self.assertEqual(len(mg.meters), 2)

    def test_full_results_with_no_sections_raises_runtime_error(self):
        mg = MeterGroup([ElecMeter(), ElecMeter()])
        with self.assertRaises(RuntimeError):
            mg.dropout_rate(full_results=True)

    def test_total_energy(self):
        filename = join(data_dir(), 'random.h5')
        ds = DataSet(filename)
        ds.buildings[1].elec.total_energy()
        ds.buildings[1].elec.total_energy() # test cache
        ds.buildings[1].elec.clear_cache()
        ds.store.close()

    def test_load(self):
        filename = join(data_dir(), 'energy.h5')
        ds = DataSet(filename)
        elec = ds.buildings[1].elec
        df = next(elec.load())
        self.assertEqual(len(df), 13)
        df = next(elec.load(chunksize=5))
        self.assertEqual(len(df), 5)
        df = next(elec.load(physical_quantity='energy'))
        self.assertEqual(len(df), 13)
        self.assertEqual(df.columns.levels, [['energy'], ['reactive']])
        df = next(elec.load(ac_type='active'))
        self.assertEqual(df.columns.levels, [['power'], ['active']])
        

if __name__ == '__main__':
    unittest.main()
