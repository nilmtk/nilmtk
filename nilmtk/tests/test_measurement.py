#!/usr/bin/python
from __future__ import print_function, division
import unittest
import pandas as pd
import numpy as np
import nilmtk.measurement as measure
from nilmtk import ElecMeter

BAD_AC_TYPES = ['foo', '', None, True, {'a':'b'}, 
                (1,2), [], ['reactive'], 'reaactive']

class TestMeasurement(unittest.TestCase):
    def test_check_ac_type(self):
        for ac_type in measure.AC_TYPES:
            measure.check_ac_type(ac_type)
        for bad_ac_type in BAD_AC_TYPES:
            with self.assertRaises(ValueError):
                measure.check_ac_type(bad_ac_type)

    def _test_ac_class(self, cls):
        for ac_type in measure.AC_TYPES:
            obj = cls(ac_type=ac_type)
            self.assertIsInstance(obj, cls)
        for bad_ac_type in BAD_AC_TYPES:
            with self.assertRaises(ValueError):
                cls(ac_type=bad_ac_type)

    def test_power_constructor(self):
        self._test_ac_class(measure.Power)

    def test_energy_constructor(self):
        self._test_ac_class(measure.Energy)
        for cumulative in [True, False]:
            energy = measure.Energy('reactive', cumulative=cumulative)
            self.assertIsInstance(energy, measure.Energy)
        for bad_cumulative in ['foo', '', [], [True], ['bar'], (1,2), {'a':'b'}]:
            with self.assertRaises(TypeError):
                measure.Energy('reactive', cumulative=bad_cumulative)

    def test_voltage_constructor(self):
        v = measure.Voltage()
        self.assertIsInstance(v, measure.Voltage)
        with self.assertRaises(TypeError):
            measure.Voltage('blah')

    def test_as_dataframe_columns(self):
        N_ROWS = 5
        columns = []

        # Create columns using every permutation of ac_type and cumulative
        for ac_type in measure.AC_TYPES:
            columns.append(measure.Power(ac_type=ac_type))
            for cumulative in [True, False]:
                columns.append(measure.Energy(ac_type=ac_type, 
                                              cumulative=cumulative))
        columns.append(measure.Voltage())

        # Create DataFrame
        N_COLS = len(columns)
        df = pd.DataFrame(np.arange(N_COLS).reshape((1,N_COLS)), columns=columns)

        # Try accessing columns
        i = 0
        for column in columns:
            series = df[column]
            self.assertIsInstance(series, pd.Series)
            self.assertEqual(series.name, column)
            self.assertEqual(series.shape, (1,))
            self.assertEqual(series.sum(), i)
            i += 1

    def test_select_best_ac_type(self):
        self.assertEqual(measure.select_best_ac_type(['reactive']), 'reactive')

        self.assertEqual(measure.select_best_ac_type(['active', 'reactive', 'apparent']), 'active')

        ElecMeter.meter_devices.update(
            {'test model': {'measurements': [measure.Power('apparent')]}})
        meter = ElecMeter(metadata={'device_model': 'test model',
                                    'dataset': 'REDD', 'building': 1, 'instance': 1})
        
        self.assertEqual(measure.select_best_ac_type(['reactive'], 
                                                     meter.available_ac_types()),
                         'reactive')

if __name__ == '__main__':
    unittest.main()
