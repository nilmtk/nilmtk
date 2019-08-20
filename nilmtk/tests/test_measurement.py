import unittest
import pandas as pd
import numpy as np
import nilmtk.measurement as measure
from nilmtk.elecmeter import ElecMeter, ElecMeterID
from nilmtk.exceptions import MeasurementError

BAD_AC_TYPES = ['foo', '', None, True, {'a':'b'}, 
                (1,2), [], ['reactive'], 'reaactive']

class TestMeasurement(unittest.TestCase):
    def test_check_ac_type(self):
        for ac_type in measure.AC_TYPES:
            measure.check_ac_type(ac_type)
        for bad_ac_type in BAD_AC_TYPES:
            with self.assertRaises(MeasurementError):
                measure.check_ac_type(bad_ac_type)

    def _test_ac_class(self, cls):
        for ac_type in measure.AC_TYPES:
            obj = cls(ac_type=ac_type)
            self.assertIsInstance(obj, cls)
        for bad_ac_type in BAD_AC_TYPES:
            with self.assertRaises(MeasurementError):
                cls(ac_type=bad_ac_type)

    def test_as_dataframe_columns(self):
        N_ROWS = 5
        columns = []

        # Create columns using every permutation of ac_type and cumulative
        for ac_type in measure.AC_TYPES:
            columns.append(('power', ac_type))
            for cumulative in [True, False]:
                if cumulative:
                    columns.append(('cumulative energy', ac_type))
                else:
                    columns.append(('energy', ac_type))
        columns.append(('voltage', ''))

        # Create DataFrame
        N_COLS = len(columns)
        df = pd.DataFrame(np.arange(N_COLS).reshape((1,N_COLS)), 
                          columns=measure.measurement_columns(columns))

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
            {'test model': {'measurements': [{'physical_quantity': 'power', 
                                              'type': 'apparent'}]}})
        meter_id = ElecMeterID(1, 1, 'REDD')
        meter = ElecMeter(metadata={'device_model': 'test model'}, meter_id=meter_id)
        
        self.assertEqual(measure.select_best_ac_type(['reactive'], 
                                                     meter.available_power_ac_types()),
                         'reactive')

if __name__ == '__main__':
    unittest.main()
