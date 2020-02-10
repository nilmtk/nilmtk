import unittest
from ..totalenergyresults import TotalEnergyResults
from ... import TimeFrame

class TestEnergyResults(unittest.TestCase):

    def test_append(self):
        er = TotalEnergyResults()
        tf = TimeFrame('2012-01-01','2012-01-02')
        er.append(tf, {'apparent':40, 'reactive':30, 'active':20})
        self.assertEqual(er._data.index.size, 1)
        self.assertEqual(er._data.index[0], tf.start)
        self.assertEqual(er._data['end'][tf.start], tf.end)
    
    def test_combined(self):
        er = TotalEnergyResults()
        tf = TimeFrame('2012-01-01','2012-01-02')
        er.append(tf, {'apparent':40})
        tf2 = TimeFrame('2012-01-02','2012-01-03')
        er.append(tf2, {'apparent':40, 'reactive':50, 'active':30})
        self.assertEqual(er.combined()['apparent'], 80)
        self.assertEqual(er.combined()['reactive'], 50)
        self.assertEqual(er.combined()['active'], 30)

        # Try a junk measurement name
        tf3 = TimeFrame('2012-01-03','2012-01-04')
        with self.assertRaises(KeyError):
            er.append(tf3, {'blah':40})

        # Try a duplicate start date
        tf4 = TimeFrame('2012-01-01','2012-01-04')
        with self.assertRaises(ValueError):
            er.append(tf4, {'active':20})

        # Try a duplicate end date
        tf5 = TimeFrame('2010-01-01','2012-01-03')
        with self.assertRaises(ValueError):
            er.append(tf5, {'active':20})

        # Try inserting an entry which overlaps
        tf6 = TimeFrame('2012-01-02 06:00', '2012-01-04')
        with self.assertRaises(ValueError):
            er.append(tf6, {'active':20})

        tf7 = TimeFrame('2010-01-02 06:00', '2012-01-04')
        with self.assertRaises(ValueError):
            er.append(tf7, {'active':20})

if __name__ == '__main__':
    unittest.main()
