import unittest
import pandas as pd
from nilmtk.timeframe import TimeFrame, merge_timeframes


class TestTimeFrame(unittest.TestCase):
    def test_date_setting(self):
        TimeFrame()
        TimeFrame("2012-01-01", "2013-01-01")

        # test identical start and end dates
        with self.assertRaises(ValueError):
            TimeFrame("2012-01-01", "2012-01-01")

        TimeFrame(start="2011-01-01")
        TimeFrame(end="2011-01-01")

        # test end date after start date
        with self.assertRaises(ValueError):
            TimeFrame("2012-01-01", "2011-01-01")

        tf = TimeFrame()
        tf.end = "2011-01-01"
        tf.start = "2010-01-01"
        with self.assertRaises(ValueError):
            tf.start = "2012-01-01"

    def test_time_delta(self):
        tf = TimeFrame("2012-01-01 00:00:00", "2013-01-01 00:00:00")
        self.assertAlmostEqual(tf.timedelta.total_seconds(), 60*60*24*366)

    def test_intersection(self):
        tf = TimeFrame("2012-01-01 00:00:00", "2013-01-01 00:00:00")
        self.assertFalse(tf.empty)

        new_tf = tf.intersection(tf)
        self.assertEqual(tf, new_tf)
        self.assertFalse(new_tf.empty)

        new_tf = tf.intersection(TimeFrame())
        self.assertEqual(tf, new_tf)
        self.assertFalse(new_tf.empty)

        new_tf = tf.intersection(TimeFrame(start="1990-01-01"))
        self.assertEqual(tf, new_tf)
        self.assertFalse(new_tf.empty)

        new_tf = tf.intersection(TimeFrame(end="2100-01-01"))
        self.assertEqual(tf, new_tf)
        self.assertFalse(new_tf.empty)

        small_tf = TimeFrame("2012-01-05 00:00:00", "2012-01-06 00:00:00")
        new_tf = tf.intersection(small_tf)
        self.assertEqual(small_tf, new_tf)
        self.assertFalse(new_tf.empty)

        large_tf = TimeFrame("2010-01-01 00:00:00", "2014-01-01 00:00:00")
        new_tf = tf.intersection(large_tf)
        self.assertEqual(tf, new_tf)
        self.assertFalse(new_tf.empty)

        disjoint = TimeFrame("2015-01-01", "2016-01-01")
        new_tf = tf.intersection(disjoint)
        self.assertTrue(new_tf.empty)

        # try intersecting with emtpy TF
        new_tf = tf.intersection(new_tf)
        self.assertTrue(new_tf.empty)

        disjoint = TimeFrame("2015-01-01", "2016-01-01")
        tf.enabled = False
        new_tf = tf.intersection(disjoint)
        self.assertEqual(new_tf, disjoint)
        self.assertFalse(new_tf.empty)
        tf.enabled = True

        # crop into the start of tf
        new_start = "2012-01-05 04:05:06"
        new_tf = tf.intersection(TimeFrame(start=new_start, end="2014-01-01"))
        self.assertEqual(new_tf, TimeFrame(start=new_start, end=tf.end))
        self.assertFalse(new_tf.empty)

        # crop into the end of tf
        new_end = "2012-01-07 04:05:06"
        new_tf = tf.intersection(TimeFrame(start="2011-01-01", end=new_end))
        self.assertEqual(new_tf, TimeFrame(start=tf.start, end=new_end))
        self.assertFalse(new_tf.empty)

    def test_adjacent(self):
        # overlap
        tf1 = TimeFrame("2011-01-01 00:00:00", "2011-02-01 00:00:00")
        tf2 = TimeFrame("2011-02-01 00:00:00", "2011-03-01 00:00:00")
        self.assertTrue(tf1.adjacent(tf2))
        self.assertTrue(tf2.adjacent(tf1))

        # no overlap
        tf1 = TimeFrame("2011-01-01 00:00:00", "2011-02-01 00:00:00")
        tf2 = TimeFrame("2011-02-01 00:00:01", "2011-03-01 00:00:00")
        self.assertFalse(tf1.adjacent(tf2))
        self.assertFalse(tf2.adjacent(tf1))

        # no overlap but gap specified
        tf1 = TimeFrame("2011-01-01 00:00:00", "2011-02-01 00:00:00")
        tf2 = TimeFrame("2011-02-01 00:00:01", "2011-03-01 00:00:00")
        self.assertTrue(tf1.adjacent(tf2, gap=1))
        self.assertTrue(tf2.adjacent(tf1, gap=1))
        self.assertTrue(tf1.adjacent(tf2, gap=100))
        self.assertTrue(tf2.adjacent(tf1, gap=100))

    def test_union(self):
        # overlap
        def test_u(ts1, ts2, ts3, ts4):
            ts1 = pd.Timestamp(ts1)
            ts2 = pd.Timestamp(ts2)
            ts3 = pd.Timestamp(ts3)
            ts4 = pd.Timestamp(ts4)
            tf1 = TimeFrame(ts1, ts2)
            tf2 = TimeFrame(ts3, ts4)
            union = tf1.union(tf2)
            self.assertEqual(union.start, ts1)
            self.assertEqual(union.end, ts4)

        test_u("2011-01-01 00:00:00", "2011-02-01 00:00:00",
               "2011-02-01 00:00:00", "2011-03-01 00:00:00")
        test_u("2011-01-01 00:00:00", "2011-01-15 00:00:00",
               "2011-02-01 00:00:00", "2011-03-01 00:00:00")

    def test_merge_timeframes(self):
        tfs = [TimeFrame("2010-01-01", "2011-01-01"),
               TimeFrame("2011-01-01", "2011-06-01"),
               TimeFrame("2012-01-01", "2013-01-01")]

        merged = merge_timeframes(tfs)
        correct_answer = [TimeFrame("2010-01-01", "2011-06-01"),
                          TimeFrame("2012-01-01", "2013-01-01")]
        self.assertEqual(merged, correct_answer)


if __name__ == '__main__':
    unittest.main()
