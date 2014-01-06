#!/usr/bin/python

"""
   Copyright 2013 nilmtk authors.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

from __future__ import print_function, division
import unittest
import nilmtk.stats.electricity.single as single
import numpy as np
import pandas as pd

class TestSingle(unittest.TestCase):
    def test_tz_to_naive(self):
        START = '2010/1/1'
        F = 'D' # frequency
        P = 48 # periods
        dti_naive = pd.date_range(START, freq=F, periods=P)

        def test_dti(tz):
            dti = pd.date_range(START, freq=F, periods=P, tz=tz)
            dti_calc = single._tz_to_naive(dti)
            for ts_calc, ts_naive in zip(dti_calc, dti_naive):
                self.assertEqual(ts_calc, ts_naive)

        tzs = ['US/Eastern', 'UTC', 'WET', 'CET', 'Asia/Kolkata']
        for tz in tzs:
            test_dti(tz)

if __name__ == '__main__':
    unittest.main()
