import unittest
from ..node import find_unsatisfied_requirements

class TestNode(unittest.TestCase):

    def test_unsatisfied_requirements(self):
        requirements = {'gaps_located':True, 'energy_computed':True}

        state = {'gaps_located':True}
        unsatisfied = find_unsatisfied_requirements(state, requirements)
        self.assertEqual(len(unsatisfied), 1)

        state = {'gaps_located':True, 'energy_computed':False}
        unsatisfied = find_unsatisfied_requirements(state, requirements)
        self.assertEqual(len(unsatisfied), 1)

        state = {'gaps_located':False, 'energy_computed':False}
        unsatisfied = find_unsatisfied_requirements(state, requirements)
        self.assertEqual(len(unsatisfied), 2)

        state = {}
        unsatisfied = find_unsatisfied_requirements(state, requirements)
        self.assertEqual(len(unsatisfied), 2)

        state = {'gaps_located':True, 'energy_computed':True}
        unsatisfied = find_unsatisfied_requirements(state, requirements)
        self.assertEqual(unsatisfied, [])

        requirements = {'preprocessing': {'gaps_located':True, 'energy_computed':False}}
        state = {'preprocessing': {'gaps_located':True, 'energy_computed':False},
                 'sample_period': 6}
        unsatisfied = find_unsatisfied_requirements(state, requirements)
        self.assertEqual(unsatisfied, [])        
        
        requirements = {'preprocessing': {'gaps_located':True, 'energy_computed':True}}
        state = {'preprocessing': {'gaps_located':True, 'energy_computed':False},
                 'sample_period': 6}
        unsatisfied = find_unsatisfied_requirements(state, requirements)
        self.assertEqual(len(unsatisfied), 1)

        requirements = {'preprocessing': {'gaps_located':True, 'energy_computed':True},
                        'sample_period': 100}
        state = {'preprocessing': {'gaps_located':True, 'energy_computed':False},
                 'sample_period': 6}
        unsatisfied = find_unsatisfied_requirements(state, requirements)
        self.assertEqual(len(unsatisfied), 2)        

        requirements = {'preprocessing': {'gaps_located':True, 'energy_computed':True},
                        'sample_period': 100}
        state = {'preprocessing': {'gaps_located':True, 'energy_computed':False}}
        unsatisfied = find_unsatisfied_requirements(state, requirements)
        self.assertEqual(len(unsatisfied), 2)        

        requirements = {'preprocessing': {'gaps_located':True, 'energy_computed':True},
                        'sample_period': 100}
        state = {}
        unsatisfied = find_unsatisfied_requirements(state, requirements)
        self.assertEqual(len(unsatisfied), 2)        

        requirements = {}
        state = {'preprocessing': {'gaps_located':True, 'energy_computed':True},
                 'sample_period': 100}
        unsatisfied = find_unsatisfied_requirements(state, requirements)
        self.assertEqual(len(unsatisfied), 0)

if __name__ == '__main__':
    unittest.main()
