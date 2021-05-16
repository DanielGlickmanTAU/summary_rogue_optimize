from unittest import TestCase

from experiments.gridsearch import gridsearch


class Test(TestCase):
    def test_gridsearch(self):
        defaults = {'one': 1, 'two': -99}
        to_search = {'two': [1, 2], 'three': ['A', 'B', 'C']}
        options = gridsearch(defaults, to_search)

        self.assert_contains_exactly_once(options, {'one': 1, 'two': 1, 'three': 'A'})
        self.assert_contains_exactly_once(options, {'one': 1, 'two': 2, 'three': 'A'})

        self.assert_contains_exactly_once(options, {'one': 1, 'two': 1, 'three': 'B'})
        self.assert_contains_exactly_once(options, {'one': 1, 'two': 2, 'three': 'B'})

        self.assert_contains_exactly_once(options, {'one': 1, 'two': 1, 'three': 'C'})
        self.assert_contains_exactly_once(options, {'one': 1, 'two': 2, 'three': 'C'})

    def assert_contains_exactly_once(self, options, params):
        params_in_options = [d for d in options if d == params]
        assert len(params_in_options) == 1
