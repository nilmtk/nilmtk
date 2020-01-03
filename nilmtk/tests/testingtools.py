"""
Tools to help with testing.
"""

import os, inspect, warnings

def data_dir():
    # Taken from http://stackoverflow.com/a/6098238/732596
    current_file_path = os.path.dirname(inspect.getfile(inspect.currentframe()))
    data_dir = os.path.join(current_file_path, '..', '..', 'data')
    data_dir = os.path.abspath(data_dir)
    assert os.path.isdir(data_dir), data_dir + " does not exist."
    return data_dir


class WarningTestMixin(object):
    """A test which checks if the specified warning was raised.

    Taken from http://stackoverflow.com/a/12935176/732596
    """
    def assertWarns(self, warning, callable, *args, **kwds):
        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter('always')
            result = callable(*args, **kwds)
            self.assertTrue(any(item.category == warning for item in warning_list),
                            msg="Warning '{}' not raised.".format(warning))
