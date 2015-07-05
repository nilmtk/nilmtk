"""File defining custom nilmtk exception classes.
"""


class TooFewSamplesError(Exception):
    pass


class PerformanceWarning(RuntimeWarning):
    pass


class MeasurementError(Exception):
    pass


class VampirePowerAlreadyInModelError(Exception):
    pass
