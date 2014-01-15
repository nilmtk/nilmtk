"""File defining custom nilmtk exception classes.
"""

class TooFewSamplesError(Exception):
    pass

class NoSuitableMeasurementError(Exception):
    pass

class NoCommonMeasurementError(Exception):
    pass
