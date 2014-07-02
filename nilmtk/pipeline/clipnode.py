from __future__ import print_function, division
from node import Node
from warnings import warn
from nilmtk.utils import index_of_column_name

class ClipNode(Node):

    # Not very well specified.  Really want to specify that 
    # we need 'lower_limit' and 'upper_limit' to be specified in
    # each measurement...
    requirements = {'device': {'measurements': 'ANY VALUE'}}
    postconditions =  {'preprocessing_applied': {'clip': {}}}
    name = 'clip'

    def process(self, df, metadata):
        measurements = metadata['device']['measurements']
        for measurement in df:
            lower, upper = _find_limits(measurement, measurements)
            if lower is not None and upper is not None:
                icol = index_of_column_name(df, measurement)
                df.iloc[:,icol] = df.iloc[:,icol].clip(lower, upper)

        return df

def _find_limits(measurement, measurements):
    """
    Returns
    -------
    lower, upper : numbers
    """
    for m in measurements:
        if ((m['physical_quantity'], m['type']) == measurement):
            return m.get('lower_limit'), m.get('upper_limit')

    warn('No measurement limits for {}.'.format(measurement), RuntimeWarning)
    return None, None
