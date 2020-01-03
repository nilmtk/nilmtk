from warnings import warn
from ..node import Node
from ..utils import index_of_column_name

class Clip(Node):

    """Ensures that no value is below a lower limit or above an upper limit.
    If self.lower and self.upper are None then will use clip settings from
    'device': {'measurements': {'upper_limit' and 'lower_limit'}}.
    """

    # Not very well specified.  Really want to specify that 
    # we need 'lower_limit' and 'upper_limit' to be specified in
    # each measurement...
    requirements = {'device': {'measurements': 'ANY VALUE'}}
    postconditions =  {'preprocessing_applied': {'clip': {}}}

    def reset(self):
        self.lower = None
        self.upper = None

    def process(self):
        self.check_requirements()
        metadata = self.upstream.get_metadata()
        measurements = metadata['device']['measurements']
        for chunk in self.upstream.process():
            for measurement in chunk:
                lower, upper = _find_limits(measurement, measurements)
                lower = lower if self.lower is None else self.lower
                upper = upper if self.upper is None else self.upper
                if lower is not None and upper is not None:
                    # We use `chunk.iloc[:,icol]` instead of iterating
                    # through each column so we can to the clipping in place
                    icol = index_of_column_name(chunk, measurement)
                    chunk.iloc[:,icol] = chunk.iloc[:,icol].clip(lower, upper)

            yield chunk

def _find_limits(measurement, measurements):
    """
    Returns
    -------
    lower, upper : numbers
    """
    for m in measurements:
        if ((m.get('physical_quantity'), m.get('type')) == measurement):
            return m.get('lower_limit'), m.get('upper_limit')

    warn('No measurement limits for {}.'.format(measurement), RuntimeWarning)
    return None, None
