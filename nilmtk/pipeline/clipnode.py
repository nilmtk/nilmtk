from __future__ import print_function, division
from node import Node
from warnings import warn

class ClipNode(Node):

    requirements = {'device': {'measurement_limits': 'ANY VALUE'}}
    postconditions =  {'preprocessing': {'clipped': True}}
    name = 'clip'

    def process(self, df, metadata):
        limits = metadata['device']['measurement_limits']
        for measurement in df:
            try:
                lim_for_measurement = limits[measurement]
            except KeyError:
                warn('No measurement limits for {}.'.format(measurement),
                     RuntimeWarning)
            else:
                lower = lim_for_measurement['lower']
                upper = lim_for_measurement['upper']
                df[measurement] = df[measurement].clip(lower, upper)

        return df
