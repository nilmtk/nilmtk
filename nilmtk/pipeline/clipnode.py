from __future__ import print_function, division
from node import Node
from warnings import warn
from nilmtk.utils import index_of_column_name

class ClipNode(Node):

    requirements = {'device': {'measurement_limits': 'ANY VALUE'}}
    postconditions =  {'preprocessing_applied': {'clip': {}}}
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
                lower = lim_for_measurement['lower_limit']
                upper = lim_for_measurement['upper_limit']
                icol = index_of_column_name(df, measurement)
                df.iloc[:,icol] = df.iloc[:,icol].clip(lower, upper)

        return df
