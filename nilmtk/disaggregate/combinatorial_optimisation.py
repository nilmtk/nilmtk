from __future__ import print_function, division
import pandas as pd
from sklearn.utils.extmath import cartesian
import numpy as np
import json
from ..appliance import ApplianceID
from ..utils import find_nearest
from ..feature_detectors import cluster

# Fix the seed for repeatability of experiments
SEED = 42
np.random.seed(SEED)


class CombinatorialOptimisation(object):
    """1 dimensional combinatorial optimisation NILM algorithm.

    Attributes
    ----------
    model : dict
        Each key is either the instance integer for an ElecMeter, 
        or a tuple of instances for a MeterGroup.
        Each value is a sorted list of power in different states.
    """

    def __init__(self):
        self.model = {}

    def train(self, metergroup):
        """Train using 1D CO. Places the learnt model in `model` attribute.

        Parameters
        ----------
        metergroup : a nilmtk.MeterGroup object

        Notes
        -----
        * only uses first chunk for each meter (TODO: handle all chunks).
        """

        num_meters = len(metergroup.meters)
        if num_meters > 12:
            max_num_clusters = 2
        else:
            max_num_clusters = 3

        # TODO: Preprocessing!
        preprocessing = [] # TODO
        for i, meter in enumerate(metergroup.submeters()):
            for chunk in meter.power_series(preprocessing=preprocessing):
                self.model[meter.instance()] = cluster(chunk, max_num_clusters)
                break # TODO handle multiple chunks per appliance

    def disaggregate(self, mains, output_datastore, **load_kwargs):
        '''Disaggregate mains according to the model learnt previously.

        Parameters
        ----------
        mains : nilmtk.ElecMeter or nilmtk.MeterGroup
        output_datastore : nilmtk.DataStore subclass
            For storing chan power predictions.
        **load_kwargs : key word arguments
            Passed to `mains.power_series(**kwargs)`
        '''

        centroids = self.model.values()
        state_combinations = cartesian(centroids)
        # state_combinations is a 2D array
        # each column is a chan
        # each row is a possible combination of power demand values e.g.
        # [[0, 0, 0, 0], [0, 0, 0, 100], [0, 0, 50, 0], [0, 0, 50, 100], ...]

        summed_power_of_each_combination = np.sum(state_combinations, axis=1)
        # summed_power_of_each_combination is now an array where each 
        # value is the total power demand for each combination of states.

        # TODO preprocessing??
        for chunk in mains.power_series(**load_kwargs):

            indices_of_state_combinations, residual_power = find_nearest(
                summed_power_of_each_combination, chunk.values)

            for i, chan in enumerate(self.model.keys()):
                predicted_power = state_combinations[
                    indices_of_state_combinations, i].flatten()
                if isinstance(chan, tuple):
                    chan = '_'.join([str(element) for element in chan])
                output_datastore.append('/building1/elec/meter{}'.format(chan),
                                        pd.Series(predicted_power,
                                                  index=chunk.index))
            # TODO: save predicted_states
            #   * need to store all metadata from training to re-use
            #   * need to know meter instance and building
            #   * save metadata. Need to be careful about dual supply appliances.

    def export_model(self, filename):
        model_copy = {}
        for appliance, appliance_states in self.model.iteritems():
            model_copy[
                "{}_{}".format(appliance.name, appliance.instance)] = appliance_states
        j = json.dumps(model_copy)
        with open(filename, 'w+') as f:
            f.write(j)

    def import_model(self, filename):
        with open(filename, 'r') as f:
            temp = json.loads(f.read())
        for appliance, centroids in temp.iteritems():
            appliance_name = appliance.split("_")[0].encode("ascii")
            appliance_instance = int(appliance.split("_")[1])
            appliance_name_instance = ApplianceID(
                appliance_name, appliance_instance)
            self.model[appliance_name_instance] = centroids
