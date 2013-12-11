import pandas as pd
import itertools
import numpy as np
from nilmtk.utils import find_nearest



def decode_co(length_sequence, centroids, appliance_list, states,
             residual_power):
    '''Decode a Combination Sequence and map K^N back to each of the K
    appliances

    Parameters
    ----------

    length_sequence: int, shape
    Length of the series for which decoding needs to be done

    centroids: dict, form: {appliance: [sorted list of power
    in different states]}

    appliance_list: list, form: [appliance_i...,]

    states: nd.array, Contains the state in overall combinations (K^N), i.e.
    at each time instance in [0, length_sequence] what is the state of overall
    system [0, K^N-1]

    residual_power: nd.array
    '''

    co_states = {}
    co_power = {}
    total_num_combinations = 1
    for appliance in appliance_list:
        total_num_combinations*= len(centroids[appliance])

    for appliance in appliance_list:
        co_states[appliance] = np.zeros(length_sequence, dtype=np.int)
        co_power[appliance] = np.zeros(length_sequence)

    for i in range(length_sequence):
        factor = total_num_combinations
        for appliance in appliance_list:
            #assuming integer division (will cause errors in Python 3x)
            factor = factor // len(centroids[appliance])

            temp = int(states[i]) / factor
            co_states[appliance][i] = temp % len(centroids[appliance])
            co_power[appliance][i] = centroids[appliance][co_states[appliance][i]]

    return [co_states, co_power]


class CO_1d(object):

    def __init__(self):

        self.model = {}
        self.predictions = pd.DataFrame()

    def train(self, train_mains, train_appliances, cluster_algo='kmeans++',
             num_states=None):

        """Train using 1d CO. Places the learnt model in `model` attribute

        Attributes
        ----------

        train_mains : 1d Pandas series (indexed on DateTime) corresponding to an
            attribute of mains such as power_active, power_reactive etc.

        train_appliances : Pandas DataFrame (indexed on DateTime);
            Each attibute (column)
            is a series corresponding to an attribute of each appliance
            such as power_active. This attribute must be the same as
            that used by the mains

        cluster_algo : string, optional
            Clustering algorithm used for learning the states of the appliances

        num_states :  dict
            Dictionarty corresponding to number of states for each appliance
            This can be passed by the user
        """

    def disaggregate(self, test_mains):
        appliance_list = [appliance for appliance in self.model]
        list_of_appliances_centroids = [self.model[appliance]
                                for appliance in appliance_list]
        states_combination = list(itertools.product
                        (*list_of_appliances_centroids))
        sum_combination = np.array(np.zeros(len(states_combination)))
        for i in range(0, len(states_combination)):
            sum_combination[i] = sum(states_combination[i])

        length_sequence = len(test_mains.values)
        states = np.zeros(length_sequence)
        residual_power = np.zeros(length_sequence)
        for i in range(length_sequence):
            [states[i], residual_power[i]] = find_nearest(
                sum_combination, test_mains.values[i])
        [predicted_states, predicted_power] = decode_co(length_sequence,
                            self.model, appliance_list, states, residual_power)
        self.predictions = pd.DataFrame(predicted_power)


