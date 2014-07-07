from __future__ import print_function, division
#from nilmtk.utils import find_nearest_vectorized

import pandas as pd
import itertools
import numpy as np
from sklearn import metrics
from sklearn.cluster import KMeans
from copy import deepcopy
import json
from ..appliance import ApplianceID

# For some reason, importing sklearn causes PyTables to raise lots
# of DepreciatedWarnings for Pandas code.
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

MAX_VALUES_TO_CONSIDER = 100
MAX_POINT_THRESHOLD = 2000
MIN_POINT_THRESHOLD = 20
SEED = 42

# Fix the seed for repeatibility of experiments
np.random.seed(SEED)


class CombinatorialOptimisation(object):
    """1 dimensional combinatorial optimisation NILM algorithm."""

    def __init__(self):
        self.model = {}
        self.predictions = pd.DataFrame()

    def train(self, metergroup):
        """Train using 1D CO. Places the learnt model in `model` attribute.

        Parameters
        ----------
        metergroup : a nilmtk.MeterGroup object
        """

        num_meters = len(metergroup.meters)
        if num_meters > 12:
            max_num_clusters = 2
        else:
            max_num_clusters = 3

        centroids = {}
        # TODO: only use downstream meters
        # Preprocessing!
        for i, meter in enumerate(metergroup.submeters()):
            preprocessing = [] # TODO
            dominant_appliance = meter.dominant_appliance()
            if dominant_appliance is None:
                raise RuntimeError('No dominant appliance for {}'.format(meter))

            for chunk in meter.power_series(preprocessing=preprocessing):

                # Finding the points where power consumption is greater than 10
                data = _transform_data(chunk)

                # Now for each meter we find the clusters
                cluster_centers = _apply_clustering(data, max_num_clusters)
                flattened = cluster_centers.flatten()
                flattened = np.append(flattened, 0)
                sorted_list = np.sort(flattened)
                sorted_list = sorted_list.astype(np.int)
                sorted_list = list(set(sorted_list.tolist()))
                sorted_list.sort()

                # Merge clusters TODO
                sorted_list = _merge_clusters(sorted_list)

                centroids[dominant_appliance.identifier] = sorted_list

        self.model = centroids
        return centroids

    def disaggregate(self, mains):
        '''Disaggregate the test data according to the model learnt previously

        Parameters
        ----------

        test_mains : Pandas DataFrame
            containing appliances as columns and their 1D power draw  as values
            NB: All appliances must have the same index

        Returns
        -------

        None
         '''
        test_mains = building.utility.electric.get_dataframe_of_mains(
            measurement=disagg_features[0])
        # Find put appliances which have more than one state. For others we do
        # not need to decode; they have only a single state. This can simplify
        # the amount of computations needed
        appliance_list = [
            appliance for appliance in self.model if len(self.model[appliance]) > 1]
        list_of_appliances_centroids = [self.model[appliance]
                                        for appliance in appliance_list]
        states_combination = list(itertools.product
                                  (*list_of_appliances_centroids))
        sum_combination = np.array(np.zeros(len(states_combination)))
        for i in range(0, len(states_combination)):
            sum_combination[i] = sum(states_combination[i])

        # We get a memory error if there are too many samples in test_mains; so
        # we divide them into chunks and do the processing on smaller chunks
        # and later combine these chunks
        nvalues = len(test_mains.index)
        start = 0
        states = np.array([])
        residual_power = np.array([])
        while start + min(nvalues, MAX_VALUES_TO_CONSIDER) - 1 < nvalues:
            [states_temp, residual_power_temp] = find_nearest_vectorized(
                sum_combination, test_mains.values[start:start + MAX_VALUES_TO_CONSIDER])
            states = np.append(states, states_temp)
            residual_power = np.append(residual_power, residual_power_temp)
            start += MAX_VALUES_TO_CONSIDER

        # If some values are still left
        [states_temp, residual_power_temp] = find_nearest_vectorized(
            sum_combination, test_mains.values[start:nvalues])
        states = np.append(states, states_temp)
        residual_power = np.append(residual_power, residual_power_temp)

        length_sequence = len(test_mains.index)
        [predicted_states, predicted_power] = _decode_co(length_sequence,
                                                        self.model, appliance_list, states, residual_power)

        # Now predicting for appliances with a single state
        single_state_appliance = [
            appliance for appliance in self.model if appliance not in appliance_list]
        for appliance in single_state_appliance:
            predicted_states[appliance] = np.zeros(
                length_sequence, dtype=np.int)
            predicted_power[appliance] = np.zeros(
                length_sequence, dtype=np.int)

        self.predictions = pd.DataFrame(
            predicted_power, index=test_mains.index)

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


def _transform_data(df_appliance):
    '''Subsamples if needed and converts to scikit-learn understandable format'''

    data_gt_10 = df_appliance[df_appliance > 10].dropna().values
    length = data_gt_10.size
    if length < MIN_POINT_THRESHOLD:
        return np.zeros((MAX_POINT_THRESHOLD, 1))

    if length > MAX_POINT_THRESHOLD:
        # Subsample
        temp = data_gt_10[
            np.random.randint(0, len(data_gt_10), MAX_POINT_THRESHOLD)]
        return temp.reshape(MAX_POINT_THRESHOLD, 1)
    else:
        return data_gt_10.reshape(length, 1)


def _merge_clusters(appliance_centroids):
    '''Merges clusters which are within a certain threshold in order to reduce the
    complexity of the learnt model'''

    # TODO: Implement
    return appliance_centroids


def _apply_clustering(X, max_num_clusters=3):
    '''Applies clustering on reduced data, i.e. data where power is greater than threshold

    Returns
    -------
    centroids:
        list
        List of power in different states of an appliance
    '''

    # Finds whether 2 or 3 gives better Silhouellete coefficient
    # Whichever is higher serves as the number of clusters for that
    # appliance
    num_clus = -1
    sh = -1
    k_means_labels = {}
    k_means_cluster_centers = {}
    k_means_labels_unique = {}
    for n_clusters in range(1, max_num_clusters):

        try:
            k_means = KMeans(init='k-means++', n_clusters=n_clusters)
            k_means.fit(X)
            k_means_labels[n_clusters] = k_means.labels_
            k_means_cluster_centers[n_clusters] = k_means.cluster_centers_
            k_means_labels_unique[n_clusters] = np.unique(k_means_labels)
            try:
                sh_n = metrics.silhouette_score(
                    X, k_means_labels[n_clusters], metric='euclidean')

                if sh_n > sh:
                    sh = sh_n
                    num_clus = n_clusters
            except Exception:

                num_clus = n_clusters
        except Exception:

            if num_clus > -1:
                return k_means_cluster_centers[num_clus]
            else:
                return np.array([0])

    # TODO: REMOVE THIS LINE!!!HARDCODING IT FOR PAPER FOR CONSISTENT
    # COMPARISON
    return k_means_cluster_centers[1]
    # return k_means_cluster_centers[num_clus]


def _decode_co(length_sequence, centroids, appliance_list, states,
              residual_power):
    '''Decode a Combination Sequence and map K ^ N back to each of the K
    appliances

    Parameters
    ----------

    length_sequence:
        int, shape
    Length of the series for which decoding needs to be done

    centroids:
        dict, form:
            {appliance: [sorted list of power
                         in different states]}

    appliance_list:
        list, form:
            [appliance_i..., ]

    states:
        nd.array, Contains the state in overall combinations(K ^ N), i.e.
    at each time instance in [0, length_sequence] what is the state of overall
    system[0, K ^ N - 1]

    residual_power:
        nd.array
    '''

    # TODO: Possible Cythonize/Vectorize in the future
    co_states = {}
    co_power = {}
    total_num_combinations = 1
    for appliance in appliance_list:
        total_num_combinations *= len(centroids[appliance])

    print(total_num_combinations)
    print(centroids)

    for appliance in appliance_list:
        co_states[appliance] = np.zeros(length_sequence, dtype=np.int)
        co_power[appliance] = np.zeros(length_sequence)

    for i in range(length_sequence):
        factor = total_num_combinations
        for appliance in appliance_list:
            # assuming integer division (will cause errors in Python 3x)
            factor = factor // len(centroids[appliance])

            temp = int(states[i]) / factor
            co_states[appliance][i] = temp % len(centroids[appliance])
            co_power[appliance][i] = centroids[
                appliance][co_states[appliance][i]]

    return [co_states, co_power]


def test_co():
    from nilmtk import HDFDataStore, DataSet
    datastore = HDFDataStore('/home/jack/workspace/python/nilmtk/notebooks/redd.h5')
    dataset = DataSet()
    dataset.load(datastore)
    elec = dataset.buildings[1].elec
    co = CombinatorialOptimisation()
    model = co.train(elec)
    print(model)

