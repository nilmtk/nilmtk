from __future__ import print_function, division
import pandas as pd
from itertools import product
import numpy as np
from sklearn import metrics
from sklearn.cluster import KMeans
from copy import deepcopy
import json
from ..appliance import ApplianceID
from ..utils import find_nearest

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
    """1 dimensional combinatorial optimisation NILM algorithm.

    Attributes
    ----------
    model : dict
        Each key is the ApplianceID for the dominant appliance.
        Each value is a sorted list of power in different states.
    """

    def __init__(self):
        self.model = {}
        self.predictions = pd.DataFrame()

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
        for i, meter in enumerate(metergroup.submeters()):

            # Find dominant appliance for this meter
            dominant_appliance = meter.dominant_appliance()
            if dominant_appliance is None:
                raise RuntimeError('No dominant appliance for {}'.format(meter))

            # Load data and train model
            preprocessing = [] # TODO
            for chunk in meter.power_series(preprocessing=preprocessing):
                # Find where power consumption is greater than 10
                data = _transform_data(chunk)

                # Find clusters
                cluster_centers = _apply_clustering(data, max_num_clusters)
                flattened = cluster_centers.flatten()
                flattened = np.append(flattened, 0)
                centroids = np.sort(flattened)
                centroids = centroids.astype(np.int) # TODO: round
                centroids = list(set(centroids.tolist())) # TODO: is np.unique() faster?
                centroids.sort()
                # TODO: Merge clusters
                self.model[dominant_appliance.identifier] = centroids
                
                break # TODO handle multiple chunks per appliance


    def disaggregate(self, mains, output_datastore, **load_kwargs):
        '''Disaggregate mains according to the model learnt previously.

        Parameters
        ----------
        mains : nilmtk.ElecMeter or nilmtk.MeterGroup
        output_datastore : nilmtk.DataStore subclass
            For storing appliance power predictions.
        **load_kwargs : key word arguments
            Passed to `mains.power_series(**kwargs)`
        '''
        # Find appliances which have more than one state. For others we do
        # not need to decode; they have only a single state. This can simplify
        # the amount of computations needed

        multi_state_appliances = []
        single_state_appliances = []
        for appliance, model in self.model.iteritems():
            n_states = len(model)
            if n_states > 1:
                multi_state_appliances.append(appliance)
            else:
                single_state_appliances.append(appliance)

        multi_state_appl_centroids = [self.model[appliance]
                                      for appliance in multi_state_appliances]
        state_combinations = np.array(list(product(*multi_state_appl_centroids)))
        # state_combinations is a 2D array
        # each column is an appliance
        # each row is a possible combination of power demand values e.g.
        # (0, 0, 0, 0), (0, 0, 0, 100), (0, 0, 50, 0), (0, 0, 50, 100), ...

        summed_power_of_each_combination = np.sum(state_combinations, axis=1)
        # summed_power_of_each_combination is now an array where each 
        # value is the total power demand for each combination of states.

        # TODO preprocessing??
        for chunk in mains.power_series(**load_kwargs):
            n_samples = len(chunk)
            predicted_power = {}

            # Multi-state appliances
            indices_of_state_combinations, residual_power = find_nearest(
                summed_power_of_each_combination, chunk.values)

            for i, appliance in enumerate(multi_state_appliances):
                predicted_power[appliance] = state_combinations[
                    indices_of_state_combinations, i].flatten()

            # Single state appliances
            for appliance in single_state_appliances:
                appliance_power = self.model[appliance][0]
                predicted_power[appliance] = (np.zeros(n_samples, dtype=np.int)
                                              + appliance_power)

            predictions_for_chunk = pd.DataFrame(predicted_power, 
                                                 index=chunk.index)

            break # TODO: remove!

        return predictions_for_chunk

            # TODO: save predicted_states
            #   * need to store all metadata from training to re-use
            #   * need to know meter instance and building
            #   * save metadata. Need to be careful about dual supply appliances.
            #   * save predictions, including state.

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


def _apply_clustering(X, max_num_clusters=3):
    '''Applies clustering on reduced data, 
    i.e. data where power is greater than threshold.

    Parameters
    ----------
    X : ndarray
    max_num_clusters : int

    Returns
    -------
    centroids : list of numbers
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

    return k_means_cluster_centers[num_clus]
