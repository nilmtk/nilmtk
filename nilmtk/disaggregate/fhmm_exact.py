import itertools
from copy import deepcopy
from collections import OrderedDict
from warnings import warn
import pickle
import nilmtk
import pandas as pd
import numpy as np
from hmmlearn import hmm

from nilmtk.feature_detectors import cluster
from nilmtk.disaggregate import Disaggregator
from nilmtk.datastore import HDFDataStore
import datetime
import matplotlib.pyplot as plt

def sort_startprob(mapping, startprob):
    """ Sort the startprob according to power means; as returned by mapping
    """
    num_elements = len(startprob)
    new_startprob = np.zeros(num_elements)
    for i in range(len(startprob)):
        new_startprob[i] = startprob[mapping[i]]
    return new_startprob


def sort_covars(mapping, covars):
    new_covars = np.zeros_like(covars)
    for i in range(len(covars)):
        new_covars[i] = covars[mapping[i]]
    return new_covars


def sort_transition_matrix(mapping, A):
    """Sorts the transition matrix according to increasing order of
    power means; as returned by mapping

    Parameters
    ----------
    mapping :
    A : numpy.array of shape (k, k)
        transition matrix
    """
    num_elements = len(A)
    A_new = np.zeros((num_elements, num_elements))
    for i in range(num_elements):
        for j in range(num_elements):
            A_new[i, j] = A[mapping[i], mapping[j]]
    return A_new


def sort_learnt_parameters(startprob, means, covars, transmat):
    mapping = return_sorting_mapping(means)
    means_new = np.sort(means, axis=0)
    startprob_new = sort_startprob(mapping, startprob)
    covars_new = sort_covars(mapping, covars)
    transmat_new = sort_transition_matrix(mapping, transmat)
    assert np.shape(means_new) == np.shape(means)
    assert np.shape(startprob_new) == np.shape(startprob)
    assert np.shape(transmat_new) == np.shape(transmat)

    return [startprob_new, means_new, covars_new, transmat_new]


def compute_A_fhmm(list_A):
    """
    Parameters
    -----------
    list_pi : List of PI's of individual learnt HMMs

    Returns
    --------
    result : Combined Pi for the FHMM
    """
    result = list_A[0]
    for i in range(len(list_A) - 1):
        result = np.kron(result, list_A[i + 1])
    return result


def compute_means_fhmm(list_means):
    """
    Returns
    -------
    [mu, cov]
    """
    states_combination = list(itertools.product(*list_means))
    num_combinations = len(states_combination)
    means_stacked = np.array([sum(x) for x in states_combination])
    means = np.reshape(means_stacked, (num_combinations, 1))
    cov = np.tile(5 * np.identity(1), (num_combinations, 1, 1))
    return [means, cov]


def compute_pi_fhmm(list_pi):
    """
    Parameters
    -----------
    list_pi : List of PI's of individual learnt HMMs

    Returns
    -------
    result : Combined Pi for the FHMM
    """
    result = list_pi[0]
    for i in range(len(list_pi) - 1):
        result = np.kron(result, list_pi[i + 1])
    return result


def create_combined_hmm(model):
    list_pi = [model[appliance].startprob_ for appliance in model]
    list_A = [model[appliance].transmat_ for appliance in model]
    list_means = [model[appliance].means_.flatten().tolist()
                  for appliance in model]

    pi_combined = compute_pi_fhmm(list_pi)
    A_combined = compute_A_fhmm(list_A)
    [mean_combined, cov_combined] = compute_means_fhmm(list_means)

    combined_model = hmm.GaussianHMM(n_components=len(pi_combined), covariance_type='full')
    combined_model.startprob_ = pi_combined
    combined_model.transmat_ = A_combined
    combined_model.covars_ = cov_combined
    combined_model.means_ = mean_combined
    
    return combined_model


def return_sorting_mapping(means):
    means_copy = deepcopy(means)
    means_copy = np.sort(means_copy, axis=0)

    # Finding mapping
    mapping = {}
    for i, val in enumerate(means_copy):
        mapping[i] = np.where(val == means)[0][0]
    return mapping


def decode_hmm(length_sequence, centroids, appliance_list, states):
    """
    Decodes the HMM state sequence
    """
    hmm_states = {}
    hmm_power = {}
    total_num_combinations = 1

    for appliance in appliance_list:
        total_num_combinations *= len(centroids[appliance])

    for appliance in appliance_list:
        hmm_states[appliance] = np.zeros(length_sequence, dtype=np.int)
        hmm_power[appliance] = np.zeros(length_sequence)

    for i in range(length_sequence):

        factor = total_num_combinations
        for appliance in appliance_list:
            # assuming integer division (will cause errors in Python 3x)
            factor = factor // len(centroids[appliance])

            temp = int(states[i]) / factor
            hmm_states[appliance][i] = temp % len(centroids[appliance])
            hmm_power[appliance][i] = centroids[
                appliance][hmm_states[appliance][i]]
    return [hmm_states, hmm_power]


class FHMMExact(Disaggregator):

    def __init__(self,params):
        self.model = {}
        self.MODEL_NAME = 'FHMM'  # Add the name for the algorithm
        self.save_model_path = params.get('save-model-path', None)
        self.load_model_path = params.get('pretrained-model-path',None)
        self.chunk_wise_training = params.get('chunk_wise_training', False)
        self.num_of_states = params.get('num_of_states', 2)
        if self.load_model_path:
            self.load_model(self.load_model_path)
        self.app_names = []

    def partial_fit(self, train_main, train_appliances, **load_kwargs):
        """
        Train using 1d FHMM.
        """
        print(".........................FHMM partial_fit.................")

        train_main = pd.concat(train_main, axis=0)
        train_app_tmp = []

        for app_name, df_list in train_appliances:
            df_list = pd.concat(df_list, axis=0)
            train_app_tmp.append((app_name,df_list))
            self.app_names.append(app_name)

        print (train_main.shape)

        train_appliances = train_app_tmp

        learnt_model = OrderedDict()
        num_meters = len(train_appliances)
        if num_meters > 12:
            max_num_clusters = 2
        else:
            max_num_clusters = 3

        for appliance, meter in train_appliances:
            
            meter_data = meter.dropna()
            X = meter_data.values.reshape((-1, 1))
            
            if not len(X):
                print("Submeter '{}' has no samples, skipping...".format(meter))
                continue
                
            assert X.ndim == 2
            self.X = X

            if self.num_of_states > 0:
                 # User has specified the number of states for this appliance
                 num_total_states = self.num_of_states

            else:
                 # Find the optimum number of states
                states = cluster(meter_data, max_num_clusters)
                num_total_states = len(states)

            print("Training model for submeter '{}'".format(appliance))
            learnt_model[appliance] = hmm.GaussianHMM(num_total_states, "full")

            # Fit
            learnt_model[appliance].fit(X)
            print("Learnt model for : "+appliance)

            # Check to see if there are any more chunks.
            # TODO handle multiple chunks per appliance.

        # Combining to make a AFHMM
        self.meters = []
        new_learnt_models = OrderedDict()
        for meter in learnt_model:
            print(meter)
            startprob, means, covars, transmat = sort_learnt_parameters(
                learnt_model[meter].startprob_, learnt_model[meter].means_,
                learnt_model[meter].covars_, learnt_model[meter].transmat_)
                
            new_learnt_models[meter] = hmm.GaussianHMM(startprob.size, "full")
            new_learnt_models[meter].startprob_ = startprob
            new_learnt_models[meter].transmat_ = transmat
            new_learnt_models[meter].means_ = means
            new_learnt_models[meter].covars_ = covars
            # UGLY! But works.
            self.meters.append(meter)

        learnt_model_combined = create_combined_hmm(new_learnt_models)
        self.individual = new_learnt_models
        self.model = learnt_model_combined

        print("print ...........",self.model)

        print("FHMM partial_fit end.................")

    def disaggregate_chunk(self, test_mains_list):

        """Disaggregate the test data according to the model learnt previously

        Performs 1D FHMM disaggregation.

        For now assuming there is no missing data at this stage.
        """
        # See v0.1 code
        # for ideas of how to handle missing data in this code if needs be.

        # Array of learnt states

        test_prediction_list = []

        for test_mains in test_mains_list:
            
            learnt_states_array = []
            if len(test_mains) == 0:
                tmp = pd.DataFrame(index = test_mains.index, columns = self.app_names)
                test_prediction_list.append(tmp)
            else:
                length = len(test_mains.index)
                temp = test_mains.values.reshape(length, 1)
                learnt_states_array.append(self.model.predict(temp))

                # Model
                means = OrderedDict()
                for elec_meter, model in self.individual.items():
                    means[elec_meter] = (
                        model.means_.round().astype(int).flatten().tolist())
                    means[elec_meter].sort()

                decoded_power_array = []
                decoded_states_array = []

                for learnt_states in learnt_states_array:
                    [decoded_states, decoded_power] = decode_hmm(
                        len(learnt_states), means, means.keys(), learnt_states)
                    decoded_states_array.append(decoded_states)
                    decoded_power_array.append(decoded_power)

                appliance_powers = pd.DataFrame(decoded_power_array[0], dtype='float32')
                test_prediction_list.append(appliance_powers)
        
        return test_prediction_list