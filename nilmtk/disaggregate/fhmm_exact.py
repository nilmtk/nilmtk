from nilmtk.utils import find_nearest
from nilmtk.utils import find_nearest_vectorized
from nilmtk.disaggregate.disaggregator import Disaggregator
from nilmtk.sensors.electricity import Measurement
from nilmtk.preprocessing.electricity.single import contiguous_blocks
from nilmtk.stats.electricity.single import get_sample_period


import pandas as pd
import itertools
import numpy as np
from sklearn import metrics
from sklearn import hmm

from copy import deepcopy
from collections import OrderedDict

SEED = 42

# Fix the seed for repeatibility of experiments
np.random.seed(SEED)


def sort_startprob(mapping, startprob):
    """ Sort the startprob according to power means; as returned by mapping
    """
    num_elements = len(startprob)
    new_startprob = np.zeros(num_elements)
    for i in xrange(len(startprob)):
        new_startprob[i] = startprob[mapping[i]]
    return new_startprob


def sort_covars(mapping, covars):
    num_elements = len(covars)
    new_covars = np.zeros_like(covars)
    for i in xrange(len(covars)):
        new_covars[i] = covars[mapping[i]]
    return new_covars


def sort_transition_matrix(mapping, A):
    """ Sorts the transition matrix according to increasing order of 
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

    #list_of_appliances_centroids=[ [appliance[i][0] for i in range(len(appliance))] for appliance in list_B]
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
    #model_fhmm=create_combined_hmm(len(pi_combined),pi_combined, A_combined, mean_combined, cov_combined)
    combined_model = hmm.GaussianHMM(n_components=len(
        pi_combined), covariance_type='full', startprob=pi_combined, transmat=A_combined)
    combined_model.covars_ = cov_combined
    combined_model.means_ = mean_combined
    return combined_model


def return_sorting_mapping(means):
    means_copy = deepcopy(means)
    # Sorting
    means_copy = np.sort(means_copy, axis=0)
    # Finding mapping
    mapping = {}
    for i, val in enumerate(means_copy):
        #assert val == means[np.where(val == means)[0]]
        mapping[i] = np.where(val == means)[0][0]
    return mapping


def decode_hmm(length_sequence, centroids, appliance_list, states):
    """
    Decodes the HMM state sequence
    """
    power_states_dict = {}
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


class FHMM(Disaggregator):

    def __init__(self):

        self.model = {}
        self.predictions = pd.DataFrame()

    def train(self, building, aggregate='mains', submetered='appliances',
              disagg_features=[Measurement('power', 'active')],
              environmental=None):
        """Train using 1d FHMM. Places the learnt model in `model` attribute
        """

         # Get a dataframe of appliances; Since the algorithm is 1D, we need
        # only the first Measurement
        train_appliances = building.utility.electric.get_dataframe_of_appliances(
            measurement=disagg_features[0])

        train_mains = building.utility.electric.get_dataframe_of_mains(
            measurement=disagg_features[0])

        # Setting frequency
        self.freq = str(int(get_sample_period(train_mains.index))) + 's'

        learnt_model = OrderedDict()
        for appliance in train_appliances:
            #print(appliance)
            learnt_model[appliance] = hmm.GaussianHMM(
                2, "full")

            # Data to fit
            X = []

            # Breaking data into contiguous blocks
            for start, end in contiguous_blocks(train_mains.index):
                #print(start, end)
                length = train_appliances[appliance][start:end].values.size
                # print(length)
                # Ignore small sequences
                if length > 50:
                    temp = train_appliances[appliance][
                        start:end].values.reshape(length, 1)
                    X.append(temp)
            # print(X)
            # Fit
            learnt_model[appliance].fit(X)

        # Combining to make a AFHMM
        new_learnt_models = OrderedDict()
        for appliance in learnt_model:
            startprob, means, covars, transmat = sort_learnt_parameters(
                learnt_model[appliance].startprob_, learnt_model[appliance].means_, learnt_model[appliance].covars_, learnt_model[appliance].transmat_)
            new_learnt_models[appliance] = hmm.GaussianHMM(
                startprob.size, "full", startprob, transmat)
            new_learnt_models[appliance].means_ = means
            new_learnt_models[appliance].covars_ = covars

        learnt_model_combined = create_combined_hmm(new_learnt_models)
        self.individual = new_learnt_models
        self.model = learnt_model_combined

    def disaggregate(
        self, building, disagg_features=[Measurement('power', 'active')],
            environmental=None):
        """Disaggregate the test data according to the model learnt previously
        Performs 1D FHMM disaggregation        
        """
        test_mains = building.utility.electric.get_dataframe_of_mains(
            measurement=disagg_features[0])

        # Array of learnt states
        learnt_states_array = []

        # Break down test_data into chunks and disaggregate separately on them
        for start, end in contiguous_blocks(test_mains.index):
            length = test_mains[start:end].values.size
            temp = test_mains[start:end].values.reshape(length, 1)
            learnt_states_array.append(self.model.predict(temp))

        # Model
        means = OrderedDict()
        for appliance in self.individual:
            means[appliance] = self.individual[appliance].means_
        means_copy = deepcopy(means)
        for appliance in means:
            means_copy[appliance] = means[
                appliance].astype(int).flatten().tolist()
            means_copy[appliance].sort()

        decoded_power_array = []
        decoded_states_array = []
        for learnt_states in learnt_states_array:
            [decoded_states, decoded_power] = decode_hmm(
                len(learnt_states), means_copy, means_copy.keys(), learnt_states)
            decoded_states_array.append(decoded_states)
            decoded_power_array.append(decoded_power)

        # Combining to make a DataFrame with correct index, based on the start,
        # end time and the frequency of the data
        dfs_list = []
        count = 0
        cont_blocks = contiguous_blocks(test_mains.index)
        for i, (start, end) in enumerate(contiguous_blocks(test_mains.index)):
            index = pd.DatetimeIndex(start=start, end=end,
                                     freq=self.freq)

            df = pd.DataFrame(decoded_power_array[i], index=index)
            dfs_list.append(df)

        self.predictions = pd.concat(dfs_list).sort_index()
