from nilmtk.utils import find_nearest
from nilmtk.utils import find_nearest_vectorized

import pandas as pd
import itertools
import numpy as np
from sklearn import metrics
from sklearn import hmm
from copy import deepcopy

from collections import OrderedDict

MAX_VALUES_TO_CONSIDER = 100
MAX_POINT_THRESHOLD = 2000
MIN_POINT_THRESHOLD = 20
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
    """ Sorts the transition matrix according to power means; as returned by mapping
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
    '''
    Input: list_pi: List of PI's of individual learnt HMMs
    Output: Combined Pi for the FHMM
    '''
    result = list_A[0]
    for i in range(len(list_A) - 1):
        result = np.kron(result, list_A[i + 1])
    return result


def compute_means_fhmm(list_means):
    '''
    Returns [mu, sigma]
    '''

    #list_of_appliances_centroids=[ [appliance[i][0] for i in range(len(appliance))] for appliance in list_B]
    states_combination = list(itertools.product(*list_means))
    print states_combination
    num_combinations = len(states_combination)
    print num_combinations
    means_stacked = np.array([sum(x) for x in states_combination])
    means = np.reshape(means_stacked, (num_combinations, 1))
    cov = np.tile(5 * np.identity(1), (num_combinations, 1, 1))
    return [means, cov]


def compute_pi_fhmm(list_pi):
    '''
    Input: list_pi: List of PI's of individual learnt HMMs
    Output: Combined Pi for the FHMM
    '''
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
    '''
    Decodes the HMM state sequence
    '''
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


def transform_data(df_appliance):
    '''Subsamples if needed and converts to scikit-learn understandable format'''

    data_gt_10 = df_appliance[df_appliance > 10].values
    length = data_gt_10.size
    if length < MIN_POINT_THRESHOLD:
        return np.zeros((MAX_POINT_THRESHOLD, 1))

    if length > MAX_POINT_THRESHOLD:
        # Subsample
        temp = data_gt_10[
            np.random.randint(0, len(data_gt_10), MAX_POINT_THRESHOLD)]
        return temp.reshape(MAX_POINT_THRESHOLD, 1)
    else:
        temp = data_gt_10
    return temp.reshape(length, 1)


class FHMM(object):

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
            Dictionary corresponding to number of states for each appliance
            This can be passed by the user
        """

        learnt_model = OrderedDict()
        for appliance in train_appliances:
            learnt_model[appliance] = hmm.GaussianHMM(
                2, "full")
            print "Learning model for appliance", appliance
            print [train_appliances[appliance].values]
            #print [train_appliances[appliance].values].shape
            length = train_appliances[appliance].values.size
            print length
            temp = train_appliances[appliance].values.reshape(length, 1)
            learnt_model[appliance].fit([temp])

        # Combining to make a AFHMM
        new_learnt_models = OrderedDict()
        for appliance in learnt_model:
            startprob, means, covars, transmat = sort_learnt_parameters(
                learnt_model[appliance].startprob_, learnt_model[appliance].means_, learnt_model[appliance].covars_, learnt_model[appliance].transmat_)
            new_learnt_models[appliance] = hmm.GaussianHMM(
                startprob.size, "full", startprob, transmat)
            new_learnt_models[appliance].means_ = means
            new_learnt_models[appliance].covars_ = covars

        raw_input('Now I am going to hang!!')
        learnt_model_combined = create_combined_hmm(new_learnt_models)
        self.individual = new_learnt_models
        self.model = learnt_model_combined

    def disaggregate(self, test_mains):
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
        # Find put appliances which have more than one state. For others we do
        # not need to decode; they have only a single state. This can simplify
        # the amount of computations needed

        learnt_states = self.model.predict(test_mains.values)

        [decoded_states, decoded_power] = decode_hmm(
            len(learnt_states), self.individual, [appliance for appliance in self.model], learnt_states)

        self.predictions = pd.DataFrame(decoded_power)
