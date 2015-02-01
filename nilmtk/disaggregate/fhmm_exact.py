from __future__ import print_function, division
from ..utils import find_nearest
import pandas as pd
import itertools
import numpy as np
from sklearn import metrics
from hmmlearn import hmm
import pandas as pd
import numpy as np
import json
from datetime import datetime
from ..appliance import ApplianceID
from ..utils import find_nearest, container_to_string
from ..feature_detectors import cluster
from ..timeframe import merge_timeframes, list_of_timeframe_dicts, TimeFrame
from ..preprocessing import Apply, Clip

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


class FHMM(object):

    def __init__(self):

        self.model = {}
        self.predictions = pd.DataFrame()

    def train(self, metergroup):
        """
        Train using 1d FHMM. Places the learnt model in `model` attribute
        The current version performs training ONLY on the first chunk.
        Online HMMs are welcome is someone can contribute :)
        """
        learnt_model = OrderedDict()
        for i, meter in enumerate(metergroup.submeters().meters):
            print("Training model for submeter '{}'".format(meter))
            learnt_model[meter] = hmm.GaussianHMM(2, "full")

            # Data to fit
            X = []
            meter_data = meter.power_series().next().dropna()
            """
            This was the behaviour in v0.1. Now assuming all
            preprocessing has been done

            # Breaking data into contiguous blocks
            for start, end in contiguous_blocks(meter_data.index):
                #print(start, end)
                length = appliance_data[start:end].values.size
                # print(length)
                # Ignore small sequences
                if length > 50:
                    temp = meter_data[
                        start:end].values.reshape(length, 1)
                    X.append(temp)
            # print(X)
            # Fit
            """
            length = len(meter_data.index)
            X = meter_data.values.reshape(length, 1)
            self.X = X
            learnt_model[meter].fit([X])

        # Combining to make a AFHMM
        self.meters = []
        new_learnt_models = OrderedDict()
        for meter in learnt_model:
            startprob, means, covars, transmat = sort_learnt_parameters(
                learnt_model[meter].startprob_, learnt_model[meter].means_, learnt_model[meter].covars_, learnt_model[meter].transmat_)
            new_learnt_models[meter] = hmm.GaussianHMM(
                startprob.size, "full", startprob, transmat)
            new_learnt_models[meter].means_ = means
            new_learnt_models[meter].covars_ = covars
            # UGLY! But works.
            self.meters.append(meter)

        learnt_model_combined = create_combined_hmm(new_learnt_models)
        self.individual = new_learnt_models
        self.model = learnt_model_combined


    def disaggregate_chunk(self, test_mains):
        """
        Disaggregate the test data according to the model learnt previously
        Performs 1D FHMM disaggregation        
        """
        
        # Array of learnt states
        learnt_states_array = []

        """For now assuming there is no missing data at this stage
        # Break down test_data into chunks and disaggregate separately on them
        for start, end in contiguous_blocks(test_mains.index):
            length = test_mains[start:end].values.size
            temp = test_mains[start:end].values.reshape(length, 1)
            learnt_states_array.append(self.model.predict(temp))
        """
        length = len(test_mains.index)
        temp = test_mains.values.reshape(length, 1)
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

        """
        The following code is different from its 0.1 version which is
        mentioned below this
        """
        dfs_list = []
        count = 0
        #cont_blocks = contiguous_blocks(test_mains.index)
        
        prediction = pd.DataFrame(decoded_power_array[0], index=test_mains.index)
        #prediction.index = test.index
        return prediction
        """
        The following code was used in v0.1
        For now, I am assuming that we have no missing data by the time
        we reach the disaggregate stage. This will greatly simplify our design
        

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
        """

    def disaggregate(self, mains, output_datastore, **load_kwargs):
        '''Disaggregate mains according to the model learnt previously.

        Parameters
        ----------
        mains : nilmtk.ElecMeter or nilmtk.MeterGroup
        output_datastore : instance of nilmtk.DataStore subclass
            For storing power predictions from disaggregation algorithm.
        output_name : string, optional
            The `name` to use in the metadata for the `output_datastore`.
            e.g. some sort of name for this experiment.  Defaults to 
            "NILMTK_FHMM_<date>"
        resample_seconds : number, optional
            The desired sample period in seconds.
        **load_kwargs : key word arguments
            Passed to `mains.power_series(**kwargs)`
        '''
        import warnings
        warnings.filterwarnings("ignore", category=Warning)
        MIN_CHUNK_LENGTH =100
        if not self.model:
            raise RuntimeError("The model needs to be instantiated before"
                               " calling `disaggregate`.  For example, the"
                               " model can be instantiated by running `train`.")

        
        
        # Extract optional parameters from load_kwargs
        date_now = datetime.now().isoformat().split('.')[0]
        output_name = load_kwargs.pop('output_name', 'NILMTK_FHMM_' + date_now)
        resample_seconds = load_kwargs.pop('resample_seconds', 60)

        
        sections = load_kwargs.pop('sections',
                                                  mains.good_sections())
        resample_rule = '{:d}S'.format(resample_seconds)
        timeframes = []
        building_path = '/building{}'.format(mains.building())
        mains_data_location = '{}/elec/meter1'.format(building_path)

        for chunk in mains.power_series(**load_kwargs):

            # Check that chunk is sensible size before resampling
            if len(chunk) < MIN_CHUNK_LENGTH:
                continue

            # Record metadata
            timeframes.append(chunk.timeframe)
            measurement = chunk.name

            chunk = chunk.resample(rule=resample_rule)
            # Check chunk size *again* after resampling
            if len(chunk) < MIN_CHUNK_LENGTH:
                continue

            # Start disaggregation
            predictions = self.disaggregate_chunk(chunk)
            for meter in predictions.columns:
            
                meter_instance = meter.instance()
                cols = pd.MultiIndex.from_tuples([chunk.name])

                predicted_power = predictions[[meter]].values
                output_datastore.append('{}/elec/meter{}'
                                        .format(building_path, meter_instance),
                                        pd.DataFrame(predicted_power,
                                                     index=chunk.index,
                                                     columns=cols))

            # Copy mains data to disag output
            output_datastore.append(key=mains_data_location,
                                    value=pd.DataFrame(chunk, columns=cols))

        ##################################
        # Add metadata to output_datastore

        # TODO: `preprocessing_applied` for all meters
        # TODO: split this metadata code into a separate function
        # TODO: submeter measurement should probably be the mains
        #       measurement we used to train on, not the mains measurement.

        # DataSet and MeterDevice metadata:
        meter_devices = {
            'FHMM': {
                'model': 'FHMM',
                'sample_period': resample_seconds,
                'max_sample_period': resample_seconds,
                'measurements': [{
                    'physical_quantity': measurement[0],
                    'type': measurement[1]
                }]
            },
            'mains': {
                'model': 'mains',
                'sample_period': resample_seconds,
                'max_sample_period': resample_seconds,
                'measurements': [{
                    'physical_quantity': measurement[0],
                    'type': measurement[1]
                }]
            }
        }

        merged_timeframes = merge_timeframes(timeframes, gap=resample_seconds)
        total_timeframe = TimeFrame(merged_timeframes[0].start,
                                    merged_timeframes[-1].end)

        dataset_metadata = {'name': output_name, 'date': date_now,
                            'meter_devices': meter_devices,
                            'timeframe': total_timeframe.to_dict()}
        output_datastore.save_metadata('/', dataset_metadata)

        # Building metadata

        # Mains meter:
        elec_meters = {
            1: {
                'device_model': 'mains',
                'site_meter': True,
                'data_location': mains_data_location,
                'preprocessing_applied': {},  # TODO
                'statistics': {
                    'timeframe': total_timeframe.to_dict(),
                    'good_sections': list_of_timeframe_dicts(merged_timeframes)
                }
            }
        }

        # TODO: FIX THIS! Ugly hack for now
        # Appliances and submeters:
        appliances = []
        for i, meter in enumerate(self.meters):
            meter_instance = meter.instance()

            for app in meter.appliances:
                meters = app.metadata['meters']
                appliance = {
                    'meters': [meter_instance], 
                    'type': app.identifier.type,
                    'instance': app.identifier.instance
                    # TODO this `instance` will only be correct when the
                    # model is trained on the same house as it is tested on.
                    # https://github.com/nilmtk/nilmtk/issues/194
                }
                appliances.append(appliance)

            elec_meters.update({
                meter_instance: {
                    'device_model': 'FHMM',
                    'submeter_of': 1,
                    'data_location': ('{}/elec/meter{}'
                                      .format(building_path, meter_instance)),
                    'preprocessing_applied': {},  # TODO
                    'statistics': {
                        'timeframe': total_timeframe.to_dict(),
                        'good_sections': list_of_timeframe_dicts(merged_timeframes)
                    }
                }
            })

            
           #Setting the name if it exists
            if meter.name:
                if len(meter.name) > 0:
                    elec_meters[meter_instance]['name'] = meter.name
        building_metadata = {
            'instance': mains.building(),
            'elec_meters': elec_meters,
            'appliances': appliances
        }

        output_datastore.save_metadata(building_path, building_metadata)
