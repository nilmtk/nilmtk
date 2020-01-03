from warnings import warn

import pandas as pd
import numpy as np
import pickle
import copy

from nilmtk.utils import find_nearest
from nilmtk.feature_detectors import cluster
from nilmtk.disaggregate import Disaggregator
from nilmtk.datastore import HDFDataStore


class CO(Disaggregator):
    """1 dimensional combinatorial optimisation NILM algorithm.

    Attributes
    ----------
    model : list of dicts
       Each dict has these keys:
           states : list of ints (the power (Watts) used in different states)
           training_metadata : ElecMeter or MeterGroup object used for training
               this set of states.  We need this information because we
               need the appliance type (and perhaps some other metadata)
               for each model.

    state_combinations : 2D array
        Each column is an appliance.
        Each row is a possible combination of power demand values e.g.
            [[0, 0,  0,   0],
             [0, 0,  0, 100],
             [0, 0, 50,   0],
             [0, 0, 50, 100], ...]

    MIN_CHUNK_LENGTH : int
    """

    def __init__(self, params):
        self.model = []
        self.MODEL_NAME = 'CO'
        self.save_model_path = params.get('save-model-path', None)
        self.load_model_path = params.get('pretrained-model-path',None)
        self.chunk_wise_training = params.get('chunk_wise_training', False)
        if self.load_model_path:
            self.load_model(self.load_model_path)
        self.state_combinations = None
        self.MIN_CHUNK_LENGTH = 100

    def partial_fit(
            self,
            train_main,
            train_appliances,
            do_preprocessing=True,
            **load_kwargs):

        train_main = pd.concat(train_main, axis=0)
        train_app_tmp = []

        for app_name, df_list in train_appliances:
            df_list = pd.concat(df_list, axis=0)
            train_app_tmp.append((app_name, df_list))

        train_appliances = train_app_tmp

        print("...............CO partial_fit running.............")
        num_on_states = None
        if len(train_appliances) > 12:
            max_num_clusters = 2
        else:
            max_num_clusters = 3
        appliance_in_model = [d['appliance_name'] for d in self.model]

        for appliance, readings in train_appliances:
            #print(appliance," ",readings)
            if appliance in appliance_in_model:
                #     raise RuntimeError(
                #     "Appliance {} is already in model!"
                #     "  Can't train twice on the same meter!",appliance)
                print("Trained on " + appliance + " before.")

            else:
                states = cluster(readings, max_num_clusters, num_on_states)
                self.model.append({
                    'states': states,
                    'appliance_name': appliance})

    def _set_state_combinations_if_necessary(self):
        """Get centroids"""
        # If we import sklearn at the top of the file then auto doc fails.
        if (self.state_combinations is None or
                self.state_combinations.shape[1] != len(self.model)):
            from sklearn.utils.extmath import cartesian
            centroids = [model['states'] for model in self.model]
            #print("centroids ...",centroids)
            self.state_combinations = cartesian(centroids)

    def disaggregate_chunk(self, mains):
        """In-memory disaggregation.

        Parameters
        ----------
        mains : pd.Series


        Returns
        -------
        appliance_powers : pd.DataFrame where each column represents a
            disaggregated appliance.  Column names are the integer index
            into `self.model` for the appliance in question.
        """
        '''if not self.model:
            raise RuntimeError(
                "The model needs to be instantiated before"
                " calling `disaggregate`.  The model"
                " can be instantiated by running `train`.")'''

        print("...............CO disaggregate_chunk running.............")

        # sklearn produces lots of DepreciationWarnings with PyTables
        import warnings
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # Because CombinatorialOptimisation could have been trained using
        # either train() or train_on_chunk(), we must
        # set state_combinations here.
        self._set_state_combinations_if_necessary()

        """
         # Add vampire power to the model
        if vampire_power is None:
            vampire_power = get_vampire_power(mains)
        if vampire_power > 0:
            print("Including vampire_power = {} watts to model..."
                  .format(vampire_power))
            n_rows = self.state_combinations.shape[0]
            vampire_power_array = np.zeros((n_rows, 1)) + vampire_power
            state_combinations = np.hstack(
                (self.state_combinations, vampire_power_array))
        else:
            state_combinations = self.state_combinations
        """

        state_combinations = self.state_combinations
        summed_power_of_each_combination = np.sum(state_combinations, axis=1)
        # summed_power_of_each_combination is now an array where each
        # value is the total power demand for each combination of states.

        # Start disaggregation

        test_prediction_list = []

        for test_df in mains:

            appliance_powers_dict = {}
            indices_of_state_combinations, residual_power = find_nearest(
                summed_power_of_each_combination, test_df.values)

            for i, model in enumerate(self.model):
                print("Estimating power demand for '{}'"
                      .format(model['appliance_name']),end="\r")
                predicted_power = state_combinations[
                    indices_of_state_combinations, i].flatten()
                column = pd.Series(
                    predicted_power, index=test_df.index, name=i)
                appliance_powers_dict[self.model[i]['appliance_name']] = column

            appliance_powers = pd.DataFrame(
                appliance_powers_dict, dtype='float32')
            test_prediction_list.append(appliance_powers)

        return test_prediction_list
