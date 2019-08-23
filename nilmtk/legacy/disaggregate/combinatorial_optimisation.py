from warnings import warn

import pandas as pd
import numpy as np
import pickle
import copy

from ...utils import find_nearest
from ...feature_detectors import cluster
from . import Disaggregator
from ...datastore import HDFDataStore


class CombinatorialOptimisation(Disaggregator):
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

    def __init__(self):
        self.model = []
        self.state_combinations = None
        self.MIN_CHUNK_LENGTH = 100
        self.MODEL_NAME = 'CO'

    def train(self, metergroup, num_states_dict=None, **load_kwargs):
        """Train using 1D CO. Places the learnt model in the `model` attribute.

        Parameters
        ----------
        metergroup : a nilmtk.MeterGroup object
        num_states_dict : dict
        **load_kwargs : keyword arguments passed to `meter.power_series()`

        Notes
        -----
        * only uses first chunk for each meter (TODO: handle all chunks).
        """
        if num_states_dict is None:
            num_states_dict = {}

        if self.model:
            raise RuntimeError(
                "This implementation of Combinatorial Optimisation"
                " does not support multiple calls to `train`.")

        num_meters = len(metergroup.meters)
        if num_meters > 12:
            max_num_clusters = 2
        else:
            max_num_clusters = 3

        for i, meter in enumerate(metergroup.submeters().meters):
            print("Training model for submeter '{}'".format(meter))
            power_series = meter.power_series(**load_kwargs)
            chunk = next(power_series)
            num_total_states = num_states_dict.get(meter)
            if num_total_states is not None:
                num_on_states = num_total_states - 1
            else:
                num_on_states = None
            self.train_on_chunk(chunk, meter, max_num_clusters, num_on_states)

            # Check to see if there are any more chunks.
            # TODO handle multiple chunks per appliance.
            try:
                next(power_series)
            except StopIteration:
                pass
            else:
                warn("The current implementation of CombinatorialOptimisation"
                     " can only handle a single chunk.  But there are multiple"
                     " chunks available.  So have only trained on the"
                     " first chunk!")

        print("Done training!")

    def train_on_chunk(self, chunk, meter, max_num_clusters, num_on_states):
        # Check if we've already trained on this meter
        meters_in_model = [d['training_metadata'] for d in self.model]
        if meter in meters_in_model:
            raise RuntimeError(
                "Meter {} is already in model!"
                "  Can't train twice on the same meter!"
                .format(meter))

        states = cluster(chunk, max_num_clusters, num_on_states)
        self.model.append({
            'states': states,
            'training_metadata': meter})

    def _set_state_combinations_if_necessary(self):
        """Get centroids"""
        # If we import sklearn at the top of the file then auto doc fails.
        if (self.state_combinations is None or
                self.state_combinations.shape[1] != len(self.model)):
            from sklearn.utils.extmath import cartesian
            centroids = [model['states'] for model in self.model]
            self.state_combinations = cartesian(centroids)

    def disaggregate(self, mains, output_datastore,**load_kwargs):
        '''Disaggregate mains according to the model learnt previously.

        Parameters
        ----------
        mains : nilmtk.ElecMeter or nilmtk.MeterGroup
        output_datastore : instance of nilmtk.DataStore subclass
            For storing power predictions from disaggregation algorithm.
        sample_period : number, optional
            The desired sample period in seconds.  Set to 60 by default.
        sections : TimeFrameGroup, optional
            Set to mains.good_sections() by default.
        **load_kwargs : key word arguments
            Passed to `mains.power_series(**kwargs)`
        '''
        load_kwargs = self._pre_disaggregation_checks(load_kwargs)

        load_kwargs.setdefault('sample_period', 60)
        load_kwargs.setdefault('sections', mains.good_sections())

        timeframes = []
        building_path = '/building{}'.format(mains.building())
        mains_data_location = building_path + '/elec/meter1'
        data_is_available = False

        for chunk in mains.power_series(**load_kwargs):
            # Check that chunk is sensible size
            if len(chunk) < self.MIN_CHUNK_LENGTH:
                continue

            # Record metadata
            timeframes.append(chunk.timeframe)
            measurement = chunk.name

            appliance_powers = self.disaggregate_chunk(chunk)

            for i, model in enumerate(self.model):
                appliance_power = appliance_powers.iloc[:, i]
                if len(appliance_power) == 0:
                    continue
                data_is_available = True
                cols = pd.MultiIndex.from_tuples([chunk.name])
                meter_instance = model['training_metadata'].instance()
                df = pd.DataFrame(
                    appliance_power.values, index=appliance_power.index,
                    columns=cols)
                key = '{}/elec/meter{}'.format(building_path, meter_instance)
                output_datastore.append(key, df)

            # Copy mains data to disag output
            mains_df = pd.DataFrame(chunk, columns=cols)
            output_datastore.append(key=mains_data_location, value=mains_df)

        if data_is_available:
            self._save_metadata_for_disaggregation(
                output_datastore=output_datastore,
                sample_period=load_kwargs['sample_period'],
                measurement=measurement,
                timeframes=timeframes,
                building=mains.building(),
                meters=[d['training_metadata'] for d in self.model]
            )

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
        if not self.model:
            raise RuntimeError(
                "The model needs to be instantiated before"
                " calling `disaggregate`.  The model"
                " can be instantiated by running `train`.")

        if len(mains) < self.MIN_CHUNK_LENGTH:
            raise RuntimeError("Chunk is too short.")

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
        indices_of_state_combinations, residual_power = find_nearest(
            summed_power_of_each_combination, mains.values)

        appliance_powers_dict = {}
        for i, model in enumerate(self.model):
            print("Estimating power demand for '{}'"
                  .format(model['training_metadata']))
            predicted_power = state_combinations[
                indices_of_state_combinations, i].flatten()
            column = pd.Series(predicted_power, index=mains.index, name=i)
            appliance_powers_dict[self.model[i]['training_metadata']] = column
        appliance_powers = pd.DataFrame(appliance_powers_dict, dtype='float32')
        return appliance_powers

    def import_model(self, filename):
        with open(filename, 'rb') as in_file:
            imported_model = pickle.load(in_file)
            
        self.model = imported_model.model
        
        # Recreate datastores from filenames
        for pair in self.model:
            store_filename = pair['training_metadata'].store
            pair['training_metadata'].store = HDFDataStore(store_filename)
            
        self.state_combinations = imported_model.state_combinations
        self.MIN_CHUNK_LENGTH = imported_model.MIN_CHUNK_LENGTH

    def export_model(self, filename):
        # Can't pickle datastore, so convert to filenames
        original_stores = []
        for pair in self.model:
            original_store = pair['training_metadata'].store
            original_stores.append(original_store)
            pair['training_metadata'].store = original_store.store.filename
    
        try:
            with open(filename, 'wb') as out_file:
                pickle.dump(self, out_file)
        finally:
            # Restore the stores even if the pickling fails
            for original_store, pair in zip(original_stores, self.model):
                pair['training_metadata'].store = original_store
                
