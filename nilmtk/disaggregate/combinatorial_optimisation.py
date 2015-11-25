from __future__ import print_function, division
from datetime import datetime
from warnings import warn

import pandas as pd
import numpy as np
import pickle
import copy

from ..utils import find_nearest
from ..feature_detectors import cluster
from ..timeframe import merge_timeframes, TimeFrame
from ..disaggregate import Disaggregator
from ..datastore import HDFDataStore
from nilmtk.electric import get_vampire_power

# Fix the seed for repeatability of experiments
SEED = 42
np.random.seed(SEED)


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
    """

    def __init__(self):
        self.model = []
        self.state_combinations = None
        self.MIN_CHUNK_LENGTH = 100

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
            chunk = power_series.next()
            num_total_states = num_states_dict.get(meter)
            if num_total_states is not None:
                num_on_states = num_total_states - 1
            else:
                num_on_states = None
            self.train_on_chunk(chunk, meter, max_num_clusters, num_on_states)

            # Check to see if there are any more chunks.
            # TODO handle multiple chunks per appliance.
            try:
                power_series.next()
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

    def disaggregate(self, mains, output_datastore, output_name=None,
                     vampire_power=None, **load_kwargs):
        '''Disaggregate mains according to the model learnt previously.

        Parameters
        ----------
        mains : nilmtk.ElecMeter or nilmtk.MeterGroup
        output_datastore : instance of nilmtk.DataStore subclass
            For storing power predictions from disaggregation algorithm.
        output_name : string, optional
            The `name` to use in the metadata for the `output_datastore`.
            e.g. some sort of name for this experiment.  Defaults to
            "NILMTK_CO_<date>"
        vampire_power : None or number (watts)
            If None then will automatically determine vampire power
            from data.  If you do not want to use vampire power then
            set vampire_power = 0.
        sample_period : number, optional
            The desired sample period in seconds.
        **load_kwargs : key word arguments
            Passed to `mains.power_series(**kwargs)`
        '''
        date_now = datetime.now().isoformat().split('.')[0]
        if output_name is None:
            output_name = 'NILMTK_CO_' + date_now

        if 'resample_seconds' in load_kwargs:
            warn("'resample_seconds' is deprecated."
                 "  Please use 'sample_period' instead.")
            load_kwargs['sample_period'] = load_kwargs.pop('resample_seconds')

        load_kwargs.setdefault('sample_period', 60)
        resample_seconds = load_kwargs['sample_period']
        load_kwargs['sections'] = load_kwargs.pop(
            'sections', mains.good_sections())

        timeframes = []
        building_path = '/building{}'.format(mains.building())
        mains_data_location = '{}/elec/meter1'.format(building_path)
        data_is_available = False

        for chunk in mains.power_series(**load_kwargs):
            # Check that chunk is sensible size before resampling
            if len(chunk) < self.MIN_CHUNK_LENGTH:
                continue

            # Record metadata
            timeframes.append(chunk.timeframe)
            measurement = chunk.name

            appliance_powers = self.disaggregate_chunk(chunk, vampire_power)

            for i, model in enumerate(self.model):
                appliance_power = appliance_powers[i]
                data_is_available = True
                cols = pd.MultiIndex.from_tuples([chunk.name])
                meter_instance = model['training_metadata'].instance()
                df = pd.DataFrame(
                    appliance_power.values, index=appliance_power.index,
                    columns=cols)
                key = '{}/elec/meter{}'.format(building_path, meter_instance)
                output_datastore.append(key, df)

            # Copy mains data to disag output
            output_datastore.append(key=mains_data_location,
                                    value=pd.DataFrame(chunk, columns=cols))

        if not data_is_available:
            return

        ##################################
        # Add metadata to output_datastore

        # TODO: `preprocessing_applied` for all meters
        # TODO: split this metadata code into a separate function
        # TODO: submeter measurement should probably be the mains
        #       measurement we used to train on, not the mains measurement.

        # DataSet and MeterDevice metadata:
        meter_devices = {
            'CO': {
                'model': 'CO',
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
                    'timeframe': total_timeframe.to_dict()
                }
            }
        }

        # Appliances and submeters:
        appliances = []
        for model in self.model:
            meter = model['training_metadata']

            meter_instance = meter.instance()

            for app in meter.appliances:
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
                    'device_model': 'CO',
                    'submeter_of': 1,
                    'data_location': ('{}/elec/meter{}'
                                      .format(building_path, meter_instance)),
                    'preprocessing_applied': {},  # TODO
                    'statistics': {
                        'timeframe': total_timeframe.to_dict()
                    }
                }
            })

            # Setting the name if it exists
            if meter.name:
                if len(meter.name) > 0:
                    elec_meters[meter_instance]['name'] = meter.name

        building_metadata = {
            'instance': mains.building(),
            'elec_meters': elec_meters,
            'appliances': appliances
        }

        output_datastore.save_metadata(building_path, building_metadata)

    def disaggregate_chunk(self, mains, vampire_power=None):
        """In-memory disaggregation.

        Parameters
        ----------
        mains : pd.Series
        vampire_power : None or number (watts)
            If None then will automatically determine vampire power
            from data.  If you do not want to use vampire power then
            set vampire_power = 0.

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

        # sklearn produces lots of DepreciationWarnings with PyTables
        import warnings
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # Because CombinatorialOptimisation could have been trained using
        # either train() or train_on_chunk(), we must
        # set state_combinations here.
        self._set_state_combinations_if_necessary()

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
            appliance_powers_dict[i] = column

        appliance_powers = pd.DataFrame(appliance_powers_dict)
        return appliance_powers

    def import_model(self, filename):
        imported_model = pickle.load(open(filename, 'r'))
        self.model = imported_model.model
        # recreate datastores from filenames
        for pair in self.model:
            pair['training_metadata'].store = HDFDataStore(
                pair['training_metadata'].store)
        self.state_combinations = imported_model.state_combinations
        self.MIN_CHUNK_LENGTH = imported_model.MIN_CHUNK_LENGTH

    def export_model(self, filename):
        # Can't pickle datastore, so convert to filenames
        exported_model = copy.deepcopy(self)
        for pair in exported_model.model:
            pair['training_metadata'].store = (
                pair['training_metadata'].store.store.filename)
        pickle.dump(exported_model, open(filename, 'wb'))
