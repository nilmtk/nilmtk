from __future__ import print_function, division
import pandas as pd
import numpy as np
import json
from datetime import datetime
from ..appliance import ApplianceID
from ..utils import find_nearest, container_to_string
from ..feature_detectors import cluster
from ..timeframe import merge_timeframes, list_of_timeframe_dicts, TimeFrame
from ..preprocessing import Apply, Clip

# Fix the seed for repeatability of experiments
SEED = 42
np.random.seed(SEED)


class CombinatorialOptimisation(object):

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
    """

    def __init__(self):
        self.model = []

    def train(self, metergroup):
        """Train using 1D CO. Places the learnt model in the `model` attribute.

        Parameters
        ----------
        metergroup : a nilmtk.MeterGroup object

        Notes
        -----
        * only uses first chunk for each meter (TODO: handle all chunks).
        """

        if self.model:
            raise RuntimeError("This implementation of Combinatorial Optimisation"
                               " does not support multiple calls to `train`.")

        num_meters = len(metergroup.meters)
        if num_meters > 12:
            max_num_clusters = 2
        else:
            max_num_clusters = 3

        for i, meter in enumerate(metergroup.submeters().meters):
            print("Training model for submeter '{}'".format(meter))
            for chunk in meter.power_series(preprocessing=[Clip()]):
                states = cluster(chunk, max_num_clusters)
                self.model.append({
                    'states': states,
                    'training_metadata': meter})
                break  # TODO handle multiple chunks per appliance 
        print("Done training!")

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
            "NILMTK_CO_<date>"
        resample_seconds : number, optional
            The desired sample period in seconds.
        **load_kwargs : key word arguments
            Passed to `mains.power_series(**kwargs)`
        '''
        MIN_CHUNK_LENGTH = 100

        if not self.model:
            raise RuntimeError("The model needs to be instantiated before"
                               " calling `disaggregate`.  For example, the"
                               " model can be instantiated by running `train`.")

        # If we import sklearn at the top of the file then auto doc fails.
        from sklearn.utils.extmath import cartesian

        # sklearn produces lots of DepreciationWarnings with PyTables
        import warnings
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # Extract optional parameters from load_kwargs
        date_now = datetime.now().isoformat().split('.')[0]
        output_name = load_kwargs.pop('output_name', 'NILMTK_CO_' + date_now)
        resample_seconds = load_kwargs.pop('resample_seconds', 60)

        # Get centroids
        centroids = [model['states'] for model in self.model]
        state_combinations = cartesian(centroids)
        # state_combinations is a 2D array
        # each column is a chan
        # each row is a possible combination of power demand values e.g.
        # [[0, 0, 0, 0], [0, 0, 0, 100], [0, 0, 50, 0], [0, 0, 50, 100], ...]

        # Add vampire power to the model
        vampire_power = mains.vampire_power()
        print("vampire_power = {} watts".format(vampire_power))
        n_rows = state_combinations.shape[0]
        vampire_power_array = np.zeros((n_rows, 1)) + vampire_power
        state_combinations = np.hstack((state_combinations, vampire_power_array))

        summed_power_of_each_combination = np.sum(state_combinations, axis=1)
        # summed_power_of_each_combination is now an array where each
        # value is the total power demand for each combination of states.

        load_kwargs['sections'] = load_kwargs.pop('sections',
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
            indices_of_state_combinations, residual_power = find_nearest(
                summed_power_of_each_combination, chunk.values)

            for i, model in enumerate(self.model):
                print("Estimating power demand for '{}'".format(model['training_metadata']))
                predicted_power = state_combinations[
                    indices_of_state_combinations, i].flatten()
                cols = pd.MultiIndex.from_tuples([chunk.name])
                meter_instance = model['training_metadata'].instance()
                if isinstance(meter_instance, tuple):
                    meter_instance = meter_instance[0] # nasty hack until we fix #195
                output_datastore.append('{}/elec/meter{:d}'
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
                    'timeframe': total_timeframe.to_dict(),
                    'good_sections': list_of_timeframe_dicts(merged_timeframes)
                }
            }
        }

        # Appliances and submeters:
        appliances = []
        for model in self.model:
            meter = model['training_metadata']

            meter_instance = meter.instance()
            if isinstance(meter_instance, tuple):
                # nasty hack until we fix #195
                meter_instance = meter_instance[0]

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
                    'device_model': 'CO',
                    'submeter_of': 1,
                    'data_location': ('{}/elec/meter{:d}'
                                      .format(building_path, meter_instance)),
                    'preprocessing_applied': {},  # TODO
                    'statistics': {
                        'timeframe': total_timeframe.to_dict(),
                        'good_sections': list_of_timeframe_dicts(merged_timeframes)
                    }
                }
            })

        building_metadata = {
            'instance': mains.building(),
            'elec_meters': elec_meters,
            'appliances': appliances
        }

        output_datastore.save_metadata(building_path, building_metadata)

    # TODO: fix export and import!
    # https://github.com/nilmtk/nilmtk/issues/193
    #
    # def export_model(self, filename):
    #     model_copy = {}
    #     for appliance, appliance_states in self.model.iteritems():
    #         model_copy[
    #             "{}_{}".format(appliance.name, appliance.instance)] = appliance_states
    #     j = json.dumps(model_copy)
    #     with open(filename, 'w+') as f:
    #         f.write(j)

    # def import_model(self, filename):
    #     with open(filename, 'r') as f:
    #         temp = json.loads(f.read())
    #     for appliance, centroids in temp.iteritems():
    #         appliance_name = appliance.split("_")[0].encode("ascii")
    #         appliance_instance = int(appliance.split("_")[1])
    #         appliance_name_instance = ApplianceID(
    #             appliance_name, appliance_instance)
    #         self.model[appliance_name_instance] = centroids
