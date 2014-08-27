from __future__ import print_function, division
import pandas as pd
import numpy as np
import json
from datetime import datetime
from ..appliance import ApplianceID
from ..utils import find_nearest, container_to_string
from ..feature_detectors import cluster, steady_states
from ..feature_detectors.cluster import hart85_means_shift_cluster
from ..timeframe import merge_timeframes, list_of_timeframe_dicts, TimeFrame
from ..preprocessing import Apply, Clip

# Fix the seed for repeatability of experiments
SEED = 42
np.random.seed(SEED)


from collections import deque
import numpy as np


class MyDeque(deque):

    def popmiddle(self, pos):
        self.rotate(-pos)
        ret = self.popleft()
        self.rotate(pos)
        return ret


class PairBuffer(object):

    '''
    Attributes:
    * transitionList (list of tuples)
    * matchedPairs (dataframe containing matched pairs of transitions)
    '''

    def __init__(self, bufferSize=20, minTolerance=35, percentTolerance=0.035, largeTransition=1000, num_measurements=3):
        # We use a deque here, because it allows us quick access to start and end popping
        # and additionally, we can set a maxlen which drops oldest items. This nicely
        # suits Hart's recomendation that the size should be tunable.
        self._bufferSize = bufferSize
        self._minTol = minTolerance
        self._percentTol = percentTolerance
        self._largeTransition = largeTransition
        self.transitionList = MyDeque([], maxlen=self._bufferSize)
        self._num_measurements = num_measurements
        if self._num_measurements == 3:
            # Both active and reactive power is available
            self.pairColumns = ['T1 Time', 'T1 Active', 'T1 Reactive',
                                'T2 Time', 'T2 Active', 'T2 Reactive']
        elif self._num_measurements == 2:
            # Only active power is available
            self.pairColumns = ['T1 Time', 'T1 Active',
                                'T2 Time', 'T2 Active']
        self.matchedPairs = pd.DataFrame(columns=self.pairColumns)

    ''' # Slower method to clean the buffer
    def cleanBuffer2(self):
        newList = []
        for entry in self.transitionList:
            if entry[3] == False:
                 newList.append(entry)
        self.transitionList = MyDeque(newList, maxlen=self._bufferSize)
    '''

    def cleanBuffer(self):
        # Remove any matched transactions
        for idx, entry in enumerate(self.transitionList):
            if entry[self._num_measurements] == True:
                self.transitionList.popmiddle(idx)
                self.cleanBuffer()
                break
        # Remove oldest transaction if buffercleaning didn't remove anything
        # if len(self.transitionList) == self._bufferSize:
        #    self.transitionList.popleft()

    def addTransition(self, transition):

        # Check transition is as expected.
        assert isinstance(transition, (tuple, list))

        # Check that we have both active and reactive powers.
        assert len(transition) == self._num_measurements

        # Convert as appropriate
        if isinstance(transition, tuple):
            mTransition = list(transition)

        # Add transition to List of transitions (set marker as unpaired)
        mTransition.append(False)
        self.transitionList.append(mTransition)

        # checking for pairs
        # self.pairTransitions()
        # self.cleanBuffer()

    def pairTransitions(self):
        '''
        Hart 85, P 33.
        When searching the working buffer for pairs, the order in which 
        entries are examined is very important. If an Appliance has 
        on and off several times in succession, there can be many 
        pairings between entries in the buffer. The algorithm must not
        allow an 0N transition to match an OFF which occurred at the end 
        of a different cycle, so that only ON/OFF pairs which truly belong 
        together are paired up. Otherwise the energy consumption of the 
        appliance will be greatly overestimated. The most straightforward 
        search procedures can make errors of this nature when faced with 
        types of transition sequences.

        Hart 85, P 32.
        For the two-state load onitor, a pair is defined as two entries
        which meet the following four conditions:
        (1) They are on the same leg, or are both 240 V,
        (2) They are both unmarked, 
        (3) The earlier has a positive real power component, and 
        (4) When added together, they result in a vector in which the 
        absolute value of the real power component is less than 35 
        Watts (or 3.5% of the real power, if the transitions are 
        over 1000 W) and the absolute value of the reactive power 
        component is less than 35 VAR (or 3.5%).

        ... the correct way to search the buffer is to start by checking 
        elements which are close together in the buffer, and gradually 
        increase the distance. First, adjacent  elements are checked for 
        pairs which meet all four requirements above; if any are found 
        they are processed and marked. Then elements two entries apart 
        are checked, then three, and so on, until the first and last 
        element are checked...

        '''

        tLength = len(self.transitionList)
        pairMatched = False

        if tLength < 2:
            return pairMatched

        # Can we reduce the running time of this algorithm?
        # My gut feeling is no, because we can't re-order the list...
        # I wonder if we sort but then check the time... maybe. TO DO
        # (perhaps!).

        # Start the element distance at 1, go up to current length of buffer
        for eDistance in range(1, tLength):

            idx = 0
            while idx < tLength - 1:

                # We don't want to go beyond length of array
                compIndex = idx + eDistance

                if compIndex < tLength:

                    val = self.transitionList[idx]
                    # val[1] is the active power and
                    # val[self._num_measurements] is match status
                    if (val[1] > 0) and (val[self._num_measurements] == False):

                        compVal = self.transitionList[compIndex]
                        if compVal[self._num_measurements] == False:
                            # Add the two elements for comparison
                            vSum = np.add(
                                val[1:self._num_measurements],
                                compVal[1:self._num_measurements])

                            # Set the allowable tolerance for reactive and
                            # active
                            matchTols = [self._minTol, self._minTol]
                            for ix in range(1, self._num_measurements):
                                matchTols[ix - 1] = self._minTol if (max(np.fabs([val[ix], compVal[ix]]))
                                                                     < self._largeTransition) else (self._percentTol
                                                                                                    * max(np.fabs([val[ix], compVal[ix]])))
                            if self._num_measurements == 3:
                                condition = (np.fabs(vSum[0]) < matchTols[0]) and (
                                    np.fabs(vSum[1]) < matchTols[1])

                            elif self._num_measurements == 2:
                                condition = np.fabs(vSum[0]) < matchTols[0]

                            if condition:
                                # Mark the transition as complete
                                self.transitionList[idx][
                                    self._num_measurements] = True
                                self.transitionList[compIndex][
                                    self._num_measurements] = True
                                pairMatched = True

                                # Append the OFF transition to the ON. Add to
                                # dataframe.
                                matchedPair = val[
                                    0:self._num_measurements] + compVal[0:self._num_measurements]
                                self.matchedPairs.loc[
                                    len(self.matchedPairs)] = matchedPair

                    # Iterate Index
                    idx += 1
                else:
                    break

        return pairMatched


class Hart85(object):

    """1 or 2 dimensional Hart 1985 algorithm.

    Attributes
    ----------
    model : dict
        Each key is either the instance integer for an ElecMeter, 
        or a tuple of instances for a MeterGroup.
        Each value is a sorted list of power in different states.
    """

    def __init__(self):
        self.model = {}

    def train(self, metergroup, bsize=20, minTol=35, cluster_features=['active']):
        """Train using Hart85. Places the learnt model in `model` attribute.

        Parameters
        ----------
        metergroup : a nilmtk.MeterGroup object


        """
        steady_states_list = []
        transients_list = []

        for power_df in metergroup.power_series_all_columns():
            if len(power_df.columns) <= 2:
                # Use whatever is available
                power_dataframe = power_df
            else:
                # Active, reactive and apparent are available
                power_dataframe = power_df[['active', 'reactive']]

            power_dataframe = power_dataframe.dropna()

            x, y = steady_states.find_steady_states(power_dataframe)
            steady_states_list.append(x)
            transients_list.append(y)

        self.steady_states = pd.concat(steady_states_list)
        self.transients = pd.concat(transients_list)

        self.pair_df = self.pair(bsize, minTol)

        self.centroids = hart85_means_shift_cluster(self.pair_df, cluster_features)

    def pair(self, bsize, minTol):
        subset = list(self.transients.itertuples())
        buffer = PairBuffer(minTolerance=minTol,
                            bufferSize=bsize, percentTolerance=0.035,
                            num_measurements=len(self.transients.columns) + 1)
        for s in subset:
            # if len(buffer.transitionList) < bsize
            if len(buffer.transitionList) == bsize:
                buffer.cleanBuffer()
            buffer.addTransition(s)
            buffer.pairTransitions()
       
        return buffer.matchedPairs

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

        # If we import sklearn at the top of the file then it makes autodoc
        # fail
        from sklearn.utils.extmath import cartesian

        # sklearn produces lots of DepreciationWarnings with PyTables
        import warnings
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # Extract optional parameters from load_kwargs
        date_now = datetime.now().isoformat().split('.')[0]
        output_name = load_kwargs.pop('output_name', 'NILMTK_CO_' + date_now)
        resample_seconds = load_kwargs.pop('resample_seconds', 60)

        # Get centroids
        centroids = self.model.values()
        state_combinations = cartesian(centroids)
        # state_combinations is a 2D array
        # each column is a chan
        # each row is a possible combination of power demand values e.g.
        # [[0, 0, 0, 0], [0, 0, 0, 100], [0, 0, 50, 0], [0, 0, 50, 100], ...]

        summed_power_of_each_combination = np.sum(state_combinations, axis=1)
        # summed_power_of_each_combination is now an array where each
        # value is the total power demand for each combination of states.

        load_kwargs['sections'] = load_kwargs.pop('sections',
                                                  mains.good_sections())
        resample_rule = '{:d}S'.format(resample_seconds)
        timeframes = []
        building_path = '/building{}'.format(mains.building())
        mains_data_location = ('{}/elec/meter{}'
                               .format(building_path,
                                       container_to_string(mains.instance())))

        for chunk in mains.power_series(**load_kwargs):

            if len(chunk) < MIN_CHUNK_LENGTH:

                continue

            # Record metadata
            timeframes.append(chunk.timeframe)
            measurement = chunk.name

            chunk = chunk.resample(rule=resample_rule)
            if len(chunk) < MIN_CHUNK_LENGTH:
                continue

            # Start disaggregation
            indices_of_state_combinations, residual_power = find_nearest(
                summed_power_of_each_combination, chunk.values)

            for i, chan in enumerate(self.model.keys()):
                predicted_power = state_combinations[
                    indices_of_state_combinations, i].flatten()
                chan_str = container_to_string(chan)
                cols = pd.MultiIndex.from_tuples([chunk.name])
                output_datastore.append('{}/elec/meter{}'
                                        .format(building_path, chan_str),
                                        pd.DataFrame(predicted_power,
                                                     index=chunk.index,
                                                     columns=cols))

            # Copy mains data to disag output
            output_datastore.append(key=mains_data_location,
                                    value=pd.DataFrame(chunk, columns=cols))

        ##################################
        # Add metadata to output_datastore

        # TODO: `preprocessing_applied` for all meters
        # TODO: appliance metadata
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
            mains.instance(): {
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

        # Submeters:
        for chan in self.model.keys():
            elec_meters.update({
                chan: {
                    'device_model': 'CO',
                    'submeter_of': mains.instance(),
                    'data_location': ('{}/elec/meter{}'
                                      .format(building_path,
                                              container_to_string(chan))),
                    'preprocessing_applied': {},  # TODO
                    'statistics': {
                        'timeframe': total_timeframe.to_dict(),
                        'good_sections': list_of_timeframe_dicts(merged_timeframes)
                    }
                }
            })

        # Appliances:
        appliances = []  # TODO

        building_metadata = {
            'instance': mains.building(),
            'elec_meters': elec_meters,
            'appliances': appliances
        }

        output_datastore.save_metadata(building_path, building_metadata)

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
