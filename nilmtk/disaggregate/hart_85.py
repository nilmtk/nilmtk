import numpy as np
from collections import OrderedDict, deque
import pandas as pd
from sklearn.metrics import mean_squared_error
import pickle
from nilmtk.feature_detectors.cluster import hart85_means_shift_cluster
from nilmtk.feature_detectors.steady_states import find_steady_states
from nilmtk.disaggregate import Disaggregator

class MyDeque(deque):
    def popmiddle(self, pos):
        self.rotate(-pos)
        ret = self.popleft()
        self.rotate(pos)
        return ret


class PairBuffer(object):
    """
    Attributes:
    * transitionList (list of tuples)
    * matchedPairs (dataframe containing matched pairs of transitions)
    """

    def __init__(self, columns,buffer_size, min_tolerance, percent_tolerance,
                 large_transition, num_measurements):
        """
        Parameters
        ----------
        buffer_size: int, optional
            size of the buffer to use for finding edges
        min_tolerance: int, optional
            variance in power draw allowed for pairing a match
        percent_tolerance: float, optional
            if transition is greater than large_transition, then use percent of large_transition
        large_transition: float, optional
            power draw of a Large transition
        num_measurements: int, optional
            2 if only active power
            3 if both active and reactive power
        """
        # We use a deque here, because it allows us quick access to start and end popping
        # and additionally, we can set a maxlen which drops oldest items. This nicely
        # suits Hart's recomendation that the size should be tunable.
        self._buffer_size = buffer_size
        self._min_tol = min_tolerance
        self._percent_tol = percent_tolerance
        self._large_transition = large_transition
        self.transition_list = MyDeque([], maxlen=self._buffer_size)
        self._num_measurements = num_measurements
        if self._num_measurements == 3:
            # Both active and reactive power is available
            self.pair_columns = ['T1 Time', 'T1 Active', 'T1 Reactive',
                                 'T2 Time', 'T2 Active', 'T2 Reactive']
        elif self._num_measurements == 2:
            # Only active power is available
            if columns[0][1]=='active':
                self.pair_columns = ['T1 Time', 'T1 Active',
                                    'T2 Time', 'T2 Active']
            elif columns[0][1]=='apparent':
                self.pair_columns = ['T1 Time', 'T1 Apparent',
                                    'T2 Time', 'T2 Apparent']
        self.matched_pairs = pd.DataFrame(columns=self.pair_columns)

    def clean_buffer(self):
        # Remove any matched transactions
        for idx, entry in enumerate(self.transition_list):
            if entry[self._num_measurements]:
                self.transition_list.popmiddle(idx)
                self.clean_buffer()
                break
                # Remove oldest transaction if buffer cleaning didn't remove anything
                # if len(self.transitionList) == self._bufferSize:
                #    self.transitionList.popleft()

    def add_transition(self, transition):
        # Check transition is as expected.
        assert isinstance(transition, (tuple, list))
        # Check that we have both active and reactive powers.
        assert len(transition) == self._num_measurements
        # Convert as appropriate
        if isinstance(transition, tuple):
            mtransition = list(transition)
        # Add transition to List of transitions (set marker as unpaired)
        mtransition.append(False)
        self.transition_list.append(mtransition)
        # checking for pairs
        # self.pairTransitions()
        # self.cleanBuffer()

    def pair_transitions(self):
        """
        Hart 85, P 33.
        The algorithm must not allow an 0N transition to match an OFF which occurred at the end
        of a different cycle, so that only ON/OFF pairs which truly belong
        together are paired up. Otherwise the energy consumption of the
        appliance will be greatly overestimated.

        Hart 85, P 32.
        For the two-state load monitor, a pair is defined as two entries
        which meet the following four conditions:
        (1) They are on the same leg, or are both 240 V,
        (2) They are both unmarked,
        (3) The earlier has a positive real power component, and
        (4) When added together, they result in a vector in which the
        absolute value of the real power component is less than 35
        Watts (or 3.5% of the real power, if the transitions are
        over 1000 W) and the absolute value of the reactive power
        component is less than 35 VAR (or 3.5%).

        """

        tlength = len(self.transition_list)
        pairmatched = False
        if tlength < 2:
            return pairmatched

        # Can we reduce the running time of this algorithm?
        # My gut feeling is no, because we can't re-order the list...
        # I wonder if we sort but then check the time... maybe. TO DO
        # (perhaps!).

        new_matched_pairs = []

        # Start the element distance at 1, go up to current length of buffer
        for eDistance in range(1, tlength):
            idx = 0
            while idx < tlength - 1:
                # We don't want to go beyond length of array
                compindex = idx + eDistance
                if compindex < tlength:
                    val = self.transition_list[idx]
                    # val[1] is the active power and
                    # val[self._num_measurements] is match status
                    if (val[1] > 0) and (val[self._num_measurements] is False):
                        compval = self.transition_list[compindex]
                        if compval[self._num_measurements] is False:
                            # Add the two elements for comparison
                            vsum = np.add(
                                val[1:self._num_measurements],
                                compval[1:self._num_measurements])
                            # Set the allowable tolerance for reactive and
                            # active
                            matchtols = [self._min_tol, self._min_tol]
                            for ix in range(1, self._num_measurements):
                                matchtols[ix - 1] = (
                                    self._min_tol
                                    if (max(np.fabs([val[ix], compval[ix]])) < self._large_transition)
                                    else (self._percent_tol * max(np.fabs([val[ix], compval[ix]])))
                                )
                            if self._num_measurements == 3:
                                condition = (
                                    np.fabs(
                                        vsum[0]) < matchtols[0]) and (
                                    np.fabs(
                                        vsum[1]) < matchtols[1])

                            elif self._num_measurements == 2:
                                condition = np.fabs(vsum[0]) < matchtols[0]

                            if condition:
                                # Mark the transition as complete
                                self.transition_list[idx][self._num_measurements] = True
                                self.transition_list[compindex][self._num_measurements] = True
                                pairmatched = True

                                # Append the OFF transition to the ON. Add to
                                # the list.
                                matchedpair = val[0:self._num_measurements] + \
                                    compval[0:self._num_measurements]
                                new_matched_pairs.append(matchedpair)

                    # Iterate Index
                    idx += 1
                else:
                    break

        # Process new pairs in a single operation (faster than growing the
        # dataframe)
        if pairmatched:
            if self.matched_pairs.empty:
                self.matched_pairs = pd.DataFrame(
                    new_matched_pairs, columns=self.pair_columns)
            else:
                self.matched_pairs = pd.concat([self.matched_pairs, pd.DataFrame(new_matched_pairs, columns=self.pair_columns)], ignore_index=True)

        return pairmatched


class Hart85(Disaggregator):
    """1 or 2 dimensional Hart 1985 algorithm.

    Attributes
    ----------
    model : dict
        Each key is either the instance integer for an ElecMeter,
        or a tuple of instances for a MeterGroup.
        Each value is a sorted list of power in different states.
    """

    def __init__(self,params):
        self.model = {}
        self.MODEL_NAME = 'Hart85'  # Add the name for the algorithm
        self.save_model_path = params.get('save-model-path', None)
        self.load_model_path = params.get('pretrained-model-path',None)
        self.chunk_wise_training = params.get('chunk_wise_training', True)
        if self.load_model_path:
            self.load_model(self.load_model_path)


    def partial_fit(
        self, train_main,train_appliances,
        buffer_size=20,
        noise_level=70,
        state_threshold=15,
        min_tolerance=100,
        percent_tolerance=0.035,
        large_transition=1000,
            **kwargs):
        """
        Train using Hart85. Places the learnt model in `model` attribute.

        Parameters
        ----------
        metergroup : a nilmtk.MeterGroup object
        columns: nilmtk.Measurement, should be one of the following
            [('power','active')]
            [('power','apparent')]
            [('power','reactive')]
            [('power','active'), ('power', 'reactive')]
        buffer_size: int, optional
            size of the buffer to use for finding edges
        min_tolerance: int, optional
            variance in power draw allowed for pairing a match
        percent_tolerance: float, optional
            if transition is greater than large_transition,
            then use percent of large_transition
        large_transition: float, optional
            power draw of a Large transition
        """
        
        # Train_appliances : list of tuples [('appliance',df),('appliance',df)]

        self.appliances=[]
        for row in train_appliances:
            self.appliances.append(row[0])
        print("...........................Hart_85 Partial Fit Running...............")
      
        train_main=train_main[0]
        l=[]
        l.append(train_main.columns[0])
        columns=l
        self.columns=columns
        self.state_threshold = state_threshold
        self.noise_level = noise_level
        [self.steady_states, self.transients] = find_steady_states(
            train_main,  noise_level=noise_level, state_threshold=state_threshold)  
        self.pair_df = self.pair(
            buffer_size, min_tolerance, percent_tolerance, large_transition)
        self.centroids = hart85_means_shift_cluster(self.pair_df, columns)
      
        print('..............................Predicting Centroid Matching..........................')
        chunk=train_main
        
        transients=self.transients
        temp_df=pd.DataFrame()
        # For now ignoring the first transient
        # transients = transients[1:]

        # Initially all appliances/meters are in unknown state (denoted by -1)
        prev = OrderedDict()
        learnt_meters = self.centroids.index.values
        for meter in learnt_meters:
            prev[meter] = -1

        
        states = pd.DataFrame(
            -1, index=chunk.index, columns=self.centroids.index.values)
           
        for transient_tuple in transients.itertuples():
            if transient_tuple[0] < chunk.index[0]:
                # Transient occurs before chunk has started; do nothing
                pass
            elif transient_tuple[0] > chunk.index[-1]:
                # Transient occurs after chunk has ended; do nothing
                pass
            else:
                # Absolute value of transient
                abs_value = np.abs(transient_tuple[1:])
                positive = transient_tuple[1] > 0
                abs_value_transient_minus_centroid = pd.DataFrame(
                    (self.centroids - abs_value).abs())
                if len(transient_tuple) == 2:
                    # 1d data
                    index_least_delta = (abs_value_transient_minus_centroid.idxmin().values[0])
                else:
                    # 2d data.
                    # Need to find absolute value before computing minimum
                    columns = abs_value_transient_minus_centroid.columns
                    abs_value_transient_minus_centroid["multidim"] = (
                        abs_value_transient_minus_centroid[columns[0]] ** 2
                        +
                        abs_value_transient_minus_centroid[columns[1]] ** 2)
                    index_least_delta = (
                        abs_value_transient_minus_centroid["multidim"].idxmin())
                if positive:
                    # Turned on
                    states.loc[transient_tuple[0]][index_least_delta] = 1
                else:
                    # Turned off
                    states.loc[transient_tuple[0]][index_least_delta] = 0
        prev = states.iloc[-1].to_dict()
        power_chunk_dict = self.assign_power_from_states(states, prev)
        self.power_dict = power_chunk_dict
        self.chunk_index = chunk.index
        # Check whether 1d data or 2d data and converting dict to dataframe
        #print('LEN of Transient Tuple',len(transient_tuple))
        if len(transient_tuple) == 2:
            
            temp_df = pd.DataFrame(power_chunk_dict, index=chunk.index)  
        else:
            tuples = []

            for i in range(len(self.centroids.index.values)):
                for j in range(0, 2):
                    tuples.append([i, j])

            columns = pd.MultiIndex.from_tuples(tuples)

            temp_df = pd.DataFrame(
                power_chunk_dict,
                index=chunk.index,
                columns=columns)

            for i in range(len(chunk.index)):
                for j in range(len(self.centroids.index.values)):
                    for k in range(0, 2):
                        temp_df.iloc[i,j,k] = power_chunk_dict[j,i,k]
        
        self.best_matches={} 
        temp_df=temp_df.fillna(0)
        best_matches={}
        
        for row in train_appliances:
            appliance=row[0]
            appliance_df=row[1][0]
            matched_col=self.min_rmse_column(temp_df,appliance_df['power'])
            best_matches[appliance]=matched_col
        self.best_matches=best_matches        
              
                         


        
        print('...................................End Centroid Matching............................')

        self.model = dict(
                best_matches=best_matches,
                columns=columns,
                state_threshold=state_threshold,
                noise_level=noise_level,
                steady_states=self.steady_states,
                transients=self.transients,
                # pair_df=self.pair_df,
                centroids=self.centroids
            )
            
            

    def pair(self, buffer_size, min_tolerance, percent_tolerance,
             large_transition):
        subset = list(self.transients.itertuples())
        buffer = PairBuffer(columns=self.columns,
            min_tolerance=min_tolerance, buffer_size=buffer_size,
            percent_tolerance=percent_tolerance,
            large_transition=large_transition,
            num_measurements=len(self.transients.columns) + 1)
        for s in subset:
            # if len(buffer.transitionList) < bsize
            if len(buffer.transition_list) == buffer_size:
                buffer.clean_buffer()
            buffer.add_transition(s)
            buffer.pair_transitions()
        print(".........................Hart_85 Partial Fit End....................")
        return buffer.matched_pairs

    def disaggregate_chunk(self, test_mains):
        
        """
        Parameters
        ----------
        chunk : pd.DataFrame
            mains power
        prev
        transients : returned by find_steady_state_transients

        Returns
        -------
        states : pd.DataFrame
            with same index as `chunk`.
        """
        #print(test_mains)
        test_predictions_list = []
        
        for chunk in test_mains:
          
            [_, transients] = find_steady_states(
            test_mains[0],  state_threshold=self.state_threshold,
            noise_level=self.noise_level)
            #print('Transients:',transients)
            # For now ignoring the first transient
            # transients = transients[1:]

            # Initially all appliances/meters are in unknown state (denoted by -1)
            prev = OrderedDict()
            learnt_meters = self.centroids.index.values
            for meter in learnt_meters:
                prev[meter] = -1
            states = pd.DataFrame(
                -1, index=chunk.index, columns=self.centroids.index.values)
            #print('STATES:',states)    
            for transient_tuple in transients.itertuples():
                if transient_tuple[0] < chunk.index[0]:
                    # Transient occurs before chunk has started; do nothing
                    pass
                elif transient_tuple[0] > chunk.index[-1]:
                    # Transient occurs after chunk has ended; do nothing
                    pass
                else:
                    # Absolute value of transient
                    abs_value = np.abs(transient_tuple[1:])
                    positive = transient_tuple[1] > 0
                    abs_value_transient_minus_centroid = pd.DataFrame(
                        (self.centroids - abs_value).abs())
                    if len(transient_tuple) == 2:
                        # 1d data
                        index_least_delta = (abs_value_transient_minus_centroid.idxmin().values[0])
                    else:
                        # 2d data.
                        # Need to find absolute value before computing minimum
                        columns = abs_value_transient_minus_centroid.columns
                        abs_value_transient_minus_centroid["multidim"] = (
                            abs_value_transient_minus_centroid[columns[0]] ** 2
                            +
                            abs_value_transient_minus_centroid[columns[1]] ** 2)
                        index_least_delta = (
                            abs_value_transient_minus_centroid["multidim"].idxmin())
                    if positive:
                        # Turned on
                        states.loc[transient_tuple[0]][index_least_delta] = 1
                    else:
                        # Turned off
                        states.loc[transient_tuple[0]][index_least_delta] = 0
            prev = states.iloc[-1].to_dict()
            power_chunk_dict = self.assign_power_from_states(states, prev)
            self.power_dict = power_chunk_dict
            self.chunk_index = chunk.index
            # Check whether 1d data or 2d data and converting dict to dataframe
            if len(transient_tuple) == 2:
                
                temp_df = pd.DataFrame(power_chunk_dict, index=chunk.index)  
                

            else:
                tuples = []

                for i in range(len(self.centroids.index.values)):
                    for j in range(0, 2):
                        tuples.append([i, j])

                columns = pd.MultiIndex.from_tuples(tuples)

                temp_df = pd.DataFrame(
                    power_chunk_dict,
                    index=chunk.index,
                    columns=columns)

                for i in range(len(chunk.index)):
                    for j in range(len(self.centroids.index.values)):
                        for k in range(0, 2):
                            temp_df.iloc[i,j,k] = power_chunk_dict[j,i,k]
         
        temp_df=temp_df.fillna(0)
 
        temp=pd.DataFrame()
        for appliance in self.appliances:
            matched_col=self.best_matches[appliance]
            temp[appliance]=temp_df[matched_col]
        test_predictions_list.append(temp)    
          
        
        return test_predictions_list
    
    def min_rmse_column(self,temp_df,gt_df):
       
        rmse_all=[]
        a=len(temp_df)
        b=len(gt_df)
        if a>b:
            temp_df=temp_df[:b]
        elif b>a:
            gt_df=gt_df[:a]
        for col in temp_df.columns:
            temp_value=(np.sqrt(mean_squared_error(temp_df[col], gt_df)))
            rmse_all.append(temp_value)
        minpos = rmse_all.index(min(rmse_all))
        return minpos     
           



    def assign_power_from_states(self, states_chunk, prev):
        di = {}
        ndim = len(self.centroids.columns)
        for appliance in states_chunk.columns:
            values = states_chunk[[appliance]].values.flatten()
            if ndim == 1:
                power = np.zeros(len(values), dtype=int)
            else:
                power = np.zeros((len(values), 2), dtype=int)
            # on = False
            i = 0
            while i < len(values) - 1:
                if values[i] == 1:
                    # print("A", values[i], i)
                    on = True
                    i = i + 1
                    power[i] = self.centroids.loc[appliance].values
                    while values[i] != 0 and i < len(values) - 1:
                        # print("B", values[i], i)
                        power[i] = self.centroids.loc[appliance].values
                        i = i + 1
                elif values[i] == 0:
                    # print("C", values[i], i)
                    on = False
                    i = i + 1
                    power[i] = 0
                    while values[i] != 1 and i < len(values) - 1:
                        # print("D", values[i], i)
                        if ndim == 1:
                            power[i] = 0
                        else:
                            power[i] = [0, 0]
                        i = i + 1
                else:
                    # print("E", values[i], i)
                    # Unknown state. If previously we know about this
                    # appliance's state, we can
                    # use that. Else, it defaults to 0
                    if prev[appliance] == -1 or prev[appliance] == 0:
                        # print("F", values[i], i)
                        on = False
                        power[i] = 0
                        while values[i] != 1 and i < len(values) - 1:
                            # print("G", values[i], i)
                            if ndim == 1:
                                power[i] = 0
                            else:
                                power[i] = [0, 0]
                            i = i + 1
                    else:
                        # print("H", values[i], i)
                        on = True
                        power[i] = self.centroids.loc[appliance].values
                        while values[i] != 0 and i < len(values) - 1:
                            # print("I", values[i], i)
                            power[i] = self.centroids.loc[appliance].values
                            i = i + 1

            di[appliance] = power
            # print(power.sum())
        return di

 
    # filename=model.pickle

    def export_model(self, filename):

        example_dict = self.model

        pickle_out = open(filename, "wb")
        pickle.dump(example_dict, pickle_out)
        pickle_out.close()

    def import_model(self, filename):
        pickle_in = open(filename)
        self.model = pickle.load(pickle_in)
        self.columns = self.model['columns']
        self.state_threshold = self.model['state_threshold']
        self.noise_level = self.model['noise_level']
        self.steady_states = self.model['steady_states']
        self.transients = self.model['transients']
        # pair_df=self.pair_df,
        self.centroids = self.model['centroids']
    
    def best_matched_appliance(self,submeters, pred_df):
        
        """
        Parameters
        ----------
        submeters : elec.submeters object
        pred_df : predicted dataframe returned by disaggregate()

        Returns
        -------
        list : containing best matched pairs to disaggregated output

        """

        rms_error = {}
        submeters_df=submeters.dataframe_of_meters()
        new_df=pd.merge(pred_df, submeters_df, left_index=True, right_index=True)

        rmse_all=[]
        for pred_appliance in pred_df.columns:
            rmse={}
            for appliance in submeters_df.columns:
                temp_value=(np.sqrt(mean_squared_error(new_df[pred_appliance], new_df[appliance])))
                rmse[appliance]=temp_value         
            rmse_all.append(rmse)
        match=[]    
        for i in range(len(rmse_all)) :
            key_min = min(rmse_all[i].keys(), key=(lambda k: rmse_all[i][k]))
            print('Best Matched Pair is',(i,key_min))    
 
    