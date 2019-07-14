from __future__ import print_function, division
from warnings import warn

import pandas as pd
import numpy as np
import pickle
import os
from collections import OrderedDict
from nilmtk.disaggregate import Disaggregator
from nilmtk.datastore import HDFDataStore


class Zero(Disaggregator):

    def __init__(self, d):
        self.model = OrderedDict()
        self.MIN_CHUNK_LENGTH = 100
        self.MODEL_NAME = 'Zero'

    def partial_fit(self, train_main, train_appliances, **load_kwargs):
        '''
                train_main :- pandas DataFrame. It will contain the mains reading.
                train_appliances :- [('appliance1',df1),('appliance2',df2),...]

        '''
        train_main = pd.concat(train_main, axis=0)
        train_app_tmp = []

        for app_name, df_list in train_appliances:
            df_list = pd.concat(df_list, axis=0)
            train_app_tmp.append((app_name, df_list))

        train_appliances = train_app_tmp

        print("...............Zero partial_fit running...............")
        for appliance, readings in train_appliances:

            # there will be only off state for all appliances.
            # the algorithm will always predict zero

            self.model[appliance] = {'states': 0}

        # Saving Model
        if not os.path.exists('zero'):
            os.mkdir('zero')

        for app in self.model:
            pickle_out = open("zero/" + app + ".pickle", "wb")
            pickle.dump(self.model[app], pickle_out)
            pickle_out.close()

    def disaggregate_chunk(self, test_mains, model=None):

        if model is not None:
            self.model = model

        print("...............Zero disaggregate_chunk running...............")

        test_predictions_list = []

        for test_df in test_mains:

            appliance_powers_dict = {}
            for i in self.model:
                print("Estimating power demand for '{}'"
                      .format(i))

                # a list of predicted power values for ith appliance
                predicted_power = [self.model[i]['states']
                                   for j in range(0, test_df.shape[0])]
                column = pd.Series(
                    predicted_power, index=test_df.index, name=i)
                appliance_powers_dict[i] = column

            appliance_powers = pd.DataFrame(
                appliance_powers_dict, dtype='float32')

            test_predictions_list.append(appliance_powers)

        return test_predictions_list
