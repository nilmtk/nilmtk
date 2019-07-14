from __future__ import print_function, division
from warnings import warn, filterwarnings

from matplotlib import rcParams
import matplotlib.pyplot as plt
from collections import OrderedDict
import random
import sys
import pandas as pd
import numpy as np
import h5py
import os
import pickle

from keras.models import Sequential
from keras.layers import Dense, Conv1D, GRU, Bidirectional, Dropout
from keras.utils import plot_model
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
import keras.backend as K
from nilmtk.utils import find_nearest
from nilmtk.feature_detectors import cluster
from nilmtk.disaggregate import Disaggregator
from nilmtk.datastore import HDFDataStore


class WindowGRU(Disaggregator):

    def __init__(self, d):

        self.batch_size = 128
        self.epochs = 1
        self.sequence_length = 100
        self.MIN_CHUNK_LENGTH = 100
        self.MODEL_NAME = "WindowGRU"
        self.max_val = 12000
        self.models = OrderedDict()
        if 'sequence_length' in d:
            self.sequence_length = d['sequence_length']
        if 'batch_size' in d:
            self.batch_size = d['batch_size']
        if 'n_epochs' in d:
            self.epochs = d['n_epochs']
        self.MIN_CHUNK_LENGTH = self.sequence_length

    def mse(self, y_true, y_pred):

        return self.max_val * K.sqrt(K.mean((y_pred - y_true)**2))

    def partial_fit(
            self,
            train_main,
            train_appliances,
            do_preprocessing=True,
            **load_kwargs):

        if do_preprocessing:
            train_main, train_appliances = self.call_preprocessing(
                train_main, train_appliances, 'train')

        mainchunk = [np.array(x) for x in train_main]
        mainchunk = np.array(mainchunk)

        for app_name, app_df in train_appliances:

            if app_name not in self.models:
                print("First model training for ", app_name)
                self.models[app_name] = self.return_network()
            else:
                print("Started re-training model for ", app_name)

            model = self.models[app_name]

            meterchunk = [np.array(x) for x in app_df]
            meterchunk = np.array(meterchunk)
            meterchunk = meterchunk.reshape(-1, meterchunk.shape[0])
            meterchunk = meterchunk[0]
            filepath = 'temp-weights.h5'
            checkpoint = ModelCheckpoint(
                filepath,
                monitor='val_loss',
                verbose=1,
                save_best_only=True,
                mode='min')
            train_x, v_x, train_y, v_y = train_test_split(
                mainchunk, meterchunk, test_size=.15)
            model.fit(
                train_x,
                train_y,
                validation_data=[
                    v_x,
                    v_y],
                epochs=self.epochs,
                callbacks=[checkpoint],
                shuffle=True,
                batch_size=self.batch_size)
            model.load_weights(filepath)

            if not os.path.exists('onlineGRU'):
                os.mkdir('onlineGRU')

            pickle_out = open("onlineGRU/" + app_name + ".pickle", "wb")
            pickle.dump(model, pickle_out)
            pickle_out.close()

    def disaggregate_chunk(
            self,
            test_main_list,
            model=None,
            do_preprocessing=True):

        if model is not None:
            self.models = model

        if do_preprocessing:
            test_main_list = self.call_preprocessing(
                test_main_list, submeters=None, method='test')

        test_mains = [np.array(x) for x in test_main_list]
        test_mains = np.array(test_mains)
        test_predictions = []
        disggregation_dict = {}
        for appliance in self.models:

            prediction = self.models[appliance].predict(test_mains)
            prediction = np.reshape(prediction, len(prediction))
            valid_predictions = prediction.flatten()
            valid_predictions = np.where(
                valid_predictions > 0, valid_predictions, 0)
            valid_predictions = self._denormalize(
                valid_predictions, self.max_val)
            df = pd.Series(valid_predictions)
            disggregation_dict[appliance] = df

        results = pd.DataFrame(disggregation_dict, dtype='float32')
        test_predictions.append(results)
        print("test predictions ", test_predictions, len(test_predictions))
        return test_predictions

    def call_preprocessing(self, mains, submeters, method):

        max_val = self.max_val
        if method == 'train':
            print("Training processing")
            mains = pd.concat(mains, axis=0)

            # add padding values
            padding = [0 for i in range(0, self.sequence_length - 1)]
            paddf = pd.DataFrame({mains.columns.values[0]: padding})
            mains = mains.append(paddf)

            mainsarray = self.preprocess_train_mains(mains)
            mains_df_list = [pd.DataFrame(window) for window in mainsarray]

            tuples_of_appliances = []
            for (appliance_name, df) in submeters:
                df = pd.concat(df, axis=1)
                data = self.preprocess_train_appliances(df)
                appliance_df_list = [pd.DataFrame(window) for window in data]
                tuples_of_appliances.append(
                    (appliance_name, appliance_df_list))

            return mains_df_list, tuples_of_appliances

        if method == 'test':

            mains = pd.concat(mains, axis=0)

            # add padding values
            padding = [0 for i in range(0, self.sequence_length - 1)]
            paddf = pd.DataFrame({mains.columns.values[0]: padding})
            mains = mains.append(paddf)

            mainsarray = self.preprocess_test_mains(mains)
            mains_df_list = [pd.DataFrame(window) for window in mainsarray]
            return mains_df_list

    def preprocess_test_mains(self, mains):

        mains = self._normalize(mains, self.max_val)
        mainsarray = np.array(mains)
        indexer = np.arange(self.sequence_length)[
            None, :] + np.arange(len(mainsarray) - self.sequence_length + 1)[:, None]
        mainsarray = mainsarray[indexer]
        return mainsarray

    def preprocess_train_appliances(self, appliance):

        appliance = self._normalize(appliance, self.max_val)
        appliancearray = np.array(appliance)
        return appliancearray

    def preprocess_train_mains(self, mains):

        mains = self._normalize(mains, self.max_val)
        mainsarray = np.array(mains)
        indexer = np.arange(self.sequence_length)[
            None, :] + np.arange(len(mainsarray) - self.sequence_length + 1)[:, None]
        mainsarray = mainsarray[indexer]
        return mainsarray

    def _normalize(self, chunk, mmax):
        '''Normalizes timeseries

        Parameters
        ----------
        chunk : the timeseries to normalize
        max : max value of the powerseries

        Returns: Normalized timeseries
        '''

        tchunk = chunk / mmax
        return tchunk

    def _denormalize(self, chunk, mmax):
        '''Deormalizes timeseries
        Note: This is not entirely correct

        Parameters
        ----------
        chunk : the timeseries to denormalize
        max : max value used for normalization

        Returns: Denormalized timeseries
        '''
        tchunk = chunk * mmax
        return tchunk

    def return_network(self):
        '''Creates the GRU architecture described in the paper
        '''
        model = Sequential()

        # 1D Conv
        model.add(
            Conv1D(
                16,
                4,
                activation='relu',
                input_shape=(
                    self.sequence_length,
                    1),
                padding="same",
                strides=1))

        # Bi-directional GRUs
        model.add(Bidirectional(GRU(64, activation='relu',
                                    return_sequences=True), merge_mode='concat'))
        model.add(Dropout(0.5))
        model.add(Bidirectional(GRU(128, activation='relu',
                                    return_sequences=False), merge_mode='concat'))
        model.add(Dropout(0.5))

        # Fully Connected Layers
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='linear'))

        model.compile(loss='mse', optimizer='adam')

        return model
