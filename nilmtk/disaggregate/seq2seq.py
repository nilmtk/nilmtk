from __future__ import print_function, division
from warnings import warn

from nilmtk.disaggregate import Disaggregator
from keras.layers import Conv1D, Dense, Dropout, Reshape, Flatten

import os
import pandas as pd
import numpy as np
import pickle
from collections import OrderedDict

from keras.optimizers import SGD
from keras.models import Sequential, load_model
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
import keras.backend as K


class Seq2Seq(Disaggregator):

    def __init__(self, d):

        self.sequence_length = 99
        self.n_epochs = 100
        self.trained = False
        self.models = OrderedDict()
        #self.max_value = 6000
        self.mains_mean = 1000
        self.mains_std = 1800
        self.batch_size = 512

        self.appliance_std = None

        if 'sequence_length' in d:
            if d['sequence_length'] % 2 == 0:
                raise ValueError("Sequence length should be a odd number!!!")
            self.sequence_length = d['sequence_length']

        if 'n_epochs' in d:
            self.n_epochs = d['n_epochs']

        if 'mains_mean' in d:
            self.mains_mean = d['mains_mean']

        if 'mains_std' in d:
            self.mains_std = d['mains_std']

        if 'appliance_params' in d:
            self.appliance_params = d['appliance_params']

    def partial_fit(
            self,
            train_main,
            train_appliances,
            do_preprocessing=True,
            **load_kwargs):

        print("...............Seq2Seq partial_fit running...............")

        if do_preprocessing:
            train_main, train_appliances = self.call_preprocessing(
                train_main, train_appliances, 'train')

        train_main = np.array(
            [i.values.reshape((self.sequence_length, 1)) for i in train_main])

        new_train_appliances = []

        for app_name, app_df in train_appliances:
            app_df = np.array([i.values for i in app_df]).reshape(
                (-1, self.sequence_length))
            new_train_appliances.append((app_name, app_df))

        train_appliances = new_train_appliances

        for appliance_name, power in train_appliances:

            if appliance_name not in self.models:
                print("First model training for ", appliance_name)
                self.models[appliance_name] = self.return_network()

            else:

                print("Started Retraining model for ", appliance_name)

            model = self.models[appliance_name]

            if train_main.size > 0:
                # Sometimes chunks can be empty after dropping NANS
                if len(train_main) > 10:
                    # Do validation when you have sufficient samples
                    filepath = 'temp-weights.h5'

                    checkpoint = ModelCheckpoint(
                        filepath,
                        monitor='val_loss',
                        verbose=1,
                        save_best_only=True,
                        mode='min')
                    train_x, v_x, train_y, v_y = train_test_split(
                        train_main, power, test_size=.15)

                    model.fit(
                        train_x,
                        train_y,
                        validation_data=[
                            v_x,
                            v_y],
                        epochs=self.n_epochs,
                        callbacks=[checkpoint],
                        batch_size=self.batch_size)
                    model.load_weights(filepath)
                    if not os.path.exists('seq2seq'):
                        os.mkdir('seq2seq')
                    pickle_out = open(
                        "seq2seq/" + appliance_name + ".pickle", "wb")
                    pickle.dump(model, pickle_out)
                    pickle_out.close()
                    pred = model.predict(v_x)

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

        print("New testing")
        disggregation_dict = {}
        test_predictions = []

        #print (test_main_list.shape)
        print("Length")
        print(len(test_main_list))
        test_main_array = np.array([window.values.flatten()
                                    for window in test_main_list])
        test_main_array = test_main_array.reshape(
            (-1, self.sequence_length, 1))
        print("Max input")
        print(np.max(test_main_array), np.min(test_main_array))
        for appliance in self.models:

            prediction = []
            model = self.models[appliance]
            prediction = model.predict(test_main_array, self.batch_size)
            l = self.sequence_length
            n = len(prediction) + l - 1
            val_arr = np.zeros((n))
            counts_arr = np.zeros((n))
            o = len(val_arr)
            for i in range(len(prediction)):
                val_arr[i:i + l] += prediction[i].flatten()
                counts_arr[i:i + l] += 1

            for i in range(len(val_arr)):
                val_arr[i] = val_arr[i] / counts_arr[i]

            prediction = self.appliance_params[appliance]['mean'] + (
                val_arr * self.appliance_params[appliance]['std'])
            valid_predictions = prediction.flatten()
            valid_predictions = np.where(
                valid_predictions > 0, valid_predictions, 0)
            df = pd.Series(valid_predictions)
            disggregation_dict[appliance] = df
        results = pd.DataFrame(disggregation_dict, dtype='float32')
        test_predictions.append(results)

        return test_predictions

    def return_network(self):

        model = Sequential()
        # 1D Conv
        model.add(
            Conv1D(
                30,
                10,
                activation="relu",
                input_shape=(
                    self.sequence_length,
                    1),
                strides=2))
        model.add(Conv1D(30, 8, activation='relu', strides=2))
        model.add(Conv1D(40, 6, activation='relu', strides=1))
        model.add(Conv1D(50, 5, activation='relu', strides=1))
        model.add(Dropout(.2))
        model.add(Conv1D(50, 5, activation='relu', strides=1))
        model.add(Dropout(.2))
        model.add(Flatten())
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(.2))
        model.add(Dense(self.sequence_length))
        model.compile(loss='mse', optimizer='adam')

        return model

    def call_preprocessing(self, mains, submeters, method):

        if method == 'train':
            mains = pd.concat(mains, axis=1)

            new_mains = mains.values.flatten()
            n = self.sequence_length
            units_to_pad = n // 2
            #new_mains = np.pad(new_mains, (units_to_pad,units_to_pad),'constant',constant_values = (0,0))
            new_mains = np.array([new_mains[i:i + n]
                                  for i in range(len(new_mains) - n + 1)])
            new_mains = (new_mains - self.mains_mean) / self.mains_std
            mains_df_list = [pd.DataFrame(window) for window in new_mains]
            #new_mains = pd.DataFrame(new_mains)
            appliance_list = []
            for app_index, (app_name, app_df) in enumerate(submeters):

                if app_name in self.appliance_params:
                    app_mean = self.appliance_params[app_name]['mean']
                    app_std = self.appliance_params[app_name]['std']

                app_df = pd.concat(app_df, axis=1)
                #mean_appliance = self.means[app_index]
                #std_appliance  = self.stds[app_index]
                new_app_readings = app_df.values.flatten()
                #new_app_readings = np.pad(app_df.values.flatten(), (units_to_pad,units_to_pad),'constant',constant_values = (0,0))
                new_app_readings = np.array(
                    [new_app_readings[i:i + n] for i in range(len(new_app_readings) - n + 1)])

                #new_app_readings = np.pad(new_app_readings, (units_to_pad,units_to_pad),'constant',constant_values = (0,0))

                # This is for choosing windows

                #new_mains = (new_mains - self.mean_mains)/self.std_mains

                #new_app_readings = (new_app_readings - mean_appliance)/std_appliance
                new_app_readings = (
                    new_app_readings - app_mean) / app_std  # /self.max_val
                # I know that the following window has only one value
                app_df_list = [pd.DataFrame(window)
                               for window in new_app_readings]
                #new_app_readings = pd.DataFrame(new_app_readings)
                # aggregate_list.append(new_mains)
                appliance_list.append((app_name, app_df_list))
                #new_app_readings = np.array([ new_app_readings[i:i+n] for i in range(len(new_app_readings)-n+1) ])
                #print (new_mains.shape, new_app_readings.shape, app_name)

            return mains_df_list, appliance_list

        else:

            mains = pd.concat(mains, axis=1)

            new_mains = mains.values.flatten()
            n = self.sequence_length
            units_to_pad = n // 2
            #new_mains = np.pad(new_mains, (units_to_pad,units_to_pad),'constant',constant_values = (0,0))
            new_mains = np.array([new_mains[i:i + n]
                                  for i in range(len(new_mains) - n + 1)])
            new_mains = (new_mains - self.mains_mean) / self.mains_std
            new_mains = new_mains.reshape((-1, self.sequence_length, 1))
            print(" test New mains shape")
            print(new_mains.shape)
            mains_df_list = [pd.DataFrame(window) for window in new_mains]

            return mains_df_list
