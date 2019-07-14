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


class DAE(Disaggregator):

    def __init__(self, d):

        self.sequence_length = 300
        self.n_epochs = 100
        self.trained = False
        self.models = OrderedDict()
        self.mains_mean = 1000
        self.mains_std = 6000

        if 'sequence_length' in d:
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

        print("...............DAE partial_fit running...............")

        #print (train_main[0])

        if do_preprocessing:
            train_main, train_appliances = self.call_preprocessing(
                train_main, train_appliances, 'train')

        train_main = np.array(
            [i.values.reshape((self.sequence_length, 1)) for i in train_main])

        new_train_appliances = []

        for app_name, app_df in train_appliances:
            app_df = np.array(
                [i.values.reshape((self.sequence_length, 1)) for i in app_df])
            new_train_appliances.append((app_name, app_df))

        train_appliances = new_train_appliances

        for appliance_name, power in train_appliances:

            if appliance_name not in self.models:
                print("First model training for ", appliance_name)
                self.models[appliance_name] = self.return_network()

            else:

                print("Started Retraining model for ", appliance_name)

            model = self.models[appliance_name]

            if len(train_main) > 3:
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
                    shuffle=True)
                model.load_weights(filepath)
                if not os.path.exists('dae'):
                    os.mkdir('dae')
                pickle_out = open("dae/" + appliance_name + ".pickle", "wb")
                pickle.dump(model, pickle_out)
                pickle_out.close()

            else:
                print("This chunk has small number of samples, so skipping the training")

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

        test_predictions = []

        for test_main in test_main_list:

            test_main = test_main.values
            test_main = test_main.reshape((-1, self.sequence_length, 1))
            disggregation_dict = {}

            for appliance in self.models:
                prediction = self.models[appliance].predict(test_main)
                prediction = prediction * self.mains_std
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
                8,
                4,
                activation="linear",
                input_shape=(
                    self.sequence_length,
                    1),
                padding="same",
                strides=1))
        model.add(Flatten())

        # Fully Connected Layers
        model.add(Dropout(0.2))
        model.add(Dense((self.sequence_length) * 8, activation='relu'))

        model.add(Dropout(0.2))
        model.add(Dense(128, activation='relu'))

        model.add(Dropout(0.2))

        model.add(Dense((self.sequence_length) * 8, activation='relu'))

        model.add(Dropout(0.2))

        # 1D Conv
        model.add(Reshape(((self.sequence_length), 8)))
        model.add(Conv1D(1, 4, activation="linear", padding="same", strides=1))

        model.compile(loss='mse', optimizer='adam')

        return model

    def call_preprocessing(self, mains, submeters, method):
        sequence_length = self.sequence_length

        #max_val = self.max_val
        if method == 'train':
            print("Training processing")
            print(mains[0].shape, submeters[0][1][0].shape)
            mains = pd.concat(mains, axis=1)
            mains = self.neural_nilm_preprocess_input(
                mains.values, sequence_length, self.mains_mean, self.mains_std, False)
            print("Means is ")
            print(np.mean(mains))
            print(mains.shape, np.max(mains))
            mains_df_list = [pd.DataFrame(window) for window in mains]

            tuples_of_appliances = []

            for (appliance_name, df) in submeters:
                # if appliance_name in self.appliance_params:
                #     app_mean =

                if appliance_name in self.appliance_params:
                    app_mean = self.appliance_params[appliance_name]['mean']
                    app_std = self.appliance_params[appliance_name]['std']

                df = pd.concat(df, axis=1)

                #data = self.neural_nilm_preprocess_output(df.values, sequence_length,app_mean,app_std,False)
                data = self.neural_nilm_preprocess_output(
                    df.values, sequence_length, self.mains_mean, self.mains_std, False)

                appliance_df_list = [pd.DataFrame(window) for window in data]

                tuples_of_appliances.append(
                    (appliance_name, appliance_df_list))

            return mains_df_list, tuples_of_appliances

        if method == 'test':
            print("Testing processing")
            mains = pd.concat(mains, axis=1)
            mains = self.neural_nilm_preprocess_input(
                mains.values, sequence_length, self.mains_mean, self.mains_std, False)
            print("Means is ")
            print(np.mean(mains))
            print(mains.shape, np.max(mains))
            mains_df_list = [pd.DataFrame(window) for window in mains]
            return mains_df_list

    # def neural_nilm_choose_windows(self, data, sequence_length):

    #     excess_entries =  sequence_length - (data.size % sequence_length)
    #     lst = np.array([0] * excess_entries)
    #     arr = np.concatenate((data.flatten(), lst),axis=0)

    #     return arr.reshape((-1,sequence_length))

    def neural_nilm_preprocess_input(
            self,
            data,
            sequence_length,
            mean,
            std,
            overlapping=False):

        #mean_sequence = np.mean(windowed_x,axis=1).reshape((-1,1))
        n = sequence_length
        excess_entries = sequence_length - (data.size % sequence_length)
        lst = np.array([0] * excess_entries)
        arr = np.concatenate((data.flatten(), lst), axis=0)
        if overlapping:
            windowed_x = np.array([arr[i:i + n]
                                   for i in range(len(arr) - n + 1)])
        else:
            windowed_x = arr.reshape((-1, sequence_length))

        #windowed_x = windowed_x - mean
        # mean_sequence # Mean centering each sequence
        print("Just flat")
        plt.plot(windowed_x.flatten()[:1000])
        plt.ylim(0, 2000)
        plt.show()
        return (windowed_x / std).reshape((-1, sequence_length))

    def neural_nilm_preprocess_output(
            self,
            data,
            sequence_length,
            mean,
            std,
            overlapping=False):

        n = sequence_length
        excess_entries = sequence_length - (data.size % sequence_length)
        lst = np.array([0] * excess_entries)
        arr = np.concatenate((data.flatten(), lst), axis=0)

        if overlapping:
            windowed_y = np.array([arr[i:i + n]
                                   for i in range(len(arr) - n + 1)])
        else:
            windowed_y = arr.reshape((-1, sequence_length))

        #self.appliance_wise_max[appliance_name] = self.default_max_reading
        windowed_y = windowed_y  # - mean
        plt.plot(windowed_y.flatten()[:1000])
        plt.ylim(0, 2000)
        plt.show()
        return (windowed_y / std).reshape((-1, sequence_length))
        # return
        # (windowed_y/max_value_of_reading).reshape((-1,sequence_length))
