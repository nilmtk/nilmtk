from nilmtk.dataset import DataSet
from nilmtk.metergroup import MeterGroup
import pandas as pd
from nilmtk.disaggregate import *
from nilmtk.disaggregate import Disaggregator
from six import iteritems
from sklearn.metrics import mean_squared_error, mean_absolute_error, f1_score
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
import os
import pickle
import datetime


class API():

    def __init__(self, d):

        self.chunk_size = 40000
        self.power = {}
        self.sample_period = 1
        self.appliances = []
        self.methods = {}
        self.chunk_size = None
        self.method_dict = {
            'CO': {},
            'FHMM': {},
            'Hart85': {},
            'DAE': {},
            'Mean': {},
            'Zero': {},
            'WindowGRU': {},
            'Seq2Point': {},
            'RNN': {},
            'Seq2Seq': {}
        }
        self.pre_trained = False
        self.metrics = []
        self.pre_trained = None
        self.train_datasets_dict = {}
        self.test_datasets_dict = {}
        self.artificial_aggregate = False
        self.train_submeters = []
        self.train_mains = pd.DataFrame()
        self.test_submeters = []
        self.test_mains = pd.DataFrame()
        self.gt_overall = {}
        self.pred_overall = {}
        self.classifiers = []
        self.dictionary = {}
        self.DROP_ALL_NANS = True
        self.FILL_ALL_NANS = False
        self.mae = pd.DataFrame()
        self.rmse = pd.DataFrame()

        self.experiment(d)

    def experiment(self, d):

        self.dictionary = d
        self.initialise(d)
        if d['preprocessing']:
            print('oo Training')
            self.train_test_preprocessed_data()

        elif d['chunk_size']:
            print('oo Chunk Training')
            self.load_datasets_chunks()

        else:
            self.load_datasets()

    def initialise(self, d):

        for elems in d['power']:
            self.power = d['power']
        self.sample_period = d['sample_rate']
        for elems in d['appliances']:
            self.appliances.append(elems)
        self.pre_trained = d['pre_trained']
        self.train_datasets_dict = d['train']['datasets']
        self.test_datasets_dict = d['test']['datasets']
        self.pre_trained = d['pre_trained']
        self.methods = d['methods']

        if "artificial_aggregate" in d:
            self.artificial_aggregate = d['artificial_aggregate']
        if "chunk_size" in d:
            self.chunk_size = d['chunk_size']

        self.metrics = d['test']['metrics']

    def train_test_preprocessed_data(self):

        # chunkwise training and testing from preprocessed file
        # Training
        self.store_classifier_instances()
        d = self.dictionary

        train_file = pd.HDFStore(d['preprocess_train_path'], "r")
        keys = train_file.keys()

        # Processed HDF5 keys will be of the following format
        # /dataset_name/building_name/

        tuples = [i.split('/')[1:] for i in keys]
        datasets_list = list(set([i[0] for i in tuples]))

        for dataset_name in datasets_list:
            # Choose the buildings for the selected dataset
            available_buildings_in_current_dataset = list(
                set([i[1] for i in tuples if i[0] == dataset_name]))

            for building_id in available_buildings_in_current_dataset:

                available_chunks = list(
                    set([i[2] for i in tuples if (i[0] == dataset_name) and i[1] == building_id]))

                for chunk_id in available_chunks:
                    mains_list = []
                    total_num_windows = len([i for i in tuples if (
                        i[0] == dataset_name and i[1] == building_id and i[2] == chunk_id and i[3] == 'mains')])
                    for window_id in range(total_num_windows):
                        mains_df = train_file['/%s/%s/%s/%s/%s' %
                                              (dataset_name, building_id, chunk_id, 'mains', str(window_id))]
                        mains_list.append(mains_df)

                    list_of_appliances = []

                    for app_name in self.appliances:
                        list_of_readings_for_current_appliance = []
                        for window_id in range(total_num_windows):
                            appliance_df = train_file['/%s/%s/%s/%s/%s' % (
                                dataset_name, building_id, chunk_id, app_name, str(window_id))]
                            list_of_readings_for_current_appliance.append(
                                appliance_df)

                        list_of_appliances.append(
                            (app_name, list_of_readings_for_current_appliance))

                    self.train_mains = mains_list
                    self.train_submeters = list_of_appliances
                    print("Training on ", dataset_name, building_id, chunk_id)
                    self.call_partial_fit()

        train_file.close()

    def load_datasets_chunks(self):

        self.store_classifier_instances()
        d = self.train_datasets_dict

        print("............... Loading Data for preprocessing ...................")
        # store the train_main readings for all buildings

        print("............... Loading Train_Mains for preprocessing ...................")

        for dataset in d:
            print("Loading data for ", dataset, " dataset")

            for building in d[dataset]['buildings']:
                train = DataSet(d[dataset]['path'])
                print("Loading building ... ", building)
                train.set_window(
                    start=d[dataset]['buildings'][building]['start_time'],
                    end=d[dataset]['buildings'][building]['end_time'])
                mains_iterator = train.buildings[building].elec.mains().load(
                    chunksize=self.chunk_size,
                    physical_quantity='power',
                    ac_type=self.power['mains'],
                    sample_period=self.sample_period)
                print(self.appliances)
                appliance_iterators = [
                    train.buildings[building].elec.select_using_appliances(
                        type=app_name).load(
                        chunksize=self.chunk_size,
                        physical_quantity='power',
                        ac_type=self.power['appliance'],
                        sample_period=self.sample_period) for app_name in self.appliances]
                for chunk_num, chunk in enumerate(train.buildings[building].elec.mains().load(
                        chunksize=self.chunk_size, physical_quantity='power', ac_type=self.power['mains'], sample_period=self.sample_period)):

                        # Dummry loop for executing on outer level. Just for
                        # looping till end of a chunk
                    print("starting enumeration..........")
                    train_df = next(mains_iterator)
                    appliance_readings = []

                    for i in appliance_iterators:
                        try:
                            appliance_df = next(i)
                        except StopIteration:
                            pass
                        appliance_readings.append(appliance_df)

                    if self.DROP_ALL_NANS:
                        train_df, appliance_readings = self.dropnans(
                            train_df, appliance_readings)

                    if self.FILL_ALL_NANS:
                        train_df, appliance_readings = self.fillnans(
                            train_df, appliance_readings)

                    if self.artificial_aggregate:
                        print("Creating an Artificial Aggregate")
                        train_df = pd.DataFrame(
                            np.zeros(
                                appliance_readings[0].shape),
                            index=appliance_readings[0].index,
                            columns=appliance_readings[0].columns)
                        for app_reading in appliance_readings:
                            train_df += app_reading

                    train_appliances = []

                    for cnt, i in enumerate(appliance_readings):
                        train_appliances.append((self.appliances[cnt], [i]))

                    self.train_mains = [train_df]
                    self.train_submeters = train_appliances
                    self.call_partial_fit()

        print("...............Finished Loading Train mains and Appliances for preprocessing ...................")

        # store train submeters reading
        d = self.test_datasets_dict
        # store the test_main readings for all buildings

        for dataset in d:
            print("Loading data for ", dataset, " dataset")
            for building in d[dataset]['buildings']:
                test = DataSet(d[dataset]['path'])
                test.set_window(
                    start=d[dataset]['buildings'][building]['start_time'],
                    end=d[dataset]['buildings'][building]['end_time'])
                mains_iterator = test.buildings[building].elec.mains().load(
                    chunksize=self.chunk_size,
                    physical_quantity='power',
                    ac_type=self.power['mains'],
                    sample_period=self.sample_period)
                appliance_iterators = [
                    test.buildings[building].elec.select_using_appliances(
                        type=app_name).load(
                        chunksize=self.chunk_size,
                        physical_quantity='power',
                        ac_type=self.power['appliance'],
                        sample_period=self.sample_period) for app_name in self.appliances]
                for chunk_num, chunk in enumerate(test.buildings[building].elec.mains().load(
                        chunksize=self.chunk_size, physical_quantity='power', ac_type=self.power['mains'], sample_period=self.sample_period)):

                    test_df = next(mains_iterator)
                    appliance_readings = []
                    for i in appliance_iterators:
                        try:
                            appliance_df = next(i)
                        except StopIteration:
                            pass
                        appliance_readings.append(appliance_df)
                    if self.DROP_ALL_NANS:
                        test_df, appliance_readings = self.dropnans(
                            test_df, appliance_readings)

                    if self.FILL_ALL_NANS:
                        test_df, appliance_readings = self.fillnans(
                            test_df, appliance_readings)

                    if self.artificial_aggregate:
                        print("Creating an Artificial Aggregate")

                        test_df = pd.DataFrame(
                            np.zeros(
                                appliance_readings[0].shape),
                            index=appliance_readings[0].index,
                            columns=appliance_readings[0].columns)
                        for app_reading in appliance_readings:
                            test_df += app_reading

                    test_appliances = []

                    for cnt, i in enumerate(appliance_readings):
                        test_appliances.append((self.appliances[cnt], [i]))

                    self.test_mains = test_df
                    self.test_submeters = test_appliances
                    print(
                        "Dataset %s Building %s chunk %s" %
                        (dataset, building, chunk_num))

                    self.test_mains = [self.test_mains]
                    self.call_predict(self.classifiers)

    def dropnans(self, mains_df, appliance_dfs):

        print("Droppping NANS")
        mains_df = mains_df.dropna()

        for i in range(len(appliance_dfs)):
            appliance_dfs[i] = appliance_dfs[i].dropna()

        ix = mains_df.index
        for app_df in appliance_dfs:
            ix = ix.intersection(app_df.index)

        mains_df = mains_df.loc[ix]

        new_appliances_list = []
        for app_df in appliance_dfs:
            new_appliances_list.append(app_df.loc[ix])

        return mains_df, new_appliances_list

    def fillnans(self, mains_df, appliance_dfs):

        print("Filling NANS")

        # First find indexes in main where it has nans

        datelist = pd.date_range(mains_df.index[0].date(), datetime.timedelta(
            days=1) + mains_df.index[-1].date(), freq='60S', tz='US/Eastern').tolist()[:-1]
        mains_df = mains_df.reindex(index=datelist)
        indexes = pd.isnull(mains_df)
        mains_df = mains_df.fillna(0)

        for i in range(len(appliance_dfs)):
            appliance_dfs[i] = appliance_dfs[i].reindex(index=datelist)
            appliance_dfs[i] = appliance_dfs[i].fillna(0)
            appliance_dfs[i][indexes] = 0

        new_appliances_list = []
        for app_df in appliance_dfs:
            new_appliances_list.append(app_df)

        return mains_df, new_appliances_list

    def load_datasets(self):

        self.store_classifier_instances()
        d = self.train_datasets_dict

        print("............... Loading Data for preprocessing ...................")
        # store the train_main readings for all buildings

        print("............... Loading Train_Mains for preprocessing ...................")

        for dataset in d:
            print("Loading data for ", dataset, " dataset")

            for building in d[dataset]['buildings']:
                train = DataSet(d[dataset]['path'])
                print("Loading building ... ", building)
                train.set_window(
                    start=d[dataset]['buildings'][building]['start_time'],
                    end=d[dataset]['buildings'][building]['end_time'])
                mains_iterator = train.buildings[building].elec.mains().load(
                    physical_quantity='power',
                    ac_type=self.power['mains'],
                    sample_period=self.sample_period)
                print(self.appliances)
                appliance_iterators = [
                    train.buildings[building].elec.select_using_appliances(
                        type=app_name).load(
                        physical_quantity='power',
                        ac_type=self.power['appliance'],
                        sample_period=self.sample_period) for app_name in self.appliances]
                for chunk_num, chunk in enumerate(train.buildings[building].elec.mains().load(
                        physical_quantity='power', ac_type=self.power['mains'], sample_period=self.sample_period)):

                        # Dummry loop for executing on outer level. Just for
                        # looping till end of a chunk
                    print("starting enumeration..........")
                    train_df = next(mains_iterator)
                    appliance_readings = []

                    for i in appliance_iterators:
                        try:
                            appliance_df = next(i)
                        except StopIteration:
                            pass
                        appliance_readings.append(appliance_df)

                    if self.DROP_ALL_NANS:
                        train_df, appliance_readings = self.dropnans(
                            train_df, appliance_readings)

                    if self.FILL_ALL_NANS:
                        train_df, appliance_readings = self.fillnans(
                            train_df, appliance_readings)

                    if self.artificial_aggregate:
                        print("Creating an Artificial Aggregate")
                        train_df = pd.DataFrame(
                            np.zeros(
                                appliance_readings[0].shape),
                            index=appliance_readings[0].index,
                            columns=appliance_readings[0].columns)
                        for app_reading in appliance_readings:
                            train_df += app_reading

                    train_appliances = []

                    for cnt, i in enumerate(appliance_readings):
                        train_appliances.append((self.appliances[cnt], [i]))

                    self.train_mains = [train_df]
                    self.train_submeters = train_appliances
                    self.call_partial_fit()

        print("...............Finished Loading Train mains and Appliances for preprocessing ...................")

        # store train submeters reading
        d = self.test_datasets_dict
        # store the test_main readings for all buildings

        for dataset in d:
            print("Loading data for ", dataset, " dataset")
            for building in d[dataset]['buildings']:
                test = DataSet(d[dataset]['path'])
                test.set_window(
                    start=d[dataset]['buildings'][building]['start_time'],
                    end=d[dataset]['buildings'][building]['end_time'])
                mains_iterator = test.buildings[building].elec.mains().load(
                    physical_quantity='power',
                    ac_type=self.power['mains'],
                    sample_period=self.sample_period)
                appliance_iterators = [
                    test.buildings[building].elec.select_using_appliances(
                        type=app_name).load(
                        physical_quantity='power',
                        ac_type=self.power['appliance'],
                        sample_period=self.sample_period) for app_name in self.appliances]
                for chunk_num, chunk in enumerate(test.buildings[building].elec.mains().load(
                        physical_quantity='power', ac_type=self.power['mains'], sample_period=self.sample_period)):

                    test_df = next(mains_iterator)
                    appliance_readings = []
                    for i in appliance_iterators:
                        try:
                            appliance_df = next(i)
                        except StopIteration:
                            pass
                        appliance_readings.append(appliance_df)
                    if self.DROP_ALL_NANS:
                        test_df, appliance_readings = self.dropnans(
                            test_df, appliance_readings)

                    if self.FILL_ALL_NANS:
                        test_df, appliance_readings = self.fillnans(
                            test_df, appliance_readings)

                    if self.artificial_aggregate:
                        print("Creating an Artificial Aggregate")

                        test_df = pd.DataFrame(
                            np.zeros(
                                appliance_readings[0].shape),
                            index=appliance_readings[0].index,
                            columns=appliance_readings[0].columns)
                        for app_reading in appliance_readings:
                            test_df += app_reading

                    test_appliances = []

                    for cnt, i in enumerate(appliance_readings):
                        test_appliances.append((self.appliances[cnt], [i]))

                    self.test_mains = test_df
                    self.test_submeters = test_appliances
                    print(
                        "Dataset %s Building %s chunk %s" %
                        (dataset, building, chunk_num))

                    self.test_mains = [self.test_mains]
                    self.call_predict(self.classifiers)

    def store_classifier_instances(self):

        method_dict = {}
        for i in self.method_dict:
            if i in self.methods:
                self.method_dict[i].update(self.methods[i])

        print(self)
        method_dict = {'CO': CombinatorialOptimisation(self.method_dict['CO']),
                       'FHMM': FHMM(self.method_dict['FHMM']),
                       'DAE': DAE(self.method_dict['DAE']),
                       'Mean': Mean(self.method_dict['Mean']),
                       'Zero': Zero(self.method_dict['Zero']),
                       'WindowGRU': WindowGRU(self.method_dict['WindowGRU']),
                       'Seq2Point': Seq2Point(self.method_dict['Seq2Point']),
                       'RNN': RNN(self.method_dict['RNN']),
                       'Seq2Seq': Seq2Seq(self.method_dict['Seq2Seq'])
                       }

        for name in self.methods:
            if name in method_dict:
                clf = method_dict[name]
                self.classifiers.append((name, clf))

    def call_predict(self, classifiers):

        pred_overall = {}
        gt_overall = {}
        for name, clf in classifiers:
            if self.pre_trained is not None:
                if name in self.pre_trained:
                    model = OrderedDict()
                    model_path = self.pre_trained[name]
                    files = [
                        f for f in os.listdir(model_path) if os.path.isfile(
                            os.path.join(
                                model_path, f))]

                    for appliances in files:
                        pickle_in = open(model_path + '/' + appliances, "rb")
                        model[appliances[:-7]] = pickle.load(pickle_in)

                    gt_overall, pred_overall[name] = self.predict(
                        clf, self.test_mains, self.test_submeters, self.sample_period, 'Europe/London', model=model)

            else:
                gt_overall, pred_overall[name] = self.predict(
                    clf, self.test_mains, self.test_submeters, self.sample_period, 'Europe/London')

        self.gt_overall = gt_overall

        self.pred_overall = pred_overall

        for i in gt_overall.columns:
            plt.figure(figsize=(20, 8))
            plt.plot(gt_overall[i].values, label='truth')
            for clf in pred_overall:
                plt.plot(pred_overall[clf][i].values, label=clf)
            plt.title(i)
            plt.legend()
            plt.show()

        # metrics

        if gt_overall.size > 0:

            for metrics in self.metrics:

                if metrics == 'f1-score':
                    f1_score = {}

                    for clf_name, clf in classifiers:
                        f1_score[clf_name] = self.compute_f1_score(
                            gt_overall, pred_overall[clf_name])
                    f1_score = pd.DataFrame(f1_score)
                    print("............ ", metrics, " ..............")
                    print(f1_score)

                if metrics == 'rmse':
                    rmse = {}
                    for clf_name, clf in classifiers:
                        rmse[clf_name] = self.compute_rmse(
                            gt_overall, pred_overall[clf_name])
                    rmse = pd.DataFrame(rmse)
                    self.rmse = rmse
                    print("............ ", metrics, " ..............")
                    print(rmse)

                if metrics == 'mae':
                    mae = {}
                    for clf_name, clf in classifiers:
                        mae[clf_name] = self.compute_mae(
                            gt_overall, pred_overall[clf_name])
                    mae = pd.DataFrame(mae)
                    self.mae = mae
                    print("............ ", metrics, " ..............")
                    print(mae)

                if metrics == 'rel_error':
                    rel_error = {}
                    for clf_name, clf in classifiers:
                        rel_error[clf_name] = self.compute_rel_error(
                            gt_overall, pred_overall[clf_name])
                    rel_error = pd.DataFrame(rel_error)
                    print("............ ", metrics, " ..............")
                    print(rel_error)

        else:
            print("No samples found in this chunk!")

    def call_partial_fit(self):

        print("Called Partial fit")

        # training models
        for name, clf in self.classifiers:
            if self.pre_trained is None:
                clf.partial_fit(self.train_mains, self.train_submeters)
            else:
                print(" Using pre trained model ")

    def predict(
            self,
            clf,
            test_elec,
            test_submeters,
            sample_period,
            timezone,
            model=None):

        pred_list = clf.disaggregate_chunk(test_elec, model)
        concat_pred_df = pd.concat(pred_list, axis=0)
        gt = {}

        for meter, data in test_submeters:
            concatenated_df_app = pd.concat(data, axis=1)
            index = concatenated_df_app.index
            gt[meter] = pd.Series(
                concatenated_df_app.values.flatten(), index=index)

        gt_overall = pd.DataFrame(gt, dtype='float32')
        pred = {}

        for app_name in concat_pred_df.columns:
            app_series_values = concat_pred_df[app_name].values.flatten()
            app_series_values = app_series_values[:len(gt_overall[app_name])]
            pred[app_name] = pd.Series(
                app_series_values, index=gt_overall.index)

        pred_overall = pd.DataFrame(pred, dtype='float32')
        return gt_overall, pred_overall

    def compute_rmse(self, gt, pred):

        rms_error = {}
        for appliance in gt.columns:
            rms_error[appliance] = np.sqrt(
                mean_squared_error(
                    gt[appliance], pred[appliance]))
        #print (gt['sockets'])
        # print (pred[])
        return pd.Series(rms_error)
