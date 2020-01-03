from datetime import datetime
import pandas as pd
import numpy as np
from ...timeframe import merge_timeframes, TimeFrame
from .disaggregator import Disaggregator
from matplotlib import pyplot as plt
from datetime import timedelta
from scipy.stats import poisson, norm
from sklearn import mixture


class MLE(Disaggregator):

    """
    Disaggregation of a single appliance based on its features and
    using the maximum likelihood of all features.

    Attributes
    ----------
    appliance: str
        Name of the appliance
    stats: list of dicts
        One dict for feature with:
    units: tuple
        For instance: ('power','active')
    resistive: boolean
        To decide if 'apparent' == 'active'
    thDelta: int
        Treshold for delta values on the power. Used on train_on_chunk method
    thLikelihood: int
        Treshold for the maximum likelihood
    sample_period: str
        For resampling in training and disaggregate methods
    sample_method: str
        Pandas method for resampling
    onpower: dict
        {'name':str, 'gmm': str, 'model': sklearn model}
    offpower: dict
        {'name':str, 'gmm': str, 'model': sklearn model}
    duration: dict
        {'name':str, 'gmm': str, 'model': sklearn model}
    onpower_train: pandas.Dataframe()
        Training samples of onpower
    offpower_train: pandas.Dataframe()
        Training samples of offpower
    duaration_train: pandas.Dataframe()
        Training samples of duration
    powerNoise: int
        For the disaggregate_chunk method, minimum delta value of a event to be
         considered, otherwise is noise.
    powerPair: int
        For the disaggregate_chunk method, max delta value difference between
         onpower and offpower
    timeWindow: int
        For the disaggregate_chunk method, a time frame to speed up
         disaggregate_chunk method.

    TODO:
    -----
    * Build a method for choosing thLikelihood automatically based on its
     optimization using ROC curve.
    * Method for measuring ROC curve.

    """

    def __init__(self):
        """
        Inizialise of the model by default

        """
        super(MLE, self).__init__()

        # Metadata
        self.appliance = None
        self.stats = []
        self.units = None
        self.resistive = False
        self.thDelta = 0
        self.thLikelihood = 0
        self.sample_period = None
        self.sampling_method = None
        # FEATURES:
        self.onpower = {'name': 'gmm', 'model': mixture.GMM(n_components=1)}
        self.offpower = {'name': 'gmm', 'model': mixture.GMM(n_components=1)}
        self.duration = {'name': 'poisson', 'model': poisson(0)}

        # Trainings:
        self.onpower_train = pd.DataFrame(columns=['onpower'])
        self.offpower_train = pd.DataFrame(columns=['offpower'])
        self.duration_train = pd.DataFrame(columns=['duration'])

        # Constrains
        self.powerNoise = 0    # Background noise in the main
        self.powerPair = 0  # Max diff between onpower and offpower
        self.timeWindow = 0        # To avoid high computation

    def __retrain(self, feature, feature_train):

        print("Training " + feature_train.columns[0])
        mu, std = norm.fit(feature_train)
        feature['model'] = norm(loc=mu, scale=std)
        '''if feature['name'] == 'gmm':
            feature['model'].fit(feature_train)
        elif feature['name'] == 'norm':
            mu, std = norm.fit(feature_train)
            feature['model'] = norm(loc=mu, scale=std)
        elif feature['name'] == 'poisson':
            self.onpower['model'] = poisson(feature_train.mean())
        else:
            raise NameError(
                "Name of the model for " + 
                str(feature_train.columns[0]) + 
                " unknown or not implemented")       ''' 

    def __physical_quantity(self, chunk): 

        if not self.resistive:
            print("Checking units")
            units_mismatched = True
            for name in chunk.columns:
                if name == self.units:
                    units = name
                    units_mismatched = False
            if units_mismatched:
                stringError = self.appliance + " cannot be disaggregated. " + self.appliance + \
                    " is a non-resistive element and  units mismatches: disaggregated data is in " + \
                    str(self.units) + \
                    " and aggregated data is " + str(units)
                raise ValueError(stringError)
        else:
            units = chunk.columns[0]
        return units

    def __pdf(self, feature, delta):

        if feature['name'] == 'norm':
            score = feature['model'].pdf(delta)
        elif feature['name'] == 'gmm':
            #score = np.exp(feature['model'].score([delta]))[0]
            score = feature['model'].pdf(delta)
        elif feature['name'] == 'poisson':
            # Decimal values produce odd values in poisson (bug)
            delta = np.round(delta)
            #score = feature['model'].pmf(delta)
            score = feature['model'].pdf(delta)
        else:
            raise AttributeError("Wrong model for" + feature['name'] +
                                 " It must be: gmm, norm or poisson")
        return score   

    def __pdf2(self, feature, delta):

        if feature['name'] == 'norm':
            score = feature['model'].pdf(delta)
        elif feature['name'] == 'gmm':
            score = np.exp(feature['model'].score([delta]))
        elif feature['name'] == 'poisson':
            # Decimal values produce odd values in poisson (bug)
            delta = np.round(delta)
            score = feature['model'].pmf(delta)
        else:
            raise AttributeError("Wrong model for" + feature['name'] +
                                 " It must be: gmm, norm or poisson")
        return score   

    def update(self, **kwargs):
        """
        This method will update attributes of the model passed by kwargs.

        Parameters
        ----------
        kwargs : key word arguments

        Notes
        -----

        """
        print("Updating model")
        print(kwargs)
        for key in kwargs:
            setattr(self, key, kwargs[key])

    def train(self, metergroup):
        """
        Train using ML.
        Call disaggregate_chunk method

        Parameters
        ----------
        metergroup : a nilmtk.MeterGroup object

        Notes
        -----
        * Inizialise "stats" and "feature_train" on the model.
        * Instance is initialised to 1. Use meter.instance to provide more
         information (TODO)

        """
        # Inizialise stats and training data:
        self.stats = []
        self.onpower_train = pd.DataFrame(columns=['onpower'])
        self.offpower_train = pd.DataFrame(columns=['offpower'])
        self.duration_train = pd.DataFrame(columns=['duration'])

        # Calling train_on_chunk by meter:
        instance = 1    # initial instance.
        for meter in metergroup.meters:
            for chunk in meter.power_series():
                if chunk.empty:
                    print("Chunk empty")
                else:
                    print("Training on chunk")
                    if self.sampling_method is not None:
                        how = lambda df: getattr(df, self.sampling_method)()
                    else:
                        how = lambda df: df.mean()
                        
                    self.train_on_chunk(how(pd.DataFrame(chunk.resample(
                        self.sample_period))),
                        meter
                    )

            instance += 1

    def train_on_chunk(self, chunk, meter):
        """
        Extracts features  from chunk, concatenates feature_train
        (onpower_train, offpower_train and duration_train) with new features
        and retrains feature
        models.
        Updates stats attribute.

        Parameters
        ----------
        chunk : pd.DataFrame where each column represents a disaggregated
        meter : ElecMeter for this chunk

        Notes
        -----
        * Disaggregates only the selected appliance.(TODO: Disaggregates many)

        """
        # EXTRACT FEATURES:
        # find units:
        self.__setattr__('units', chunk.columns[0])
        # Loading treshold for getting events:
        thDelta = getattr(self, 'thDelta')
        chunk.index.name = 'date_time'
        # To prevent learning many samples at the middle of a edge:
        chunk.ix[:, 0][chunk.ix[:, 0] < thDelta] = 0
        # Learning edges
        chunk['delta'] = chunk.ix[:, 0].diff()
        chunk.delta.fillna(0, inplace=True)
        edges = chunk[np.abs(chunk['delta']) > thDelta].delta
        # Pairing on/off events
        #print(chunk)
        if len(edges) > 1:
            offpower = edges[edges.apply(np.sign).diff() == -2]
            onpower = edges[edges.apply(np.sign).diff(-1) == 2]
            duration = offpower.reset_index().date_time - \
                onpower.reset_index().date_time
            duration = duration.astype('timedelta64[s]')

            # Set consistent index for concatenation:
            onpower = pd.DataFrame(onpower).reset_index(drop=True)
            onpower.columns = ['onpower']
            offpower = pd.DataFrame(offpower).reset_index(drop=True)
            offpower.columns = ['offpower']
            duration = pd.DataFrame(duration).reset_index(drop=True)
            duration.columns = ['duration']

            # Len of samples:
            print("Samples of onpower: " + str(len(onpower)))
            print("Samples of offpower: " + str(len(offpower)))
            print("Samples of duration: " + str(len(duration)))

            number_of_events = len(onpower)
            # Features (concatenation)
            self.onpower_train = pd.concat(
                [self.onpower_train, onpower]).reset_index(drop=True)
            self.offpower_train = pd.concat(
                [self.offpower_train, offpower]).reset_index(drop=True)
            self.duration_train = pd.concat(
                [self.duration_train, duration]).reset_index(drop=True)
        
        else:
            number_of_events = 0
            print("""WARNING: No paired events found on this chunk.
            Is it thDelta too high?""")
        
        self.duration_train = self.duration_train[self.duration_train.duration<400]

        # RE-TRAIN FEATURE MODELS:
        self.__retrain(self.onpower, self.onpower_train)
        self.__retrain(self.offpower, self.offpower_train)
        self.__retrain(self.duration, self.duration_train)

        # UPDATE STATS:
        stat_dict = {'appliance': meter.identifier[
            0], 'instance': meter.identifier[1], 'Nevents': number_of_events}
        instanceFound = False
        if len(self.stats) == 0:
            self.stats.append(stat_dict)
        else:
            for stat in self.stats:
                if ((stat['appliance'] == stat_dict['appliance']) and
                        (stat['instance'] == stat_dict['instance'])):
                    index = self.stats.index(stat)
                    self.stats[index]['Nevents'] = self.stats[
                        index]['Nevents'] + number_of_events
                    instanceFound = True
            if not instanceFound:
                self.stats.append(stat_dict)

    def disaggregate(self, mains, output_datastore):
        """
        Passes each chunk from mains generator to disaggregate_chunk()
        and passes the output to _write_disaggregated_chunk_to_datastore()
        Will have a default implementation in super class.
        Can be overridden for more simple in-memory disaggregation,
        or more complex out-of-core disaggregation.

        Parameters
        ----------
        mains : nilmtk.ElecMeter (single-phase) or nilmtk.MeterGroup (multi-phase)
        output_datastore : instance of nilmtk.DataStore or str of datastore location

        """
        
        building_path = '/building{}'.format(mains.building())
        # only writes one appliance and meter per building
        meter_instance = 2
        mains_data_location = '{}/elec/meter1'.format(building_path)
        
        #dis_main = pd.DataFrame()
        chunk_number = 0
        timeframes = []

        for chunk in mains.power_series():
        
            # Record metadata
            timeframes.append(chunk.timeframe)
            measurement = chunk.name
            cols = pd.MultiIndex.from_tuples([chunk.name])
            
            dis_chunk = self.disaggregate_chunk(
                pd.DataFrame(chunk.resample(self.sample_period, how=self.sampling_method)))
            #dis_main = pd.concat([dis_main, dis_chunk])
            chunk_number += 1
            print(str(chunk_number) + " chunks disaggregated")
            
            # Write appliance data to disag output
            key = '{}/elec/meter{}'.format(building_path, meter_instance)
            df = pd.DataFrame(
                    dis_chunk.values, index=dis_chunk.index,
                    columns=cols)
            output_datastore.append(key, df)

            # Copy mains data to disag output
            output_datastore.append(key=mains_data_location,
                                    value=pd.DataFrame(chunk, columns=cols))

        # Saving output datastore:
        #output_datastore.append(key=mains.key, value=dis_main)
        
        ##################################
        # Add metadata to output_datastore

        # TODO: `preprocessing_applied` for all meters
        # TODO: split this metadata code into a separate function
        # TODO: submeter measurement should probably be the mains
        #       measurement we used to train on, not the mains measurement.
        
        date_now = datetime.now().isoformat().split('.')[0]
        output_name = 'NILMTK_MLE_' + date_now
        resample_seconds = 10
        mains_data_location = '{}/elec/meter1'.format(building_path)

        # DataSet and MeterDevice metadata:
        meter_devices = {
            'MLE': {
                'model': 'MLE',
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
        appliance = {
            'meters': [meter_instance],
            'type': 'kettle',
            'instance': 1
            # TODO this `instance` will only be correct when the
            # model is trained on the same house as it is tested on.
            # https://github.com/nilmtk/nilmtk/issues/194
        }
        appliances.append(appliance)

        elec_meters.update({
            meter_instance: {
                'device_model': 'MLE',
                'submeter_of': 1,
                'data_location': ('{}/elec/meter{}'
                                      .format(building_path, meter_instance)),
                'preprocessing_applied': {},  # TODO
                'statistics': {
                    'timeframe': total_timeframe.to_dict()
                }
            }
        })
        elec_meters[meter_instance]['name'] = 'kettle'

        building_metadata = {
            'instance': mains.building(),
            'elec_meters': elec_meters,
            'appliances': appliances
        }

        output_datastore.save_metadata(building_path, building_metadata)

    def disaggregate_chunk(self, chunk):
        """
        Checks units.
        Disaggregates "chunk" with MaximumLikelihood algorithm.

        Optimization:
        Filters events with powerNoise.
        Filters paired-events with powerPair.
        Windowing with timeWindow for speeding up.

        Parameters
        ----------
        chunk : pd.DataFrame (in NILMTK format)

        Returns
        -------
        chunk : pd.DataFrame where each column represents a disaggregated appliance

        Notes
        -----
        * Disaggregation is not prooved. (TODO: verify the process with the Groundtruth)
        * Disaggregates only the selected appliance.(TODO: Disaggregates many)

        """

        # An resistive element has active power equal to apparent power.
        # Checking power units.
        units = self.__physical_quantity(chunk)

        # EVENTS OUT OF THE CHUNK:
        # Delta values:
        column_name = 'diff_' + units[1]
        chunk[column_name] = chunk.loc[:, units].diff()

        # Filter the noise.
        chunk['onpower'] = (chunk[column_name] > self.powerNoise)
        chunk['offpower'] = (chunk[column_name] < -self.powerNoise)
        events = chunk[(chunk.onpower == True) | (chunk.offpower == True)]

        detection_list = []
        singleOnevent = 0
        # Max Likelihood algorithm (optimized):
        for onevent in events[events.onpower == True].iterrows():
            # onTime = onevent[0]
            # deltaOn = onevent[1][1]
            # windowning:
            offevents = events[(events.offpower == True) & (events.index > onevent[0]) & (
                events.index < onevent[0] + timedelta(seconds=self.timeWindow))]
            # Filter paired events:
            offevents = offevents[
                abs(onevent[1][1] - offevents[column_name].abs()) < self.powerPair]

            # Max likelihood computation:
            if not offevents.empty:
                # pon = self.__pdf(self.onpower, onevent[1][1])
                for offevent in offevents.iterrows():
                    # offTime = offevent[0]
                    # deltaOff = offevent[1][1]
                    # poff = self.__pdf(self.offpower, offevent[1][1])
                    # duration = offevent[0] - onTime
                    # pduration = self.__pdf(self.duration, (offevent[0] - onTime).total_seconds())
                    likelihood = self.__pdf(self.onpower, onevent[1][1]) * \
                                    self.__pdf(self.offpower, offevent[1][1]) * \
                                    self.__pdf(self.duration, (offevent[0] - \
                                        onevent[0]).total_seconds())
                    detection_list.append(
                        {'likelihood': likelihood, 'onTime': onevent[0], 
                        'offTime': offevent[0], 'deltaOn': onevent[1][1]})
            else:
                singleOnevent += 1

        # Passing detections to a pandas.DataFrame
        detections = pd.DataFrame(
            columns=('onTime', 'offTime', 'likelihood', 'deltaOn'))

        for i in range(len(detection_list)):
            detections.loc[i] = [detection_list[i]['onTime'], detection_list[i][
                'offTime'], detection_list[i]['likelihood'], detection_list[i]['deltaOn']]

        detections = detections[detections.likelihood >= self.thLikelihood]

        # Constructing dis_chunk (power of disaggregated appliance)
        dis_chunk = pd.DataFrame(
            index=chunk.index, columns=[str(units[0]) + '_' + str(units[1])])
        dis_chunk.fillna(0, inplace=True)

        # Ruling out overlapped detecttions ordering by likelihood value.
        detections = detections.sort('likelihood', ascending=False)
        for row in detections.iterrows():
            # onTime = row[1][0] offTime = row[1][1] deltaOn = row[1][3]
            #import ipdb
            #ipdb.set_trace()
            if ((dis_chunk[(dis_chunk.index >= row[1][0]) &
                    (dis_chunk.index < row[1][1])].sum().values[0]) == 0):
                # delta = chunk[chunk.index == onTime][column_name].values[0]
                dis_chunk[(dis_chunk.index >= row[1][0]) & (
                    dis_chunk.index < row[1][1])] = row[1][3]

        # Stat information:
        print(str(len(events)) + " events found.")
        print(str(len(events[events.onpower == True])) + " onEvents found")
        print(str(singleOnevent) + " onEvents no paired.")

        return dis_chunk

    def no_overfitting(self):
        """
        Crops feature_train(onpower_train, offpower_train and duration_train)
        to get same samples from different appliances(same model-appliance) 
        and avoids overfittings to a many samples appliance.
        Updates stats attribute.
        Does the retraining.
        """

        # Instance with minimun length should be the maximum length
        train_len = []
        [train_len.append(st['Nevents']) for st in self.stats]
        train_len = np.array(train_len)
        max_len = train_len[train_len != 0].min()

        # CROPS FEATURE SAMPLES
        onpower_train = pd.DataFrame()
        offpower_train = pd.DataFrame()
        duration_train = pd.DataFrame()
        start = 0
        end = 0
        for ind in np.arange(len(self.stats)):
            if self.stats[ind]['Nevents'] != 0:
                if ind == 0:
                    start = 0
                else:
                    start = end
                end += self.stats[ind]['Nevents']

                aux = self.onpower_train[start:end]
                aux = aux[:max_len]
                onpower_train = pd.concat([onpower_train, aux])

                aux = self.offpower_train[start:end]
                aux = aux[:max_len]
                offpower_train = pd.concat([offpower_train, aux])

                aux = self.duration_train[start:end]
                aux = aux[:max_len]
                duration_train = pd.concat([duration_train, aux])

                # udating stats:
                self.stats[ind]['Nevents'] = max_len

        self.onpower_train = onpower_train
        self.offpower_train = offpower_train
        self.duration_train = duration_train

        # RE-TRAINS FEATURES:
        self.__retrain(self.onpower, self.onpower_train)
        self.__retrain(self.offpower, self.offpower_train)
        self.__retrain(self.duration, self.duration_train)

    def check_cdfIntegrity(self, step):
        """
        Cheks integrity of feature model distributions.
        CDF has to be bounded by one.

        Parameters
        ----------
        step: resolution step size on the x-axis for pdf and cdf functions.
        """
        # Selecting bins automatically:
        x_max = self.onpower_train.max().values[0]
        x_min = 0
        step = 1
        x_onpower = np.arange(x_min, x_max, step).reshape(-1, 1)

        x_max = 0
        x_min = self.offpower_train.min().values[0]
        step = 1
        x_offpower = np.arange(x_min, x_max, step).reshape(-1, 1)

        x_max = self.duration_train.max().values[0]
        x_min = 0
        step = 1
        x_duration = np.arange(x_min, x_max, step).reshape(-1, 1)

        # Evaluating score for:
        # Onpower
        y_onpower = self.__pdf2(self.onpower, x_onpower)
        print("Onpower cdf: " + str(y_onpower.sum()))

        # Offpower
        y_offpower = self.__pdf2(self.offpower, x_offpower)
        print("Offpower cdf: " + str(y_offpower.sum()))

        # duration
        y_duration = self.__pdf2(self.duration, x_duration)
        print("Duration cdf: " + str(y_duration.sum()))

        # Plots:
        # fig1 = plt.figure()
        # ax1 = fig1.add_subplot(311)
        # ax2 = fig1.add_subplot(312)
        # ax3 = fig1.add_subplot(313)

        # ax1.plot(x_onpower, y_onpower)
        # ax1.set_title("PDF CDF: Onpower")
        # ax1.set_ylabel("density")
        # ax1.set_xlabel("Watts")

        # ax2.plot(x_offpower, y_offpower)
        # ax2.set_title(" PDF CDF: Offpower")
        # ax2.set_ylabel("denisty")
        # ax2.set_xlabel("Watts")

        # ax3.plot(x_duration, y_duration)
        # ax3.set_title("PDF CDF: Duration")
        # ax3.set_ylabel("density")
        # ax3.set_xlabel("Seconds")

    def featuresHist(self, **kwargs):
        """
        Visualization tool to check if feature model distributions fit
        to samples for feature training (onpower_train, offpower_train
         and duration_train)

        Parameters
        ----------
        kwargs : keyword arguments list with bins_onpower, bins_offpower and bin_duration.
            bins_feature: numpy.arange for plotting the hist with specified bin sizes.
        """

        # Selecting bins automatically:
        bins_onpower = np.arange(self.onpower_train.min().values[0],
                                 self.onpower_train.max().values[0],
                                 (self.onpower_train.max().values[0] -
                                  self.onpower_train.min().values[0]) / 50)

        bins_offpower = np.arange(self.offpower_train.min().values[0],
                                  self.offpower_train.max().values[0],
                                  (self.offpower_train.max().values[0] -
                                   self.offpower_train.min().values[0]) / 50)

        bins_duration = np.arange(self.duration_train.min().values[0],
                                  self.duration_train.max().values[0],
                                  (self.duration_train.max().values[0] -
                                   self.duration_train.min().values[0]) / 50)

        # If a bin has been specified update the bin sizes.
        for key in kwargs:
            if key == 'bins_onpower':
                bins_onpower = kwargs[key]
            elif key == 'bins_offpower':
                bins_offpower = kwargs[key]
            elif key == 'bins_duration':
                bins_duration = kwargs[key]
            else:
                print("Non valid kwarg")

        # Plot structure:
        fig = plt.figure()
        ax1 = fig.add_subplot(311)
        ax2 = fig.add_subplot(312)
        ax3 = fig.add_subplot(313)

        # Evaluating score for:
        # Onpower
        x = np.arange(bins_onpower.min(), bins_onpower.max() + \
            np.diff(bins_onpower)[0], np.diff(bins_onpower)[0] / float(1000)).reshape(-1, 1)
        y = self.__pdf(self.onpower, x)
        norm = pd.cut(
            self.onpower_train.onpower, bins=bins_onpower).value_counts().max() / max(y)
        # Plots for Onpower
        ax1.hist(
            self.onpower_train.onpower.values, bins=bins_onpower, alpha=0.5)
        ax1.plot(x, y * norm)
        #ax1.set_title("Feature: Onpower")
        #ax1.set_ylabel("Counts")
        #ax1.set_xlabel("On power (W)")
        ax1.set_ylabel("On power counts")

        # Offpower
        x = np.arange(bins_offpower.min(), bins_offpower.max() + \
            np.diff(bins_offpower)[0], np.diff(bins_offpower)[0] / float(1000)).reshape(-1, 1)
        y = self.__pdf(self.offpower, x)
        norm = pd.cut(self.offpower_train.offpower,
                      bins=bins_offpower).value_counts().max() / max(y)
        # Plots for Offpower
        ax2.hist(self.offpower_train.offpower.values,
                 bins=bins_offpower, alpha=0.5)
        ax2.plot(x, y * norm)
        #ax2.set_title("Feature: Offpower")
        #ax2.set_ylabel("Counts")
        #ax2.set_xlabel("Off power (W)")
        ax2.set_ylabel("Off power counts")

        # Duration
        x = np.arange(bins_duration.min(), bins_duration.max() + \
            np.diff(bins_duration)[0], np.diff(bins_duration)[0] / float(1000)).reshape(-1, 1)
        y = self.__pdf(self.duration, x)
        norm = pd.cut(self.duration_train.duration,
                      bins=bins_duration).value_counts().max() / max(y)
        # Plots for duration
        ax3.hist(self.duration_train.duration.values,
                 bins=bins_duration, alpha=0.5)
        ax3.plot(x, y * norm)
        #ax3.set_title("Feature: Duration")
        #ax3.set_ylabel("Counts")
        #ax3.set_xlabel("Duration (seconds)")
        ax3.set_ylabel("Duration counts")
    

    def featuresHist_colors(self, **kwargs):
        """
        Visualization tool to check if samples for feature training 
        (onpower_train, offpower_train and duration_train) are equal 
        for each appliance (same model appliance).
        Each appliance represented by a different color.

        Parameters
        ----------
        kwargs : keyword arguments list with bins_onpower, bins_offpower and bin_duration.
            bins_feature: numpy.arange for plotting the hist with specified bin sizes.
        """
        # Selecting bins automatically:
        bins_onpower = np.arange(self.onpower_train.min().values[0],
                                 self.onpower_train.max().values[0],
                                 (self.onpower_train.max().values[0] -
                                  self.onpower_train.min().values[0]) / 50)

        bins_offpower = np.arange(self.offpower_train.min().values[0],
                                  self.offpower_train.max().values[0],
                                  (self.offpower_train.max().values[0] -
                                   self.offpower_train.min().values[0]) / 50)

        bins_duration = np.arange(self.duration_train.min().values[0],
                                  self.duration_train.max().values[0],
                                  (self.duration_train.max().values[0] -
                                   self.duration_train.min().values[0]) / 50)

        # If a bin has been specified update the bin sizes.
        # Updating bins with specified values.
        for key in kwargs:
            if key == 'bins_onpower':
                bins_onpower = kwargs[key]
            elif key == 'bins_offpower':
                bins_offpower = kwargs[key]
            elif key == 'bins_duration':
                bins_duration = kwargs[key]
            else:
                print("Non valid kwarg")

        # Plot:
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(311)
        ax2 = fig1.add_subplot(312)
        ax3 = fig1.add_subplot(313)

        start = 0
        end = 0
        for ind in np.arange(len(self.stats)):

            if self.stats[ind]['Nevents'] != 0:
                if ind == 0:
                    start = 0
                else:
                    start = end
                end += self.stats[ind]['Nevents']
                ax1.hist(
                    self.onpower_train[start:end].onpower.values, bins=bins_onpower, alpha=0.5)
                ax2.hist(
                    self.offpower_train[start:end].offpower.values, bins=bins_offpower, alpha=0.5)
                ax3.hist(
                    self.duration_train[start:end].duration.values, bins=bins_duration, alpha=0.5)

        ax1.set_title("Feature: Onpower")
        ax1.set_xlabel("Watts")
        ax1.set_ylabel("Counts")

        ax2.set_title("Feature: Offpower")
        ax2.set_xlabel("Watts")
        ax2.set_ylabel("Counts")

        ax3.set_title("Feature: Duration")
        ax3.set_xlabel("Seconds")
        ax3.set_ylabel("Counts")
