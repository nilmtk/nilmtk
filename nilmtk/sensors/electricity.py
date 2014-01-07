from collections import namedtuple
import copy
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

Measurement = namedtuple('Measurement', ['physical_quantity', 'type'])
"""
physical_quantity : string
    One of: {power, energy, voltage}
type : string
    One of: {active, reactive, apparent, ''}
"""

ApplianceName = namedtuple('ApplianceName', ['name', 'instance'])
MainsName = namedtuple('MainsName', ['split', 'meter'])
CircuitName = namedtuple('CircuitName', ['name', 'split', 'meter'])
DualSupply = namedtuple('DualSupply', ['measurement', 'supply'])


def get_two_dataframes_of_dualsupply(appliance_df):
    """Creates two appliance dataframes from a dual supply

    Parameters
    ----------
    appliance_df :

    Returns
    -------
    df_1
    df_2
    
    """
    df_1 = pd.DataFrame(index=appliance_df.index)
    df_2 = pd.DataFrame(index=appliance_df.index)
    split_1_supply, split_2_supply = appliance_df.columns[
        0].supply, appliance_df.columns[1].supply
    for column in appliance_df.columns:
        if column.supply == split_1_supply:
            df_1[column.measurement] = appliance_df[[column]]
        else:
            df_2[column.measurement] = appliance_df[[column]]
    return [df_1, df_2, split_1_supply, split_2_supply]


class Electricity(object):

    """Store and process electricity for a building.

    Attributes
    ----------

    mains : dict of DataFrames, optional
        The power measurements taken from the level furthest upstream.
        Each key is a MainsName namedtuple with keys:

        * `split` is the phase or split.  Indexed from 1.
        * `meter` is the numeric ID of the meter. Indexed from 1.

        Each value is a DataFrame of shape (n_samples, n_features) 
        where each column name is a Measurement namedtuple (please
        definition of Measurement at the top of this file for a
        description of what can go into a Measurement namedtuple).

        DataFrame.index is a timezone-aware pd.DateTimeIndex.
        Power values are of type np.float32.
        DataFrame.name should be identical to the `mains` dict key which 
        maps to this DataFrame.

        For example, if we had a dataset recorded in the UK where the home has
        only a single phase supply but uses two separate meters, the first of
        which measures active and reactive power; the second of which measures only
        active, then we'd use:

        `mains = {MainsName(split=1, meter=1): 
                      DataFrame(columns=[Measurement('power','active'),
                                         Measurement('power','reactive')]),
                  MainsName(split=1, meter=2):
                      DataFrame(columns=[Measurement('power','active')])
                 }`

    circuits : dict of DataFrames, optional
        The power measurements taken from midstream.
        Each key is a CircuitName namedtuple with fields:

        * `name` is the standard name for this circuit.
        * `split` is the index for this circuit.  Indexed from 1.
        * `meter` is the numeric ID of the meter. Indexed from 1.
          Indexes into the `meters` dict.

        Each value is a DataFrame of shape (n_samples, n_features) 
        where each column name is a Measurement namedtuple (please
        definition of Measurement at the top of this file for a
        description of what can go into a Measurement namedtuple).

        DataFrame.index is a timezone-aware pd.DateTimeIndex.
        Power values are of type np.float32.
        DataFrame.name should be identical to the `circuits` dict key which 
        maps to this DataFrame.

        For example: 

        `circuits = {CircuitName(circuit='lighting', split=1):
                         DataFrame(columns=[Measurement('power','active')])}`

    appliances : dict of DataFrames, optional
        Power measurements taken from the furthest downstream in the dataset.
        Each key is an ApplianceName namedtuple with fields:
        * `name` is a standard appliance name.  For a list of valid
           names, see nilmtk/docs/standard_names/appliances.txt
        * `instance` is the index for that appliance within this building.
           Indexed from 1.

        For example, if a house has two TVs, whose power is recorded separately,
        then use these two different keys: `('tv', 1)` and `('tv', 2)`

        If multiple appliances are monitored on one channel (e.g. tv + dvd)
        then use a tuple of appliances as the key, e.g.:

        `(('tv', 1), ('dvd player', 1))`
        
        Each value of the `appliances` dict is a DataFrame of 
        shape (n_samples, n_features) where each column name is either:

        * a Measurement namedtuple (please definition of Measurement at the top
          of this file for a description of what can go into 
          a Measurement namedtuple).
        * a DualSupply namedtuple with fields:
          * measurement : Measurement namedtuple
          * supply : int. Index of supply. Start from 1. Does not have to map
            directly to the index used to number the mains splits if this 
            information is not known.
          DualSupply is used for appliances like American ovens which are
          are single appliances but pull power from both "splits".
        * Or 'state' (where 0 is 'off'). This is used when the ground-truth of
          the appliance state is known; for example in the BLUEd dataset.

        DataFrame.index is a timezone-aware pd.DateTimeIndex.
        Power values are of type np.float32; `state` values are np.int32
        DataFrame.name should be identical to the `appliances` dict key which 
        maps to this DataFrame.

        Note that `appliances` always stores measurements from the meters
        in the dataset placed furthest downstream.  So, for example, in the case
        of the REDD dataset where we have measurements of 'mains' and 'circuits',
        these 'circuits' channels are put into `Electricity.appliances` because
        REDD's 'circuits' channels are the furthest downstream.

    appliance_estimates : Panel (3D matrix), optional
        Output from the NILM disaggregation algorithm.
        The index is a timezone-aware pd.DateTimeIndex
        The first two dimensions are time and ApplianceName.
        The third dimension describes, for each appliance and for each time:
        * `power` : np.float32. Estimated power consumption in Watts.
        * `state` : np.int32, optional.
        * `confidence` : np.float32 [0,1], optional.
        * `power_prob_dist` : object describing the probability dist, optional
        * `state_prob_dist` : object describing the probability dist, optional

    metadata : dict, optional.  All this information should be ground-truth, 
        not automatically inferred.  Automatically inferred data should go into
        `inferred_metadata`.

        nominal_mains_voltage : np.float32, optional
            The nominal mains voltage in volts.

        mains_wiring : networkx.DiGraph, optional
            Nodes are ApplianceNames or CircuitNames or MainsNames.
            Edges describe the power wiring between mains, circuits and 
            appliances.
            Edge direction indicates the flow of energy.  i.e. edges point
            towards loads.

        control_connections : networkx.DiGraph, optional
            Each node is an ApplianceName.
            Edges represent data / audio / video / control connections between
            appliances.  Edges can be single-directional or bi-directional.
            For example, a dvd player would have an edge pointing towards the
            tv.  A wifi router would have bi-directional edges to laptops,
            printers, smart TVs etc.

        appliances : dict, optional
            Metadata describing each appliance.  This should be ground truth data,
            not automatically inferred information.
            Each key is an ApplianceName(<appliance name>, <instance>) namedtuple.
            Each value is list of dicts. Each dict describes metadata for that 
            specific appliance.  Multiple dicts are used to express replacing 
            appliances over time (in which case the items should be in 
            chronological order so the last element of the list is always the
            most recent.)  Each dict has 'general' appliance fields (which
            all appliances can have) and fields which are specific to that
            class of appliance.

            General fields (all of which are optional)
            ------------------------------------------

            'room': (<room name>, <room instance>) tuple (as used in this `Building.rooms`).

            'meter': (<manufacturer>, <model>) tuple which maps into global Meters DB.

            'start date', 'end date': datetime to represent the period during which
                this appliance configuration was active. Set 'start date' to 0
                if this appliance was active from the start of the dataset. Set
                'end date' to 0 if this appliance is still active at the end of 
                the dataset.

            Any machine-readable field specified in the communally-defined 
            appliance controlled vocabulary may be overridden.
    
            DualSupply appliances may have a 'supply1' and 'supply2'
            key which maps to a list of strings describing the main
            components supplied by that supply.  e.g.

            {('washer dryer', 1): {'supply1': ['motor'], 
                                   'supply2': ['heating element']} }

            Valid component names are specified on the wiki at 
            electricity-disaggregation.org.

            Appliance-specific fields
            -------------------------
            The permitted fields and values for each appliance name are 
            described in `nilmtk/docs/standard_names/appliances.txt`.  e.g.

            Appliances not directly metered
            -------------------------------
            Appliances which are not directly metered can be listed. For 
            example, if a dataset records the lighting circuit (but
            not each individual ceiling light) then we can specify each
            ceiling light in `metadata['appliances']` and then specify
            the wiring from the lighting circuit to each ceiling light in
            the `metadata['mains_wiring']` graph.

            Example
            -------
            `{('tv', 1): 
                [{'original name in source dataset': 'Television LCD',
                  'display': 'lcd', 
                  'backlight': 'led'
                  'screen size in inches': 42,
                  'year of manufacture': 2001,
                  'start date': '3/4/2012',
                  'end date': '4/5/2013',
                  'quantity installed': 1,
                  'room': ('livingroom', 1),
                  'meter': ('Current Cost', 'IAM')
                }],
             ('lights', 1):
                [{'room': ('kitchen', 1),
                  'placing': 'ceiling',
                  'lamp': 'tungsten',
                  'dimmable': True,
                  'nominal Wattage each': 50,
                  'quantity installed': 10,
                  'meter': ('Current Cost', 'EnviR')
                 }]
            }`

        metadata_authorship_date : datetime, optional
            The controlled vocabulary specified on the energy-disaggregation.org
            wiki will evolve with time.  Hence state the date when the metadata
            was authored.

    inferred_metadata : dict, optional
        Has the same structure as `metadata` but contains information which has
        been automatically inferred from the data.

    """

    def __init__(self):
        self.mains = {}
        self.circuits = {}
        self.appliances = {}
        self.appliance_estimates = None
        self.metadata = {}
        self.inferred_metadata = {}

    def get_dataframe_of_mains(self, measurement=Measurement('power', 'active')):
        """Get a pandas.DataFrame of all mains data


        """
        first_mains = self.mains.values()[0]
        shape = first_mains.shape
        columns = first_mains.columns
        index = first_mains.index
        data = np.zeros(shape)
        sum_df = pd.DataFrame(data, index=index, columns=columns)

        for main_df in self.mains.itervalues():
            sum_df += main_df
        return sum_df[[measurement]]

    def get_dataframe_of_appliances(self,
                                    measurement=Measurement('power', 'active')):
        """Get a pandas.DataFrame of all appliance data.

        If any DualSupply appliances are present then sum together the two
        supplies (after checking if `dualsupply.measurement == measurement`).

        Parameters
        ----------
        measurement : Measurement, optional
            default=Measurement('power', 'active')
            if `measurement=None` then just get the first column per DataFrame.

        Returns
        -------
        pandas.DataFrame
            Index is the same as the index used in the appliances DataFrames.
            Each column name is an ApplianceName namedtuple.
        """
        if measurement is None:
            appliance_dict = {
                appliance_name: appliance_df.icol[0]
                for appliance_name, appliance_df in self.appliances.iteritems()}
        else:
            appliance_dict = {
                appliance_name: appliance_df[measurement]
                for appliance_name, appliance_df in self.appliances.iteritems()
                if measurement in appliance_df}

            # Handle DualSupply appliances
            for appliance_name, appliance_df in self.appliances.iteritems():
                dual_supply_columns = []
                for column_name in appliance_df:
                    if (isinstance(column_name, DualSupply) and
                            column_name.measurement == measurement):
                        dual_supply_columns.append(appliance_df[column_name])

                if dual_supply_columns:
                    appliance_dict[appliance_name] = (dual_supply_columns[0] +
                                                      dual_supply_columns[1])

        return pd.DataFrame(appliance_dict)

    def get_appliance(self, appliance_name, measurement="all"):
        """
        Parameters
        ----------
        appliance_name : string
        measurement : string or list of strings, optional
            {'apparent', 'active', 'reactive', 'voltage', 'all'}

        Returns
        -------
        appliance_data : DataFrame
        """
        raise NotImplementedError

    def count_appliances(self, appliance_name):
        """
        Returns
        -------
        n_appliances : int
        """
        raise NotImplementedError

    def get_vampire_power(self):
        raise NotImplementedError

    def get_diff_between_aggregate_and_appliances(self):
        raise NotImplementedError

    def plot_appliance_activity(self, source):
        """Plot a compact representation of all appliance activity."""
        raise NotImplementedError

    def get_start_and_end_dates(self):
        """Returns the start and end dates covering the data in
        appliances, circuits and mains.

        Returns
        -------
        [start, end] : pd.Timestamp
        """
        start = None
        end = None
        for dict_of_dfs in [self.appliances, self.circuits, self.mains]:
            for df in dict_of_dfs.values():
                df_start = df.index[0]
                if start is None or df_start < start:
                    start = df_start

                df_end = df.index[-1]
                if end is None or df_end > end:
                    end = df_end

        return [start, end]

    def __str__(self):
        return ""

    def to_json(self):
        representation = copy.copy(self.metadata)
        representation["mains"] = ""
        representation["appliances"] = ""
        representation["circuits"] = ""
        return json.dumps(representation)
