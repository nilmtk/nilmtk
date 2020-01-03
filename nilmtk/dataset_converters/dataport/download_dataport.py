import os
import re
import datetime
import sys
from os.path import join, isdir, isfile, dirname, abspath
import pandas as pd
import yaml
import psycopg2 as db
from nilmtk.measurement import measurement_columns
from nilmtk.measurement import LEVEL_NAMES
from nilmtk.datastore import Key
from nilm_metadata import convert_yaml_to_hdf5
from nilmtk.utils import get_module_directory
import shutil
import tempfile

"""
MANUAL:

dataport is a large dataset hosted in a remote SQL database. This
file provides a function to download the dataset and save it to disk
as NILMTK-DF. Since downloading the entire dataset will likely take >
24 hours, this function provides some options to allow you to download
only a subset of the data.

'''''''''''''''' Previous Version '''''''''''''''''''''
For example, to only load house 26 for April 2014:
    from nilmtk.dataset_converters.dataport.download_dataport
    import download_dataport
    download_dataport(
        'username',
        'password',
        '/path/output_filename.h5',
        periods_to_load={26: ('2014-04-01', '2014-05-01')}
    )
'''''''''''''''' Previous Version '''''''''''''''''''''

'''''''''''''''' New Version '''''''''''''''''''''

    from nilmtk.dataset_converters.dataport.download_dataport
    import download_dataport,
            _dataport_dataframe_to_hdf,
            view_database_tables,
            view_buildings,
            view_data_window

    # see all available tables in the dataport database.
    view_database_tables(
        'username',
        'password',
        'database_schema'    # electricity 
    )

    # show the list of all available buildings
    view_buildings(
        'username',
        'password',
        'database_schema',  # electricity 
        'table_name'        # for example 'eg_realpower_1min', 'eg_current_15min'
    )

    # view data collection window of selected buildings
    view_data_window(
        'username',
        'password',
        'database_schema',  # electricity
        'table_name',       # for example 'eg_realpower_1min','eg_current_1hr'
        [18,26,43,44]       # data collection window of building 18,26,43 and 44 respectively
    )

    # download the dataset.
    For example, loading electricity_egauge_hours from 2018-11-17 to
    2019-12-17 of building 26
    download_dataport(
        'username',
        'password',
        '/path/output_filename.h5',
        'electricity',
        'eg_realpower_1hr',
        periods_to_load={ 26: ('2018-11-17', '2019-12-17')})


'''''''''''''''' New Version '''''''''''''''''''''

REQUIREMENTS:

On Ubuntu:
* sudo apt-get install libpq-dev
* sudo pip install psycopg2

TODO:
* intelligently handle queries that fail due to network
* integrate 'grid' (use - gen) and 'gen'

"""
feed_mapping = {
                'air1': {'type': 'air conditioner'},
                'air2': {'type': 'air conditioner'},
                'air3': {'type': 'air conditioner'},
                'airwindowunit1': {'type': 'air conditioner'},
                'aquarium1': {'type': 'appliance'},
                'bathroom1': {'type': 'sockets', 'room': 'bathroom'},
                'bathroom2': {'type': 'sockets', 'room': 'bathroom'},
                'bedroom1': {'type': 'sockets', 'room': 'bedroom'},
                'bedroom2': {'type': 'sockets', 'room': 'bedroom'},
                'bedroom3': {'type': 'sockets', 'room': 'bedroom'},
                'bedroom4': {'type': 'sockets', 'room': 'bedroom'},
                'bedroom5': {'type': 'sockets', 'room': 'bedroom'},
                'battery1': {},                       #new field, need mapping
                'car1': {'type': 'electric vehicle'},
                'circpump1': {},                      #new field, need mapping
                'clotheswasher1': {'type': 'washing machine'},
                'clotheswasher_dryg1': {'type': 'washer dryer'},
                'diningroom1': {'type': 'sockets', 'room': 'dining room'},
                'diningroom2': {'type': 'sockets', 'room': 'dining room'},
                'dishwasher1': {'type': 'dish washer'},
                'disposal1': {'type': 'waste disposal unit'},
                'drye1': {'type': 'spin dryer'},
                'dryg1': {'type': 'spin dryer'},
                'freezer1': {'type': 'freezer'},
                'furnace1': {'type': 'electric furnace'},
                'furnace2': {'type': 'electric furnace'},
                'garage1': {'type': 'sockets', 'room': 'dining room'},
                'garage2': {'type': 'sockets', 'room': 'dining room'},
                'grid': {},
                'heater1': {'type': 'electric space heater'},
                'heater2': {'type': 'electric space heater'},
                'heater3': {'type': 'electric space heater'},
                'housefan1': {'type': 'electric space heater'},
                'icemaker1': {'type': 'appliance'},
                'jacuzzi1': {'type': 'electric hot tub heater'},
                'kitchen1': {'type': 'sockets', 'room': 'kitchen'},
                'kitchen2': {'type': 'sockets', 'room': 'kitchen'},
                'kitchenapp1': {'type': 'sockets', 'room': 'kitchen'},
                'kitchenapp2': {'type': 'sockets', 'room': 'kitchen'},
                'lights_plugs1': {'type': 'light'},
                'lights_plugs2': {'type': 'light'},
                'lights_plugs3': {'type': 'light'},
                'lights_plugs4': {'type': 'light'},
                'lights_plugs5': {'type': 'light'},
                'lights_plugs6': {'type': 'light'},
                'livingroom1': {'type': 'sockets', 'room': 'living room'},
                'livingroom2': {'type': 'sockets', 'room': 'living room'},
                'microwave1': {'type': 'microwave'},
                'office1': {'type': 'sockets', 'room': 'office'},
                'outsidelights_plugs1': {'type': 'sockets', 'room': 'outside'},
                'outsidelights_plugs2': {'type': 'sockets', 'room': 'outside'},
                'oven1': {'type': 'oven'},
                'oven2': {'type': 'oven'},
                'pool1': {'type': 'electric swimming pool heater'},
                'pool2': {'type': 'electric swimming pool heater'},
                'poollight1': {'type': 'light'},
                'poolpump1': {'type': 'electric swimming pool heater'},
                'pump1': {'type': 'appliance'},
                'range1': {'type': 'stove'},
                'refrigerator1': {'type': 'fridge'},
                'refrigerator2': {'type': 'fridge'},
                'security1': {'type': 'security alarm'},
                'sewerpump1': {},               #new field, need mapping
                'shed1': {'type': 'sockets', 'room': 'shed'},
                'solar': {},
                'solar2': {},
                'sprinkler1': {'type': 'appliance'},
                'sumppump1': {},                #new field, need mapping
                'utilityroom1': {'type': 'sockets', 'room': 'utility room'},
                'venthood1': {'type': 'appliance'},
                'waterheater1': {'type': 'electric water heating appliance'},
                'waterheater2': {'type': 'electric water heating appliance'},
                'wellpump1': {},                #new field, need mapping
                'winecooler1': {'type': 'appliance'},
                'leg1v':{},
                'leg2v':{}
                }

feed_ignore = ['solar', 'solar2', 'grid', 'leg1v', 'leg2v', 'battery1', 'circpump1',
               'sewerpump1', 'sumppump1', 'wellpump1']


def database_assert(database_table):
    assert (
            database_table == 'eg_angle_15min' or
            database_table == 'eg_angle_1hr' or
            database_table == 'eg_angle_1min' or
            database_table == 'eg_angle_1s' or
            database_table == 'eg_apparentpower_15min' or
            database_table == 'eg_apparentpower_1hr' or
            database_table == 'eg_apparentpower_1min' or
            database_table == 'eg_apparentpower_1s' or
            database_table == 'eg_current_15min' or
            database_table == 'eg_current_1hr' or
            database_table == 'eg_current_1min' or
            database_table == 'eg_current_1s' or
            database_table == 'eg_realpower_15min' or
            database_table == 'eg_realpower_1hr' or
            database_table == 'eg_realpower_1min' or
            database_table == 'eg_realpower_1s' or
            database_table == 'eg_thd_15min' or
            database_table == 'eg_thd_1hr' or
            database_table == 'eg_thd_1min' or
            database_table == 'eg_thd_1s' or
            database_table == 'eg_realpower_1s_40homes_dataset'
            ), "Table not compatible with NILMTK"


def view_database_tables(
    database_username,
    database_password,
    database_schema
):
    
    database_host = 'dataport.pecanstreet.org'
    database_port = '5434'
    database_name = 'dataport'

    try:
        conn = db.connect('host=' + database_host +
                          ' port=' + database_port +
                          ' dbname=' + database_name +
                          ' user=' + database_username +
                          ' password=' + database_password)
    except:
        print('Could not connect to remote database')
        raise

    # Loading university schemas
    sql_query = ("SELECT table_name" +
                 " FROM information_schema.views" +
                 " WHERE table_schema ='" + database_schema + "'" +
                 " ORDER BY table_name")
    database_tables = pd.read_sql(sql_query, conn)['table_name'].tolist()

    df = pd.DataFrame({database_schema: database_tables})
    print(df)
    conn.close()


def view_buildings(
    database_username,
    database_password,
    database_schema,
    database_table
):

    database_assert(database_table)
    database_host = 'dataport.pecanstreet.org'
    database_port = '5434'
    database_name = 'dataport'

    # try to connect to database
    try:
        conn = db.connect('host=' + database_host +
                          ' port=' + database_port +
                          ' dbname=' + database_name +
                          ' user=' + database_username +
                          ' password=' + database_password)
    except:
        print('Could not connect to remote database')
        raise

    # select all buildings for the database_table
    sql_query = ('SELECT DISTINCT dataid' +
                 ' FROM other_datasets.metadata' +
                 ' ORDER BY dataid')
    
    buildings_in_table = pd.read_sql(sql_query, conn)['dataid'].tolist()
    print(buildings_in_table)
    conn.close()


def view_data_window(
                     database_username,
                     database_password,
                     database_schema,
                     database_table,
                     building_no=None):

    database_assert(database_table)
    database_host = 'dataport.pecanstreet.org'
    database_port = '5434'
    database_name = 'dataport'

    # try to connect to database
    try:
        conn = db.connect('host=' + database_host +
                          ' port=' + database_port +
                          ' dbname=' + database_name +
                          ' user=' + database_username +
                          ' password=' + database_password)
    except:
        print('Could not connect to remote database')
        raise

    # select all buildings for the database_table
    sql_query = ('SELECT DISTINCT dataid' +
                 ' FROM other_datasets.metadata' +
                 ' ORDER BY dataid')

    if(not (building_no)):
        print(" Please provide the list of building numbers ")
    else:
        for each_building in building_no:
            #seperate columns for 1s and 1min data
            if(database_table.find('_1s') > 0):
                sql_query = ('SELECT MIN(egauge_1s_min_time) AS minlocalminute,' +
                         ' MAX(egauge_1s_max_time) AS maxlocalminute' +
                         ' FROM other_datasets.metadata' +
                         ' WHERE dataid=' + str(each_building))
            else:
                sql_query = ('SELECT MIN(egauge_1min_min_time) AS minlocalminute,' +
                         ' MAX(egauge_1min_max_time) AS maxlocalminute' +
                         ' FROM other_datasets.metadata' +
                         ' WHERE dataid=' + str(each_building))

            timestamps = pd.read_sql(sql_query, conn)
            first_timestamp_in_table = timestamps['minlocalminute'][0]
            last_timestamp_in_table = timestamps['maxlocalminute'][0]
            print(str(each_building),
                  "\t\t", first_timestamp_in_table,
                  "\t\t", last_timestamp_in_table)
        print("Done loading all the buildings!!")

    conn.close()


def download_dataport(database_username,
                      database_password, hdf_filename,
                      database_schema='electricity',
                      user_selected_table='eg_realpower_1min',
                      periods_to_load=None):
    """
    Downloads data from dataport database into an HDF5 file.

    Parameters
    ----------
    hdf_filename : str
        Output HDF filename.  If file exists already then will be deleted.
    database_username, database_password, database_schema,user_selected_table, hdf_filename : str
    periods_to_load : dict of tuples, optional
       Key of dict is the building number (int).
       Values are (<start date>, <end date>)
       e.g. ("2013-04-01", None) or ("2013-04-01", "2013-08-01")
       defaults to all buildings and all date ranges
    """

    database_assert(user_selected_table)
    # dataport database settings
    database_host = 'dataport.pecanstreet.org'
    database_port = '5434'
    database_name = 'dataport'

    # try to connect to database
    try:
        conn = db.connect('host=' + database_host +
                          ' port=' + database_port +
                          ' dbname=' + database_name +
                          ' user=' + database_username +
                          ' password=' + database_password)
    except:
        print('Could not connect to remote database')
        raise

    # map user_selected_table and timestamp column
    timestamp_map = {"eg_angle_15min": "local_15min",
                     "eg_angle_1hr": "localhour",
                     "eg_angle_1min": "localminute",
                     "eg_angle_1s": "localminute",
                     "eg_apparentpower_15min": "local_15min",
                     "eg_apparentpower_1hr": "localhour",
                     "eg_apparentpower_1min": "localminute",
                     "eg_apparentpower_1s": "localminute",
                     "eg_current_15min": "local_15min",
                     "eg_current_1hr": "localhour",
                     "eg_current_1min": "localminute",
                     "eg_current_1s": "localminute",
                     "eg_realpower_15min": "local_15min",
                     "eg_realpower_1hr": "localhour",
                     "eg_realpower_1min": "localminute",
                     "eg_realpower_1s": "localminute",
                     "eg_thd_15min": "local_15min",
                     "eg_thd_1hr": "localhour",
                     "eg_thd_1min": "localminute",
                     "eg_thd_1s": "localminute",
                     "eg_realpower_1s_40homes_dataset": "localminute"}

    # set up a new HDF5 datastore (overwrites existing store)
    
    store = pd.HDFStore(hdf_filename, 'w', complevel=9, complib='zlib')
    

    # Create a temporary metadata dir, remove existing building
    # yaml files in module dir (if any)
    original_metadata_dir = join(get_module_directory(),
                                 'dataset_converters',
                                 'dataport',
                                 'metadata')
    tmp_dir = tempfile.mkdtemp()
    metadata_dir = join(tmp_dir, 'metadata')
    shutil.copytree(original_metadata_dir, metadata_dir)
    print("Using temporary dir for metadata:", metadata_dir)

    for f in os.listdir(metadata_dir):
        if re.search('^building', f):
            os.remove(join(metadata_dir, f))

    """
    TODO:
    The section below can be altered or removed,
    since the restructured Dataport
    now has only one electricity_egauge_minutes table.
    """
    # get tables in database schema
    sql_query = ("SELECT table_name" +
                 " FROM information_schema.views" +
                 " WHERE table_schema ='" + database_schema + "'" +
                 " ORDER BY table_name")
    database_tables = pd.read_sql(sql_query, conn)['table_name'].tolist()
    database_tables = [t for t in database_tables if user_selected_table in t]
    # if user has specified buildings
    if periods_to_load:
        buildings_to_load = list(periods_to_load.keys())
    else:
        # get buildings present in all tables
        sql_query = ''
        for table in database_tables:
            sql_query = (sql_query + '(SELECT DISTINCT dataid' +
                         ' FROM "' + database_schema + '".' + table +
                         ') UNION ')
        sql_query = sql_query[:-7]
        sql_query = (sql_query + ' ORDER BY dataid')
        buildings_to_load = pd.read_sql(sql_query, conn)['dataid'].tolist()

    # for each user specified building or all buildings in database
    for building_id in buildings_to_load:
        print("Loading building {:d} @ {}"
              .format(building_id, datetime.datetime.now()))
        sys.stdout.flush()

        # create new list of chunks for concatenating later
        dataframe_list = []

        # for each table of 1 month data
        for database_table in database_tables:
            print("  Loading table {:s}".format(database_table))
            sys.stdout.flush()

            # get buildings present in electricity_egauge_minutes table
            if(user_selected_table.find('_1s') > 0):
                sql_query = ('SELECT DISTINCT dataid' +
                         ' FROM other_datasets.metadata' +
                         ' WHERE egauge_1s_min_time IS NOT NULL' +
                         ' ORDER BY dataid')
            else:
                sql_query = ('SELECT DISTINCT dataid' +
                         ' FROM other_datasets.metadata' +
                         ' WHERE egauge_1min_min_time IS NOT NULL' +
                         ' ORDER BY dataid')
                

            buildings_in_table = pd.read_sql(sql_query,
                                             conn)['dataid'].tolist()
            if building_id in buildings_in_table:
                # get first and last timestamps for this
                # house in electricity_egauge_minutes table
                if(user_selected_table.find('_1s') > 0):
                    sql_query = ('SELECT MIN(egauge_1s_min_time) AS minlocalminute,' +
                             ' MAX(egauge_1s_max_time) AS maxlocalminute' +
                             ' FROM other_datasets.metadata' +
                             ' WHERE dataid=' + str(building_id))
                else:
                    sql_query = ('SELECT MIN(egauge_1min_min_time) AS minlocalminute,' +
                             ' MAX(egauge_1min_max_time) AS maxlocalminute' +
                             ' FROM other_datasets.metadata' +
                             ' WHERE dataid=' + str(building_id))

                range = pd.read_sql(sql_query, conn)

                first_timestamp_in_table = range['minlocalminute'][0]
                last_timestamp_in_table = range['maxlocalminute'][0]

                # get requested start and end and localize them
                requested_start = None
                requested_end = None
                database_timezone = 'US/Central'
                if periods_to_load:
                    if periods_to_load[building_id][0]:
                        requested_start = pd.Timestamp(periods_to_load[building_id][0])
                        requested_start = requested_start.tz_localize(database_timezone)
                    if periods_to_load[building_id][1]:
                        requested_end = pd.Timestamp(periods_to_load[building_id][1])
                        requested_end = requested_end.tz_localize(database_timezone)

                # check user start is not after end
                if requested_start > requested_end:
                    print('requested end is before requested start')
                    sys.stdout.flush()
                else:
                    # clip data to smallest range
                    if requested_start:
                        start = max(requested_start, first_timestamp_in_table)
                    else:
                        start = first_timestamp_in_table
                    if requested_end:
                        end = min(requested_end, last_timestamp_in_table)
                    else:
                        end = last_timestamp_in_table

                    # download data in chunks
                    chunk_start = start
                    chunk_size = datetime.timedelta(10)  # 1 day
                    while chunk_start < end:
                        chunk_end = chunk_start + chunk_size
                        if chunk_end > end:
                            chunk_end = end
                        # subtract 1 second so end is exclusive
                        chunk_end = chunk_end - datetime.timedelta(0, 1)

                        # query power data for all channels
                        format = '%Y-%m-%d %H:%M:%S'
                        sql_query = ('SELECT *' +
                                     ' FROM "' + database_schema + '".' + user_selected_table +
                                     ' WHERE dataid=' + str(building_id) +
                                     'and "' + timestamp_map[user_selected_table] + '" between ' +
                                     "'" + chunk_start.strftime(format) + "'" +
                                     " and " +
                                     "'" + chunk_end.strftime(format) +
                                     "' ORDER BY "+timestamp_map[user_selected_table]
                                     )
                        chunk_dataframe = pd.read_sql(sql_query, conn)
                        # nilmtk requires building indices to start at 1
                        nilmtk_building_id = buildings_to_load.index(building_id) + 1
                        # convert to nilmtk-df and save to disk
                        nilmtk_dataframe = _dataport_dataframe_to_hdf(
                            chunk_dataframe, store,
                            nilmtk_building_id,
                            building_id,
                            timestamp_map[user_selected_table],
                            metadata_dir,
                            user_selected_table                            
                        )

                        # print progress
                        print('    ' + str(chunk_start) + ' -> ' +
                              str(chunk_end) + ': ' +
                              str(len(chunk_dataframe.index)) + ' rows')
                        sys.stdout.flush()

                        # append all chunks into list for csv writing
                        # dataframe_list.append(chunk_dataframe)

                        # move on to next chunk
                        chunk_start = chunk_start + chunk_size

        # saves all chunks in list to csv
        # if len(dataframe_list) > 0:
            # dataframe_concat = pd.concat(dataframe_list)
            # dataframe_concat.to_csv(output_directory + str(building_id) + '.csv')
    store.close()
    conn.close()

    # write yaml to hdf5
    # dataset.yaml and meter_devices.yaml are static, building<x>.yaml are dynamic
    convert_yaml_to_hdf5(metadata_dir, hdf_filename)

    # remote the temporary dir when finished
    shutil.rmtree(tmp_dir)


def _dataport_dataframe_to_hdf(dataport_dataframe,
                               store,
                               nilmtk_building_id,
                               dataport_building_id,
                               timestamp_name,
                               metadata_dir,
                               user_selected_table):
    local_dataframe = dataport_dataframe.copy()

    # remove timezone information to avoid append errors
    local_dataframe[timestamp_name] = pd.DatetimeIndex([i.replace(tzinfo=None)
                                                       for i in local_dataframe[timestamp_name]])
    # set timestamp as frame index
    local_dataframe = local_dataframe.set_index(timestamp_name)

    # set timezone
    local_dataframe = local_dataframe.tz_localize('US/Central')
    # remove timestamp column from dataframe
    feeds_dataframe = local_dataframe.drop('dataid', axis=1)
    # Column names for dataframe
    column_names = [('power', 'active')]
    # convert from kW to W for realpower data
    if(user_selected_table.find('_realpower_') > 0):
        feeds_dataframe = feeds_dataframe.mul(1000)
    # building metadata
    building_metadata = {}
    building_metadata['instance'] = nilmtk_building_id
    building_metadata['original_name'] = int(dataport_building_id)  # use python int
    building_metadata['elec_meters'] = {}
    building_metadata['appliances'] = []

    # initialise dict of instances of each appliance type
    instance_counter = {}

    meter_id = 1
    for column in feeds_dataframe.columns:
        if feeds_dataframe[column].notnull().sum() > 0 and not column in feed_ignore:

            # convert timeseries into dataframe
            feed_dataframe = pd.DataFrame(feeds_dataframe[column])

            # set column names
            feed_dataframe.columns = pd.MultiIndex.from_tuples(column_names)

            # Modify the column labels to reflect the power measurements recorded.
            feed_dataframe.columns.set_names(LEVEL_NAMES, inplace=True)

            key = Key(building=nilmtk_building_id, meter=meter_id)

            # store dataframe
            store.put(str(key), feed_dataframe, format='table', append=True)
            store.flush()

            # elec_meter metadata
            if column == 'use':
                meter_metadata = {'device_model': 'eGauge',
                                  'site_meter': True}
            else:
                meter_metadata = {'device_model': 'eGauge',
                                  'submeter_of': 0}
            building_metadata['elec_meters'][meter_id] = meter_metadata
            # appliance metadata
            if column != 'use':
                # original name and meter id
                appliance_metadata = {'original_name': column,
                                      'meters': [meter_id]}
                # appliance type and room if available
                appliance_metadata.update(feed_mapping[column])
                # appliance instance number
                if instance_counter.get(appliance_metadata['type']) == None:
                    instance_counter[appliance_metadata['type']] = 0
                instance_counter[appliance_metadata['type']] += 1
                appliance_metadata['instance'] = instance_counter[appliance_metadata['type']]
                building_metadata['appliances'].append(appliance_metadata)

            meter_id += 1

    # write building yaml to file
    building = 'building{:d}'.format(nilmtk_building_id)
    yaml_full_filename = join(metadata_dir, building + '.yaml')
    with open(yaml_full_filename, 'w') as outfile:
        outfile.write(yaml.dump(building_metadata))

    return 0


