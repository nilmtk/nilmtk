"""
Produces a LaTeX table summarising the datasets
"""
from __future__ import print_function, division
from os.path import expanduser, join
import pandas as pd
from nilmtk.dataset import DataSet
from nilmtk.utils import summary_stats_string
import nilmtk.preprocessing.electricity.building as prepb
from collections import OrderedDict

"""
TODO:
* see Jack's to-do list on our 'plan of action' google doc
"""

LOAD_DATASETS = True
OUTPUT_LATEX = True
DATASET_PATH = expanduser('~/Dropbox/Data/nilmtk_datasets/')

# Maps from human-readable name to path
DATASETS = OrderedDict()
# DATASETS['REDD'] = join(DATASET_PATH, 'redd/low_freq')
DATASETS['Smart*'] = join(DATASET_PATH, 'smart')
# DATASETS['Pecan Street'] = join(DATASET_PATH, 'pecan_1min')
# DATASETS['AMPds'] = join(DATASET_PATH, 'ampds')
# DATASETS['iAWE'] = join(DATASET_PATH, 'iawe')
# DATASETS['UKPD'] = '/data/mine/vadeec/h5_cropped'

# Maps from short col name to human-readable name
COLUMNS = OrderedDict()
COLUMNS['n_appliances'] = """Number of\\\\appliances"""
COLUMNS['energy_submetered'] = """Percentage\\\\energy\\\\sub-metered"""
COLUMNS['dropout_rate_ignoring_gaps'] = """Percentage\\\\missing samples\\\\(ignoring gaps)"""
COLUMNS['uptime'] = """Mains up-time\\\\per building\\\\(days)"""
COLUMNS['proportion_up'] = """Percentage\\\\up-time"""
# COLUMNS['prop_timeslices'] = ("""% timeslices\\\\where energy\\\\"""
#                              """submetered > 70%""")

for key, value in COLUMNS.iteritems():
    if OUTPUT_LATEX:
        COLUMNS[key] = """\textbf{\specialcell[h]{""" + value + """}}"""
    else:
        COLUMNS[key] = key

stats_df = pd.DataFrame(index=DATASETS.keys(), columns=COLUMNS.values())

for ds_name, ds_path in DATASETS.iteritems():
    if LOAD_DATASETS:
        dataset = DataSet()
        print("##################################################")
        print("Loading", ds_path)
        dataset.load_hdf5(ds_path)

        if ds_name == 'iAWE':
            print("Pre-processing iAWE...")
            electric = dataset.buildings[1].utility.electric
            electric.crop('2013/6/11', '2013/7/31')
        # elif ds_name == 'UKPD':
        #     electric = dataset.buildings[1].utility.electric
        #     electric.appliances = electric.remove_channels_from_appliances(
        #         ['kitchen_lights', 'LED_printer', 'DAB_radio_livingroom'])
        #     electric.crop(start_datetime='2013-10-01')
        #     electric.remove_voltage()
        #     dataset.buildings[2].utility.electric.remove_voltage()
        #     dataset.buildings[2].utility.electric.crop(start_datetime='2013-06-01')
        #     dataset.buildings[2].utility.electric.appliances = (
        #         dataset.buildings[2].utility.electric.remove_channels_from_appliances(
        #             ['modem_router']))

    print('Calculating stats for', ds_name)
    ds_stats = dataset.descriptive_stats()
    for col_short, col_long in COLUMNS.iteritems():
        if col_short in ['energy_submetered', 'proportion_up', 
                         'dropout_rate_ignoring_gaps', 'prop_timeslices']:
            fmt = '{:>.0%}'
        else:
            fmt = '{:>.0f}'
        s = ""
        if len(dataset.buildings) > 1:
            s += summary_stats_string(ds_stats[col_short], sep=""", """,
                                      stat_strings=['min', 'median', 'max'],
                                      minimal=True, fmt=fmt).replace('%','')
        else:
            s += fmt.format((ds_stats[col_short][0]))
        stats_df[col_long][ds_name] = s

if OUTPUT_LATEX:
    print("------------LATEX BEGINS-----------------")
    latex = stats_df.to_latex()
    latex = latex.replace('  ', '')
    n_columns = len(COLUMNS) + 1
    latex = latex.replace('l'*n_columns, 'c'*n_columns)
    for str_to_replace in ['midrule', 'toprule', 'bottomrule']:
        latex = latex.replace(str_to_replace, 'hline')
    print(latex)
    print("------------LATEX ENDS-------------------")
else:
    print(stats_df)
