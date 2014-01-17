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
OUTPUT_LATEX = False
DATASET_PATH = expanduser('~/Dropbox/Data/nilmtk_datasets/')

# Maps from human-readable name to path
DATASETS = OrderedDict()
#DATASETS['REDD'] = join(DATASET_PATH, 'redd/low_freq')
#DATASETS['Pecan Street'] = join(DATASET_PATH, 'pecan_1min')
#DATASETS['AMDds'] = join(DATASET_PATH, 'ampds')
DATASETS['iAWE'] = join(DATASET_PATH, 'iawe')
#DATASETS['UKPD'] = '/data/mine/vadeec/h5'
# DATASETS['Smart'] = 'smart'

# Maps from short col name to human-readable name
COLUMNS = OrderedDict()
COLUMNS['n_appliances'] = """number of\\\\appliances"""
COLUMNS['energy_submetered'] = """% energy\\\\submetered"""
COLUMNS['proportion_up'] = """uptime / \\\\ total duration"""
COLUMNS['dropout_rate_ignoring_gaps'] = """dropout rate\\\\(ignoring gaps)"""
COLUMNS['uptime'] = """mains uptime\\\\per building\\\\(days)"""
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
        elif ds_name == 'UKPD':
            electric = dataset.buildings[1].utility.electric
            electric.appliances = electric.remove_channels_from_appliances(
                ['kitchen_lights', 'LED_printer'])

    print('Calculating stats for', ds_name)
    ds_stats = dataset.descriptive_stats()
    for col_short, col_long in COLUMNS.iteritems():
        s = ""
        if OUTPUT_LATEX:
            s += """\specialcell{"""
        s += summary_stats_string(ds_stats[col_short], sep=""",""",
                                  stat_strings=['min', 'median', 'max'],
                                  minimal=True).replace(' ', '')
        if OUTPUT_LATEX:
            s += """}"""
        stats_df[col_long][ds_name] = s

if OUTPUT_LATEX:
    print("------------LATEX BEGINS-----------------")
    latex = stats_df.to_latex()
    for str_to_replace in ['midrule', 'toprule', 'bottomrule']:
        latex = latex.replace(str_to_replace, 'hline')
    print(latex)
    print("------------LATEX ENDS-------------------")
else:
    print(stats_df)
