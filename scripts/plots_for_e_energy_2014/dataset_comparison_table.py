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
DATASETS['REDD'] = 'redd/low_freq'
DATASETS['Pecan Street'] = 'pecan_1min'
DATASETS['AMDds'] = 'ampds'
DATASETS['iAWE'] = 'iawe'
DATASETS['UKPD'] = 'ukpd'
# DATASETS['Smart'] = 'smart'

# Maps from short col name to human-readable name
COLUMNS = OrderedDict()
COLUMNS['n_appliances'] = """number of\\\\appliances"""
COLUMNS['energy_submetered'] = """% energy\\\\submetered"""
COLUMNS['proportion_up'] = """uptime / \\\\ total duration"""
COLUMNS['dropout_rate_ignoring_gaps'] = """dropout rate\\\\(ignoring gaps)"""
COLUMNS['uptime'] = """mains uptime\\\\per building\\\\(days)"""
COLUMNS['prop_timeslices'] = ("""% timeslices\\\\where energy\\\\"""
                              """submetered > 70%""")

for key, value in COLUMNS.iteritems():
    if OUTPUT_LATEX:
        COLUMNS[key] = """\textbf{\specialcell[h]{""" + value + """}}"""
    else:
        COLUMNS[key] = key

stats_df = pd.DataFrame(index=DATASETS.keys(), columns=COLUMNS.values())

for ds_name, ds_path in DATASETS.iteritems():
    if LOAD_DATASETS:
        dataset = DataSet()
        full_path = join(DATASET_PATH, ds_path)
        print("##################################################")
        print("Loading", full_path)
        dataset.load_hdf5(full_path)

        if ds_name == 'iAWE':
            print("Pre-processing iAWE...")
            electric = dataset.buildings[1].utility.electric
            electric.crop('2013/6/11', '2013/7/31')
            electric.drop_duplicate_indicies() # TODO: remove when no longer necessary
        elif ds_name == 'UKPD':
            electric = dataset.buildings[1].utility.electric
            electric.appliances = electric.remove_channels_from_appliances(
                ['kitchen_lights', 'LED_printer'])
            electric.crop('2013/3/17')

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
