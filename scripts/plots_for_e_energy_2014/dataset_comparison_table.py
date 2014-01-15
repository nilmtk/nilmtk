"""
Produces a LaTeX table summarising the datasets
"""
from __future__ import print_function, division
from os.path import expanduser, join
import pandas as pd
from nilmtk.dataset import DataSet
from nilmtk.utils import summary_stats_string
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
# DATASETS['REDD'] = 'redd/low_freq'
DATASETS['Pecan Street'] = 'pecan_1min'
# DATASETS['AMDds'] = 'ampds'
# DATASETS['iAWE'] = 'iawe'
# TODO: UKPD

if LOAD_DATASETS:
    dataset_objs = OrderedDict()
    for ds_name, ds_path in DATASETS.iteritems():
        dataset = DataSet()
        full_path = join(DATASET_PATH, ds_path)
        print("Loading", full_path)
        dataset.load_hdf5(full_path)
        dataset_objs[ds_name] = dataset


# Maps from short col name to human-readable name
COLUMNS = OrderedDict()
COLUMNS['n_appliances'] = """number of\\\\appliances"""
COLUMNS['energy_submetered'] = """% energy\\\\submetered"""
COLUMNS['dropout_rate'] = 'dropout rate'
COLUMNS['dropout_rate_ignoring_gaps'] = """dropout rate\\\\(ignoring gaps)"""
COLUMNS['uptime'] = """mains uptime\\\\per building\\\\(days)"""
COLUMNS['prop_timeslices'] = ("""% timeslices\\\\where energy\\\\"""
                              """submetered > 70%""")

for key, value in COLUMNS.iteritems():
    if OUTPUT_LATEX:
        COLUMNS[key] = """\textbf{\specialcell[h]{""" + value + """}}"""
    else:
        COLUMNS[key] = key

df = pd.DataFrame(index=DATASETS.keys(), columns=COLUMNS.values())

for ds_name in DATASETS.iterkeys():
    print('Calculating stats for', ds_name)
    dataset = dataset_objs[ds_name]
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
        df[col_long][ds_name] = s

if OUTPUT_LATEX:
    print("------------LATEX BEGINS-----------------")
    latex = df.to_latex()
    for str_to_replace in ['midrule', 'toprule', 'bottomrule']:
        latex = latex.replace(str_to_replace, 'hline')
    print(latex)
    print("------------LATEX ENDS-------------------")
else:
    print(df)
