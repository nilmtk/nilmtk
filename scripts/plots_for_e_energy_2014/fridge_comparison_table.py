"""
Produces a LaTeX table summarising the datasets
"""
from __future__ import print_function, division
from os.path import expanduser, join
import pandas as pd
from nilmtk.dataset import DataSet
from collections import OrderedDict


LOAD_DATASETS = True

DATASET_PATH = expanduser('~/Dropbox/nilmtk_datasets/')

# Maps from human-readable name to path
DATASETS = OrderedDict()
DATASETS['REDD'] = 'redd/low_freq'
DATASETS['Pecan Street'] = 'pecan_1min'
DATASETS['AMDds'] = 'ampds'
DATASETS['iAWE'] = 'iawe'

for dataset_name, dataset in DATASETS:
    # Choose first home from each


# Maps from short col name to human-readable name
COLUMNS = OrderedDict()
COLUMNS['n_appliances'] = """number of\\\\appliances"""
COLUMNS['energy_submetered'] = """% energy\\\\submetered"""
COLUMNS['dropout_rate'] = 'dropout rate'
COLUMNS['dropout_rate_ignoring_gaps'] = """dropout rate\\\\(ignoring gaps)"""
COLUMNS['uptime'] = """mains uptime\\\\per building\\\\(days)"""
COLUMNS[
    'prop_timeslices'] = """% timeslices\\\\where energy\\\\submetered > 70%"""

for key, value in COLUMNS.iteritems():
    COLUMNS[key] = """\textbf{\specialcell[h]{""" + value + """}}"""

df = pd.DataFrame(index=DATASETS.keys(), columns=COLUMNS.values())

for ds_name in DATASETS.iterkeys():
    print('Calculating stats for', ds_name)
    dataset = dataset_objs[ds_name]
    ds_stats = dataset.descriptive_stats()
    for col_short, col_long in COLUMNS.iteritems():
        s = """\specialcell{"""
        s += summary_stats_string(ds_stats[col_short], sep="""\\\\""",
                                  stat_strings=['min', 'mean', 'max']).replace(' ', '')
        s += """}"""
        df[col_long][ds_name] = s

print("------------LATEX BEGINS-----------------")
latex = df.to_latex()
for str_to_replace in ['midrule', 'toprule', 'bottomrule']:
    latex = latex.replace(str_to_replace, 'hline')
print(latex)
print("------------LATEX ENDS-------------------")
