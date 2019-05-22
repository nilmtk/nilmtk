# ! python
# coding: utf-8
import time
import os
import argparse
import glob
from tqdm import tqdm
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert.preprocessors.execute import CellExecutionError
flag=1 # to check whether all notebooks executed successfully without error
# Parse args
parser = argparse.ArgumentParser(description="Runs a set of Jupyter \
                                              notebooks.")

file_text='C:/Users/Rajat/Desktop/NILMTK/nilmtk/docs/manual/user_guide/*.ipynb'

# please add path to location where all .ipynb files are located.

""" Notebook file(s) to be run, e.g. '*.ipynb' (default),
'my_nb1.ipynb', 'my_nb1.ipynb my_nb2.ipynb', 'my_dir/*.ipynb'
"""
parser.add_argument('file_list', metavar='F', type=str, nargs='*', 
    help=file_text)
parser.add_argument('-t', '--timeout', help='Length of time (in secs) a cell \
    can run before raising TimeoutError (default 600).', default=600, 
    required=False)
parser.add_argument('-p', '--run-path', help='The path the notebook will be \
    run from (default pwd).', default='.', required=False)
args = parser.parse_args()
print('Args:', args)
if not args.file_list: # Default file_list
    args.file_list = glob.glob('*.ipynb')

# Check list of notebooks
notebooks = []
print('Notebooks to run:')
for f in args.file_list:
    # Find notebooks but not notebooks previously output from this script
    if f.endswith('.ipynb') and not f.endswith('_out.ipynb'):
        print(f[:-6])
        notebooks.append(f[:-6]) # Want the filename without '.ipynb'

# Execute notebooks and output
num_notebooks = len(notebooks)
print('*****')
time_all=[]
for i, n in tqdm(enumerate(notebooks)):
    n_out = n + '_out'
    with open(n + '.ipynb') as f:
        start = time. time()
        nb = nbformat.read(f, as_version=4)
        ep = ExecutePreprocessor(timeout=int(args.timeout), kernel_name='python')
        try:
            print('Running', n, ':', i, '/', num_notebooks)
            out = ep.preprocess(nb, {'metadata': {'path': args.run_path}})
        except CellExecutionError:
            out = None
            flag=0
             
if flag==0:
   print('All Notebooks didnt run successfully')
else:
    print('All Notebooks ran successfully')    
  
