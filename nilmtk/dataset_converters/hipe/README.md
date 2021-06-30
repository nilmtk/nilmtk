# Industrial Load Disaggregation

This repository houses the data converter from the paper

- F. Kalinke, P. Bielski, S. Singh, E. Fouché, and Klemens Böhm:
[An Evaluation of NILM Approaches on Industrial Energy-Consumption Data.](https://doi.org/10.1145/3447555.3464863) e-Energy 2021: 239-243

Please cite the paper if you use the data set.

## Dependencies

We use nilmtk 4.1, pandas, numpy, matplotlib.

## Data

The dataset converter for the HIPE data set has been made available within the [NILMTK toolkit](https://github.com/nilmtk/nilmtk/tree/master/nilmtk/dataset_converters). Please refer to the README therein for usage instructions.

To create the NILMTK-compatible HIPE data:

1. Download the [data](https://www.energystatusdata.kit.edu/hipe.php), either one week or three months.
1. Run the `convert_hipe.py` script. `<infolder>` refers to the folder with the HIPE CSV files.
    
    `python convert_hipe.py <infolder> <outfile>`
	
1. Check the example notebooks.

## Notebooks

Examples reside in the notebooks:

- `examples/01_test_conversion.ipynb`
- `examples/02_run_experiments.ipynb`


