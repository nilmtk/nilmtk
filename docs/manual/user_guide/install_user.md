# Installing NILMTK

We recommend using [Anaconda](https://store.continuum.io/cshop/anaconda/), which bundles together most of the required packages. NILMTK requires Python 3.6+ due to the module it depends upon.

After Anaconda has been installed, open up the terminal (Unix) or Anaconda prompt (Windows):

1.  NILMTK should work fine in the base environment, but we recommend creating a new environment where NILMTK and related dependecies are installed.

	```bash
	conda create --name nilmtk-env 
	```

2. Add conda-forge to list of channels to be searched for packages.
	```bash
	conda config --add channels conda-forge
	```

3. Activate the new *nilmtk-env* environment.

	```bash
	conda activate nilmtk-env
	```

4. Install the NILMTK package

	```bash
	conda install -c nilmtk nilmtk
	```

5. The installed package import for python/ ipython can be  tested in the terminal using the following command:
	```bash
	python -c "import nilmtk"
	```
	or	
	```bash
	ipython -c "import nilmtk"
	```
	> Note: This might show DepreciationWarning due to the *imp* module. That will be fixed in a future release.

	* Alternatively, you can also run your IDE in *nilmtk-env* from: Anaconda Navigator > "Applications on" dropdown > nilmtk-env
	To check the current environment variables,

		```python
		import sys
		print(sys.executable)
		print(sys.version)
		```
		You will see an output similar to:
		```
		/home/ayush/anaconda3/envs/nilmtk-env/bin/python
		3.6.7 | packaged by conda-forge | (default, Feb 28 2019, 09:07:38) 
		[GCC 7.3.0]
		```
6. Run your Python IDE from this environment, for example:

	```bash
	jupyter notebook
	```
	or

	```bash
	spyder
	```

7. Import NILMTK in the IDE:

	```python
	import nilmtk
	```
	The package modules can now be used.
8. To deactivate this environment,

	```bash
	conda deactivate
	```
    
We recommend checking the [Anaconda documentation about environments](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) if the concept is new to you.


# Conda development snapshots

If you want to try out tagged development versions, you can follow the normal installation guide but use the following command for the NILMTK installation (step 4):

```bash
    conda install -c nilmtk -c nilmtk/label/dev nilmtk
```
